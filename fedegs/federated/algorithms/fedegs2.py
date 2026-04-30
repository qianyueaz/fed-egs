import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.models import SmallCNN, build_teacher_model, estimate_model_flops, load_teacher_checkpoint, model_memory_mb


@dataclass
class PublicKnowledge:
    sample_indices: List[int]
    soft_predictions: torch.Tensor
    features: torch.Tensor
    uncertainty_weights: torch.Tensor


@dataclass
class AggregatedPublicKnowledge:
    sample_indices: List[int]
    soft_predictions: torch.Tensor
    features: torch.Tensor
    uncertainty_weights: torch.Tensor
    raw_weight_sums: torch.Tensor
    client_reliability: Dict[str, float]


@dataclass
class FedEGS2ClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_knowledge: PublicKnowledge


class DistilledGeneralModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        knowledge_dim: int,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        backbone = build_teacher_model(num_classes=num_classes)
        self.initialized_from_teacher = False
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu")
                load_teacher_checkpoint(backbone, state)
                self.initialized_from_teacher = True
                LOGGER.info("Initialized shared general model from %s", checkpoint)
            else:
                LOGGER.warning("General model checkpoint not found at %s. Using random initialization.", checkpoint)

        self.num_classes = num_classes
        self.feature_dim = backbone.fc.in_features
        self.knowledge_dim = knowledge_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier.load_state_dict(backbone.fc.state_dict())
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if knowledge_dim == self.feature_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(self.feature_dim, knowledge_dim, bias=False)
            self._initialize_projector()

    def _initialize_projector(self) -> None:
        weight = torch.zeros((self.knowledge_dim, self.feature_dim))
        diagonal = min(self.knowledge_dim, self.feature_dim)
        weight[:diagonal, :diagonal] = torch.eye(diagonal)
        self.projector.weight.data.copy_(weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(features)
        return F.normalize(embedding, dim=1)

    def forward_with_embedding(self, x: torch.Tensor):
        features = self.forward_features(x)
        logits = self.classify_features(features)
        embedding = self.project_features(features)
        return features, embedding, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, logits = self.forward_with_embedding(x)
        return logits


class FedEGS2Client(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, num_classes: int, device: str, config, data_module) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)

    def train(
        self,
        general_model: DistilledGeneralModel,
        public_loader: DataLoader,
        distillation_focus: bool = False,
    ) -> FedEGS2ClientUpdate:
        train_loader = self.data_module.make_loader(self.dataset, shuffle=True)
        self._refresh_expert_from_general(public_loader, general_model, distillation_focus=distillation_focus)
        loss = self._train_local_expert(train_loader, general_model, distillation_focus=distillation_focus)
        public_knowledge = self._extract_public_knowledge(public_loader)
        return FedEGS2ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            public_knowledge=public_knowledge,
        )

    def _refresh_expert_from_general(
        self,
        loader: DataLoader,
        general_model: DistilledGeneralModel,
        distillation_focus: bool = False,
    ) -> None:
        refresh_epochs = max(int(self._effective_refresh_epochs(distillation_focus)), 0)
        if refresh_epochs == 0:
            return

        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr
            * self._effective_local_lr_scale(distillation_focus)
            * float(self.config.federated.expert_refresh_lr_scale),
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        kd_temperature = self.config.federated.client_kd_temperature
        feature_hint_weight = self._effective_feature_hint_weight(distillation_focus)

        self.expert_model.train()
        general_model.eval()
        for _ in range(refresh_epochs):
            for images, _, _ in loader:
                images = images.to(self.device)

                with torch.no_grad():
                    _, teacher_embeddings, teacher_logits = general_model.forward_with_embedding(images)
                    teacher_probs = torch.softmax(teacher_logits / kd_temperature, dim=1)
                    refresh_weights = self._build_refresh_weights(teacher_probs)

                optimizer.zero_grad(set_to_none=True)
                _, student_embeddings, student_logits = self.expert_model.forward_with_embedding(images)
                kd_sample_weights = self._boost_hard_subset_weights(
                    refresh_weights,
                    boost=float(self.config.federated.hard_subset_kd_boost),
                )
                hint_sample_weights = self._boost_hard_subset_weights(
                    refresh_weights,
                    boost=float(self.config.federated.hard_subset_hint_boost),
                )
                kd_loss = self._weighted_kd_loss(student_logits, teacher_probs, kd_sample_weights, kd_temperature)
                feature_hint_loss = self._weighted_feature_hint_loss(
                    student_embeddings,
                    teacher_embeddings,
                    hint_sample_weights,
                )
                loss = kd_loss + feature_hint_weight * feature_hint_loss
                loss.backward()
                optimizer.step()

    def _train_local_expert(
        self,
        loader: DataLoader,
        general_model: DistilledGeneralModel,
        distillation_focus: bool = False,
    ) -> float:
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr * self._effective_local_lr_scale(distillation_focus),
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        kd_weight = self._effective_kd_weight(distillation_focus)
        kd_temperature = self.config.federated.client_kd_temperature
        feature_hint_weight = self._effective_feature_hint_weight(distillation_focus)
        hard_weight = self.config.federated.client_hard_weight

        self.expert_model.train()
        general_model.eval()
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    _, teacher_embeddings, teacher_logits = general_model.forward_with_embedding(images)
                    teacher_probs = torch.softmax(teacher_logits / kd_temperature, dim=1)
                    teacher_predictions = teacher_probs.argmax(dim=1)

                optimizer.zero_grad(set_to_none=True)
                _, student_embeddings, student_logits = self.expert_model.forward_with_embedding(images)
                student_probs = torch.softmax(student_logits.detach(), dim=1)
                student_predictions = student_probs.argmax(dim=1)
                sample_weights = self._build_local_hard_weights(
                    teacher_probs=teacher_probs,
                    teacher_predictions=teacher_predictions,
                    student_probs=student_probs,
                    student_predictions=student_predictions,
                    targets=targets,
                    hard_weight=hard_weight,
                )
                kd_sample_weights = self._boost_hard_subset_weights(
                    sample_weights,
                    boost=float(self.config.federated.hard_subset_kd_boost),
                )
                hint_sample_weights = self._boost_hard_subset_weights(
                    sample_weights,
                    boost=float(self.config.federated.hard_subset_hint_boost),
                )

                ce_loss = self._weighted_mean(criterion(student_logits, targets), sample_weights)
                kd_loss = self._weighted_kd_loss(student_logits, teacher_probs, kd_sample_weights, kd_temperature)
                feature_hint_loss = self._weighted_feature_hint_loss(
                    student_embeddings,
                    teacher_embeddings,
                    hint_sample_weights,
                )

                loss = ce_loss + kd_weight * kd_loss + feature_hint_weight * feature_hint_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * weights) / weights.sum().clamp_min(1e-8)

    def _effective_kd_weight(self, distillation_focus: bool) -> float:
        if distillation_focus:
            return float(self.config.federated.freeze_client_kd_weight)
        return float(self.config.federated.client_kd_weight)

    def _effective_feature_hint_weight(self, distillation_focus: bool) -> float:
        if distillation_focus:
            return float(self.config.federated.freeze_client_feature_hint_weight)
        return float(self.config.federated.client_feature_hint_weight)

    def _effective_refresh_epochs(self, distillation_focus: bool) -> int:
        if distillation_focus:
            return int(self.config.federated.freeze_expert_refresh_epochs)
        return int(self.config.federated.expert_refresh_epochs)

    def _effective_local_lr_scale(self, distillation_focus: bool) -> float:
        if distillation_focus:
            return float(self.config.federated.freeze_local_lr_scale)
        return 1.0

    def _boost_hard_subset_weights(self, base_weights: torch.Tensor, boost: float) -> torch.Tensor:
        boost = max(float(boost), 0.0)
        ratio = min(max(float(self.config.federated.hard_subset_ratio), 0.0), 1.0)
        if boost <= 0.0 or ratio <= 0.0 or base_weights.numel() == 0:
            return base_weights

        k = max(1, int(math.ceil(base_weights.numel() * ratio)))
        top_values, _ = torch.topk(base_weights.detach(), k=k)
        cutoff = top_values[-1]
        hard_mask = (base_weights.detach() >= cutoff).float()
        return base_weights * (1.0 + boost * hard_mask)

    def _build_refresh_weights(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        hard_weight = float(self.config.federated.client_hard_weight)
        margin_threshold = float(self.config.federated.client_hard_margin_threshold)
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        uncertainty = 1.0 - teacher_confidence
        low_margin = (teacher_margin < margin_threshold).float()
        weights = 1.0 + hard_weight * (0.75 * uncertainty + 0.25 * low_margin)
        return weights.clamp_min(1.0)

    def _build_local_hard_weights(
        self,
        teacher_probs: torch.Tensor,
        teacher_predictions: torch.Tensor,
        student_probs: torch.Tensor,
        student_predictions: torch.Tensor,
        targets: torch.Tensor,
        hard_weight: float,
    ) -> torch.Tensor:
        focus_power = float(self.config.federated.client_hard_focus_power)
        margin_threshold = float(self.config.federated.client_hard_margin_threshold)
        extra_weight = float(self.config.federated.client_hard_extra_weight)

        teacher_true_probs = teacher_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        student_true_probs = student_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        student_topk = torch.topk(student_probs, k=min(2, student_probs.size(1)), dim=1)
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_true_probs)
        if student_topk.values.size(1) > 1:
            student_margin = student_topk.values[:, 0] - student_topk.values[:, 1]
        else:
            student_margin = torch.ones_like(student_true_probs)

        teacher_hardness = 1.0 - teacher_true_probs
        student_hardness = 1.0 - student_true_probs
        margin_hardness = 1.0 - 0.5 * (teacher_margin + student_margin)
        hardness = (
            0.45 * teacher_hardness
            + 0.35 * student_hardness
            + 0.20 * margin_hardness.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        focused_hardness = hardness.pow(focus_power)
        disagreement = (teacher_predictions != student_predictions).float()
        extreme_hard = (
            (teacher_true_probs < 0.45)
            | (student_true_probs < 0.45)
            | (teacher_margin < margin_threshold)
            | (student_margin < margin_threshold)
            | disagreement.bool()
        ).float()

        sample_weights = (
            1.0
            + hard_weight * (0.75 * focused_hardness + 0.25 * disagreement)
            + extra_weight * extreme_hard
        )
        return sample_weights.clamp_min(1.0)

    def _weighted_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        sample_weights: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        per_sample_kl = torch.sum(
            teacher_probs * (teacher_probs.clamp_min(1e-8).log() - student_log_probs),
            dim=1,
        ) * (temperature ** 2)
        return self._weighted_mean(per_sample_kl, sample_weights)

    def _weighted_feature_hint_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_mse = torch.mean((student_embeddings - teacher_embeddings) ** 2, dim=1)
        return self._weighted_mean(per_sample_mse, sample_weights)

    def _extract_public_knowledge(self, loader: DataLoader) -> PublicKnowledge:
        temperature = self.config.federated.distill_temperature
        min_uncertainty_weight = self.config.federated.min_uncertainty_weight
        log_num_classes = math.log(float(self.config.model.num_classes))

        sample_indices: List[torch.Tensor] = []
        soft_predictions: List[torch.Tensor] = []
        features: List[torch.Tensor] = []
        uncertainty_weights: List[torch.Tensor] = []

        self.expert_model.eval()
        with torch.no_grad():
            for images, _, indices in loader:
                images = images.to(self.device)
                _, embeddings, logits = self.expert_model.forward_with_embedding(images)
                probs = torch.softmax(logits / temperature, dim=1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / max(log_num_classes, 1e-8)
                weights = torch.exp(-entropy).clamp_min(min_uncertainty_weight)

                sample_indices.append(indices.detach().cpu())
                soft_predictions.append(probs.detach().cpu())
                features.append(embeddings.detach().cpu())
                uncertainty_weights.append(weights.detach().cpu())

        merged_indices = torch.cat(sample_indices, dim=0)
        merged_predictions = torch.cat(soft_predictions, dim=0)
        merged_features = torch.cat(features, dim=0)
        merged_weights = torch.cat(uncertainty_weights, dim=0)

        order = torch.argsort(merged_indices)
        return PublicKnowledge(
            sample_indices=merged_indices[order].tolist(),
            soft_predictions=merged_predictions[order],
            features=F.normalize(merged_features[order], dim=1),
            uncertainty_weights=merged_weights[order],
        )


class FedEGS2Server(BaseFederatedServer):
    def __init__(
        self,
        config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices,
        writer=None,
        public_dataset: Dataset | None = None,
    ) -> None:
        if public_dataset is None or len(public_dataset) == 0:
            raise ValueError("fedegs2 now requires a non-empty public_dataset for server-side distillation.")

        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )
        self.general_model = DistilledGeneralModel(
            num_classes=config.model.num_classes,
            knowledge_dim=config.model.knowledge_dim,
            checkpoint_path=(
                config.dataset.difficulty_checkpoint
                if bool(config.federated.general_init_from_teacher)
                else None
            ),
        ).to(self.device)
        if self.general_model.initialized_from_teacher:
            self.anchor_model = copy.deepcopy(self.general_model).to(self.device)
            self.anchor_model.eval()
            for parameter in self.anchor_model.parameters():
                parameter.requires_grad_(False)
            LOGGER.info("fedegs2 general model starts from teacher initialization.")
        else:
            self.anchor_model = None
            LOGGER.info("fedegs2 general model uses cold-start initialization from client knowledge only.")
        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
        self.clients = {
            client_id: FedEGS2Client(
                client_id,
                dataset,
                config.model.num_classes,
                config.federated.device,
                config,
                data_module,
            )
            for client_id, dataset in client_datasets.items()
        }
        self.public_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)
        self.public_targets_by_index = self._build_public_target_lookup()
        self.distill_optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            weight_decay=self.config.federated.local_weight_decay,
        )
        self.current_public_knowledge: Optional[AggregatedPublicKnowledge] = None
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.best_general_snapshot: Optional[Dict[str, object]] = None
        self.rounds_since_best = 0
        self.general_distillation_frozen = False
        self.last_public_teacher_gaps: Dict[str, float] = {}
        self.last_public_fallback_rates: Dict[str, float] = {}
        base_confidence = float(config.inference.confidence_threshold)
        base_margin = float(config.inference.route_distance_threshold)
        self.client_confidence_thresholds = {
            client_id: base_confidence for client_id in client_datasets.keys()
        }
        self.client_margin_thresholds = {
            client_id: base_margin for client_id in client_datasets.keys()
        }

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegs2 round %d sampled clients=%s", round_idx, sampled_ids)
            updates: List[FedEGS2ClientUpdate] = []

            for client_id in sampled_ids:
                updates.append(
                    self.clients[client_id].train(
                        self.general_model,
                        self.public_loader,
                        distillation_focus=self.general_distillation_frozen,
                    )
                )

            self._update_personalized_routing_thresholds(updates)
            self.current_public_knowledge = self._aggregate_public_knowledge(updates)
            distill_stats = self._distill_general_model(self.current_public_knowledge, round_idx)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs2-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs2-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs2-routed")

            avg_expert_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            aggregate = routed_eval["aggregate"]
            macro = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate["invocation_rate"],
                mode="routed",
            )
            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=avg_expert_loss,
                routed_accuracy=macro["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
            )

            public_weight_mean = float(self.current_public_knowledge.raw_weight_sums.mean().item())
            public_weight_std = float(self.current_public_knowledge.raw_weight_sums.std(unbiased=False).item())
            public_weight_min = float(self.current_public_knowledge.raw_weight_sums.min().item())
            public_weight_max = float(self.current_public_knowledge.raw_weight_sums.max().item())
            client_reliabilities = list(self.current_public_knowledge.client_reliability.values())
            client_weight_mean = float(sum(client_reliabilities) / max(len(client_reliabilities), 1))
            client_weight_min = float(min(client_reliabilities)) if client_reliabilities else 0.0
            client_weight_max = float(max(client_reliabilities)) if client_reliabilities else 0.0
            confidence_threshold_mean = float(
                sum(self.client_confidence_thresholds.values()) / max(len(self.client_confidence_thresholds), 1)
            )
            margin_threshold_mean = float(
                sum(self.client_margin_thresholds.values()) / max(len(self.client_margin_thresholds), 1)
            )
            public_teacher_gap_mean = float(
                sum(self.last_public_teacher_gaps.values()) / max(len(self.last_public_teacher_gaps), 1)
            )
            public_fallback_rate_mean = float(
                sum(self.last_public_fallback_rates.values()) / max(len(self.last_public_fallback_rates), 1)
            )
            LOGGER.info(
                "fedegs2 round %d | expert_loss=%.4f | distill_loss=%.4f | distill_scale=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | invocation=%.4f | savings=%.4f | public_weight=%.4f | public_teacher_gap=%.4f | public_fallback=%.4f",
                round_idx,
                avg_expert_loss,
                distill_stats["total_loss"],
                distill_stats["scale"],
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                compute_profile["savings_ratio"],
                public_weight_mean,
                public_teacher_gap_mean,
                public_fallback_rate_mean,
            )
            LOGGER.info(
                "fedegs2 distill round %d | client_weight_mean=%.4f | client_weight_min=%.4f | client_weight_max=%.4f | public_weight_std=%.4f | public_weight_min=%.4f | public_weight_max=%.4f | ce=%.4f | logit=%.4f | feature=%.4f | relation=%.4f | anchor=%.4f",
                round_idx,
                client_weight_mean,
                client_weight_min,
                client_weight_max,
                public_weight_std,
                public_weight_min,
                public_weight_max,
                distill_stats["ce_loss"],
                distill_stats["logit_loss"],
                distill_stats["feature_loss"],
                distill_stats["relation_loss"],
                distill_stats["anchor_loss"],
            )
            LOGGER.info(
                "fedegs2 auxiliary round %d | expert_acc=%.4f | general_acc=%.4f",
                round_idx,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegs2", avg_expert_loss, round_idx)
                self.writer.add_scalar("distill_loss/fedegs2", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar("distill_scale/fedegs2", distill_stats["scale"], round_idx)
                self.writer.add_scalar(
                    "general_distillation_frozen/fedegs2",
                    float(self.general_distillation_frozen),
                    round_idx,
                )
                self.writer.add_scalar("public_weight/fedegs2", public_weight_mean, round_idx)
                self.writer.add_scalar("public_weight_std/fedegs2", public_weight_std, round_idx)
                self.writer.add_scalar("client_reliability_mean/fedegs2", client_weight_mean, round_idx)
                self.writer.add_scalar("route_confidence_threshold_mean/fedegs2", confidence_threshold_mean, round_idx)
                self.writer.add_scalar("route_margin_threshold_mean/fedegs2", margin_threshold_mean, round_idx)
                self.writer.add_scalar("public_teacher_gap_mean/fedegs2", public_teacher_gap_mean, round_idx)
                self.writer.add_scalar("public_fallback_rate_mean/fedegs2", public_fallback_rate_mean, round_idx)
                self.writer.add_scalar("distill_ce_loss/fedegs2", distill_stats["ce_loss"], round_idx)
                self.writer.add_scalar("distill_logit_loss/fedegs2", distill_stats["logit_loss"], round_idx)
                self.writer.add_scalar("distill_feature_loss/fedegs2", distill_stats["feature_loss"], round_idx)
                self.writer.add_scalar("distill_relation_loss/fedegs2", distill_stats["relation_loss"], round_idx)
                self.writer.add_scalar("distill_anchor_loss/fedegs2", distill_stats["anchor_loss"], round_idx)
                self._log_auxiliary_accuracy_metrics(
                "fedegs2",
                round_idx,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

            self._log_round_metrics("fedegs2", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best_general_snapshot(
                round_idx=round_idx,
                general_accuracy=general_eval["macro"]["accuracy"],
            )
            improved = self._maybe_update_best_snapshot(
                round_idx=round_idx,
                round_metrics=round_metrics,
                expert_accuracy=expert_eval["macro"]["accuracy"],
                general_accuracy=general_eval["macro"]["accuracy"],
            )
            self._update_general_freeze_state(improved, round_idx)

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best_snapshot()
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        route_export_path = self._build_route_export_path("fedegs2_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            prefix="fedegs2_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs2_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs2_final_general")
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = 0
        if self.best_snapshot is not None:
            final_loss = float(self.best_snapshot["avg_client_loss"])
            best_round = int(self.best_snapshot["round_idx"])

        routed_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            routed_eval["aggregate"]["invocation_rate"],
            mode="routed",
        )
        expert_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            0.0,
            mode="expert_only",
        )
        general_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            1.0,
            mode="general_only",
        )

        public_weight_mean = 0.0
        if self.current_public_knowledge is not None:
            public_weight_mean = float(self.current_public_knowledge.raw_weight_sums.mean().item())

        return {
            "algorithm": "fedegs2",
            "metrics": {
                "accuracy": routed_eval["macro"]["accuracy"],
                "personalized_accuracy": routed_eval["macro"]["accuracy"],
                "weighted_accuracy": routed_eval["aggregate"]["accuracy"],
                "global_accuracy": routed_eval["macro"]["accuracy"],
                "local_accuracy": routed_eval["macro"]["accuracy"],
                "routed_accuracy": routed_eval["macro"]["accuracy"],
                "hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "hard_sample_recall": routed_eval["aggregate"]["hard_recall"],
                "routed_hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["macro"]["accuracy"],
                "general_only_accuracy": general_eval["macro"]["accuracy"],
                "expert_only_recall_macro": expert_eval["aggregate"]["recall_macro"],
                "general_only_recall_macro": general_eval["aggregate"]["recall_macro"],
                "public_dataset_size": len(self.public_dataset),
                "mean_public_uncertainty_weight": public_weight_mean,
                "final_training_loss": final_loss,
                "best_round": best_round,
            },
            "client_metrics": {
                "routed": routed_eval["clients"],
                "expert_only": expert_eval["clients"],
                "general_only": general_eval["clients"],
            },
            "group_metrics": {
                "routed": routed_eval["groups"],
                "expert_only": expert_eval["groups"],
                "general_only": general_eval["groups"],
            },
            "compute": {
                "routed": routed_compute,
                "expert_only": expert_compute,
                "general_only": general_compute,
            },
            "memory_mb": {
                "expert": model_memory_mb(self.reference_expert),
                "general": model_memory_mb(self.general_model),
            },
            "artifacts": {
                "route_csv": str(route_export_path),
            },
        }

    def _aggregate_public_knowledge(self, updates: List[FedEGS2ClientUpdate]) -> AggregatedPublicKnowledge:
        if not updates:
            raise RuntimeError("No client updates were provided for public distillation.")

        reference_indices = updates[0].public_knowledge.sample_indices
        num_public = len(reference_indices)
        num_classes = self.config.model.num_classes
        knowledge_dim = self.config.model.knowledge_dim
        index_to_position = {sample_index: idx for idx, sample_index in enumerate(reference_indices)}

        prediction_sums = torch.zeros((num_public, num_classes), dtype=torch.float32)
        feature_sums = torch.zeros((num_public, knowledge_dim), dtype=torch.float32)
        weight_sums = torch.zeros(num_public, dtype=torch.float32)
        client_reliability: Dict[str, float] = {}

        for update in updates:
            knowledge = update.public_knowledge
            reliability = self._compute_client_reliability(update.loss)
            client_reliability[update.client_id] = reliability
            client_weight = float(update.num_samples) * reliability
            for row_idx, sample_index in enumerate(knowledge.sample_indices):
                position = index_to_position[sample_index]
                sample_weight = client_weight * float(knowledge.uncertainty_weights[row_idx].item())
                prediction_sums[position] += knowledge.soft_predictions[row_idx] * sample_weight
                feature_sums[position] += knowledge.features[row_idx] * sample_weight
                weight_sums[position] += sample_weight

        normalized_predictions = torch.zeros_like(prediction_sums)
        normalized_features = torch.zeros_like(feature_sums)
        safe_weights = weight_sums.clamp_min(1e-8)
        for position in range(num_public):
            normalized_predictions[position] = prediction_sums[position] / safe_weights[position]
            normalized_predictions[position] = normalized_predictions[position] / normalized_predictions[position].sum().clamp_min(1e-8)
            normalized_features[position] = F.normalize(
                (feature_sums[position] / safe_weights[position]).unsqueeze(0),
                dim=1,
            ).squeeze(0)

        uncertainty_weights = weight_sums / weight_sums.mean().clamp_min(1e-8)
        uncertainty_weights = uncertainty_weights.clamp_min(self.config.federated.min_uncertainty_weight)

        return AggregatedPublicKnowledge(
            sample_indices=reference_indices,
            soft_predictions=normalized_predictions,
            features=normalized_features,
            uncertainty_weights=uncertainty_weights,
            raw_weight_sums=weight_sums,
            client_reliability=client_reliability,
        )

    def _distill_general_model(self, public_knowledge: AggregatedPublicKnowledge, round_idx: int) -> Dict[str, float]:
        if self.general_distillation_frozen:
            return {
                "total_loss": 0.0,
                "ce_loss": 0.0,
                "logit_loss": 0.0,
                "feature_loss": 0.0,
                "relation_loss": 0.0,
                "anchor_loss": 0.0,
                "scale": 0.0,
            }

        temperature = self.config.federated.distill_temperature
        logit_weight = self.config.federated.logit_align_weight
        feature_weight = self.config.federated.feature_align_weight
        relation_weight = self.config.federated.relation_align_weight
        anchor_weight = (
            float(self.config.federated.general_anchor_weight)
            if self.anchor_model is not None
            else 0.0
        )
        criterion = torch.nn.CrossEntropyLoss()
        index_to_position = {sample_index: idx for idx, sample_index in enumerate(public_knowledge.sample_indices)}
        distill_scale = self._compute_distill_scale(round_idx)

        if distill_scale <= 0.0:
            return {
                "total_loss": 0.0,
                "ce_loss": 0.0,
                "logit_loss": 0.0,
                "feature_loss": 0.0,
                "relation_loss": 0.0,
                "anchor_loss": 0.0,
                "scale": 0.0,
            }

        self.general_model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_logit_loss = 0.0
        total_feature_loss = 0.0
        total_relation_loss = 0.0
        total_anchor_loss = 0.0
        total_batches = 0
        for _ in range(self.config.federated.distill_epochs):
            for images, targets, indices in self.public_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                positions = torch.tensor(
                    [index_to_position[int(sample_index)] for sample_index in indices.tolist()],
                    device=self.device,
                    dtype=torch.long,
                )

                target_probs = public_knowledge.soft_predictions[positions.cpu()].to(self.device)
                target_features = public_knowledge.features[positions.cpu()].to(self.device)
                sample_weights = public_knowledge.uncertainty_weights[positions.cpu()].to(self.device)

                self.distill_optimizer.zero_grad(set_to_none=True)
                _, student_embeddings, student_logits = self.general_model.forward_with_embedding(images)

                ce_loss = criterion(student_logits, targets)
                logit_loss = self._weighted_logit_loss(student_logits, target_probs, sample_weights, temperature)
                feature_loss = self._weighted_feature_loss(student_embeddings, target_features, sample_weights)
                relation_loss = self._relation_loss(student_embeddings, target_features, sample_weights)
                if anchor_weight > 0 and self.anchor_model is not None:
                    with torch.no_grad():
                        anchor_logits = self.anchor_model(images)
                        anchor_probs = torch.softmax(anchor_logits / temperature, dim=1)
                    anchor_loss = self._weighted_logit_loss(student_logits, anchor_probs, sample_weights, temperature)
                else:
                    anchor_loss = ce_loss.new_zeros(())

                loss = (
                    ce_loss
                    + distill_scale
                    * (
                        logit_weight * logit_loss
                        + feature_weight * feature_loss
                        + relation_weight * relation_loss
                        + anchor_weight * anchor_loss
                    )
                )
                loss.backward()
                self.distill_optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_ce_loss += float(ce_loss.detach().cpu().item())
                total_logit_loss += float(logit_loss.detach().cpu().item())
                total_feature_loss += float(feature_loss.detach().cpu().item())
                total_relation_loss += float(relation_loss.detach().cpu().item())
                total_anchor_loss += float(anchor_loss.detach().cpu().item())
                total_batches += 1

        denominator = max(total_batches, 1)
        return {
            "total_loss": total_loss / denominator,
            "ce_loss": total_ce_loss / denominator,
            "logit_loss": total_logit_loss / denominator,
            "feature_loss": total_feature_loss / denominator,
            "relation_loss": total_relation_loss / denominator,
            "anchor_loss": total_anchor_loss / denominator,
            "scale": distill_scale,
        }

    def _compute_client_reliability(self, loss: float) -> float:
        if not math.isfinite(loss):
            return self.config.federated.min_client_reliability
        reliability = 1.0 / (1.0 + max(loss, 0.0))
        return max(self.config.federated.min_client_reliability, reliability)

    def _compute_distill_scale(self, round_idx: int) -> float:
        warmup_rounds = max(int(self.config.federated.general_warmup_rounds), 0)
        ramp_rounds = max(int(self.config.federated.general_distill_ramp_rounds), 0)
        max_scale = max(float(self.config.federated.general_distill_max_scale), 0.0)
        if round_idx <= warmup_rounds:
            return 0.0
        if ramp_rounds == 0:
            return max_scale
        return min(max_scale, float(round_idx - warmup_rounds) / float(ramp_rounds))

    def _clone_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _build_public_target_lookup(self) -> Dict[int, int]:
        targets_by_index: Dict[int, int] = {}
        for _, targets, indices in self.public_loader:
            for sample_index, target in zip(indices.tolist(), targets.tolist()):
                targets_by_index[int(sample_index)] = int(target)
        return targets_by_index

    def _predict_public_general_labels(self) -> Dict[int, int]:
        predictions_by_index: Dict[int, int] = {}
        self.general_model.eval()
        with torch.no_grad():
            for images, _, indices in self.public_loader:
                images = images.to(self.device)
                logits = self.general_model(images)
                predictions = logits.argmax(dim=1).detach().cpu().tolist()
                for sample_index, prediction in zip(indices.tolist(), predictions):
                    predictions_by_index[int(sample_index)] = int(prediction)
        return predictions_by_index

    def _fallback_floor_for_client(self, client_id: str) -> float:
        if client_id.startswith("complex_"):
            return float(self.config.inference.complex_client_fallback_floor)
        return float(self.config.inference.simple_client_fallback_floor)

    def _maybe_update_best_general_snapshot(self, round_idx: int, general_accuracy: float) -> None:
        if self.best_general_snapshot is not None:
            previous_accuracy = float(self.best_general_snapshot["general_accuracy"])
            if general_accuracy <= previous_accuracy + 1e-8:
                return

        self.best_general_snapshot = {
            "round_idx": round_idx,
            "general_accuracy": general_accuracy,
            "general_model_state": self._clone_model_state(self.general_model),
        }
        LOGGER.info(
            "Updated fedegs2 best general checkpoint | round=%d | general_acc=%.4f",
            round_idx,
            general_accuracy,
        )

    def _maybe_update_best_snapshot(
        self,
        round_idx: int,
        round_metrics: RoundMetrics,
        expert_accuracy: float,
        general_accuracy: float,
    ) -> bool:
        if self.best_snapshot is None:
            is_better = True
        else:
            previous_routed = float(self.best_snapshot["routed_accuracy"])
            previous_general = float(self.best_snapshot["general_accuracy"])
            is_better = round_metrics.routed_accuracy > previous_routed + 1e-8
            if not is_better and abs(round_metrics.routed_accuracy - previous_routed) <= 1e-8:
                is_better = general_accuracy > previous_general + 1e-8

        if not is_better:
            return False

        self.best_snapshot = {
            "round_idx": round_idx,
            "routed_accuracy": round_metrics.routed_accuracy,
            "general_accuracy": general_accuracy,
            "expert_accuracy": expert_accuracy,
            "avg_client_loss": round_metrics.avg_client_loss,
            "general_model_state": self._clone_model_state(self.general_model),
            "client_expert_states": {
                client_id: self._clone_model_state(client.expert_model)
                for client_id, client in self.clients.items()
            },
            "client_confidence_thresholds": dict(self.client_confidence_thresholds),
            "client_margin_thresholds": dict(self.client_margin_thresholds),
        }
        LOGGER.info(
            "Updated fedegs2 best checkpoint | round=%d | routed_acc=%.4f | general_acc=%.4f | expert_acc=%.4f",
            round_idx,
            round_metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )
        return True

    def _update_personalized_routing_thresholds(self, updates: List[FedEGS2ClientUpdate]) -> None:
        step = max(float(self.config.inference.personalized_threshold_step), 0.0)
        margin_step = max(float(self.config.inference.personalized_margin_step), 0.0)
        target_strength = float(self.config.inference.expert_priority_reliability_target)
        floor_strength = float(self.config.inference.expert_priority_reliability_floor)
        teacher_gap_guard = float(self.config.inference.public_teacher_gap_guard)
        min_conf = float(self.config.inference.min_confidence_threshold)
        max_conf = float(self.config.inference.max_confidence_threshold)
        min_margin = float(self.config.inference.min_margin_threshold)
        max_margin = float(self.config.inference.max_margin_threshold)
        general_predictions = self._predict_public_general_labels()
        self.last_public_teacher_gaps = {}
        self.last_public_fallback_rates = {}

        for update in updates:
            client_id = update.client_id
            threshold = self.client_confidence_thresholds[client_id]
            margin = self.client_margin_thresholds[client_id]
            reliability = self._compute_client_reliability(update.loss)
            knowledge = update.public_knowledge
            uncertainty_mean = float(knowledge.uncertainty_weights.mean().item())
            normalized_uncertainty = min(max(uncertainty_mean / 1.5, 0.0), 1.0)
            strength = 0.5 * reliability + 0.5 * normalized_uncertainty
            expert_probs = knowledge.soft_predictions
            expert_predictions = expert_probs.argmax(dim=1)
            expert_topk = torch.topk(expert_probs, k=min(2, expert_probs.size(1)), dim=1)
            expert_confidence = expert_topk.values[:, 0]
            if expert_topk.values.size(1) > 1:
                expert_margin = expert_topk.values[:, 0] - expert_topk.values[:, 1]
            else:
                expert_margin = torch.ones_like(expert_confidence)

            public_targets = torch.tensor(
                [self.public_targets_by_index[int(sample_index)] for sample_index in knowledge.sample_indices],
                dtype=torch.long,
            )
            general_public_predictions = torch.tensor(
                [general_predictions[int(sample_index)] for sample_index in knowledge.sample_indices],
                dtype=torch.long,
            )
            expert_public_accuracy = float((expert_predictions == public_targets).float().mean().item())
            general_public_accuracy = float((general_public_predictions == public_targets).float().mean().item())
            teacher_gap = general_public_accuracy - expert_public_accuracy
            fallback_mask = self._build_fallback_mask(
                expert_confidence=expert_confidence,
                expert_margin=expert_margin,
                confidence_threshold=threshold,
                margin_threshold=margin,
            )
            fallback_rate = float(fallback_mask.float().mean().item())
            fallback_floor = self._fallback_floor_for_client(client_id)
            if teacher_gap > max(teacher_gap_guard, 0.25):
                fallback_floor += 0.02
            self.last_public_teacher_gaps[client_id] = teacher_gap
            self.last_public_fallback_rates[client_id] = fallback_rate

            if teacher_gap > teacher_gap_guard:
                threshold += step
                margin += margin_step
            elif fallback_rate < fallback_floor:
                threshold += step
                margin += margin_step * 0.5
            elif strength >= target_strength:
                if fallback_rate > fallback_floor * 1.25:
                    threshold -= step
                    margin -= margin_step
            elif strength < floor_strength:
                threshold += step * 0.5
                margin += margin_step * 0.5
            elif fallback_rate > fallback_floor * 1.10:
                threshold -= step * 0.5
                margin -= margin_step * 0.5

            self.client_confidence_thresholds[client_id] = min(max(threshold, min_conf), max_conf)
            self.client_margin_thresholds[client_id] = min(max(margin, min_margin), max_margin)

    def _update_general_freeze_state(self, improved: bool, round_idx: int) -> None:
        if improved:
            self.rounds_since_best = 0
            return

        self.rounds_since_best += 1
        patience = max(int(self.config.federated.general_freeze_patience), 0)
        if self.general_distillation_frozen or patience == 0:
            return

        if round_idx <= max(int(self.config.federated.general_warmup_rounds), 0):
            return

        if self.rounds_since_best < patience:
            return

        if self.best_general_snapshot is not None:
            self.general_model.load_state_dict(self.best_general_snapshot["general_model_state"])
            if self.anchor_model is not None:
                self.anchor_model.load_state_dict(self.best_general_snapshot["general_model_state"])
            LOGGER.info(
                "Restored best general teacher before freezing | best_round=%d | general_acc=%.4f",
                int(self.best_general_snapshot["round_idx"]),
                float(self.best_general_snapshot["general_accuracy"]),
            )

        self.general_distillation_frozen = True
        LOGGER.info(
            "Frozen fedegs2 general distillation | round=%d | rounds_since_best=%d",
            round_idx,
            self.rounds_since_best,
        )

    def _restore_best_snapshot(self) -> None:
        if self.best_snapshot is None:
            return

        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        client_states = self.best_snapshot["client_expert_states"]
        for client_id, state_dict in client_states.items():
            self.clients[client_id].expert_model.load_state_dict(state_dict)
        self.client_confidence_thresholds = dict(self.best_snapshot["client_confidence_thresholds"])
        self.client_margin_thresholds = dict(self.best_snapshot["client_margin_thresholds"])

        LOGGER.info(
            "Restored fedegs2 best checkpoint | round=%d | routed_acc=%.4f | general_acc=%.4f | expert_acc=%.4f",
            int(self.best_snapshot["round_idx"]),
            float(self.best_snapshot["routed_accuracy"]),
            float(self.best_snapshot["general_accuracy"]),
            float(self.best_snapshot["expert_accuracy"]),
        )

    def _weighted_logit_loss(
        self,
        student_logits: torch.Tensor,
        target_probs: torch.Tensor,
        sample_weights: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        per_sample_kl = torch.sum(
            target_probs * (target_probs.clamp_min(1e-8).log() - student_log_probs),
            dim=1,
        ) * (temperature ** 2)
        return torch.sum(per_sample_kl * sample_weights) / sample_weights.sum().clamp_min(1e-8)

    def _weighted_feature_loss(
        self,
        student_embeddings: torch.Tensor,
        target_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_mse = torch.mean((student_embeddings - target_features) ** 2, dim=1)
        return torch.sum(per_sample_mse * sample_weights) / sample_weights.sum().clamp_min(1e-8)

    def _relation_loss(
        self,
        student_embeddings: torch.Tensor,
        target_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if student_embeddings.size(0) < 2:
            return student_embeddings.new_zeros(())

        student_norm = F.normalize(student_embeddings, dim=1)
        target_norm = F.normalize(target_features, dim=1)
        student_relation = student_norm @ student_norm.T
        target_relation = target_norm @ target_norm.T
        pair_weights = torch.sqrt(sample_weights.unsqueeze(1) * sample_weights.unsqueeze(0))
        squared_error = (student_relation - target_relation) ** 2
        return torch.sum(squared_error * pair_weights) / pair_weights.sum().clamp_min(1e-8)

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        logits = self.general_model(images)
        return logits.argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        logits = client.expert_model(images)
        return logits.argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        return self._confidence_route(
            client.expert_model,
            self.general_model,
            images,
            confidence_threshold=self.client_confidence_thresholds[client_id],
            margin_threshold=self.client_margin_thresholds[client_id],
        )
