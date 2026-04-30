import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.models import SmallCNN, build_teacher_model, estimate_model_flops, load_teacher_checkpoint, model_memory_mb


@dataclass
class KnowledgePacket:
    sample_indices: List[int]
    soft_predictions: torch.Tensor
    embeddings: torch.Tensor
    uncertainty_weights: torch.Tensor
    expert_confidence: torch.Tensor
    expert_margin: torch.Tensor
    route_gain_targets: torch.Tensor
    class_prototypes: Dict[int, torch.Tensor]
    class_counts: Dict[int, int]


@dataclass
class AggregatedKnowledgePacket:
    sample_indices: List[int]
    soft_predictions: torch.Tensor
    embeddings: torch.Tensor
    uncertainty_weights: torch.Tensor
    route_gain_targets: torch.Tensor
    raw_weight_sums: torch.Tensor
    client_reliability: Dict[str, float]
    class_prototypes: Dict[int, torch.Tensor]
    class_route_priors: torch.Tensor


@dataclass
class FedEGSBGClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    knowledge_packet: KnowledgePacket
    kd_gate_ratio: float
    refresh_gate_ratio: float
    route_stats: Dict[str, float]
    raw_upload_bytes: int
    compressed_upload_bytes: int


class GainRoutedGeneralModel(nn.Module):
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
                LOGGER.info("Initialized fedegsbg general model from %s", checkpoint)
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

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.classify_features(features)
        embedding = self.project_features(features)
        return features, embedding, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, logits = self.forward_with_embedding(x)
        return logits


class FedEGSBGClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, num_classes: int, device: str, config, data_module) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        self.local_prototypes: Dict[int, torch.Tensor] = {}
        self.local_class_counts: Dict[int, int] = {}

    def train(
        self,
        round_idx: int,
        general_model: GainRoutedGeneralModel,
        public_loader: DataLoader,
        use_teacher: bool,
    ) -> FedEGSBGClientUpdate:
        train_loader = self.data_module.make_loader(self.dataset, shuffle=True)
        refresh_gate_ratio = self._refresh_expert_from_general(public_loader, general_model, use_teacher=use_teacher)
        loss, kd_gate_ratio = self._train_local_expert(
            round_idx=round_idx,
            loader=train_loader,
            general_model=general_model,
            use_teacher=use_teacher,
        )
        prototype_loader = self.data_module.make_loader(self.dataset, shuffle=False)
        self.local_prototypes, self.local_class_counts = self._extract_local_prototypes(prototype_loader)
        knowledge_packet, route_stats = self._extract_knowledge_packet(public_loader, general_model)
        raw_upload_bytes = self._estimate_packet_nbytes(knowledge_packet, quantized=False)
        compressed_upload_bytes = self._estimate_packet_nbytes(
            knowledge_packet,
            quantized=bool(getattr(self.config.federated, "communication_quantization_enabled", False)),
        )
        return FedEGSBGClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            knowledge_packet=knowledge_packet,
            kd_gate_ratio=kd_gate_ratio,
            refresh_gate_ratio=refresh_gate_ratio,
            route_stats=route_stats,
            raw_upload_bytes=raw_upload_bytes,
            compressed_upload_bytes=compressed_upload_bytes,
        )

    def _refresh_expert_from_general(
        self,
        loader: DataLoader,
        general_model: GainRoutedGeneralModel,
        use_teacher: bool,
    ) -> float:
        refresh_epochs = max(int(getattr(self.config.federated, "expert_refresh_epochs", 0)), 0)
        logit_weight = float(getattr(self.config.federated, "expert_refresh_logit_weight", 0.0))
        hint_weight = float(getattr(self.config.federated, "expert_refresh_feature_hint_weight", 0.0))
        if not use_teacher or refresh_epochs == 0 or (logit_weight <= 0.0 and hint_weight <= 0.0):
            return 0.0

        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=float(self.config.federated.local_lr) * float(getattr(self.config.federated, "expert_refresh_lr_scale", 1.0)),
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        temperature = float(getattr(self.config.federated, "expert_kd_temperature", self.config.federated.client_kd_temperature))
        conf_thr = float(getattr(self.config.federated, "expert_refresh_confidence_threshold", 0.0))
        margin_thr = float(getattr(self.config.federated, "expert_refresh_margin_threshold", 0.0))
        gain_thr = float(getattr(self.config.federated, "expert_refresh_gain_threshold", 0.0))
        hard_boost = float(getattr(self.config.federated, "expert_refresh_hard_boost", 0.0))

        total_selected = 0
        total_seen = 0
        self.expert_model.train()
        general_model.eval()

        for _ in range(refresh_epochs):
            for images, _, _ in loader:
                images = images.to(self.device)
                with torch.no_grad():
                    _, teacher_embeddings, teacher_logits = general_model.forward_with_embedding(images)
                    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
                    teacher_conf, teacher_margin = self._confidence_and_margin(teacher_probs)

                _, student_embeddings, student_logits = self.expert_model.forward_with_embedding(images)
                student_probs = torch.softmax(student_logits.detach(), dim=1)
                student_conf, student_margin = self._confidence_and_margin(student_probs)

                score = (
                    (teacher_conf - student_conf)
                    + 0.5 * (teacher_margin - student_margin)
                    + 0.25 * (1.0 - student_conf)
                )
                base_mask = (
                    (teacher_conf >= conf_thr)
                    & (teacher_margin >= margin_thr)
                    & ((teacher_conf - student_conf) >= gain_thr)
                )
                selected_mask = self._expand_gate_mask(
                    score=score,
                    base_mask=base_mask,
                    target_coverage=float(getattr(self.config.federated, "expert_refresh_target_coverage", 0.0)),
                    adaptive_enabled=bool(getattr(self.config.federated, "expert_refresh_adaptive_coverage_enabled", False)),
                    min_gate_ratio=max(
                        float(getattr(self.config.federated, "expert_refresh_min_gate_ratio", 0.0)),
                        float(getattr(self.config.federated, "expert_refresh_gate_floor", 0.0)),
                    ),
                )
                total_selected += int(selected_mask.sum().item())
                total_seen += int(selected_mask.numel())
                if not selected_mask.any():
                    continue

                selected_teacher_probs = teacher_probs[selected_mask]
                selected_student_logits = student_logits[selected_mask]
                sample_weights = (1.0 + hard_boost * (1.0 - teacher_conf[selected_mask])).clamp_min(1.0)
                loss = student_logits.new_zeros(())
                if logit_weight > 0.0:
                    loss = loss + logit_weight * self._weighted_kd_loss(
                        selected_student_logits,
                        selected_teacher_probs,
                        sample_weights,
                        temperature,
                    )
                if hint_weight > 0.0:
                    loss = loss + hint_weight * self._weighted_feature_loss(
                        student_embeddings[selected_mask],
                        teacher_embeddings[selected_mask],
                        sample_weights,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        return float(total_selected) / max(total_seen, 1)

    def _train_local_expert(
        self,
        round_idx: int,
        loader: DataLoader,
        general_model: GainRoutedGeneralModel,
        use_teacher: bool,
    ) -> Tuple[float, float]:
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        kd_temperature = float(getattr(self.config.federated, "expert_kd_temperature", self.config.federated.client_kd_temperature))
        feature_hint_weight = float(getattr(self.config.federated, "client_feature_hint_weight", 0.0))
        personalization_weight = float(getattr(self.config.federated, "expert_personalization_weight", 0.0))
        hard_boost = float(getattr(self.config.federated, "expert_kd_hard_boost", 0.0))
        warmup_rounds = max(int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0)), 0)
        use_kd = use_teacher and kd_weight > 0.0 and round_idx > warmup_rounds
        reference_model = copy.deepcopy(self.expert_model).to(self.device) if personalization_weight > 0.0 else None
        if reference_model is not None:
            reference_model.eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)

        total_loss = 0.0
        total_batches = 0
        total_kd_selected = 0
        total_seen = 0

        self.expert_model.train()
        general_model.eval()

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                _, student_embeddings, student_logits = self.expert_model.forward_with_embedding(images)
                ce_loss = criterion(student_logits, targets).mean()
                loss = ce_loss

                if use_kd:
                    with torch.no_grad():
                        _, teacher_embeddings, teacher_logits = general_model.forward_with_embedding(images)
                        teacher_probs = torch.softmax(teacher_logits / kd_temperature, dim=1)
                        teacher_conf, teacher_margin = self._confidence_and_margin(teacher_probs)
                        teacher_predictions = teacher_probs.argmax(dim=1)

                    student_probs = torch.softmax(student_logits.detach(), dim=1)
                    student_conf, student_margin = self._confidence_and_margin(student_probs)
                    student_predictions = student_probs.argmax(dim=1)
                    teacher_correct = (teacher_predictions == targets).float()
                    student_correct = (student_predictions == targets).float()

                    score = (
                        0.50 * (teacher_correct - student_correct)
                        + 0.30 * (teacher_conf - student_conf)
                        + 0.20 * (teacher_margin - student_margin)
                        + float(getattr(self.config.inference, "route_gain_positive_margin", 0.0))
                        * (teacher_predictions != student_predictions).float()
                    )
                    base_mask = (
                        (teacher_conf >= float(getattr(self.config.federated, "expert_kd_confidence_threshold", 0.0)))
                        & (teacher_margin >= float(getattr(self.config.federated, "expert_kd_margin_threshold", 0.0)))
                        & ((teacher_conf - student_conf) >= float(getattr(self.config.federated, "expert_kd_teacher_confidence_delta", 0.0)))
                        & ((teacher_margin - student_margin) >= float(getattr(self.config.federated, "expert_kd_teacher_margin_delta", 0.0)))
                        & (student_conf <= float(getattr(self.config.federated, "expert_kd_student_confidence_ceiling", 1.0)))
                        & (student_margin <= float(getattr(self.config.federated, "expert_kd_student_margin_ceiling", 1.0)))
                    )
                    selected_mask = self._expand_gate_mask(
                        score=score,
                        base_mask=base_mask,
                        target_coverage=float(getattr(self.config.federated, "expert_kd_target_coverage", 0.0)),
                        adaptive_enabled=bool(getattr(self.config.federated, "expert_kd_adaptive_coverage_enabled", False)),
                        min_gate_ratio=max(
                            float(getattr(self.config.federated, "expert_kd_min_gate_ratio", 0.0)),
                            float(getattr(self.config.federated, "expert_kd_gate_floor", 0.0)),
                        ),
                    )
                    total_kd_selected += int(selected_mask.sum().item())
                    total_seen += int(selected_mask.numel())
                    if selected_mask.any():
                        kd_sample_weights = (1.0 + hard_boost * (1.0 - student_conf[selected_mask])).clamp_min(1.0)
                        kd_loss = self._weighted_kd_loss(
                            student_logits[selected_mask],
                            teacher_probs[selected_mask],
                            kd_sample_weights,
                            kd_temperature,
                        )
                        hint_loss = self._weighted_feature_loss(
                            student_embeddings[selected_mask],
                            teacher_embeddings[selected_mask],
                            kd_sample_weights,
                        )
                        loss = loss + kd_weight * kd_loss + feature_hint_weight * hint_loss

                if reference_model is not None:
                    loss = loss + personalization_weight * self._proximal_penalty(self.expert_model, reference_model)

                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                total_batches += 1

        return total_loss / max(total_batches, 1), float(total_kd_selected) / max(total_seen, 1)

    def _extract_local_prototypes(self, loader: DataLoader) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        feature_sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        self.expert_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                _, embeddings, _ = self.expert_model.forward_with_embedding(images)
                for embedding, target in zip(embeddings.detach().cpu(), targets.tolist()):
                    target = int(target)
                    if target not in feature_sums:
                        feature_sums[target] = embedding.clone()
                        counts[target] = 1
                    else:
                        feature_sums[target] += embedding
                        counts[target] += 1

        prototypes: Dict[int, torch.Tensor] = {}
        for class_idx, feature_sum in feature_sums.items():
            prototypes[class_idx] = F.normalize((feature_sum / max(counts[class_idx], 1)).unsqueeze(0), dim=1).squeeze(0)
        return prototypes, counts

    def _extract_knowledge_packet(
        self,
        loader: DataLoader,
        general_model: GainRoutedGeneralModel,
    ) -> Tuple[KnowledgePacket, Dict[str, float]]:
        temperature = float(self.config.federated.distill_temperature)
        min_uncertainty_weight = float(self.config.federated.min_uncertainty_weight)
        log_num_classes = math.log(float(self.config.model.num_classes))

        sample_indices: List[torch.Tensor] = []
        soft_predictions: List[torch.Tensor] = []
        embeddings_list: List[torch.Tensor] = []
        uncertainty_weights: List[torch.Tensor] = []
        expert_confidence: List[torch.Tensor] = []
        expert_margin: List[torch.Tensor] = []
        route_gain_targets: List[torch.Tensor] = []
        expert_correct_list: List[torch.Tensor] = []
        general_correct_list: List[torch.Tensor] = []

        self.expert_model.eval()
        general_model.eval()
        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                _, expert_embeddings_batch, expert_logits = self.expert_model.forward_with_embedding(images)
                _, _, general_logits = general_model.forward_with_embedding(images)

                expert_probs = torch.softmax(expert_logits / temperature, dim=1)
                general_probs = torch.softmax(general_logits / temperature, dim=1)
                expert_conf_batch, expert_margin_batch = self._confidence_and_margin(expert_probs)
                general_conf_batch, general_margin_batch = self._confidence_and_margin(general_probs)
                entropy = -(expert_probs * expert_probs.clamp_min(1e-8).log()).sum(dim=1) / max(log_num_classes, 1e-8)
                weights = torch.exp(-entropy).clamp_min(min_uncertainty_weight)

                expert_pred = expert_probs.argmax(dim=1)
                general_pred = general_probs.argmax(dim=1)
                expert_correct = (expert_pred == targets).float()
                general_correct = (general_pred == targets).float()
                gain_score = (
                    0.50 * (general_correct - expert_correct)
                    + 0.30 * (general_conf_batch - expert_conf_batch)
                    + 0.20 * (general_margin_batch - expert_margin_batch)
                )
                gain_targets = torch.sigmoid(float(getattr(self.config.federated, "drel_beta", 8.0)) * gain_score)

                sample_indices.append(indices.detach().cpu())
                soft_predictions.append(expert_probs.detach().cpu())
                embeddings_list.append(F.normalize(expert_embeddings_batch.detach().cpu(), dim=1))
                uncertainty_weights.append(weights.detach().cpu())
                expert_confidence.append(expert_conf_batch.detach().cpu())
                expert_margin.append(expert_margin_batch.detach().cpu())
                route_gain_targets.append(gain_targets.detach().cpu())
                expert_correct_list.append(expert_correct.detach().cpu())
                general_correct_list.append(general_correct.detach().cpu())

        merged_indices = torch.cat(sample_indices, dim=0)
        merged_predictions = torch.cat(soft_predictions, dim=0)
        merged_embeddings = torch.cat(embeddings_list, dim=0)
        merged_weights = torch.cat(uncertainty_weights, dim=0)
        merged_confidence = torch.cat(expert_confidence, dim=0)
        merged_margin = torch.cat(expert_margin, dim=0)
        merged_gain = torch.cat(route_gain_targets, dim=0)
        merged_expert_correct = torch.cat(expert_correct_list, dim=0)
        merged_general_correct = torch.cat(general_correct_list, dim=0)
        order = torch.argsort(merged_indices)

        packet = KnowledgePacket(
            sample_indices=merged_indices[order].tolist(),
            soft_predictions=merged_predictions[order],
            embeddings=merged_embeddings[order],
            uncertainty_weights=merged_weights[order],
            expert_confidence=merged_confidence[order],
            expert_margin=merged_margin[order],
            route_gain_targets=merged_gain[order],
            class_prototypes={class_idx: prototype.detach().cpu().clone() for class_idx, prototype in self.local_prototypes.items()},
            class_counts=dict(self.local_class_counts),
        )
        route_stats = {
            "upgrade_rate": float((packet.route_gain_targets >= 0.5).float().mean().item()),
            "mean_route_gain": float(packet.route_gain_targets.mean().item()),
            "public_expert_accuracy": float(merged_expert_correct.float().mean().item()),
            "public_general_accuracy": float(merged_general_correct.float().mean().item()),
            "teacher_advantage": float((merged_general_correct - merged_expert_correct).float().mean().item()),
        }
        return packet, route_stats

    def _expand_gate_mask(
        self,
        score: torch.Tensor,
        base_mask: torch.Tensor,
        target_coverage: float,
        adaptive_enabled: bool,
        min_gate_ratio: float,
    ) -> torch.Tensor:
        if score.numel() == 0:
            return base_mask
        selected_mask = base_mask.clone()
        desired_ratio = max(float(min_gate_ratio), 0.0)
        if adaptive_enabled and target_coverage > 0.0:
            desired_ratio = max(desired_ratio, float(target_coverage))
        if desired_ratio <= 0.0:
            return selected_mask
        required = min(score.numel(), max(1, int(math.ceil(desired_ratio * score.numel()))))
        if int(selected_mask.sum().item()) >= required:
            return selected_mask
        topk_values, _ = torch.topk(score.detach(), k=required)
        cutoff = topk_values[-1]
        return selected_mask | (score.detach() >= cutoff)

    def _confidence_and_margin(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
        confidence = topk.values[:, 0]
        if topk.values.size(1) > 1:
            margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            margin = torch.ones_like(confidence)
        return confidence, margin

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
        return torch.sum(per_sample_kl * sample_weights) / sample_weights.sum().clamp_min(1e-8)

    def _weighted_feature_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_mse = torch.mean((student_embeddings - teacher_embeddings) ** 2, dim=1)
        return torch.sum(per_sample_mse * sample_weights) / sample_weights.sum().clamp_min(1e-8)

    def _estimate_packet_nbytes(self, packet: KnowledgePacket, quantized: bool) -> int:
        bits = int(getattr(self.config.federated, "communication_quantization_bits", 8))
        total = 0
        tensors = [
            torch.tensor(packet.sample_indices, dtype=torch.int64),
            packet.soft_predictions,
            packet.embeddings,
            packet.uncertainty_weights,
            packet.expert_confidence,
            packet.expert_margin,
            packet.route_gain_targets,
        ] + list(packet.class_prototypes.values())
        for tensor in tensors:
            if quantized and tensor.is_floating_point():
                total += int(math.ceil(tensor.numel() * max(bits, 1) / 8.0))
            else:
                total += int(tensor.numel() * tensor.element_size())
        total += len(packet.class_counts) * 8
        return total


class FedEGSBGServer(BaseFederatedServer):
    def __init__(
        self,
        config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices,
        writer=None,
        public_dataset: Optional[Dataset] = None,
    ) -> None:
        if public_dataset is None or len(public_dataset) == 0:
            raise ValueError("fedegsbg requires a non-empty public_dataset.")
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )
        checkpoint_path = (
            config.dataset.difficulty_checkpoint
            if bool(getattr(config.federated, "general_init_from_teacher", False))
            else None
        )
        self.general_model = GainRoutedGeneralModel(
            num_classes=config.model.num_classes,
            knowledge_dim=config.model.knowledge_dim,
            checkpoint_path=checkpoint_path,
        ).to(self.device)
        self.deploy_general_model = copy.deepcopy(self.general_model).to(self.device)
        self.general_bootstrapped = bool(self.general_model.initialized_from_teacher)

        self.public_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)
        if bool(getattr(config.federated, "general_pretrain_on_public", False)):
            self._pretrain_general_on_public()
            self.general_bootstrapped = True
        self.anchor_model = copy.deepcopy(self.general_model).to(self.device) if self.general_bootstrapped else None
        if self.anchor_model is not None:
            self.anchor_model.eval()
            for parameter in self.anchor_model.parameters():
                parameter.requires_grad_(False)
            self.deploy_general_model.load_state_dict(self.general_model.state_dict())

        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.deploy_general_model)
        self.clients = {
            client_id: FedEGSBGClient(
                client_id,
                dataset,
                config.model.num_classes,
                config.federated.device,
                config,
                data_module,
            )
            for client_id, dataset in client_datasets.items()
        }
        self.distill_optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            weight_decay=self.config.federated.local_weight_decay,
        )
        self.public_targets_by_index = self._build_public_target_lookup()
        self.current_public_knowledge: Optional[AggregatedKnowledgePacket] = None
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.client_reliability_scores = {
            client_id: float(self.config.federated.min_client_reliability)
            for client_id in client_datasets.keys()
        }
        self.client_gain_thresholds = {
            client_id: float(getattr(self.config.inference, "route_gain_threshold", 0.0))
            for client_id in client_datasets.keys()
        }
        self.class_route_priors = torch.zeros(self.config.model.num_classes, dtype=torch.float32)

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegsbg round %d sampled clients=%s", round_idx, sampled_ids)
            updates: List[FedEGSBGClientUpdate] = []
            teacher_active = self.general_bootstrapped or round_idx > int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0))

            for client_id in sampled_ids:
                update = self.clients[client_id].train(
                    round_idx=round_idx,
                    general_model=self.deploy_general_model,
                    public_loader=self.public_loader,
                    use_teacher=teacher_active,
                )
                updates.append(update)

            self._update_client_reliability_scores(updates)
            self._update_client_gain_thresholds(updates)
            self.current_public_knowledge = self._aggregate_knowledge_packets(updates)
            distill_stats = self._distill_general_model(self.current_public_knowledge, round_idx)
            self._update_deploy_general_model()

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegsbg-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegsbg-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegsbg-routed")
            extra_metrics = self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)

            avg_client_loss = sum(update.loss for update in updates) / max(len(updates), 1)
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
                avg_client_loss=avg_client_loss,
                routed_accuracy=macro["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                extra_metrics=extra_metrics,
            )

            mean_reliability = sum(self.client_reliability_scores.values()) / max(len(self.client_reliability_scores), 1)
            mean_threshold = sum(self.client_gain_thresholds.values()) / max(len(self.client_gain_thresholds), 1)
            mean_kd_gate = sum(float(update.kd_gate_ratio) for update in updates) / max(len(updates), 1)
            mean_refresh_gate = sum(float(update.refresh_gate_ratio) for update in updates) / max(len(updates), 1)
            mean_public_gain = float(self.current_public_knowledge.route_gain_targets.mean().item())
            mean_route_prior = float(self.class_route_priors.mean().item())
            uplink_raw_bytes = sum(int(update.raw_upload_bytes) for update in updates)
            uplink_compressed_bytes = sum(int(update.compressed_upload_bytes) for update in updates)

            LOGGER.info(
                "fedegsbg round %d | client_loss=%.4f | distill_loss=%.4f | scale=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_acc=%.4f | invocation=%.4f | savings=%.4f",
                round_idx,
                avg_client_loss,
                distill_stats["total_loss"],
                distill_stats["scale"],
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                compute_profile["savings_ratio"],
            )
            LOGGER.info(
                "fedegsbg distill | round=%d | ce=%.4f | logit=%.4f | feature=%.4f | prototype=%.4f | route_prior=%.4f | anchor=%.4f | reliability=%.4f | mean_gain=%.4f | mean_prior=%.4f",
                round_idx,
                distill_stats["ce_loss"],
                distill_stats["logit_loss"],
                distill_stats["feature_loss"],
                distill_stats["prototype_loss"],
                distill_stats["route_prior_loss"],
                distill_stats["anchor_loss"],
                mean_reliability,
                mean_public_gain,
                mean_route_prior,
            )
            LOGGER.info(
                "fedegsbg local-kd | round=%d | kd_gate=%.4f | refresh_gate=%.4f | gain_threshold=%.4f | uplink_raw=%d | uplink_compressed=%d",
                round_idx,
                mean_kd_gate,
                mean_refresh_gate,
                mean_threshold,
                uplink_raw_bytes,
                uplink_compressed_bytes,
            )
            LOGGER.info(
                "fedegsbg auxiliary | round=%d | expert_acc=%.4f | general_acc=%.4f",
                round_idx,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegsbg", avg_client_loss, round_idx)
                self.writer.add_scalar("distill_loss/fedegsbg", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar("distill_scale/fedegsbg", distill_stats["scale"], round_idx)
                self.writer.add_scalar("distill_ce_loss/fedegsbg", distill_stats["ce_loss"], round_idx)
                self.writer.add_scalar("distill_logit_loss/fedegsbg", distill_stats["logit_loss"], round_idx)
                self.writer.add_scalar("distill_feature_loss/fedegsbg", distill_stats["feature_loss"], round_idx)
                self.writer.add_scalar("distill_prototype_loss/fedegsbg", distill_stats["prototype_loss"], round_idx)
                self.writer.add_scalar("distill_route_prior_loss/fedegsbg", distill_stats["route_prior_loss"], round_idx)
                self.writer.add_scalar("distill_anchor_loss/fedegsbg", distill_stats["anchor_loss"], round_idx)
                self.writer.add_scalar("client_reliability_mean/fedegsbg", mean_reliability, round_idx)
                self.writer.add_scalar("routing_gain_threshold_mean/fedegsbg", mean_threshold, round_idx)
                self.writer.add_scalar("routing_public_gain_mean/fedegsbg", mean_public_gain, round_idx)
                self.writer.add_scalar("routing_prior_mean/fedegsbg", mean_route_prior, round_idx)
                self.writer.add_scalar("local_kd_gate_mean/fedegsbg", mean_kd_gate, round_idx)
                self.writer.add_scalar("refresh_gate_mean/fedegsbg", mean_refresh_gate, round_idx)
                if uplink_raw_bytes > 0:
                    self.writer.add_scalar("comm/uplink_ratio_fedegsbg", uplink_compressed_bytes / uplink_raw_bytes, round_idx)
                self._log_auxiliary_accuracy_metrics(
                    "fedegsbg",
                    round_idx,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics("fedegsbg", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best_snapshot(
                round_idx=round_idx,
                round_metrics=round_metrics,
                expert_accuracy=expert_eval["macro"]["accuracy"],
                general_accuracy=general_eval["macro"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best_snapshot()
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        route_export_path = self._build_route_export_path("fedegsbg_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            prefix="fedegsbg_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegsbg_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegsbg_final_general")
        extra_metrics = self._build_final_extra_metrics(expert_eval, general_eval, routed_eval)
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = int(self.best_snapshot["round_idx"]) if self.best_snapshot is not None else 0
        if self.best_snapshot is not None:
            final_loss = float(self.best_snapshot["avg_client_loss"])

        routed_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            routed_eval["aggregate"]["invocation_rate"],
            mode="routed",
        )
        expert_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, mode="expert_only")
        general_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, mode="general_only")

        return {
            "algorithm": "fedegsbg",
            "metrics": {
                "accuracy": routed_eval["macro"]["accuracy"],
                "personalized_accuracy": routed_eval["macro"]["accuracy"],
                "weighted_accuracy": routed_eval["aggregate"]["accuracy"],
                "global_accuracy": routed_eval["macro"]["accuracy"],
                "local_accuracy": routed_eval["macro"]["accuracy"],
                "routed_accuracy": routed_eval["macro"]["accuracy"],
                "hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "hard_sample_recall": routed_eval["aggregate"]["hard_recall"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["macro"]["accuracy"],
                "general_only_accuracy": general_eval["macro"]["accuracy"],
                "public_dataset_size": len(self.public_dataset),
                "mean_client_reliability": sum(self.client_reliability_scores.values()) / max(len(self.client_reliability_scores), 1),
                "mean_route_prior": float(self.class_route_priors.mean().item()),
                "final_training_loss": final_loss,
                "best_round": best_round,
                **extra_metrics,
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
                "general": model_memory_mb(self.deploy_general_model),
            },
            "artifacts": {
                "route_csv": str(route_export_path),
            },
        }

    def _build_round_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._evaluate_route_effectiveness_metrics_from_predictors(
            expert_eval,
            general_eval,
            routed_eval,
            self._predict_expert_only,
            self._predict_general_only,
            self._predict_routed,
            general_route_types=("general", "fusion"),
        )

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)

    def _pretrain_general_on_public(self) -> None:
        epochs = max(int(getattr(self.config.federated, "general_pretrain_epochs", 0)), 0)
        if epochs == 0:
            return
        optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=float(getattr(self.config.federated, "general_pretrain_lr", 0.01)),
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()
        self.general_model.train()
        for _ in range(epochs):
            for images, targets, _ in self.public_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.general_model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
        LOGGER.info("Completed fedegsbg general public pretraining | epochs=%d", epochs)

    def _build_public_target_lookup(self) -> Dict[int, int]:
        targets_by_index: Dict[int, int] = {}
        for _, targets, indices in self.public_loader:
            for sample_index, target in zip(indices.tolist(), targets.tolist()):
                targets_by_index[int(sample_index)] = int(target)
        return targets_by_index

    def _aggregate_knowledge_packets(self, updates: List[FedEGSBGClientUpdate]) -> AggregatedKnowledgePacket:
        if not updates:
            raise RuntimeError("No client updates were provided for fedegsbg knowledge aggregation.")

        reference_indices = updates[0].knowledge_packet.sample_indices
        num_public = len(reference_indices)
        num_classes = self.config.model.num_classes
        knowledge_dim = self.config.model.knowledge_dim
        index_to_position = {sample_index: idx for idx, sample_index in enumerate(reference_indices)}

        prediction_sums = torch.zeros((num_public, num_classes), dtype=torch.float32)
        embedding_sums = torch.zeros((num_public, knowledge_dim), dtype=torch.float32)
        gain_sums = torch.zeros(num_public, dtype=torch.float32)
        weight_sums = torch.zeros(num_public, dtype=torch.float32)
        class_prior_sums = torch.zeros(num_classes, dtype=torch.float32)
        class_prior_weights = torch.zeros(num_classes, dtype=torch.float32)
        prototype_sums: Dict[int, torch.Tensor] = {}
        prototype_weights: Dict[int, float] = {}
        client_reliability: Dict[str, float] = {}

        for update in updates:
            packet = update.knowledge_packet
            reliability = float(self.client_reliability_scores.get(update.client_id, self.config.federated.min_client_reliability))
            client_reliability[update.client_id] = reliability
            client_weight = float(update.num_samples) * reliability
            for row_idx, sample_index in enumerate(packet.sample_indices):
                position = index_to_position[sample_index]
                sample_weight = client_weight * float(packet.uncertainty_weights[row_idx].item())
                prediction_sums[position] += packet.soft_predictions[row_idx] * sample_weight
                embedding_sums[position] += packet.embeddings[row_idx] * sample_weight
                gain_sums[position] += packet.route_gain_targets[row_idx] * sample_weight
                weight_sums[position] += sample_weight
                class_idx = self.public_targets_by_index[int(sample_index)]
                class_prior_sums[class_idx] += float(packet.route_gain_targets[row_idx].item()) * sample_weight
                class_prior_weights[class_idx] += sample_weight

            for class_idx, prototype in packet.class_prototypes.items():
                prototype_weight = client_weight * float(packet.class_counts.get(class_idx, 0))
                if prototype_weight <= 0:
                    continue
                if class_idx not in prototype_sums:
                    prototype_sums[class_idx] = prototype.clone() * prototype_weight
                    prototype_weights[class_idx] = prototype_weight
                else:
                    prototype_sums[class_idx] += prototype * prototype_weight
                    prototype_weights[class_idx] += prototype_weight

        safe_weights = weight_sums.clamp_min(1e-8)
        normalized_predictions = prediction_sums / safe_weights.unsqueeze(1)
        normalized_predictions = normalized_predictions / normalized_predictions.sum(dim=1, keepdim=True).clamp_min(1e-8)
        normalized_embeddings = F.normalize(embedding_sums / safe_weights.unsqueeze(1), dim=1)
        normalized_gain = gain_sums / safe_weights
        normalized_uncertainty = (weight_sums / weight_sums.mean().clamp_min(1e-8)).clamp_min(self.config.federated.min_uncertainty_weight)
        class_route_priors = torch.where(
            class_prior_weights > 0,
            class_prior_sums / class_prior_weights.clamp_min(1e-8),
            torch.zeros_like(class_prior_sums),
        ).clamp(0.0, 1.0)
        aggregated_prototypes = {
            class_idx: F.normalize((prototype_sum / max(prototype_weights[class_idx], 1e-8)).unsqueeze(0), dim=1).squeeze(0)
            for class_idx, prototype_sum in prototype_sums.items()
        }
        self.class_route_priors = class_route_priors.detach().cpu()

        return AggregatedKnowledgePacket(
            sample_indices=reference_indices,
            soft_predictions=normalized_predictions,
            embeddings=normalized_embeddings,
            uncertainty_weights=normalized_uncertainty,
            route_gain_targets=normalized_gain,
            raw_weight_sums=weight_sums,
            client_reliability=client_reliability,
            class_prototypes=aggregated_prototypes,
            class_route_priors=class_route_priors,
        )

    def _distill_general_model(self, knowledge: AggregatedKnowledgePacket, round_idx: int) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        ce_weight = float(getattr(self.config.federated, "public_ce_weight", 1.0))
        logit_weight = float(getattr(self.config.federated, "public_logit_align_weight", self.config.federated.logit_align_weight))
        feature_weight = float(getattr(self.config.federated, "distill_feature_weight", self.config.federated.feature_align_weight))
        prototype_weight = float(getattr(self.config.federated, "prototype_align_weight", 0.0))
        route_prior_weight = float(getattr(self.config.federated, "route_prior_align_weight", 0.0))
        anchor_weight = float(getattr(self.config.federated, "general_anchor_weight", 0.0)) if self.anchor_model is not None else 0.0
        criterion = torch.nn.CrossEntropyLoss()
        index_to_position = {sample_index: idx for idx, sample_index in enumerate(knowledge.sample_indices)}
        distill_scale = self._compute_distill_scale(round_idx)

        if distill_scale <= 0.0:
            return {
                "total_loss": 0.0,
                "ce_loss": 0.0,
                "logit_loss": 0.0,
                "feature_loss": 0.0,
                "prototype_loss": 0.0,
                "route_prior_loss": 0.0,
                "anchor_loss": 0.0,
                "scale": 0.0,
            }

        total_loss = 0.0
        total_ce_loss = 0.0
        total_logit_loss = 0.0
        total_feature_loss = 0.0
        total_prototype_loss = 0.0
        total_route_prior_loss = 0.0
        total_anchor_loss = 0.0
        total_batches = 0
        class_route_priors = knowledge.class_route_priors.to(self.device)

        self.general_model.train()
        for _ in range(self.config.federated.distill_epochs):
            for images, targets, indices in self.public_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                positions = torch.tensor(
                    [index_to_position[int(sample_index)] for sample_index in indices.tolist()],
                    device=self.device,
                    dtype=torch.long,
                )

                target_probs = knowledge.soft_predictions[positions.cpu()].to(self.device)
                target_embeddings = knowledge.embeddings[positions.cpu()].to(self.device)
                sample_weights = knowledge.uncertainty_weights[positions.cpu()].to(self.device)
                target_route_gain = knowledge.route_gain_targets[positions.cpu()].to(self.device)

                self.distill_optimizer.zero_grad(set_to_none=True)
                _, student_embeddings, student_logits = self.general_model.forward_with_embedding(images)
                student_probs = torch.softmax(student_logits, dim=1)

                ce_loss = criterion(student_logits, targets)
                logit_loss = self._weighted_logit_loss(student_logits, target_probs, sample_weights, temperature)
                feature_loss = self._weighted_feature_loss(student_embeddings, target_embeddings, sample_weights)
                prototype_loss = self._prototype_alignment_loss(student_embeddings, targets, knowledge.class_prototypes)
                predicted_route_prior = torch.sum(student_probs * class_route_priors.unsqueeze(0), dim=1)
                route_prior_loss = torch.sum(((predicted_route_prior - target_route_gain) ** 2) * sample_weights) / sample_weights.sum().clamp_min(1e-8)

                if anchor_weight > 0.0 and self.anchor_model is not None:
                    with torch.no_grad():
                        anchor_logits = self.anchor_model(images)
                        anchor_probs = torch.softmax(anchor_logits / temperature, dim=1)
                    anchor_loss = self._weighted_logit_loss(student_logits, anchor_probs, sample_weights, temperature)
                else:
                    anchor_loss = ce_loss.new_zeros(())

                loss = ce_weight * ce_loss + distill_scale * (
                    logit_weight * logit_loss
                    + feature_weight * feature_loss
                    + prototype_weight * prototype_loss
                    + route_prior_weight * route_prior_loss
                    + anchor_weight * anchor_loss
                )
                loss.backward()
                self.distill_optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_ce_loss += float(ce_loss.detach().cpu().item())
                total_logit_loss += float(logit_loss.detach().cpu().item())
                total_feature_loss += float(feature_loss.detach().cpu().item())
                total_prototype_loss += float(prototype_loss.detach().cpu().item())
                total_route_prior_loss += float(route_prior_loss.detach().cpu().item())
                total_anchor_loss += float(anchor_loss.detach().cpu().item())
                total_batches += 1

        denominator = max(total_batches, 1)
        return {
            "total_loss": total_loss / denominator,
            "ce_loss": total_ce_loss / denominator,
            "logit_loss": total_logit_loss / denominator,
            "feature_loss": total_feature_loss / denominator,
            "prototype_loss": total_prototype_loss / denominator,
            "route_prior_loss": total_route_prior_loss / denominator,
            "anchor_loss": total_anchor_loss / denominator,
            "scale": distill_scale,
        }

    def _update_client_reliability_scores(self, updates: List[FedEGSBGClientUpdate]) -> None:
        momentum = float(getattr(self.config.federated, "reliability_ema_momentum", 0.8))
        accuracy_weight = float(getattr(self.config.federated, "reliability_accuracy_weight", 0.7))
        minimum = float(self.config.federated.min_client_reliability)
        for update in updates:
            base = 1.0 / (1.0 + max(update.loss, 0.0)) if math.isfinite(update.loss) else minimum
            public_accuracy = float(update.route_stats.get("public_expert_accuracy", 0.0))
            instant = accuracy_weight * public_accuracy + (1.0 - accuracy_weight) * base
            previous = float(self.client_reliability_scores.get(update.client_id, minimum))
            smoothed = momentum * previous + (1.0 - momentum) * instant
            self.client_reliability_scores[update.client_id] = min(max(smoothed, minimum), 1.0)

    def _update_client_gain_thresholds(self, updates: List[FedEGSBGClientUpdate]) -> None:
        step = float(getattr(self.config.inference, "route_gain_threshold_step", 0.05))
        min_threshold = float(getattr(self.config.inference, "min_route_gain_threshold", -1.0))
        max_threshold = float(getattr(self.config.inference, "max_route_gain_threshold", 1.0))
        for update in updates:
            client_id = update.client_id
            threshold = float(self.client_gain_thresholds.get(client_id, 0.0))
            target_rate = self._target_invocation_rate_for_client(client_id)
            upgrade_rate = float(update.route_stats.get("upgrade_rate", 0.0))
            teacher_advantage = float(update.route_stats.get("teacher_advantage", 0.0))
            if upgrade_rate > target_rate + 0.02:
                threshold += step
            elif upgrade_rate < max(target_rate - 0.02, 0.0):
                threshold -= step * 0.5
            if teacher_advantage > float(getattr(self.config.inference, "route_gain_positive_margin", 0.0)) and upgrade_rate < target_rate:
                threshold -= step * 0.5
            self.client_gain_thresholds[client_id] = min(max(threshold, min_threshold), max_threshold)

    def _target_invocation_rate_for_client(self, client_id: str) -> float:
        if client_id.startswith("complex_"):
            return float(getattr(self.config.inference, "complex_target_general_invocation_rate", self.config.inference.target_general_invocation_rate))
        if client_id.startswith("simple_"):
            return float(getattr(self.config.inference, "simple_target_general_invocation_rate", self.config.inference.target_general_invocation_rate))
        return float(self.config.inference.target_general_invocation_rate)

    def _update_deploy_general_model(self) -> None:
        momentum = float(getattr(self.config.federated, "general_deploy_ema_momentum", 0.9))
        with torch.no_grad():
            for deploy_param, source_param in zip(self.deploy_general_model.parameters(), self.general_model.parameters()):
                deploy_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)
            for deploy_buffer, source_buffer in zip(self.deploy_general_model.buffers(), self.general_model.buffers()):
                deploy_buffer.copy_(source_buffer)

    def _compute_distill_scale(self, round_idx: int) -> float:
        warmup_rounds = max(int(getattr(self.config.federated, "general_warmup_rounds", 0)), 0)
        ramp_rounds = max(int(getattr(self.config.federated, "general_distill_ramp_rounds", 0)), 0)
        max_scale = max(float(getattr(self.config.federated, "general_distill_max_scale", 1.0)), 0.0)
        if round_idx <= warmup_rounds:
            return 0.0
        if ramp_rounds == 0:
            return max_scale
        return min(max_scale, float(round_idx - warmup_rounds) / float(ramp_rounds))

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
        target_embeddings: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_mse = torch.mean((student_embeddings - target_embeddings) ** 2, dim=1)
        return torch.sum(per_sample_mse * sample_weights) / sample_weights.sum().clamp_min(1e-8)

    def _prototype_alignment_loss(
        self,
        student_embeddings: torch.Tensor,
        targets: torch.Tensor,
        class_prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for class_idx, prototype in class_prototypes.items():
            mask = targets == int(class_idx)
            if not mask.any():
                continue
            class_mean = F.normalize(student_embeddings[mask].mean(dim=0, keepdim=True), dim=1)
            target_proto = prototype.to(self.device).unsqueeze(0)
            losses.append(torch.mean((class_mean - target_proto) ** 2))
        if not losses:
            return student_embeddings.new_zeros(())
        return torch.stack(losses).mean()

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
            "deploy_general_model_state": self._clone_model_state(self.deploy_general_model),
            "client_expert_states": {
                client_id: self._clone_model_state(client.expert_model)
                for client_id, client in self.clients.items()
            },
            "client_local_prototypes": {
                client_id: {class_idx: tensor.detach().cpu().clone() for class_idx, tensor in client.local_prototypes.items()}
                for client_id, client in self.clients.items()
            },
            "client_local_counts": {
                client_id: dict(client.local_class_counts)
                for client_id, client in self.clients.items()
            },
            "client_gain_thresholds": dict(self.client_gain_thresholds),
            "client_reliability_scores": dict(self.client_reliability_scores),
            "class_route_priors": self.class_route_priors.detach().cpu().clone(),
        }
        LOGGER.info(
            "Updated fedegsbg best checkpoint | round=%d | routed_acc=%.4f | general_acc=%.4f | expert_acc=%.4f",
            round_idx,
            round_metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )
        return True

    def _restore_best_snapshot(self) -> None:
        if self.best_snapshot is None:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        self.deploy_general_model.load_state_dict(self.best_snapshot["deploy_general_model_state"])
        for client_id, state_dict in self.best_snapshot["client_expert_states"].items():
            self.clients[client_id].expert_model.load_state_dict(state_dict)
            self.clients[client_id].local_prototypes = {
                class_idx: tensor.detach().cpu().clone()
                for class_idx, tensor in self.best_snapshot["client_local_prototypes"].get(client_id, {}).items()
            }
            self.clients[client_id].local_class_counts = dict(self.best_snapshot["client_local_counts"].get(client_id, {}))
        self.client_gain_thresholds = dict(self.best_snapshot["client_gain_thresholds"])
        self.client_reliability_scores = dict(self.best_snapshot["client_reliability_scores"])
        self.class_route_priors = self.best_snapshot["class_route_priors"].detach().cpu().clone()
        LOGGER.info(
            "Restored fedegsbg best checkpoint | round=%d | routed_acc=%.4f | general_acc=%.4f | expert_acc=%.4f",
            int(self.best_snapshot["round_idx"]),
            float(self.best_snapshot["routed_accuracy"]),
            float(self.best_snapshot["general_accuracy"]),
            float(self.best_snapshot["expert_accuracy"]),
        )

    def _clone_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _predict_general_only(self, client_id, images, indices):
        self.deploy_general_model.eval()
        logits = self.deploy_general_model(images)
        return logits.argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        logits = client.expert_model(images)
        return logits.argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        self.deploy_general_model.eval()

        _, expert_embeddings, expert_logits = client.expert_model.forward_with_embedding(images)
        expert_probs = torch.softmax(expert_logits, dim=1)
        expert_confidence, expert_margin = self._confidence_and_margin(expert_probs)
        expert_predictions = expert_probs.argmax(dim=1)
        energy = -torch.logsumexp(expert_logits, dim=1)

        confidence_score = 1.0 - expert_confidence
        margin_score = 1.0 - expert_margin.clamp(0.0, 1.0)
        energy_threshold = float(getattr(self.config.inference, "route_energy_threshold", -3.0))
        energy_score = torch.sigmoid((energy - energy_threshold) / 0.25)
        distance_score = self._prototype_distance_score(client.local_prototypes, expert_embeddings, expert_predictions)
        prior_tensor = self.class_route_priors.to(self.device)
        prior_score = prior_tensor[expert_predictions]
        reliability = float(self.client_reliability_scores.get(client_id, self.config.federated.min_client_reliability))
        reliability_score = torch.full_like(confidence_score, 1.0 - reliability)

        gain = (
            float(getattr(self.config.inference, "route_gain_confidence_weight", 0.0)) * confidence_score
            + float(getattr(self.config.inference, "route_gain_margin_weight", 0.0)) * margin_score
            + float(getattr(self.config.inference, "route_gain_energy_weight", 0.0)) * energy_score
            + float(getattr(self.config.inference, "route_gain_distance_weight", 0.0)) * distance_score
            + float(getattr(self.config.inference, "route_gain_prior_weight", 0.0)) * prior_score
            + float(getattr(self.config.inference, "route_gain_reliability_weight", 0.0)) * reliability_score
        )
        threshold = float(self.client_gain_thresholds.get(client_id, getattr(self.config.inference, "route_gain_threshold", 0.0)))
        general_mask = gain >= threshold

        predictions = expert_predictions.clone()
        route_types = ["expert"] * images.size(0)
        invoked_general = int(general_mask.sum().item())
        if general_mask.any():
            general_logits = self.deploy_general_model(images[general_mask])
            general_probs = torch.softmax(general_logits, dim=1)
            general_predictions = general_probs.argmax(dim=1)
            general_confidence = general_probs.max(dim=1).values
            selected_expert_confidence = expert_confidence[general_mask]
            fusion_band = max(float(getattr(self.config.inference, "fusion_band", 0.0)), 0.0)
            selected_predictions = general_predictions.clone()
            if fusion_band > 0.0:
                fusion_mask = (general_confidence - selected_expert_confidence).abs() <= fusion_band
                if fusion_mask.any():
                    selected_expert_logits = expert_logits[general_mask][fusion_mask]
                    selected_general_logits = general_logits[fusion_mask]
                    expert_weight = selected_expert_confidence[fusion_mask] / (
                        selected_expert_confidence[fusion_mask] + general_confidence[fusion_mask]
                    ).clamp_min(1e-8)
                    fused_logits = (
                        expert_weight.unsqueeze(1) * selected_expert_logits
                        + (1.0 - expert_weight.unsqueeze(1)) * selected_general_logits
                    )
                    selected_predictions[fusion_mask] = fused_logits.argmax(dim=1)
                local_indices = general_mask.nonzero(as_tuple=False).flatten().tolist()
                for local_idx, sample_idx in enumerate(local_indices):
                    route_types[sample_idx] = "fusion" if fusion_band > 0.0 and (general_confidence[local_idx] - selected_expert_confidence[local_idx]).abs() <= fusion_band else "general"
            else:
                for sample_idx in general_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_idx] = "general"
            predictions[general_mask] = selected_predictions

        metadata = {
            "route_type": route_types,
            "expert_confidence": expert_confidence.detach().cpu().tolist(),
        }
        return predictions, invoked_general, metadata

    def _confidence_and_margin(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
        confidence = topk.values[:, 0]
        if topk.values.size(1) > 1:
            margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            margin = torch.ones_like(confidence)
        return confidence, margin

    def _prototype_distance_score(
        self,
        prototypes: Dict[int, torch.Tensor],
        embeddings: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        distance_threshold = max(float(getattr(self.config.inference, "route_distance_threshold", 0.04)), 1e-6)
        scores = torch.zeros(embeddings.size(0), device=embeddings.device)
        if not prototypes:
            return scores
        normalized_embeddings = F.normalize(embeddings, dim=1)
        for row_idx, class_idx in enumerate(predictions.detach().cpu().tolist()):
            prototype = prototypes.get(int(class_idx))
            if prototype is None:
                continue
            prototype = prototype.to(embeddings.device)
            cosine_distance = 1.0 - torch.sum(normalized_embeddings[row_idx] * prototype)
            scores[row_idx] = torch.clamp(cosine_distance / distance_threshold, min=0.0, max=1.0)
        return scores
