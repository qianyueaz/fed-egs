"""
FedEGSD-S: Resource-asymmetric FedEGS with frozen local general fallback.

Design summary:
  - Clients only train the lightweight personalized expert (SmallCNN).
  - A stronger general model (ResNet18-style) is deployed locally but remains
    frozen on clients; it acts as a teacher during training and a fallback
    model during inference.
  - Routing uses expert prototype distance so hard / unfamiliar samples can be
    handled by the local general model without leaving the device.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.models import SmallCNN, build_teacher_model, estimate_model_flops, load_teacher_checkpoint, model_memory_mb
from fedegs.models.width_scalable_resnet import average_weighted_deltas, state_dict_delta


def drel_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Decoupled relative-entropy loss with target / non-target branches."""
    temperature = max(float(temperature), 1e-8)
    tc_teacher = logits_teacher.gather(1, targets.unsqueeze(1)) / temperature
    tc_student = logits_student.gather(1, targets.unsqueeze(1)) / temperature
    tc_loss = F.mse_loss(tc_student, tc_teacher.detach()) * (temperature ** 2)

    mask = torch.zeros_like(logits_teacher)
    mask.scatter_(1, targets.unsqueeze(1), -1e9)
    nc_teacher_probs = F.softmax((logits_teacher + mask) / temperature, dim=1)
    nc_student_log_probs = F.log_softmax((logits_student + mask) / temperature, dim=1)
    nc_loss = F.kl_div(nc_student_log_probs, nc_teacher_probs.detach(), reduction="batchmean") * (temperature ** 2)
    return alpha * tc_loss + beta * nc_loss


@dataclass
class FedEGSDSClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    expert_delta: Dict[str, torch.Tensor]


class AsymmetricGeneralModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        knowledge_dim: int,
        checkpoint_path: Optional[str] = None,
        pretrained_imagenet: bool = False,
    ) -> None:
        super().__init__()
        backbone = build_teacher_model(num_classes=num_classes, pretrained_imagenet=pretrained_imagenet)
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location="cpu")
                load_teacher_checkpoint(backbone, state)
                LOGGER.info("Initialized general model from %s", checkpoint)
            else:
                LOGGER.warning("Requested general initialization checkpoint missing at %s", checkpoint)

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
        if isinstance(self.projector, nn.Identity):
            return
        weight = torch.zeros((self.knowledge_dim, self.feature_dim))
        diagonal = min(self.knowledge_dim, self.feature_dim)
        weight[:diagonal, :diagonal] = torch.eye(diagonal)
        self.projector.weight.data.copy_(weight)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

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


class FedEGSDSClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, config, data_module) -> None:
        super().__init__(client_id, dataset, config.federated.device)
        self.config = config
        self.data_module = data_module
        self.expert_model = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        self.class_counts: Dict[int, int] = {}
        self.centroids: Dict[int, torch.Tensor] = {}
        self.centroid_stds: Dict[int, float] = {}
        self.local_classes: List[int] = []

    def initialize_personal_state(self, expert_prior: nn.Module) -> None:
        self.expert_model.load_state_dict(expert_prior.state_dict())

    def train_round(
        self,
        expert_prior: SmallCNN,
        global_general: AsymmetricGeneralModel,
        round_idx: int,
    ) -> FedEGSDSClientUpdate:
        round_expert = copy.deepcopy(expert_prior).to(self.device)
        personal_anchor = copy.deepcopy(self.expert_model).to(self.device)
        frozen_general = copy.deepcopy(global_general).to(self.device)
        frozen_general.freeze()

        before_expert = {key: value.detach().cpu().clone() for key, value in round_expert.state_dict().items()}
        train_loader = self.data_module.make_loader(self.dataset, shuffle=True)
        loss = self._optimize_expert(round_expert, personal_anchor, frozen_general, train_loader, round_idx)

        self.expert_model.load_state_dict(round_expert.state_dict())
        self._compute_prototypes()
        return FedEGSDSClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            expert_delta=state_dict_delta(self.expert_model.state_dict(), before_expert),
        )

    def _expert_kd_scale(self, round_idx: int) -> float:
        warmup_rounds = max(int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0)), 0)
        return 0.0 if round_idx <= warmup_rounds else 1.0

    def _optimize_expert(
        self,
        round_expert: SmallCNN,
        personal_anchor: SmallCNN,
        frozen_general: AsymmetricGeneralModel,
        loader,
        round_idx: int,
    ) -> float:
        cfg = self.config.federated
        drel_alpha = float(getattr(cfg, "drel_alpha", 1.0))
        drel_beta = float(getattr(cfg, "drel_beta", 8.0))
        drel_temperature = float(getattr(cfg, "expert_kd_temperature", 3.0))
        lambda_ge = float(getattr(cfg, "lambda_ge", 1.0)) * self._expert_kd_scale(round_idx)
        personalization_mu = float(getattr(cfg, "prox_mu", 0.0))
        feature_hint_weight = float(
            getattr(cfg, "client_feature_hint_weight", getattr(cfg, "feature_hint_weight", 0.0))
        )

        expert_optimizer = torch.optim.SGD(
            round_expert.parameters(),
            lr=cfg.local_lr,
            momentum=cfg.local_momentum,
            weight_decay=cfg.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        round_expert.train()
        frozen_general.freeze()

        total_loss = 0.0
        total_batches = 0
        for _ in range(cfg.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    _, general_embeddings, general_logits = frozen_general.forward_with_embedding(images)

                _, expert_embeddings, expert_logits = round_expert.forward_with_embedding(images)
                ce_loss = criterion(expert_logits, targets)

                kd_loss = ce_loss.new_zeros(())
                if lambda_ge > 0:
                    kd_loss = lambda_ge * drel_loss(
                        general_logits.detach(),
                        expert_logits,
                        targets,
                        alpha=drel_alpha,
                        beta=drel_beta,
                        temperature=drel_temperature,
                    )

                feature_hint_loss = ce_loss.new_zeros(())
                if feature_hint_weight > 0:
                    feature_hint_loss = feature_hint_weight * torch.mean((expert_embeddings - general_embeddings.detach()) ** 2)

                prox_loss = ce_loss.new_zeros(())
                if personalization_mu > 0:
                    prox_loss = 0.5 * personalization_mu * self._proximal_penalty(round_expert, personal_anchor)

                expert_loss = ce_loss + kd_loss + feature_hint_loss + prox_loss
                expert_optimizer.zero_grad(set_to_none=True)
                expert_loss.backward()
                expert_optimizer.step()

                total_loss += float(expert_loss.detach().cpu().item())
                total_batches += 1
        return total_loss / max(total_batches, 1)

    def _proximal_penalty(self, current_model: nn.Module, anchor_model: nn.Module) -> torch.Tensor:
        penalty = torch.zeros(1, device=self.device)
        for current_param, anchor_param in zip(current_model.parameters(), anchor_model.parameters()):
            penalty = penalty + torch.sum((current_param - anchor_param.detach().to(self.device)) ** 2)
        return penalty.squeeze()

    @torch.no_grad()
    def _compute_prototypes(self) -> None:
        loader = self.data_module.make_loader(self.dataset, shuffle=False)
        self.expert_model.eval()
        class_features: Dict[int, List[torch.Tensor]] = {}
        self.class_counts = {}
        for images, targets, _ in loader:
            images = images.to(self.device)
            _, embeddings, _ = self.expert_model.forward_with_embedding(images)
            embeddings = F.normalize(embeddings, dim=1)
            for embedding, label in zip(embeddings.cpu(), targets.tolist()):
                class_features.setdefault(label, []).append(embedding)
                self.class_counts[label] = self.class_counts.get(label, 0) + 1

        self.local_classes = sorted(class_features.keys())
        self.centroids = {}
        self.centroid_stds = {}
        for label, features in class_features.items():
            stacked = torch.stack(features, dim=0)
            centroid = F.normalize(stacked.mean(dim=0), dim=0)
            self.centroids[label] = centroid
            distances = 1.0 - F.cosine_similarity(stacked, centroid.unsqueeze(0), dim=1)
            self.centroid_stds[label] = float(distances.mean().item())

    @torch.no_grad()
    def distance_route(
        self,
        images: torch.Tensor,
        fallback_general: nn.Module,
        distance_threshold: float,
        std_multiplier: float = 1.5,
    ):
        self.expert_model.eval()
        fallback_general.eval()
        batch_size = images.size(0)
        if not self.centroids:
            general_logits = fallback_general(images)
            return general_logits.argmax(dim=1), batch_size, {
                "route_type": ["general"] * batch_size,
                "expert_confidence": [0.0] * batch_size,
            }

        _, embeddings, expert_logits = self.expert_model.forward_with_embedding(images)
        embeddings = F.normalize(embeddings, dim=1)
        expert_predictions = expert_logits.argmax(dim=1)

        centroid_labels = sorted(self.centroids.keys())
        centroid_stack = torch.stack([self.centroids[label] for label in centroid_labels], dim=0).to(embeddings.device)
        min_distance, nearest_index = (1.0 - embeddings @ centroid_stack.T).min(dim=1)

        thresholds = torch.full((batch_size,), distance_threshold, device=images.device)
        for idx in range(batch_size):
            nearest_label = centroid_labels[nearest_index[idx].item()]
            thresholds[idx] = distance_threshold + std_multiplier * self.centroid_stds.get(nearest_label, 0.0)

        fallback_mask = min_distance >= thresholds
        predictions = expert_predictions.clone()
        invoked_general = int(fallback_mask.sum().item())
        route_types = ["expert"] * batch_size
        if fallback_mask.any():
            general_logits = fallback_general(images[fallback_mask])
            predictions[fallback_mask] = general_logits.argmax(dim=1)
            for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                route_types[sample_idx] = "general"
        return predictions, invoked_general, {
            "route_type": route_types,
            "expert_confidence": min_distance.detach().cpu().tolist(),
        }


class FedEGSDSServer(BaseFederatedServer):
    def __init__(self, config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer=None, public_dataset=None):
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)

        checkpoint_path = (
            config.dataset.difficulty_checkpoint
            if bool(getattr(config.federated, "general_init_from_teacher", False))
            else None
        )
        self.general_model = AsymmetricGeneralModel(
            num_classes=config.model.num_classes,
            knowledge_dim=config.model.knowledge_dim,
            checkpoint_path=checkpoint_path,
            pretrained_imagenet=bool(getattr(config.federated, "general_pretrain_imagenet_init", False)),
        ).to(self.device)
        self.general_model.freeze()

        self.global_expert_prior = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)

        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
        self.clients: Dict[str, FedEGSDSClient] = {
            client_id: FedEGSDSClient(client_id, dataset, config, data_module)
            for client_id, dataset in client_datasets.items()
        }
        for client in self.clients.values():
            client.initialize_personal_state(self.global_expert_prior)

        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.route_distance_threshold = float(getattr(config.inference, "route_distance_threshold", 0.3))

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegsd-s round %d sampled clients=%s", round_idx, sampled_ids)

            updates = [
                self.clients[client_id].train_round(self.global_expert_prior, self.general_model, round_idx)
                for client_id in sampled_ids
            ]
            self._update_global_expert_prior(updates)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegsd-s-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegsd-s-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegsd-s-routed")

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
                routed_accuracy=aggregate["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
            )

            LOGGER.info(
                "fedegsd-s round %d | expert_loss=%.4f | routed=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f",
                round_idx,
                avg_expert_loss,
                aggregate["accuracy"],
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegsd_s", avg_expert_loss, round_idx)
                self._log_auxiliary_accuracy_metrics(
                    "fedegsd_s",
                    round_idx,
                    expert_eval["aggregate"]["accuracy"],
                    general_eval["aggregate"]["accuracy"],
                )

            self._log_round_metrics("fedegsd_s", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best_snapshot(
                round_idx,
                round_metrics,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best_snapshot()
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        route_export_path = self._build_route_export_path("fedegsd_s_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            prefix="fedegsd_s_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegsd_s_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegsd_s_final_general")
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
        expert_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, mode="expert_only")
        general_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, mode="general_only")

        return {
            "algorithm": "fedegsd-s",
            "metrics": {
                "accuracy": routed_eval["aggregate"]["accuracy"],
                "global_accuracy": routed_eval["aggregate"]["accuracy"],
                "local_accuracy": routed_eval["macro"]["accuracy"],
                "routed_accuracy": routed_eval["aggregate"]["accuracy"],
                "hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "hard_sample_recall": routed_eval["aggregate"]["hard_recall"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["aggregate"]["accuracy"],
                "general_only_accuracy": general_eval["aggregate"]["accuracy"],
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

    def _update_global_expert_prior(self, updates: List[FedEGSDSClientUpdate]) -> None:
        aggregated_delta = average_weighted_deltas((update.num_samples, update.expert_delta) for update in updates)
        updated_state = self.global_expert_prior.state_dict()
        for key, delta in aggregated_delta.items():
            if key in updated_state:
                updated_state[key] = updated_state[key] + delta.to(updated_state[key].device)
        self.global_expert_prior.load_state_dict(updated_state)

    def _predict_expert_only(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        logits = client.expert_model(images)
        return logits.argmax(dim=1), 0

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        logits = self.general_model(images)
        return logits.argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        std_multiplier = float(getattr(self.config.inference, "route_distance_std_multiplier", 1.5))
        return client.distance_route(images, self.general_model, self.route_distance_threshold, std_multiplier=std_multiplier)

    def _maybe_update_best_snapshot(self, round_idx: int, round_metrics: RoundMetrics, expert_accuracy: float, general_accuracy: float) -> None:
        if self.best_snapshot is not None:
            previous_accuracy = float(self.best_snapshot["routed_accuracy"])
            if round_metrics.routed_accuracy <= previous_accuracy + 1e-8:
                return

        self.best_snapshot = {
            "round_idx": round_idx,
            "routed_accuracy": round_metrics.routed_accuracy,
            "general_accuracy": general_accuracy,
            "expert_accuracy": expert_accuracy,
            "avg_client_loss": round_metrics.avg_client_loss,
            "general_model_state": self._clone_state(self.general_model),
            "global_expert_prior_state": self._clone_state(self.global_expert_prior),
            "client_expert_states": {client_id: self._clone_state(client.expert_model) for client_id, client in self.clients.items()},
        }
        LOGGER.info(
            "Updated fedegsd-s best checkpoint | round=%d | routed_acc=%.4f | general_acc=%.4f | expert_acc=%.4f",
            round_idx,
            round_metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )

    def _restore_best_snapshot(self) -> None:
        if self.best_snapshot is None:
            return

        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        self.general_model.freeze()
        self.global_expert_prior.load_state_dict(self.best_snapshot["global_expert_prior_state"])
        for client_id, client in self.clients.items():
            client.expert_model.load_state_dict(self.best_snapshot["client_expert_states"][client_id])
            client._compute_prototypes()
        LOGGER.info(
            "Restored fedegsd-s best checkpoint | round=%d | routed_acc=%.4f",
            int(self.best_snapshot["round_idx"]),
            float(self.best_snapshot["routed_accuracy"]),
        )

    def _clone_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
