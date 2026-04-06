"""FedEGS-3: Mutual Knowledge Distillation with Class Prototypes.

Extends FedEGS-2 with two key optimizations:

1. **Local knowledge distillation** — the server's general model acts as a
   teacher during client-side expert training.  The expert's loss becomes:
       total = CE(expert_logits, hard_label) + alpha * KL(expert_logits, teacher_soft_label)
   The general model is only used for forward passes on the client (no
   backpropagation), keeping the overhead minimal on edge devices.

2. **Class-wise logit prototypes** — each client computes per-class average
   logits over its local data and uploads them alongside the public-set logits.
   The server aggregates prototypes across clients and adds a regularisation
   term when distilling the general model, encouraging its class-level outputs
   to align with the true local distributions reported by all clients.

The bidirectional knowledge flow creates a virtuous cycle: a stronger general
model produces better soft labels → experts improve → experts produce better
public logits and prototypes → general model improves further.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import (
    BaseFederatedClient,
    BaseFederatedServer,
    LOGGER,
    RoundMetrics,
)
from fedegs.models import SmallCNN, WidthScalableResNet, model_memory_mb


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FedEGS3ClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_logits: torch.Tensor
    class_prototypes: Dict[int, torch.Tensor]  # class_id -> avg logit vector
    class_counts: Dict[int, int]               # class_id -> sample count


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedEGS3Client(BaseFederatedClient):
    """Client that trains an expert with local KD from the general model."""

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        num_classes: int,
        device: str,
        config,
        data_module,
    ) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.num_classes = num_classes
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)

    # ---- public API -------------------------------------------------------

    def train(
        self,
        general_model: nn.Module,
        public_loader: DataLoader,
        round_idx: int = 1,
    ) -> FedEGS3ClientUpdate:
        """Run one round of local training with mutual distillation."""
        local_loader = self.data_module.make_loader(self.dataset, shuffle=True)

        # Step 1: train expert with combined CE + KD loss
        loss = self._train_with_distillation(general_model, local_loader, round_idx)

        # Step 2: generate public-set logits (same as FedEGS-2)
        public_logits = self._predict_public_logits(public_loader)

        # Step 3: compute class-wise prototypes from local data
        # Use a fresh non-shuffled loader to ensure deterministic full coverage
        proto_loader = self.data_module.make_loader(self.dataset, shuffle=False)
        class_prototypes, class_counts = self._compute_class_prototypes(proto_loader)

        return FedEGS3ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            public_logits=public_logits,
            class_prototypes=class_prototypes,
            class_counts=class_counts,
        )

    # ---- internal ---------------------------------------------------------

    def _train_with_distillation(
        self,
        teacher_model: nn.Module,
        loader: DataLoader,
        round_idx: int = 1,
    ) -> float:
        """Train the expert model with CE + alpha * KL(expert, teacher).

        During the first ``kd_warmup_rounds`` rounds the teacher (general model)
        is too weak to provide useful guidance, so ``current_alpha`` is forced
        to 0 — the expert trains with pure CE, and the teacher forward pass is
        skipped entirely to save compute.
        """
        distill_alpha: float = getattr(self.config.federated, "distill_alpha", 0.5)
        warmup_rounds: int = getattr(self.config.federated, "kd_warmup_rounds", 10)
        temperature: float = self.config.federated.distill_temperature

        # Dynamic alpha: cut off distillation during warmup
        current_alpha = distill_alpha if round_idx > warmup_rounds else 0.0

        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.expert_model.train()
        if current_alpha > 0:
            teacher_model.eval()

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Teacher forward — only when alpha is active
                with torch.no_grad():
                    teacher_logits = teacher_model(images) if current_alpha > 0 else None

                # Student forward
                optimizer.zero_grad(set_to_none=True)
                student_logits = self.expert_model(images)

                # Hard-label loss
                ce_loss = criterion(student_logits, targets)

                # Soft-label distillation loss (skipped during warmup)
                if current_alpha > 0 and teacher_logits is not None:
                    kd_loss = F.kl_div(
                        F.log_softmax(student_logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction="batchmean",
                    ) * (temperature ** 2)
                else:
                    kd_loss = torch.tensor(0.0, device=self.device)

                loss = (1.0 - current_alpha) * ce_loss + current_alpha * kd_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _predict_public_logits(self, public_loader: DataLoader) -> torch.Tensor:
        self.expert_model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for images, _, _ in public_loader:
                images = images.to(self.device)
                outputs.append(self.expert_model(images).detach().cpu())
        return torch.cat(outputs, dim=0)

    def _compute_class_prototypes(
        self,
        loader: DataLoader,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        """Compute per-class average logit vectors over local data."""
        self.expert_model.eval()
        accumulators: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                logits = self.expert_model(images).detach().cpu()
                for logit, target in zip(logits, targets):
                    cls = int(target.item())
                    if cls not in accumulators:
                        accumulators[cls] = torch.zeros(self.num_classes)
                        counts[cls] = 0
                    accumulators[cls] += logit
                    counts[cls] += 1

        prototypes: Dict[int, torch.Tensor] = {}
        for cls in accumulators:
            prototypes[cls] = accumulators[cls] / counts[cls]

        return prototypes, counts


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedEGS3Server(BaseFederatedServer):
    """Server with prototype-regularised distillation."""

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
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )
        if self.public_dataset is None or len(self.public_dataset) == 0:
            raise ValueError("FedEGS-3 requires a non-empty public distillation dataset.")

        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)

        self.clients = {
            client_id: FedEGS3Client(
                client_id, dataset, config.model.num_classes,
                config.federated.device, config, data_module,
            )
            for client_id, dataset in client_datasets.items()
        }

        self.public_eval_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)
        self.last_history: List[RoundMetrics] = []

    # ---- training loop ----------------------------------------------------

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []

        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegs3 round %d sampled clients=%s", round_idx, sampled_ids)

            updates: List[FedEGS3ClientUpdate] = []
            for client_id in sampled_ids:
                updates.append(
                    self.clients[client_id].train(self.general_model, self.public_eval_loader, round_idx)
                )

            # Aggregate public logits (same as FedEGS-2)
            teacher_logits = self._aggregate_public_logits(updates)

            # Aggregate class prototypes across sampled clients
            global_prototypes = self._aggregate_class_prototypes(updates)

            # Distill general model with prototype regularisation
            distill_loss = self._distill_general_model(teacher_logits, global_prototypes)

            avg_loss = sum(u.loss for u in updates) / max(len(updates), 1)

            # Evaluate
            expert_eval = self._evaluate_predictor_on_client_tests(
                self._predict_expert_only, prefix="fedegs3-expert",
            )
            general_eval = self._evaluate_predictor_on_client_tests(
                self._predict_general_only, prefix="fedegs3-general",
            )
            routed_eval = self._evaluate_predictor_on_client_tests(
                self._predict_routed, prefix="fedegs3-routed",
            )
            aggregate = routed_eval["aggregate"]
            round_metrics = RoundMetrics(
                round_idx, avg_loss,
                aggregate["accuracy"],
                aggregate["hard_accuracy"],
                aggregate["invocation_rate"],
            )

            warmup_rounds = getattr(self.config.federated, "kd_warmup_rounds", 10)
            kd_active = round_idx > warmup_rounds
            current_alpha = self.config.federated.distill_alpha if kd_active else 0.0

            LOGGER.info(
                "fedegs3 round %d | local_loss=%.4f | distill_loss=%.4f "
                "| routed_acc=%.4f | hard_acc=%.4f | invocation=%.4f | kd_alpha=%.2f%s",
                round_idx, avg_loss, distill_loss,
                aggregate["accuracy"],
                aggregate["hard_accuracy"],
                aggregate["invocation_rate"],
                current_alpha,
                "" if kd_active else " [warmup]",
            )
            LOGGER.info(
                "fedegs3 auxiliary round %d | expert_acc=%.4f | general_acc=%.4f",
                round_idx,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )

            if self.writer is not None:
                self.writer.add_scalar("distill_loss/fedegs3", distill_loss, round_idx)
                self.writer.add_scalar("prototype/num_classes_covered", len(global_prototypes), round_idx)
                self.writer.add_scalar("prototype/total_samples", sum(
                    u.class_counts.get(c, 0) for u in updates for c in u.class_counts
                ), round_idx)
                self._log_auxiliary_accuracy_metrics(
                    "fedegs3", round_idx,
                    expert_eval["aggregate"]["accuracy"],
                    general_eval["aggregate"]["accuracy"],
                )

            self._log_round_metrics("fedegs3", round_metrics)
            metrics.append(round_metrics)

        self.last_history = metrics
        return metrics

    # ---- evaluation -------------------------------------------------------

    def evaluate_baselines(self, test_dataset: Dataset):
        route_export_path = self._build_route_export_path("fedegs3_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            prefix="fedegs3_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(
            self._predict_expert_only, prefix="fedegs3_final_expert",
        )
        general_eval = self._evaluate_predictor_on_client_tests(
            self._predict_general_only, prefix="fedegs3_final_general",
        )
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0

        return {
            "algorithm": "fedegs3",
            "metrics": {
                "routed_accuracy": routed_eval["aggregate"]["accuracy"],
                "routed_hard_accuracy": routed_eval["aggregate"]["hard_accuracy"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["aggregate"]["accuracy"],
                "expert_only_recall_macro": expert_eval["aggregate"]["recall_macro"],
                "general_only_accuracy": general_eval["aggregate"]["accuracy"],
                "general_only_recall_macro": general_eval["aggregate"]["recall_macro"],
                "public_dataset_size": len(self.public_dataset),
                "final_training_loss": final_loss,
            },
            "client_metrics": {
                "routed": routed_eval["clients"],
                "expert_only": expert_eval["clients"],
                "general_only": general_eval["clients"],
            },
            "memory_mb": {
                "expert": model_memory_mb(next(iter(self.clients.values())).expert_model),
                "general": model_memory_mb(self.general_model),
            },
            "artifacts": {
                "route_csv": str(route_export_path),
            },
        }

    # ---- aggregation helpers ----------------------------------------------

    def _aggregate_public_logits(self, updates: List[FedEGS3ClientUpdate]) -> torch.Tensor:
        """Aggregate public-set logits with per-sample confidence weighting.

        Instead of a flat sample-count average, each client's contribution to
        each public sample is scaled by its prediction confidence (max softmax
        probability).  Clients that are confident about a sample get more say;
        clients that are guessing contribute less noise.
        """
        aggregate = torch.zeros_like(updates[0].public_logits)
        weight_sum = torch.zeros(updates[0].public_logits.size(0), 1)

        for u in updates:
            probs = torch.softmax(u.public_logits, dim=1)
            confidence, _ = torch.max(probs, dim=1, keepdim=True)  # [num_public, 1]
            weight = float(u.num_samples) * confidence
            aggregate += u.public_logits * weight
            weight_sum += weight

        aggregate /= weight_sum.clamp(min=1e-8)
        return aggregate

    def _aggregate_class_prototypes(
        self,
        updates: List[FedEGS3ClientUpdate],
    ) -> Dict[int, torch.Tensor]:
        """Weighted average of per-class prototypes across all sampled clients."""
        accumulators: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        for u in updates:
            for cls, proto in u.class_prototypes.items():
                n = u.class_counts.get(cls, 0)
                if n == 0:
                    continue
                if cls not in accumulators:
                    accumulators[cls] = torch.zeros_like(proto)
                    counts[cls] = 0
                accumulators[cls] += proto * n
                counts[cls] += n

        global_prototypes: Dict[int, torch.Tensor] = {}
        for cls in accumulators:
            global_prototypes[cls] = accumulators[cls] / counts[cls]

        return global_prototypes

    # ---- distillation with prototype regularisation -----------------------

    def _distill_general_model(
        self,
        teacher_logits: torch.Tensor,
        global_prototypes: Dict[int, torch.Tensor],
    ) -> float:
        """Distill the general model on public data.

        Loss = ce_weight * CE(general, hard_label)
             + (1 - ce_weight) * KL(general, aggregated_expert_logits)
             + prototype_weight * cosine_distance(general_class_mean, prototype)

        The CE term anchors the general model to ground-truth labels on the
        public split, preventing it from drifting when expert logits are noisy.
        """
        temperature = self.config.federated.distill_temperature
        prototype_weight: float = getattr(self.config.federated, "prototype_weight", 0.01)
        server_ce_weight: float = getattr(self.config.federated, "server_ce_weight", 0.5)

        optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )

        # Pre-compute prototype regularisation target tensor (num_classes, num_classes)
        num_classes = self.config.model.num_classes
        proto_target = torch.zeros(num_classes, num_classes, device=self.device)
        proto_mask = torch.zeros(num_classes, device=self.device)
        for cls, proto in global_prototypes.items():
            proto_target[cls] = proto.to(self.device)
            proto_mask[cls] = 1.0

        losses: List[float] = []
        teacher_cursor = 0

        for _ in range(self.config.federated.distill_epochs):
            self.general_model.train()
            teacher_cursor = 0

            for images, targets, _ in self.public_eval_loader:
                batch_size = images.size(0)
                batch_teacher = teacher_logits[teacher_cursor: teacher_cursor + batch_size].to(self.device)
                teacher_cursor += batch_size
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)

                # 1. Ground-truth CE loss on public data (hard labels)
                ce_loss = F.cross_entropy(student_logits, targets)

                # 2. KL distillation loss from aggregated expert logits (soft labels)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(batch_teacher / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                # 3. Prototype regularisation (cosine similarity)
                proto_loss = torch.tensor(0.0, device=self.device)
                classes_in_batch = 0
                if prototype_weight > 0 and proto_mask.sum() > 0:
                    for cls in range(num_classes):
                        if proto_mask[cls] == 0:
                            continue
                        cls_mask = targets == cls
                        if not cls_mask.any():
                            continue
                        cls_mean_logit = student_logits[cls_mask].mean(dim=0)
                        cos_sim = F.cosine_similarity(
                            cls_mean_logit.unsqueeze(0),
                            proto_target[cls].unsqueeze(0),
                        )
                        proto_loss = proto_loss + (1.0 - cos_sim.squeeze())
                        classes_in_batch += 1

                    if classes_in_batch > 0:
                        proto_loss = proto_loss / classes_in_batch

                # Combined loss: CE anchors to truth, KL absorbs expert dark knowledge
                loss = (
                    server_ce_weight * ce_loss
                    + (1.0 - server_ce_weight) * kd_loss
                    + prototype_weight * proto_loss
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        return sum(losses) / max(len(losses), 1)

    # ---- predictors -------------------------------------------------------

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
        return self._dual_threshold_route(client.expert_model, self.general_model, images)
