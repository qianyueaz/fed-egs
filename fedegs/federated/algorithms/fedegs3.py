"""FedEGS-3: Mutual Knowledge Distillation with Class Prototypes.

Architecture (v3 rewrite — five core improvements):

1. **Single-workbench GPU model** — only one SmallCNN lives on GPU at a time.
   Each client stores its weights as a CPU state_dict.  Before training, the
   server loads the client's weights into the shared workbench; afterward it
   copies the updated weights back to CPU.  This caps GPU memory at O(1)
   regardless of client count.

2. **In-train prototype interception** — prototypes are harvested from the
   *last epoch* of local training for free, using the logits that are already
   computed for the CE/KD loss.  No extra forward pass.

3. **Entropy-weighted logit aggregation** — replaces max-confidence weighting
   with temperature-smoothed Shannon entropy.  Samples whose entropy exceeds
   80% of the theoretical maximum are hard-zeroed, eliminating pure-noise
   contributions from Non-IID clients.

4. **EMA prototype memory bank** — the server maintains a persistent
   ``global_class_means`` tensor that is updated with exponential moving
   average (momentum 0.9) each batch, smoothing out per-batch class sampling
   noise in the prototype regularisation loss.

5. **Persistent server optimizer** — the SGD optimizer for the general model
   is created once in ``__init__`` and reused across all federated rounds,
   preserving momentum buffers and accelerating convergence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

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
    class_prototypes: Dict[int, torch.Tensor]
    class_counts: Dict[int, int]


# ---------------------------------------------------------------------------
# Client  (改造一：瘦身为数据容器，不持有 GPU 模型)
# ---------------------------------------------------------------------------

class FedEGS3Client:
    """Lightweight client — holds only a CPU state_dict, no GPU model."""

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        num_classes: int,
        config,
        data_module,
        initial_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.num_classes = num_classes
        self.config = config
        self.data_module = data_module
        # Store weights on CPU — never on GPU
        self.state_dict_cpu: Dict[str, torch.Tensor] = {
            k: v.cpu().clone() for k, v in initial_state_dict.items()
        }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedEGS3Server(BaseFederatedServer):
    """Server with single-workbench training and EMA prototype memory."""

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
            config, client_datasets, client_test_datasets,
            data_module, test_hard_indices, writer,
            public_dataset=public_dataset,
        )
        if self.public_dataset is None or len(self.public_dataset) == 0:
            raise ValueError("FedEGS-3 requires a non-empty public distillation dataset.")

        num_classes = config.model.num_classes

        # ---- GPU models (only these two live on GPU) ----
        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=num_classes,
        ).to(self.device)

        # Single expert workbench — shared across all clients
        self.expert_workbench = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)

        # ---- Lightweight client objects (CPU only) ----
        # Generate one random init and share it as the starting point
        init_state = {k: v.cpu().clone() for k, v in self.expert_workbench.state_dict().items()}
        self.clients = {
            cid: FedEGS3Client(
                cid, ds, num_classes, config, data_module,
                initial_state_dict=init_state,
            )
            for cid, ds in client_datasets.items()
        }

        self.public_eval_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)
        self.last_history: List[RoundMetrics] = []

        # ---- 改造四：EMA prototype memory bank ----
        self.global_class_means = torch.zeros(num_classes, num_classes, device=self.device)
        self.class_means_initialized = torch.zeros(num_classes, dtype=torch.bool, device=self.device)

        # ---- 改造五：persistent server optimizer ----
        self.server_optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=config.federated.distill_lr,
            momentum=config.federated.local_momentum,
            weight_decay=config.federated.local_weight_decay,
        )

    # ---- training loop ----------------------------------------------------

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []

        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegs3 round %d sampled clients=%s", round_idx, sampled_ids)

            updates: List[FedEGS3ClientUpdate] = []
            for client_id in sampled_ids:
                update = self._train_single_client(client_id, round_idx)
                updates.append(update)

            teacher_logits = self._aggregate_public_logits(updates)
            global_prototypes = self._aggregate_class_prototypes(updates)
            distill_loss = self._distill_general_model(teacher_logits, global_prototypes)

            avg_loss = sum(u.loss for u in updates) / max(len(updates), 1)

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
                aggregate["accuracy"], aggregate["hard_accuracy"],
                aggregate["invocation_rate"], current_alpha,
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
                self._log_auxiliary_accuracy_metrics(
                    "fedegs3", round_idx,
                    expert_eval["aggregate"]["accuracy"],
                    general_eval["aggregate"]["accuracy"],
                )

            self._log_round_metrics("fedegs3", round_metrics)
            metrics.append(round_metrics)

        self.last_history = metrics
        return metrics

    # ---- 改造一：单工作台客户端训练 ----

    def _train_single_client(self, client_id: str, round_idx: int) -> FedEGS3ClientUpdate:
        """Load client weights → train on workbench → save weights back to CPU."""
        client = self.clients[client_id]

        # Load this client's weights onto the GPU workbench
        self.expert_workbench.load_state_dict(client.state_dict_cpu)

        local_loader = client.data_module.make_loader(client.dataset, shuffle=True)

        # Train (改造二: prototypes are harvested inside the last epoch)
        loss, class_prototypes, class_counts = self._train_with_distillation(
            self.expert_workbench, self.general_model, local_loader, round_idx,
        )

        # Public logits
        public_logits = self._predict_expert_logits(self.expert_workbench)

        # Save updated weights back to CPU
        client.state_dict_cpu = {
            k: v.detach().cpu().clone() for k, v in self.expert_workbench.state_dict().items()
        }

        # Free GPU cache between clients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return FedEGS3ClientUpdate(
            client_id=client_id,
            num_samples=len(client.dataset),
            loss=loss,
            public_logits=public_logits,
            class_prototypes=class_prototypes,
            class_counts=class_counts,
        )

    # ---- 改造二：训练中截获原型 ----

    def _train_with_distillation(
        self,
        expert_model: nn.Module,
        teacher_model: nn.Module,
        loader: DataLoader,
        round_idx: int,
    ) -> Tuple[float, Dict[int, torch.Tensor], Dict[int, int]]:
        """Train expert with CE + KD.  Harvest prototypes from last epoch."""
        distill_alpha = getattr(self.config.federated, "distill_alpha", 0.5)
        warmup_rounds = getattr(self.config.federated, "kd_warmup_rounds", 10)
        temperature = self.config.federated.distill_temperature
        local_epochs = self.config.federated.local_epochs
        num_classes = self.config.model.num_classes

        current_alpha = distill_alpha if round_idx > warmup_rounds else 0.0

        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        expert_model.train()
        if current_alpha > 0:
            teacher_model.eval()

        total_loss = 0.0
        total_batches = 0

        # Prototype accumulators (only active in the last epoch)
        proto_accum: Dict[int, torch.Tensor] = {}
        proto_counts: Dict[int, int] = {}

        for epoch in range(1, local_epochs + 1):
            is_last_epoch = (epoch == local_epochs)

            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    teacher_logits = teacher_model(images) if current_alpha > 0 else None

                optimizer.zero_grad(set_to_none=True)
                student_logits = expert_model(images)

                ce_loss = criterion(student_logits, targets)

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

                # Intercept logits for prototype computation (last epoch only)
                if is_last_epoch:
                    detached = student_logits.detach().cpu()
                    for logit, target in zip(detached, targets.cpu()):
                        cls = int(target.item())
                        if cls not in proto_accum:
                            proto_accum[cls] = torch.zeros(num_classes)
                            proto_counts[cls] = 0
                        proto_accum[cls] += logit
                        proto_counts[cls] += 1

        # Finalize prototypes
        prototypes: Dict[int, torch.Tensor] = {}
        for cls in proto_accum:
            prototypes[cls] = proto_accum[cls] / proto_counts[cls]

        return total_loss / max(total_batches, 1), prototypes, proto_counts

    def _predict_expert_logits(self, expert_model: nn.Module) -> torch.Tensor:
        """Run expert on public set to get logits for server aggregation."""
        expert_model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for images, _, _ in self.public_eval_loader:
                images = images.to(self.device)
                outputs.append(expert_model(images).detach().cpu())
        return torch.cat(outputs, dim=0)

    # ---- 改造三：熵值加权聚合 ----

    def _aggregate_public_logits(self, updates: List[FedEGS3ClientUpdate]) -> torch.Tensor:
        """Entropy-weighted logit aggregation with hard noise cutoff.

        1. Temperature-smooth each client's logits (T=2.0 hardcoded).
        2. Compute per-sample Shannon entropy.
        3. Zero-out samples above 80% of max entropy (pure noise).
        4. Weight = num_samples * (1 - normalized_entropy).
        """
        num_classes = updates[0].public_logits.size(1)
        max_entropy = math.log(num_classes)  # ln(C), maximum possible entropy
        entropy_cutoff = 0.8 * max_entropy
        smooth_temp = 2.0

        aggregate = torch.zeros_like(updates[0].public_logits)
        weight_sum = torch.zeros(updates[0].public_logits.size(0), 1)

        for u in updates:
            # Temperature-smoothed probabilities
            probs = torch.softmax(u.public_logits / smooth_temp, dim=1)

            # Shannon entropy per sample: -sum(p * log(p))
            log_probs = torch.log(probs.clamp(min=1e-8))
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # [N, 1]

            # Normalized entropy in [0, 1]
            norm_entropy = entropy / max_entropy

            # Hard cutoff: zero weight for samples above threshold
            mask = (entropy.squeeze(1) <= entropy_cutoff).float().unsqueeze(1)  # [N, 1]

            # Weight = num_samples * (1 - norm_entropy) * mask
            sample_weight = float(u.num_samples) * (1.0 - norm_entropy) * mask

            aggregate += u.public_logits * sample_weight
            weight_sum += sample_weight

        aggregate /= weight_sum.clamp(min=1e-8)
        return aggregate

    def _aggregate_class_prototypes(
        self,
        updates: List[FedEGS3ClientUpdate],
    ) -> Dict[int, torch.Tensor]:
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

    # ---- 改造四 + 五：EMA 记忆库 + 持久化优化器蒸馏 ----

    def _distill_general_model(
        self,
        teacher_logits: torch.Tensor,
        global_prototypes: Dict[int, torch.Tensor],
    ) -> float:
        """Distill with CE + KL + EMA-smoothed prototype regularisation.

        Uses the persistent ``self.server_optimizer`` (改造五) and updates
        ``self.global_class_means`` with EMA (改造四) for smooth proto loss.
        """
        temperature = self.config.federated.distill_temperature
        prototype_weight = getattr(self.config.federated, "prototype_weight", 0.01)
        server_ce_weight = getattr(self.config.federated, "server_ce_weight", 0.5)
        mixup_alpha = getattr(self.config.federated, "server_mixup_alpha", 0.4)
        ema_momentum = 0.9

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

                # Mixup
                if mixup_alpha > 0 and batch_size > 1:
                    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
                    lam = max(lam, 1.0 - lam)
                    perm = torch.randperm(batch_size, device=self.device)
                    images_mixed = lam * images + (1.0 - lam) * images[perm]
                    batch_teacher_mixed = lam * batch_teacher + (1.0 - lam) * batch_teacher[perm]
                    targets_a, targets_b = targets, targets[perm]
                else:
                    images_mixed = images
                    batch_teacher_mixed = batch_teacher
                    targets_a, targets_b = targets, targets
                    lam = 1.0

                # 改造五：use persistent optimizer
                self.server_optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images_mixed)

                ce_loss = (
                    lam * F.cross_entropy(student_logits, targets_a)
                    + (1.0 - lam) * F.cross_entropy(student_logits, targets_b)
                )

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(batch_teacher_mixed / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                # 改造四：EMA update of global_class_means, then compute proto loss
                with torch.no_grad():
                    for cls in range(num_classes):
                        cls_mask = targets_a == cls
                        if not cls_mask.any():
                            continue
                        batch_cls_mean = student_logits[cls_mask].mean(dim=0).detach()
                        if self.class_means_initialized[cls]:
                            self.global_class_means[cls] = (
                                ema_momentum * self.global_class_means[cls]
                                + (1.0 - ema_momentum) * batch_cls_mean
                            )
                        else:
                            self.global_class_means[cls] = batch_cls_mean
                            self.class_means_initialized[cls] = True

                # Proto loss: EMA memory vs. client prototypes
                proto_loss = torch.tensor(0.0, device=self.device)
                classes_active = 0
                if prototype_weight > 0 and proto_mask.sum() > 0:
                    for cls in range(num_classes):
                        if proto_mask[cls] == 0 or not self.class_means_initialized[cls]:
                            continue
                        cos_sim = F.cosine_similarity(
                            self.global_class_means[cls].unsqueeze(0),
                            proto_target[cls].unsqueeze(0),
                        )
                        proto_loss = proto_loss + (1.0 - cos_sim.squeeze())
                        classes_active += 1

                    if classes_active > 0:
                        proto_loss = proto_loss / classes_active

                loss = (
                    server_ce_weight * ce_loss
                    + (1.0 - server_ce_weight) * kd_loss
                    + prototype_weight * proto_loss
                )
                loss.backward()
                self.server_optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        return sum(losses) / max(len(losses), 1)

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
                "expert": model_memory_mb(self.expert_workbench),
                "general": model_memory_mb(self.general_model),
            },
            "artifacts": {
                "route_csv": str(route_export_path),
            },
        }

    # ---- predictors (use workbench for evaluation) ----

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        logits = self.general_model(images)
        return logits.argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        client = self.clients[client_id]
        self.expert_workbench.load_state_dict(client.state_dict_cpu)
        self.expert_workbench.eval()
        logits = self.expert_workbench(images)
        return logits.argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        self.expert_workbench.load_state_dict(client.state_dict_cpu)
        return self._dual_threshold_route(self.expert_workbench, self.general_model, images)
