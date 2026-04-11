"""
FedEGS-D: Federated Expert-General System - Decoupled

Core design:
  - Client trains a lightweight SmallCNN expert on local data (FP32/FP16).
  - Client uploads expert MODEL PARAMETERS (not logits) to the server.
  - Server instantiates all received expert models, runs them on a public
    proxy dataset to extract logits, averages into pseudo-labels.
  - Server distills these pseudo-labels into a large general model (ResNet-18).
  - Server quantises general model (simulated INT8) and broadcasts it.
  - At inference the client first runs the expert; if confidence is low
    it falls back to the (quantised) general model.

Differences from FedEGS-2:
  - Clients upload expert *weights*, NOT logits/features on public data.
  - All public-data inference happens server-side (zero extra client compute).
  - No feature-hint loss, relation loss, or anchor loss on the client.
  - Server-side distillation uses simple ensemble averaging + KL divergence.
  - Simulated INT8 quantisation for the broadcast general model.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import (
    BaseFederatedClient,
    BaseFederatedServer,
    LOGGER,
    RoundMetrics,
)
from fedegs.models import (
    SmallCNN,
    build_teacher_model,
    estimate_model_flops,
    load_teacher_checkpoint,
    model_memory_mb,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FedEGSDClientUpdate:
    """What each client sends back to the server."""
    client_id: str
    num_samples: int
    loss: float
    expert_state_dict: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# General model wrapper (reused from fedegs2, simplified)
# ---------------------------------------------------------------------------

class GeneralModel(nn.Module):
    """Server-side general model built on top of a ResNet-18 teacher backbone."""

    def __init__(self, num_classes: int, pretrained_imagenet: bool = False) -> None:
        super().__init__()
        backbone = build_teacher_model(
            num_classes=num_classes,
            pretrained_imagenet=pretrained_imagenet,
        )
        self.num_classes = num_classes
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier.load_state_dict(backbone.fc.state_dict())
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedEGSDClient(BaseFederatedClient):
    """
    Client-side logic for FedEGS-D.

    Each round the client:
      1. Trains its local expert model on private data (standard CE).
      2. Returns the expert model state_dict to the server.
    No public data inference happens on the client.
    """

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
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)

    def train_local(self) -> FedEGSDClientUpdate:
        """Run local SGD training on private data and return expert weights."""
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        loss = self._optimize_model(
            model=self.expert_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        # Clone state dict to CPU for upload
        state = {
            k: v.detach().cpu().clone()
            for k, v in self.expert_model.state_dict().items()
        }
        return FedEGSDClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            expert_state_dict=state,
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedEGSDServer(BaseFederatedServer):
    """
    Server-side logic for FedEGS-D.

    Each round:
      1. Sample clients, collect expert weights.
      2. Instantiate expert replicas, run on proxy dataset → ensemble logits.
      3. Distill ensemble logits into the general model (KL + CE).
      4. Evaluate routed / expert-only / general-only.
    """

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
            raise ValueError("FedEGS-D requires a non-empty public_dataset for server-side distillation.")

        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )

        # --- General model (server-side, full precision) ---
        self.general_model = GeneralModel(
            num_classes=config.model.num_classes,
            pretrained_imagenet=bool(config.federated.general_pretrain_imagenet_init),
        ).to(self.device)

        # --- Reference expert for FLOPs estimation ---
        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)

        # --- Clients ---
        self.clients: Dict[str, FedEGSDClient] = {
            cid: FedEGSDClient(cid, ds, config.model.num_classes, config.federated.device, config, data_module)
            for cid, ds in client_datasets.items()
        }

        # --- Public data loader ---
        self.public_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)

        # --- Distillation optimiser ---
        self.distill_optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            weight_decay=self.config.federated.local_weight_decay,
        )

        # --- Tracking ---
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict] = None

        # --- Per-client routing thresholds ---
        base_conf = float(config.inference.confidence_threshold)
        base_margin = float(config.inference.route_distance_threshold)
        self.client_confidence_thresholds = {cid: base_conf for cid in client_datasets}
        self.client_margin_thresholds = {cid: base_margin for cid in client_datasets}

    # ------------------------------------------------------------------
    # Optional: pretrain general model on public data before federation
    # ------------------------------------------------------------------

    def _pretrain_general_on_public(self) -> None:
        epochs = max(int(self.config.federated.general_pretrain_epochs), 0)
        if epochs == 0:
            return
        lr = float(self.config.federated.general_pretrain_lr)
        LOGGER.info("fedegsd pretrain general | epochs=%d | lr=%.4f", epochs, lr)

        optimizer = torch.optim.SGD(
            self.general_model.parameters(), lr=lr, momentum=0.9,
            weight_decay=float(self.config.federated.local_weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        shuffle_loader = self.data_module.make_loader(self.public_dataset, shuffle=True)

        best_acc, best_state = -1.0, None
        for epoch in range(1, epochs + 1):
            self.general_model.train()
            correct = total = 0
            for images, targets, _ in shuffle_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.general_model(images), targets)
                loss.backward()
                optimizer.step()
                correct += int((self.general_model(images).argmax(1) == targets).sum())
                total += targets.size(0)
            scheduler.step()
            acc = correct / max(total, 1)
            LOGGER.info("fedegsd pretrain epoch %d/%d | acc=%.4f", epoch, epochs, acc)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()}

        if best_state is not None:
            self.general_model.load_state_dict(best_state)
            LOGGER.info("fedegsd pretrain done | best_acc=%.4f", best_acc)

        # Reset distill optimizer
        self.distill_optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            weight_decay=self.config.federated.local_weight_decay,
        )

    # ------------------------------------------------------------------
    # Server-side ensemble logit extraction  (FedDF core)
    # ------------------------------------------------------------------

    def _extract_ensemble_logits(
        self,
        updates: List[FedEGSDClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        """
        Instantiate each client's expert model on the server, run on the
        public dataset, and produce per-sample averaged soft-labels.

        Returns dict with keys:
          - 'soft_labels': Tensor [N_public, num_classes]
          - 'entropy_weights': Tensor [N_public]  (higher = more certain)
        """
        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)

        # Collect per-client logits
        all_client_probs: List[torch.Tensor] = []
        all_client_entropy: List[torch.Tensor] = []

        for update in updates:
            expert = SmallCNN(
                num_classes=num_classes,
                base_channels=self.config.model.expert_base_channels,
            ).to(self.device)
            expert.load_state_dict(update.expert_state_dict)
            expert.eval()

            sample_probs_list: List[torch.Tensor] = []
            with torch.no_grad():
                for images, _, _ in self.public_loader:
                    images = images.to(self.device)
                    logits = expert(images)
                    probs = F.softmax(logits / temperature, dim=1)
                    sample_probs_list.append(probs.cpu())

            client_probs = torch.cat(sample_probs_list, dim=0)  # [N, C]
            # Entropy for uncertainty weighting
            log_num_classes = math.log(float(num_classes))
            entropy = -(client_probs * client_probs.clamp_min(1e-8).log()).sum(dim=1) / max(log_num_classes, 1e-8)
            weights = torch.exp(-entropy)  # low entropy → high weight

            all_client_probs.append(client_probs)
            all_client_entropy.append(weights)

            del expert  # free GPU memory

        # Uncertainty-weighted averaging across clients
        stacked_probs = torch.stack(all_client_probs, dim=0)      # [K, N, C]
        stacked_weights = torch.stack(all_client_entropy, dim=0)   # [K, N]

        # Per-sample per-client weight
        weight_sum = stacked_weights.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, N]
        normalised_weights = stacked_weights / weight_sum  # [K, N]

        # Weighted average: [K, N, C] * [K, N, 1] → sum over K → [N, C]
        soft_labels = (stacked_probs * normalised_weights.unsqueeze(-1)).sum(dim=0)
        # Re-normalise
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Overall certainty per sample (mean weight across clients)
        sample_certainty = stacked_weights.mean(dim=0)

        return {
            "soft_labels": soft_labels,
            "entropy_weights": sample_certainty,
        }

    # ------------------------------------------------------------------
    # Server-side distillation of general model
    # ------------------------------------------------------------------

    def _distill_general_model(
        self,
        ensemble: Dict[str, torch.Tensor],
        round_idx: int,
    ) -> Dict[str, float]:
        """Distill ensemble soft-labels into the general model."""
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)

        soft_labels_all = ensemble["soft_labels"]       # [N, C]
        weights_all = ensemble["entropy_weights"]       # [N]

        # Build index mapping for the public loader
        # The public loader yields (images, targets, indices) in the same order
        # as soft_labels_all because the loader is not shuffled.
        criterion_ce = nn.CrossEntropyLoss()

        self.general_model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        total_batches = 0
        offset = 0

        for _ in range(distill_epochs):
            offset = 0
            for images, targets, _ in self.public_loader:
                bs = images.size(0)
                images = images.to(self.device)
                targets = targets.to(self.device)

                batch_soft = soft_labels_all[offset:offset + bs].to(self.device)
                batch_weights = weights_all[offset:offset + bs].to(self.device)
                offset += bs

                self.distill_optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)

                # CE on true labels
                ce_loss = criterion_ce(student_logits, targets)

                # KL on soft labels (weighted)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                per_sample_kl = F.kl_div(
                    student_log_probs, batch_soft, reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                kl_loss = (per_sample_kl * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)

                loss = ce_loss + float(self.config.federated.logit_align_weight) * kl_loss
                loss.backward()
                self.distill_optimizer.step()

                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_kl += kl_loss.item()
                total_batches += 1

        denom = max(total_batches, 1)
        return {
            "total_loss": total_loss / denom,
            "ce_loss": total_ce / denom,
            "kl_loss": total_kl / denom,
        }

    # ------------------------------------------------------------------
    # Predictors for evaluation
    # ------------------------------------------------------------------

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        return self.general_model(images).argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        return self.clients[client_id].expert_model(images).argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        return self._confidence_route(
            self.clients[client_id].expert_model,
            self.general_model,
            images,
            confidence_threshold=self.client_confidence_thresholds[client_id],
            margin_threshold=self.client_margin_thresholds[client_id],
        )

    # ------------------------------------------------------------------
    # Adaptive routing threshold update
    # ------------------------------------------------------------------

    def _update_routing_thresholds(self, updates: List[FedEGSDClientUpdate]) -> None:
        step = max(float(self.config.inference.personalized_threshold_step), 0.0)
        margin_step = max(float(self.config.inference.personalized_margin_step), 0.0)
        teacher_gap_guard = float(self.config.inference.public_teacher_gap_guard)
        min_conf = float(self.config.inference.min_confidence_threshold)
        max_conf = float(self.config.inference.max_confidence_threshold)
        min_margin = float(self.config.inference.min_margin_threshold)
        max_margin = float(self.config.inference.max_margin_threshold)

        # General model predictions on public data
        general_preds = self._predict_public_general_labels()
        public_targets = self._build_public_target_lookup()

        for update in updates:
            cid = update.client_id
            threshold = self.client_confidence_thresholds[cid]
            margin = self.client_margin_thresholds[cid]

            # Expert accuracy on public data (via uploaded weights)
            expert = SmallCNN(
                num_classes=self.config.model.num_classes,
                base_channels=self.config.model.expert_base_channels,
            ).to(self.device)
            expert.load_state_dict(update.expert_state_dict)
            expert.eval()

            expert_correct = 0
            general_correct = 0
            total = 0
            with torch.no_grad():
                for images, targets, indices in self.public_loader:
                    images = images.to(self.device)
                    ep = expert(images).argmax(1).cpu()
                    for i, idx in enumerate(indices.tolist()):
                        t = public_targets[idx]
                        expert_correct += int(ep[i].item() == t)
                        general_correct += int(general_preds[idx] == t)
                        total += 1

            expert_acc = expert_correct / max(total, 1)
            general_acc = general_correct / max(total, 1)
            teacher_gap = general_acc - expert_acc

            if teacher_gap > teacher_gap_guard:
                threshold += step
                margin += margin_step
            elif teacher_gap < -teacher_gap_guard:
                threshold -= step
                margin -= margin_step

            self.client_confidence_thresholds[cid] = min(max(threshold, min_conf), max_conf)
            self.client_margin_thresholds[cid] = min(max(margin, min_margin), max_margin)
            del expert

    def _predict_public_general_labels(self) -> Dict[int, int]:
        preds: Dict[int, int] = {}
        self.general_model.eval()
        with torch.no_grad():
            for images, _, indices in self.public_loader:
                images = images.to(self.device)
                p = self.general_model(images).argmax(1).cpu().tolist()
                for idx, pred in zip(indices.tolist(), p):
                    preds[idx] = pred
        return preds

    def _build_public_target_lookup(self) -> Dict[int, int]:
        targets: Dict[int, int] = {}
        for _, t, indices in self.public_loader:
            for idx, target in zip(indices.tolist(), t.tolist()):
                targets[idx] = target
        return targets

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def _maybe_update_best_snapshot(
        self, round_idx: int, round_metrics: RoundMetrics,
        expert_acc: float, general_acc: float,
    ) -> bool:
        if self.best_snapshot is None:
            is_better = True
        else:
            is_better = round_metrics.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not is_better:
            return False

        self.best_snapshot = {
            "round_idx": round_idx,
            "routed_accuracy": round_metrics.routed_accuracy,
            "general_accuracy": general_acc,
            "expert_accuracy": expert_acc,
            "avg_client_loss": round_metrics.avg_client_loss,
            "general_model_state": {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()},
            "client_expert_states": {
                cid: {k: v.cpu().clone() for k, v in c.expert_model.state_dict().items()}
                for cid, c in self.clients.items()
            },
            "client_confidence_thresholds": dict(self.client_confidence_thresholds),
            "client_margin_thresholds": dict(self.client_margin_thresholds),
        }
        LOGGER.info(
            "fedegsd best snapshot | round=%d | routed=%.4f | general=%.4f | expert=%.4f",
            round_idx, round_metrics.routed_accuracy, general_acc, expert_acc,
        )
        return True

    def _restore_best_snapshot(self) -> None:
        if self.best_snapshot is None:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for cid, state in self.best_snapshot["client_expert_states"].items():
            self.clients[cid].expert_model.load_state_dict(state)
        self.client_confidence_thresholds = dict(self.best_snapshot["client_confidence_thresholds"])
        self.client_margin_thresholds = dict(self.best_snapshot["client_margin_thresholds"])
        LOGGER.info("fedegsd restored best snapshot from round %d", self.best_snapshot["round_idx"])

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        # Optional pretrain
        if bool(self.config.federated.general_pretrain_on_public):
            self._pretrain_general_on_public()

        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            # 1. Sample clients
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegsd round %d | clients=%s", round_idx, sampled_ids)

            # 2. Local expert training → upload weights
            updates: List[FedEGSDClientUpdate] = []
            for cid in sampled_ids:
                updates.append(self.clients[cid].train_local())

            # 3. Server-side: extract ensemble logits from uploaded experts
            ensemble = self._extract_ensemble_logits(updates)

            # 4. Server-side: distill into general model
            distill_stats = self._distill_general_model(ensemble, round_idx)

            # 5. Update routing thresholds
            self._update_routing_thresholds(updates)

            # 6. Evaluate
            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegsd-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegsd-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegsd-routed")

            avg_loss = sum(u.loss for u in updates) / max(len(updates), 1)
            agg = routed_eval["aggregate"]
            macro = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops, self.general_flops, agg["invocation_rate"], mode="routed",
            )

            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=avg_loss,
                routed_accuracy=agg["accuracy"],
                hard_accuracy=agg["hard_recall"],
                invocation_rate=agg["invocation_rate"],
                local_accuracy=macro["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
            )

            LOGGER.info(
                "fedegsd round %d | loss=%.4f | distill=%.4f | routed=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f | savings=%.4f",
                round_idx, avg_loss, distill_stats["total_loss"],
                agg["accuracy"], expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"], agg["hard_recall"],
                agg["invocation_rate"], compute_profile["savings_ratio"],
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegsd", avg_loss, round_idx)
                self.writer.add_scalar("distill_loss/fedegsd", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar("distill_ce/fedegsd", distill_stats["ce_loss"], round_idx)
                self.writer.add_scalar("distill_kl/fedegsd", distill_stats["kl_loss"], round_idx)
                self._log_auxiliary_accuracy_metrics(
                    "fedegsd", round_idx,
                    expert_eval["aggregate"]["accuracy"],
                    general_eval["aggregate"]["accuracy"],
                )

            self._log_round_metrics("fedegsd", round_metrics)
            metrics.append(round_metrics)

            self._maybe_update_best_snapshot(
                round_idx, round_metrics,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best_snapshot()
        return metrics

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def evaluate_baselines(self, test_dataset: Dataset):
        route_path = self._build_route_export_path("fedegsd_final_routed")
        routed = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsd_final_routed", route_export_path=route_path)
        expert = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd_final_expert")
        general = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd_final_general")

        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = 0
        if self.best_snapshot:
            final_loss = float(self.best_snapshot["avg_client_loss"])
            best_round = int(self.best_snapshot["round_idx"])

        routed_compute = self._build_compute_profile(self.expert_flops, self.general_flops, routed["aggregate"]["invocation_rate"], "routed")
        expert_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, "expert_only")
        general_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, "general_only")

        return {
            "algorithm": "fedegsd",
            "metrics": {
                "accuracy": routed["aggregate"]["accuracy"],
                "global_accuracy": routed["aggregate"]["accuracy"],
                "local_accuracy": routed["macro"]["accuracy"],
                "routed_accuracy": routed["aggregate"]["accuracy"],
                "hard_accuracy": routed["aggregate"]["hard_recall"],
                "hard_sample_recall": routed["aggregate"]["hard_recall"],
                "routed_hard_accuracy": routed["aggregate"]["hard_recall"],
                "general_invocation_rate": routed["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed["aggregate"]["precision_macro"],
                "recall_macro": routed["aggregate"]["recall_macro"],
                "f1_macro": routed["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert["aggregate"]["accuracy"],
                "general_only_accuracy": general["aggregate"]["accuracy"],
                "public_dataset_size": len(self.public_dataset),
                "final_training_loss": final_loss,
                "best_round": best_round,
            },
            "client_metrics": {
                "routed": routed["clients"],
                "expert_only": expert["clients"],
                "general_only": general["clients"],
            },
            "group_metrics": {
                "routed": routed["groups"],
                "expert_only": expert["groups"],
                "general_only": general["groups"],
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
                "route_csv": str(route_path),
            },
        }
