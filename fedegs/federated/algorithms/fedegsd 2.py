"""
FedEGS-D: Federated Expert-General System - Decoupled

Aligned with FedDF (NeurIPS 2020) ensemble distillation:
  - Client trains SmallCNN expert on local data (pure CE), uploads weights.
  - Server instantiates experts, runs on an EXTERNAL UNLABELED proxy dataset
    (e.g. CIFAR-100 or ImageNet-32, NOT a subset of CIFAR-10 training data).
  - Server averages raw logits across experts (AVGLOGITS), then applies softmax.
  - Server distills ensemble soft-labels into general model via pure KL divergence
    (no CE on true labels — proxy dataset labels are unused / unavailable).
  - Adam + cosine annealing scheduler for distillation (per FedDF Section 4.1).
  - At inference: expert-first with confidence routing to general model fallback.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

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
# General model wrapper
# ---------------------------------------------------------------------------

class GeneralModel(nn.Module):
    """Server-side general model built on a ResNet-18 backbone."""

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
# External distillation dataset loader
# ---------------------------------------------------------------------------

def load_distillation_dataset(
    name: str,
    root: str,
    max_samples: int = 0,
) -> Dataset:
    """
    Load an external unlabeled dataset for server-side ensemble distillation.
    Labels exist in the object but are NOT used during distillation.

    Supported: 'cifar100', 'svhn', or an ImageFolder path for ImageNet-32.
    All images are resized/normalised to match CIFAR-10 input distribution.
    """
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    normalised = name.lower().replace("-", "").replace("_", "")

    if normalised in ("cifar100",):
        ds = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    elif normalised in ("svhn",):
        ds = datasets.SVHN(root=root, split="train", download=True, transform=transform)
    elif normalised in ("imagenet32", "imagenet"):
        folder = Path(root) / "imagenet32" / "train"
        if not folder.exists():
            raise FileNotFoundError(
                f"ImageNet-32 folder not found at {folder}. "
                "Download ImageNet-32 and organise as ImageFolder."
            )
        ds = datasets.ImageFolder(str(folder), transform=transform)
    else:
        folder = Path(root) / name
        if folder.exists():
            ds = datasets.ImageFolder(str(folder), transform=transform)
        else:
            raise ValueError(f"Unsupported distillation dataset: {name}")

    total = len(ds)
    if 0 < max_samples < total:
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))

    LOGGER.info("Distillation dataset '%s': %d samples (max_samples=%d)", name, len(ds), max_samples)
    return ds


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedEGSDClient(BaseFederatedClient):
    """
    Local expert training with optional KD from the general model.

    Loss = CE(expert, true_label) + kd_weight * KL(expert, general_soft_label)

    The general model runs in inference-only mode on the client's private data,
    providing soft labels as a regulariser. This lets the expert absorb global
    knowledge accumulated in the general model each round, breaking the
    "information island" problem.
    """

    def __init__(self, client_id, dataset, num_classes, device, config, data_module):
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)

    def train_local(self, general_model: nn.Module = None) -> FedEGSDClientUpdate:
        """
        Train expert on private data.
        If general_model is provided and kd_weight > 0, add KL distillation loss.
        """
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        kd_temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))

        use_kd = general_model is not None and kd_weight > 0.0

        if not use_kd:
            # Pure CE, use base class helper
            loss = self._optimize_model(
                model=self.expert_model,
                loader=loader,
                epochs=self.config.federated.local_epochs,
                lr=self.config.federated.local_lr,
                momentum=self.config.federated.local_momentum,
                weight_decay=self.config.federated.local_weight_decay,
            )
        else:
            # CE + KD from general model
            loss = self._train_with_kd(loader, general_model, kd_weight, kd_temperature)

        state = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        return FedEGSDClientUpdate(
            client_id=self.client_id, num_samples=len(self.dataset),
            loss=loss, expert_state_dict=state,
        )

    def _train_with_kd(
        self,
        loader: DataLoader,
        general_model: nn.Module,
        kd_weight: float,
        kd_temperature: float,
    ) -> float:
        """Train expert with CE + KD from general model on private data."""
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        self.expert_model.train()
        general_model.eval()

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = general_model(images)
                    teacher_probs = F.softmax(teacher_logits / kd_temperature, dim=1)

                # Student forward
                optimizer.zero_grad(set_to_none=True)
                student_logits = self.expert_model(images)

                # CE loss on true labels
                ce_loss = criterion(student_logits, targets)

                # KL divergence loss (student learns from general model's soft labels)
                student_log_probs = F.log_softmax(student_logits / kd_temperature, dim=1)
                kl_loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean",
                ) * (kd_temperature ** 2)

                loss = ce_loss + kd_weight * kl_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedEGSDServer(BaseFederatedServer):

    def __init__(
        self, config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module, test_hard_indices,
        writer=None, public_dataset: Dataset | None = None,
    ) -> None:
        super().__init__(config, client_datasets, client_test_datasets,
                         data_module, test_hard_indices, writer, public_dataset=public_dataset)

        # General model — random init (no pretrain, aligned with FedDF)
        self.general_model = GeneralModel(
            num_classes=config.model.num_classes,
            pretrained_imagenet=bool(getattr(config.federated, "general_pretrain_imagenet_init", False)),
        ).to(self.device)

        # FLOPs
        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)

        # Clients
        self.clients: Dict[str, FedEGSDClient] = {
            cid: FedEGSDClient(cid, ds, config.model.num_classes,
                               config.federated.device, config, data_module)
            for cid, ds in client_datasets.items()
        }

        # --- External distillation dataset ---
        distill_name = str(getattr(config.federated, "distill_dataset", "cifar100"))
        distill_root = str(getattr(config.federated, "distill_dataset_root", config.dataset.root))
        distill_max = int(getattr(config.federated, "distill_max_samples", 0))
        self.distill_dataset = load_distillation_dataset(distill_name, distill_root, distill_max)
        self.distill_loader = DataLoader(
            self.distill_dataset, batch_size=config.dataset.batch_size,
            shuffle=False, num_workers=config.dataset.num_workers,
        )

        # Public loader (for routing threshold calibration only)
        if public_dataset is not None and len(public_dataset) > 0:
            self.public_loader = self.data_module.make_loader(public_dataset, shuffle=False)
        else:
            self.public_loader = None

        # Tracking
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict] = None

        # Per-client thresholds
        bc = float(config.inference.confidence_threshold)
        bm = float(config.inference.route_distance_threshold)
        self.client_confidence_thresholds = {c: bc for c in client_datasets}
        self.client_margin_thresholds = {c: bm for c in client_datasets}

    # ================================================================
    # AVGLOGITS extraction (FedDF core)
    # ================================================================

    def _extract_ensemble_logits(self, updates: List[FedEGSDClientUpdate]) -> Dict[str, torch.Tensor]:
        """Average raw logits across experts, THEN softmax (FedDF AVGLOGITS)."""
        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)

        all_logits: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        for update in updates:
            expert = SmallCNN(num_classes=num_classes,
                              base_channels=self.config.model.expert_base_channels).to(self.device)
            expert.load_state_dict(update.expert_state_dict)
            expert.eval()

            batches: List[torch.Tensor] = []
            with torch.no_grad():
                for batch in self.distill_loader:
                    images = batch[0].to(self.device)
                    batches.append(expert(images).cpu())

            client_logits = torch.cat(batches, dim=0)  # [N, C]

            # Entropy-based uncertainty weight
            probs = F.softmax(client_logits / temperature, dim=1)
            log_C = math.log(float(num_classes))
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(log_C, 1e-8)
            w = torch.exp(-ent)

            all_logits.append(client_logits)
            all_weights.append(w)
            del expert

        stacked_l = torch.stack(all_logits, dim=0)   # [K, N, C]
        stacked_w = torch.stack(all_weights, dim=0)   # [K, N]

        norm_w = stacked_w / stacked_w.sum(0, keepdim=True).clamp_min(1e-8)  # [K, N]
        avg_logits = (stacked_l * norm_w.unsqueeze(-1)).sum(0)                # [N, C]
        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        sample_w = stacked_w.mean(0)

        return {"soft_labels": soft_labels, "sample_weights": sample_w}

    # ================================================================
    # Distillation: pure KL + Adam + cosine annealing
    # ================================================================

    def _distill_general_model(self, ensemble: Dict[str, torch.Tensor], round_idx: int) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        bs = self.config.dataset.batch_size

        soft_all = ensemble["soft_labels"]
        w_all = ensemble["sample_weights"]

        # Collect distillation images (same order as soft_all)
        all_imgs: List[torch.Tensor] = []
        for batch in self.distill_loader:
            all_imgs.append(batch[0])
        distill_images = torch.cat(all_imgs, dim=0)
        N = distill_images.size(0)

        # Fresh optimizer + cosine schedule per round
        optimizer = torch.optim.Adam(self.general_model.parameters(),
                                     lr=float(self.config.federated.distill_lr))
        total_steps = distill_epochs * ((N + bs - 1) // bs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        self.general_model.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                imgs = distill_images[idx].to(self.device)
                b_soft = soft_all[idx].to(self.device)
                b_w = w_all[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                s_logits = self.general_model(imgs)

                s_log_p = F.log_softmax(s_logits / temperature, dim=1)
                per_kl = F.kl_div(s_log_p, b_soft, reduction="none").sum(1) * (temperature ** 2)
                loss = (per_kl * b_w).sum() / b_w.sum().clamp_min(1e-8)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_batches += 1

        d = max(total_batches, 1)
        return {"total_loss": total_loss / d, "kl_loss": total_loss / d}

    # ================================================================
    # Predictors
    # ================================================================

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        return self.general_model(images).argmax(1), 0

    def _predict_expert_only(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        return self.clients[client_id].expert_model(images).argmax(1), 0

    def _predict_routed(self, client_id, images, indices):
        return self._confidence_route(
            self.clients[client_id].expert_model, self.general_model, images,
            confidence_threshold=self.client_confidence_thresholds[client_id],
            margin_threshold=self.client_margin_thresholds[client_id],
        )

    # ================================================================
    # Routing threshold adaptation
    # ================================================================

    def _update_routing_thresholds(self, updates):
        if self.public_loader is None:
            return
        step = max(float(self.config.inference.personalized_threshold_step), 0.0)
        mstep = max(float(self.config.inference.personalized_margin_step), 0.0)
        guard = float(self.config.inference.public_teacher_gap_guard)
        gp = self._predict_public_general()
        pt = self._get_public_targets()

        for u in updates:
            cid = u.client_id
            thr = self.client_confidence_thresholds[cid]
            mar = self.client_margin_thresholds[cid]
            expert = SmallCNN(num_classes=self.config.model.num_classes,
                              base_channels=self.config.model.expert_base_channels).to(self.device)
            expert.load_state_dict(u.expert_state_dict); expert.eval()
            eo = go = tot = 0
            with torch.no_grad():
                for imgs, _, idxs in self.public_loader:
                    imgs = imgs.to(self.device)
                    ep = expert(imgs).argmax(1).cpu()
                    for i, idx in enumerate(idxs.tolist()):
                        t = pt[idx]; eo += int(ep[i].item() == t); go += int(gp[idx] == t); tot += 1
            gap = (go - eo) / max(tot, 1)
            if gap > guard: thr += step; mar += mstep
            elif gap < -guard: thr -= step; mar -= mstep
            self.client_confidence_thresholds[cid] = min(max(thr, float(self.config.inference.min_confidence_threshold)),
                                                         float(self.config.inference.max_confidence_threshold))
            self.client_margin_thresholds[cid] = min(max(mar, float(self.config.inference.min_margin_threshold)),
                                                     float(self.config.inference.max_margin_threshold))
            del expert

    def _predict_public_general(self):
        p = {}; self.general_model.eval()
        with torch.no_grad():
            for imgs, _, idxs in self.public_loader:
                imgs = imgs.to(self.device)
                for idx, pred in zip(idxs.tolist(), self.general_model(imgs).argmax(1).cpu().tolist()):
                    p[idx] = pred
        return p

    def _get_public_targets(self):
        t = {}
        for _, targets, idxs in self.public_loader:
            for idx, target in zip(idxs.tolist(), targets.tolist()):
                t[idx] = target
        return t

    # ================================================================
    # Snapshots
    # ================================================================

    def _maybe_update_best(self, ri, rm, ea, ga):
        better = self.best_snapshot is None or rm.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not better: return False
        self.best_snapshot = {
            "round_idx": ri, "routed_accuracy": rm.routed_accuracy,
            "general_accuracy": ga, "expert_accuracy": ea,
            "avg_client_loss": rm.avg_client_loss,
            "general_model_state": {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()},
            "client_expert_states": {c: {k: v.cpu().clone() for k, v in cl.expert_model.state_dict().items()} for c, cl in self.clients.items()},
            "client_confidence_thresholds": dict(self.client_confidence_thresholds),
            "client_margin_thresholds": dict(self.client_margin_thresholds),
        }
        LOGGER.info("fedegsd best | round=%d | routed=%.4f | general=%.4f | expert=%.4f", ri, rm.routed_accuracy, ga, ea)
        return True

    def _restore_best(self):
        if not self.best_snapshot: return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for c, st in self.best_snapshot["client_expert_states"].items():
            self.clients[c].expert_model.load_state_dict(st)
        self.client_confidence_thresholds = dict(self.best_snapshot["client_confidence_thresholds"])
        self.client_margin_thresholds = dict(self.best_snapshot["client_margin_thresholds"])
        LOGGER.info("fedegsd restored best from round %d", self.best_snapshot["round_idx"])

    # ================================================================
    # Main loop
    # ================================================================

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        kd_warmup = int(getattr(self.config.federated, "expert_kd_warmup_rounds", 10))

        for ri in range(1, self.config.federated.rounds + 1):
            sids = self._sample_client_ids()

            # KD warmup: first N rounds train expert with pure CE (no KD from random general)
            # After warmup, general model has accumulated enough quality to be a useful teacher
            use_kd = ri > kd_warmup
            teacher = self.general_model if use_kd else None
            if ri == kd_warmup + 1:
                LOGGER.info("fedegsd round %d | KD warmup complete, enabling expert KD from general model", ri)

            LOGGER.info("fedegsd round %d | clients=%s | kd=%s", ri, sids, use_kd)

            updates = [self.clients[c].train_local(general_model=teacher) for c in sids]
            ensemble = self._extract_ensemble_logits(updates)
            ds = self._distill_general_model(ensemble, ri)
            self._update_routing_thresholds(updates)

            ee = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd-expert")
            ge = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd-general")
            re = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsd-routed")

            al = sum(u.loss for u in updates) / max(len(updates), 1)
            a = re["aggregate"]; m = re["macro"]
            cp = self._build_compute_profile(self.expert_flops, self.general_flops, a["invocation_rate"], "routed")

            rm = RoundMetrics(round_idx=ri, avg_client_loss=al, routed_accuracy=a["accuracy"],
                              hard_accuracy=a["hard_recall"], invocation_rate=a["invocation_rate"],
                              local_accuracy=m["accuracy"], compute_savings=cp["savings_ratio"])

            LOGGER.info("fedegsd round %d | loss=%.4f | distill=%.4f | routed=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f",
                         ri, al, ds["total_loss"], a["accuracy"], ee["aggregate"]["accuracy"],
                         ge["aggregate"]["accuracy"], a["hard_recall"], a["invocation_rate"])

            if self.writer:
                self.writer.add_scalar("expert_loss/fedegsd", al, ri)
                self.writer.add_scalar("distill_loss/fedegsd", ds["total_loss"], ri)
                self._log_auxiliary_accuracy_metrics("fedegsd", ri, ee["aggregate"]["accuracy"], ge["aggregate"]["accuracy"])

            self._log_round_metrics("fedegsd", rm)
            metrics.append(rm)
            self._maybe_update_best(ri, rm, ee["aggregate"]["accuracy"], ge["aggregate"]["accuracy"])

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best()
        return metrics

    # ================================================================
    # Final eval
    # ================================================================

    def evaluate_baselines(self, test_dataset):
        rp = self._build_route_export_path("fedegsd_final_routed")
        ro = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsd_final_routed", route_export_path=rp)
        eo = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd_final_expert")
        go = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd_final_general")
        fl = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        br = 0
        if self.best_snapshot: fl = float(self.best_snapshot["avg_client_loss"]); br = int(self.best_snapshot["round_idx"])
        rc = self._build_compute_profile(self.expert_flops, self.general_flops, ro["aggregate"]["invocation_rate"], "routed")
        ec = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, "expert_only")
        gc = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, "general_only")
        return {
            "algorithm": "fedegsd",
            "metrics": {
                "accuracy": ro["aggregate"]["accuracy"], "global_accuracy": ro["aggregate"]["accuracy"],
                "local_accuracy": ro["macro"]["accuracy"], "routed_accuracy": ro["aggregate"]["accuracy"],
                "hard_accuracy": ro["aggregate"]["hard_recall"], "hard_sample_recall": ro["aggregate"]["hard_recall"],
                "general_invocation_rate": ro["aggregate"]["invocation_rate"], "compute_savings": rc["savings_ratio"],
                "precision_macro": ro["aggregate"]["precision_macro"], "recall_macro": ro["aggregate"]["recall_macro"],
                "f1_macro": ro["aggregate"]["f1_macro"],
                "expert_only_accuracy": eo["aggregate"]["accuracy"], "general_only_accuracy": go["aggregate"]["accuracy"],
                "final_training_loss": fl, "best_round": br,
            },
            "client_metrics": {"routed": ro["clients"], "expert_only": eo["clients"], "general_only": go["clients"]},
            "group_metrics": {"routed": ro["groups"], "expert_only": eo["groups"], "general_only": go["groups"]},
            "compute": {"routed": rc, "expert_only": ec, "general_only": gc},
            "memory_mb": {"expert": model_memory_mb(self.reference_expert), "general": model_memory_mb(self.general_model)},
            "artifacts": {"route_csv": str(rp)},
        }
