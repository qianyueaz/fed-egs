"""
FedEGS-S: Federated Expert-General System - Supernet + BDD-HFL

Supernet-subnet architecture with asymmetric bidirectional decoupled distillation:
  - Expert is a width-slice of the general model (WidthScalableResNet).
  - Bidirectional DREL (Decoupled Relative Entropy Loss):
    * General→Expert: full update expert with TC+NC decomposed knowledge.
    * Expert→General: lightweight update of general's classifier head only.
  - General head delta is uploaded and aggregated on server alongside expert delta.
  - Server-side DREL distillation with confidence-thresholded pseudo-labels.
  - Sigmoid warmup for λ_eg (expert→general strength).
  - Prototype-based distance routing for OOD detection.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    WidthScalableResNet,
    apply_expert_delta_to_general,
    average_weighted_deltas,
    estimate_model_flops,
    get_num_expert_blocks,
    load_expert_state_dict,
    model_memory_mb,
)
from fedegs.models.width_scalable_resnet import state_dict_delta


# ---------------------------------------------------------------------------
# DREL: Decoupled Relative Entropy Loss
# ---------------------------------------------------------------------------

def drel_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 3.0,
) -> torch.Tensor:
    """
    Decoupled Relative Entropy Loss (BDD-HFL).

    Decomposes KL divergence into Target Class (TC) and Non-Target Class (NC).
    NC gets higher weight (beta >> alpha) to preserve inter-class structure.

    Uses masking trick: set target class logit to -1e9 for NC softmax,
    preserving tensor shape and avoiding NaN.
    """
    T = temperature
    B, C = logits_teacher.shape

    # --- Target Class (TC) ---
    tc_t = logits_teacher.gather(1, targets.unsqueeze(1)) / T  # [B, 1]
    tc_s = logits_student.gather(1, targets.unsqueeze(1)) / T
    # TC loss: MSE on scaled logits (simpler and stable)
    tc_loss = F.mse_loss(tc_s, tc_t.detach()) * (T ** 2)

    # --- Non-Target Class (NC) ---
    # Mask target class position with -inf for NC softmax
    mask = torch.zeros_like(logits_teacher)
    mask.scatter_(1, targets.unsqueeze(1), -1e9)

    nc_teacher_probs = F.softmax((logits_teacher + mask) / T, dim=1)
    nc_student_log_probs = F.log_softmax((logits_student + mask) / T, dim=1)
    nc_loss = F.kl_div(nc_student_log_probs, nc_teacher_probs.detach(),
                       reduction="batchmean") * (T ** 2)

    return alpha * tc_loss + beta * nc_loss


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FedEGSSClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    delta: Dict[str, torch.Tensor]              # expert parameter delta
    general_head_delta: Dict[str, torch.Tensor]  # general classifier head delta
    local_classes: List[int]
    centroids: Dict[int, torch.Tensor]
    centroid_stds: Dict[int, float]


# ---------------------------------------------------------------------------
# External distillation dataset
# ---------------------------------------------------------------------------

def load_distillation_dataset(name, root, max_samples=0):
    transform = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    n = name.lower().replace("-", "").replace("_", "")
    if n in ("cifar100",):
        ds = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    elif n in ("svhn",):
        ds = datasets.SVHN(root=root, split="train", download=True, transform=transform)
    else:
        folder = Path(root) / name
        if folder.exists():
            ds = datasets.ImageFolder(str(folder), transform=transform)
        else:
            raise ValueError(f"Unsupported distillation dataset: {name}")
    if 0 < max_samples < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    LOGGER.info("Distillation dataset '%s': %d samples", name, len(ds))
    return ds


# ---------------------------------------------------------------------------
# Sigmoid schedule for λ_eg warmup
# ---------------------------------------------------------------------------

def sigmoid_schedule(round_idx: int, total_rounds: int, max_value: float) -> float:
    """Sigmoid warmup: ~0 for first 20%, ramps in middle 60%, saturates last 20%."""
    x = (round_idx / max(total_rounds, 1) - 0.4) * 10.0
    return max_value / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedEGSSClient(BaseFederatedClient):
    """
    Supernet-subnet client with asymmetric bidirectional DREL:
      - Expert: full CE + DREL(general→expert) training
      - General classifier head: DREL(expert→general) lightweight update
      - Uploads both expert delta and general head delta
    """

    def __init__(self, client_id, dataset, expert_width, num_classes,
                 device, config, data_module, block_index):
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.block_index = block_index
        self.num_classes = num_classes
        self.expert_model = WidthScalableResNet(
            width_factor=expert_width, num_classes=num_classes,
        ).to(self.device)
        self.local_classes: Set[int] = set()
        self.centroids: Dict[int, torch.Tensor] = {}
        self.centroid_stds: Dict[int, float] = {}

    def train_round(self, general_model: WidthScalableResNet,
                    use_kd: bool = False, lambda_eg: float = 0.0) -> FedEGSSClientUpdate:
        # 1. Slice expert from general
        load_expert_state_dict(general_model, self.expert_model, block_index=self.block_index)
        before_expert = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        before_head = {k: v.detach().cpu().clone() for k, v in general_model.fc.state_dict().items()}

        # 2. Train
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        if use_kd:
            loss = self._train_bidirectional_drel(loader, general_model, lambda_eg)
        else:
            loss = self._optimize_model(
                model=self.expert_model, loader=loader,
                epochs=self.config.federated.local_epochs,
                lr=self.config.federated.local_lr,
                momentum=self.config.federated.local_momentum,
                weight_decay=self.config.federated.local_weight_decay,
            )

        # 3. Compute deltas
        expert_delta = state_dict_delta(self.expert_model.state_dict(), before_expert)
        head_delta = state_dict_delta(general_model.fc.state_dict(), before_head)

        # 4. Prototypes
        self._compute_prototypes()

        return FedEGSSClientUpdate(
            client_id=self.client_id, num_samples=len(self.dataset),
            loss=loss, delta=expert_delta, general_head_delta=head_delta,
            local_classes=sorted(self.local_classes),
            centroids={c: v.cpu().clone() for c, v in self.centroids.items()},
            centroid_stds=dict(self.centroid_stds),
        )

    def _train_bidirectional_drel(self, loader, general_model, lambda_eg):
        """
        Asymmetric Bidirectional DREL training:
          - Expert: CE + λ_ge × DREL(general→expert)    [full update]
          - General head: λ_eg × DREL(expert→general)    [classifier only]
        """
        cfg = self.config.federated
        drel_alpha = float(getattr(cfg, "drel_alpha", 1.0))
        drel_beta = float(getattr(cfg, "drel_beta", 8.0))
        drel_T = float(getattr(cfg, "expert_kd_temperature", 3.0))
        lambda_ge = float(getattr(cfg, "lambda_ge", 1.0))
        head_lr = float(getattr(cfg, "general_head_lr", 0.001))

        # Expert optimizer: full parameters
        expert_opt = torch.optim.SGD(
            self.expert_model.parameters(), lr=cfg.local_lr,
            momentum=cfg.local_momentum, weight_decay=cfg.local_weight_decay,
        )
        # General head optimizer: only classifier (fc layer)
        head_opt = torch.optim.SGD(general_model.fc.parameters(), lr=head_lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()

        # Freeze general backbone, unfreeze head
        for p in general_model.parameters():
            p.requires_grad_(False)
        for p in general_model.fc.parameters():
            p.requires_grad_(True)

        self.expert_model.train()
        general_model.train()  # BN in train mode for head, but backbone frozen

        total_loss = 0.0; total_batches = 0

        for _ in range(cfg.local_epochs):
            for images, targets, _ in loader:
                images, targets = images.to(self.device), targets.to(self.device)

                # Forward both models
                expert_logits = self.expert_model(images)
                general_logits = general_model(images)

                # === Expert update: CE + DREL(general→expert) ===
                ce_loss = criterion(expert_logits, targets)
                ge_drel = drel_loss(general_logits.detach(), expert_logits, targets,
                                    alpha=drel_alpha, beta=drel_beta, temperature=drel_T)
                expert_loss = ce_loss + lambda_ge * ge_drel

                expert_opt.zero_grad(set_to_none=True)
                expert_loss.backward(retain_graph=(lambda_eg > 0))
                expert_opt.step()

                # === General head update: DREL(expert→general) ===
                if lambda_eg > 0:
                    # Re-forward expert (detached) to get fresh logits after expert update
                    with torch.no_grad():
                        expert_logits_detached = self.expert_model(images)
                    general_logits_fresh = general_model(images)
                    eg_drel = drel_loss(expert_logits_detached, general_logits_fresh, targets,
                                        alpha=drel_alpha, beta=drel_beta, temperature=drel_T)
                    head_loss = lambda_eg * eg_drel

                    head_opt.zero_grad(set_to_none=True)
                    head_loss.backward()
                    head_opt.step()

                total_loss += expert_loss.item(); total_batches += 1

        # Restore general model grad state
        for p in general_model.parameters():
            p.requires_grad_(True)

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _compute_prototypes(self):
        loader = self.data_module.make_loader(self.dataset, shuffle=False)
        self.expert_model.eval()
        class_features: Dict[int, List[torch.Tensor]] = {}
        for images, targets, _ in loader:
            images = images.to(self.device)
            features = F.normalize(self.expert_model.forward_features(images), dim=1)
            for feat, label in zip(features.cpu(), targets.tolist()):
                class_features.setdefault(label, []).append(feat)
        self.local_classes = set(class_features.keys())
        self.centroids = {}; self.centroid_stds = {}
        for cls, feats in class_features.items():
            stacked = torch.stack(feats, 0)
            centroid = F.normalize(stacked.mean(0), dim=0)
            self.centroids[cls] = centroid
            self.centroid_stds[cls] = float((1.0 - F.cosine_similarity(stacked, centroid.unsqueeze(0), dim=1)).mean().item())

    @torch.no_grad()
    def distance_route(self, images, general_model, distance_threshold, std_multiplier=1.5):
        self.expert_model.eval(); general_model.eval()
        B = images.size(0)
        if not self.centroids:
            return general_model(images).argmax(1), B, {"route_type": ["general"]*B, "expert_confidence": [0.0]*B}
        features = F.normalize(self.expert_model.forward_features(images), dim=1)
        expert_preds = self.expert_model(images).argmax(1)
        cc = sorted(self.centroids.keys())
        cs = torch.stack([self.centroids[c] for c in cc], 0).to(features.device)
        md, ni = (1.0 - features @ cs.T).min(dim=1)
        thr = torch.full((B,), distance_threshold, device=images.device)
        for i in range(B):
            thr[i] = distance_threshold + std_multiplier * self.centroid_stds.get(cc[ni[i].item()], 0.0)
        fb = md >= thr
        preds = expert_preds.clone()
        inv = int(fb.sum().item())
        rt = ["expert"] * B
        if fb.any():
            preds[fb] = general_model(images[fb]).argmax(1)
            for idx in fb.nonzero(as_tuple=False).flatten().tolist(): rt[idx] = "general"
        return preds, inv, {"route_type": rt, "expert_confidence": md.cpu().tolist()}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedEGSSServer(BaseFederatedServer):

    def __init__(self, config, client_datasets, client_test_datasets,
                 data_module, test_hard_indices, writer=None, public_dataset=None):
        super().__init__(config, client_datasets, client_test_datasets,
                         data_module, test_hard_indices, writer, public_dataset=public_dataset)

        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width, num_classes=config.model.num_classes,
        ).to(self.device)
        self.reference_expert = WidthScalableResNet(
            width_factor=config.model.expert_width, num_classes=config.model.num_classes,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
        self.num_expert_blocks = get_num_expert_blocks(self.general_model, self.reference_expert)

        sorted_ids = sorted(client_datasets.keys())
        self.client_block_map = {cid: i % self.num_expert_blocks for i, cid in enumerate(sorted_ids)}
        self.clients: Dict[str, FedEGSSClient] = {
            cid: FedEGSSClient(cid, ds, config.model.expert_width, config.model.num_classes,
                               config.federated.device, config, data_module, self.client_block_map[cid])
            for cid, ds in client_datasets.items()
        }
        for cid in sorted_ids:
            LOGGER.info("FedEGS-S client %s → block %d/%d", cid, self.client_block_map[cid], self.num_expert_blocks)

        dn = str(getattr(config.federated, "distill_dataset", "cifar100"))
        dr = str(getattr(config.federated, "distill_dataset_root", config.dataset.root))
        dm = int(getattr(config.federated, "distill_max_samples", 0))
        self.distill_dataset = load_distillation_dataset(dn, dr, dm)
        self.distill_loader = DataLoader(self.distill_dataset, batch_size=config.dataset.batch_size,
                                         shuffle=False, num_workers=config.dataset.num_workers)

        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict] = None
        self.route_distance_threshold = float(getattr(config.inference, "route_distance_threshold", 0.3))

    # ================================================================
    # Server-side DREL distillation with confidence-gated pseudo-labels
    # ================================================================

    def _distill_general_model(self, updates, round_idx):
        T = float(self.config.federated.distill_temperature)
        de = max(int(self.config.federated.distill_epochs), 1)
        bs = self.config.dataset.batch_size
        nc = self.config.model.num_classes
        drel_alpha = float(getattr(self.config.federated, "drel_alpha", 1.0))
        drel_beta = float(getattr(self.config.federated, "drel_beta", 8.0))
        conf_thr = float(getattr(self.config.federated, "drel_confidence_threshold", 0.6))

        # 1. Ensemble logits
        all_logits, all_w = [], []
        for u in updates:
            expert = WidthScalableResNet(width_factor=self.config.model.expert_width, num_classes=nc).to(self.device)
            load_expert_state_dict(self.general_model, expert, block_index=self.client_block_map[u.client_id])
            expert.eval()
            bl = []
            with torch.no_grad():
                for batch in self.distill_loader:
                    bl.append(expert(batch[0].to(self.device)).cpu())
            cl = torch.cat(bl, 0)
            p = F.softmax(cl / T, dim=1)
            ent = -(p * p.clamp_min(1e-8).log()).sum(1) / max(math.log(float(nc)), 1e-8)
            all_logits.append(cl); all_w.append(torch.exp(-ent)); del expert

        sl = torch.stack(all_logits, 0); sw = torch.stack(all_w, 0)
        nw = sw / sw.sum(0, keepdim=True).clamp_min(1e-8)
        avg_logits = (sl * nw.unsqueeze(-1)).sum(0)  # [N, C]
        soft_labels = F.softmax(avg_logits / T, dim=1)
        pseudo_labels = soft_labels.argmax(dim=1)
        max_conf = soft_labels.max(dim=1).values
        sample_w = sw.mean(0)

        # 2. Collect images
        imgs = torch.cat([b[0] for b in self.distill_loader], 0)
        N = imgs.size(0)

        # 3. Distill with confidence-gated DREL
        opt = torch.optim.Adam(self.general_model.parameters(), lr=float(self.config.federated.distill_lr))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(de * ((N+bs-1)//bs), 1))
        self.general_model.train()
        tl = 0.0; tb = 0

        for _ in range(de):
            pm = torch.randperm(N)
            for s in range(0, N, bs):
                ix = pm[s:s+bs]
                im = imgs[ix].to(self.device)
                b_soft = soft_labels[ix].to(self.device)
                b_pseudo = pseudo_labels[ix].to(self.device)
                b_conf = max_conf[ix].to(self.device)
                b_w = sample_w[ix].to(self.device)

                opt.zero_grad(set_to_none=True)
                out = self.general_model(im)

                # High confidence → DREL; low confidence → standard KL
                high_mask = b_conf >= conf_thr
                if high_mask.any():
                    drel_part = drel_loss(
                        (avg_logits[ix][high_mask.cpu()]).to(self.device),
                        out[high_mask], b_pseudo[high_mask],
                        alpha=drel_alpha, beta=drel_beta, temperature=T,
                    )
                else:
                    drel_part = out.new_zeros(())

                if (~high_mask).any():
                    kl_part = F.kl_div(
                        F.log_softmax(out[~high_mask] / T, 1),
                        b_soft[~high_mask], reduction="batchmean",
                    ) * (T ** 2)
                else:
                    kl_part = out.new_zeros(())

                n_high = max(int(high_mask.sum().item()), 1)
                n_low = max(int((~high_mask).sum().item()), 1)
                loss = (drel_part * n_high + kl_part * n_low) / (n_high + n_low)

                loss.backward(); opt.step(); sch.step()
                tl += loss.item(); tb += 1

        return {"total_loss": tl / max(tb, 1)}

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
        sm = float(getattr(self.config.inference, "route_distance_std_multiplier", 1.5))
        return self.clients[client_id].distance_route(images, self.general_model, self.route_distance_threshold, sm)

    # ================================================================
    # Snapshots
    # ================================================================

    def _maybe_update_best(self, ri, rm, ea, ga):
        better = self.best_snapshot is None or rm.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not better: return False
        self.best_snapshot = {
            "round_idx": ri, "routed_accuracy": rm.routed_accuracy, "general_accuracy": ga,
            "expert_accuracy": ea, "avg_client_loss": rm.avg_client_loss,
            "general_model_state": {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()},
        }
        LOGGER.info("fedegss best | round=%d | routed=%.4f | general=%.4f | expert=%.4f", ri, rm.routed_accuracy, ga, ea)
        return True

    def _restore_best(self):
        if not self.best_snapshot: return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for cid, cl in self.clients.items():
            load_expert_state_dict(self.general_model, cl.expert_model, block_index=self.client_block_map[cid])
        LOGGER.info("fedegss restored best from round %d", self.best_snapshot["round_idx"])

    # ================================================================
    # Main loop
    # ================================================================

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        kd_warmup = int(getattr(self.config.federated, "expert_kd_warmup_rounds", 5))
        total_rounds = self.config.federated.rounds
        max_lambda_eg = float(getattr(self.config.federated, "lambda_eg", 0.5))

        for ri in range(1, total_rounds + 1):
            sids = self._sample_client_ids()
            use_kd = ri > kd_warmup
            # Sigmoid schedule for expert→general strength
            cur_lambda_eg = sigmoid_schedule(ri, total_rounds, max_lambda_eg) if use_kd else 0.0

            if ri == kd_warmup + 1:
                LOGGER.info("fedegss round %d | KD warmup done", ri)
            LOGGER.info("fedegss round %d | clients=%s | kd=%s | λ_eg=%.4f", ri, sids, use_kd, cur_lambda_eg)

            # 1. Client training with bidirectional DREL
            updates: List[FedEGSSClientUpdate] = []
            for cid in sids:
                updates.append(self.clients[cid].train_round(
                    self.general_model, use_kd=use_kd, lambda_eg=cur_lambda_eg,
                ))

            # 2. Delta writeback: expert subnet
            block_groups: Dict[int, List[FedEGSSClientUpdate]] = {}
            for u in updates:
                bi = self.client_block_map[u.client_id]
                block_groups.setdefault(bi, []).append(u)
            for bi, group in block_groups.items():
                agg = average_weighted_deltas((u.num_samples, u.delta) for u in group)
                apply_expert_delta_to_general(self.general_model, agg, self.reference_expert, block_index=bi)

            # 3. Delta writeback: general classifier head
            head_deltas = [(u.num_samples, u.general_head_delta) for u in updates]
            agg_head = average_weighted_deltas(head_deltas)
            head_state = self.general_model.fc.state_dict()
            for key, delta in agg_head.items():
                if key in head_state:
                    head_state[key] = head_state[key] + delta.to(head_state[key].device)
            self.general_model.fc.load_state_dict(head_state)

            # 4. FedDF DREL distillation on external data
            ds = self._distill_general_model(updates, ri)

            # 5. Re-slice experts + recompute prototypes
            for cid, cl in self.clients.items():
                load_expert_state_dict(self.general_model, cl.expert_model, block_index=self.client_block_map[cid])
                cl._compute_prototypes()

            # 6. Evaluate
            ee = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegss-expert")
            ge = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegss-general")
            re = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegss-routed")

            al = sum(u.loss for u in updates) / max(len(updates), 1)
            a = re["aggregate"]; m = re["macro"]
            cp = self._build_compute_profile(self.expert_flops, self.general_flops, a["invocation_rate"], "routed")
            rm = RoundMetrics(round_idx=ri, avg_client_loss=al, routed_accuracy=a["accuracy"],
                              hard_accuracy=a["hard_recall"], invocation_rate=a["invocation_rate"],
                              local_accuracy=m["accuracy"], compute_savings=cp["savings_ratio"])

            LOGGER.info("fedegss round %d | loss=%.4f | distill=%.4f | routed=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f | λ_eg=%.4f",
                         ri, al, ds["total_loss"], a["accuracy"], ee["aggregate"]["accuracy"],
                         ge["aggregate"]["accuracy"], a["hard_recall"], a["invocation_rate"], cur_lambda_eg)

            if self.writer:
                self.writer.add_scalar("expert_loss/fedegss", al, ri)
                self.writer.add_scalar("distill_loss/fedegss", ds["total_loss"], ri)
                self.writer.add_scalar("routed_accuracy/fedegss", a["accuracy"], ri)
                self.writer.add_scalar("routed_hard_accuracy/fedegss", a["hard_recall"], ri)
                self.writer.add_scalar("routed_local_accuracy/fedegss", m["accuracy"], ri)
                self.writer.add_scalar("invocation_rate/fedegss", a["invocation_rate"], ri)
                self.writer.add_scalar("lambda_eg/fedegss", cur_lambda_eg, ri)
                self.writer.add_scalar("compare/fedegss/routed_accuracy", a["accuracy"], ri)
                self.writer.add_scalar("compare/fedegss/expert_accuracy", ee["aggregate"]["accuracy"], ri)
                self.writer.add_scalar("compare/fedegss/general_accuracy", ge["aggregate"]["accuracy"], ri)
                for gn, gm in re["groups"].items():
                    self.writer.add_scalar(f"routed_accuracy_group/{gn}", gm["accuracy"], ri)
                    self.writer.add_scalar(f"invocation_rate_group/{gn}", gm["invocation_rate"], ri)
                self._log_auxiliary_accuracy_metrics("fedegss", ri, ee["aggregate"]["accuracy"], ge["aggregate"]["accuracy"])

            self._log_round_metrics("fedegss", rm)
            metrics.append(rm)
            self._maybe_update_best(ri, rm, ee["aggregate"]["accuracy"], ge["aggregate"]["accuracy"])

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint: self._restore_best()
        return metrics

    # ================================================================
    # Final eval
    # ================================================================

    def evaluate_baselines(self, test_dataset):
        rp = self._build_route_export_path("fedegss_final_routed")
        ro = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegss_final_routed", route_export_path=rp)
        eo = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegss_final_expert")
        go = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegss_final_general")
        fl = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        br = 0
        if self.best_snapshot: fl = float(self.best_snapshot["avg_client_loss"]); br = int(self.best_snapshot["round_idx"])
        rc = self._build_compute_profile(self.expert_flops, self.general_flops, ro["aggregate"]["invocation_rate"], "routed")
        ec = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, "expert_only")
        gc = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, "general_only")
        return {
            "algorithm": "fedegss",
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
