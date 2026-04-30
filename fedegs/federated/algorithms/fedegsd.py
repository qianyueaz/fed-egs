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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

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


def _stable_client_seed(base_seed: int, client_id: str) -> int:
    return int(base_seed) + sum((index + 1) * ord(char) for index, char in enumerate(client_id))


def _client_holdout_seed(base_seed: int, client_id: str, seed_offset: int) -> int:
    return _stable_client_seed(int(base_seed) + int(seed_offset), client_id)


def _split_dataset_for_holdout(
    dataset: Dataset,
    holdout_ratio: float,
    min_holdout_samples: int,
    max_holdout_samples: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    total_samples = len(dataset)
    if total_samples <= 1 or holdout_ratio <= 0.0:
        return dataset, dataset

    requested_samples = max(int(round(total_samples * holdout_ratio)), 1)
    if total_samples >= max(2 * max(min_holdout_samples, 1), 8):
        requested_samples = max(requested_samples, max(min_holdout_samples, 1))
    if max_holdout_samples > 0:
        requested_samples = min(requested_samples, max_holdout_samples)
    holdout_samples = min(max(requested_samples, 1), total_samples - 1)
    if holdout_samples <= 0 or holdout_samples >= total_samples:
        return dataset, dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total_samples, generator=generator).tolist()
    holdout_indices = sorted(permutation[:holdout_samples])
    train_indices = sorted(permutation[holdout_samples:])
    if not train_indices or not holdout_indices:
        return dataset, dataset
    return Subset(dataset, train_indices), Subset(dataset, holdout_indices)


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
            loss = self._optimize_model(
                model=self.expert_model,
                loader=loader,
                epochs=self.config.federated.local_epochs,
                lr=self.config.federated.local_lr,
                momentum=self.config.federated.local_momentum,
                weight_decay=self.config.federated.local_weight_decay,
            )
        else:
            loss = self._train_with_kd(loader, general_model, kd_weight, kd_temperature)

        state = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        return FedEGSDClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            expert_state_dict=state,
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

                with torch.no_grad():
                    teacher_logits = general_model(images)
                    teacher_probs = F.softmax(teacher_logits / kd_temperature, dim=1)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.expert_model(images)

                ce_loss = criterion(student_logits, targets)

                student_log_probs = F.log_softmax(student_logits / kd_temperature, dim=1)
                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
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
        self,
        config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices,
        writer=None,
        public_dataset: Dataset | None = None,
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

        self.general_enabled = bool(getattr(config.federated, "general_enabled", True))
        self.general_model = GeneralModel(
            num_classes=config.model.num_classes,
            pretrained_imagenet=bool(getattr(config.federated, "general_pretrain_imagenet_init", False)),
        ).to(self.device)

        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
        self.resource_profiles = self._build_dual_model_resource_profiles(
            self.reference_expert,
            self.general_model,
            self.expert_flops,
            self.general_flops,
        )

        holdout_ratio = max(float(getattr(config.inference, "routing_holdout_ratio", 0.0)), 0.0)
        holdout_min_samples = max(int(getattr(config.inference, "routing_holdout_min_samples", 0)), 0)
        holdout_max_samples = max(int(getattr(config.inference, "routing_holdout_max_samples", 0)), 0)
        holdout_seed_offset = int(getattr(config.inference, "routing_holdout_seed_offset", 17))
        self.client_routing_datasets: Dict[str, Dataset] = {}
        self.client_training_datasets: Dict[str, Dataset] = {}
        for client_id, dataset in client_datasets.items():
            train_dataset, routing_dataset = _split_dataset_for_holdout(
                dataset=dataset,
                holdout_ratio=holdout_ratio,
                min_holdout_samples=holdout_min_samples,
                max_holdout_samples=holdout_max_samples,
                seed=_client_holdout_seed(config.federated.seed, client_id, holdout_seed_offset),
            )
            self.client_training_datasets[client_id] = train_dataset
            self.client_routing_datasets[client_id] = routing_dataset

        self.clients: Dict[str, FedEGSDClient] = {
            cid: FedEGSDClient(
                cid,
                self.client_training_datasets[cid],
                config.model.num_classes,
                config.federated.device,
                config,
                data_module,
            )
            for cid in client_datasets.keys()
        }

        self.distill_dataset = None
        self.distill_loader = None
        if self.general_enabled:
            distill_name = str(getattr(config.federated, "distill_dataset", "cifar100"))
            distill_root = str(getattr(config.federated, "distill_dataset_root", config.dataset.root))
            distill_max = int(getattr(config.federated, "distill_max_samples", 0))
            self.distill_dataset = load_distillation_dataset(distill_name, distill_root, distill_max)
            self.distill_loader = DataLoader(
                self.distill_dataset,
                batch_size=config.dataset.batch_size,
                shuffle=False,
                num_workers=config.dataset.num_workers,
            )
        else:
            LOGGER.info("fedegsd general path disabled | running expert-only ablation")

        if public_dataset is not None and len(public_dataset) > 0:
            self.public_loader = self.data_module.make_loader(public_dataset, shuffle=False)
        else:
            self.public_loader = None

        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict] = None
        self.current_round = 0

        initial_route_threshold = self._initial_route_score_threshold()
        self.client_route_score_thresholds = {c: initial_route_threshold for c in client_datasets}
        self.client_expert_temperatures = {c: 1.0 for c in client_datasets}
        self.client_prototypes: Dict[str, torch.Tensor] = {}
        self.client_prototype_scales: Dict[str, torch.Tensor] = {}
        self.client_prototype_masks: Dict[str, torch.Tensor] = {}
        self.client_routing_metrics: Dict[str, Dict[str, float]] = {}
        if holdout_ratio > 0.0:
            LOGGER.info(
                "fedegsd routing holdout | ratio=%.3f min=%d max=%d avg_holdout=%.1f avg_train=%.1f",
                holdout_ratio,
                holdout_min_samples,
                holdout_max_samples,
                sum(len(dataset) for dataset in self.client_routing_datasets.values())
                / max(len(self.client_routing_datasets), 1),
                sum(len(dataset) for dataset in self.client_training_datasets.values())
                / max(len(self.client_training_datasets), 1),
            )

    # ================================================================
    # AVGLOGITS extraction (FedDF core)
    # ================================================================

    def _build_zero_eval_from_template(self, prefix: str, template_eval: Dict[str, object]) -> Dict[str, object]:
        client_results: Dict[str, Dict[str, float]] = {}
        for client_id, metrics in template_eval["clients"].items():
            client_results[client_id] = {
                "accuracy": 0.0,
                "hard_recall": 0.0,
                "hard_accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "invocation_rate": 0.0,
                "latency_ms": 0.0,
                "num_samples": int(metrics["num_samples"]),
                "num_hard_samples": int(metrics["num_hard_samples"]),
            }
        weighted = self._aggregate_metrics(client_results, weighted=True)
        macro = self._aggregate_metrics(client_results, weighted=False)
        groups = self._aggregate_group_metrics(client_results)
        self._log_client_metrics_table(prefix, client_results, weighted, macro, groups)
        return {
            "aggregate": weighted,
            "macro": macro,
            "groups": groups,
            "clients": client_results,
        }

    def _extract_ensemble_logits(self, updates: List[FedEGSDClientUpdate]) -> Dict[str, torch.Tensor]:
        """Average raw logits across experts, THEN softmax (FedDF AVGLOGITS)."""

        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)

        all_logits: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        for update in updates:
            expert = SmallCNN(
                num_classes=num_classes,
                base_channels=self.config.model.expert_base_channels,
            ).to(self.device)
            expert.load_state_dict(update.expert_state_dict)
            expert.eval()

            batches: List[torch.Tensor] = []
            with torch.no_grad():
                for batch in self.distill_loader:
                    images = batch[0].to(self.device)
                    batches.append(expert(images).cpu())

            client_logits = torch.cat(batches, dim=0)

            probs = F.softmax(client_logits / temperature, dim=1)
            log_c = math.log(float(num_classes))
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(log_c, 1e-8)
            w = torch.exp(-ent)

            all_logits.append(client_logits)
            all_weights.append(w)
            del expert

        stacked_l = torch.stack(all_logits, dim=0)
        stacked_w = torch.stack(all_weights, dim=0)

        norm_w = stacked_w / stacked_w.sum(0, keepdim=True).clamp_min(1e-8)
        avg_logits = (stacked_l * norm_w.unsqueeze(-1)).sum(0)
        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        sample_w = stacked_w.mean(0)

        return {"soft_labels": soft_labels, "sample_weights": sample_w}

    # ================================================================
    # Distillation: pure KL + Adam + cosine annealing
    # ================================================================

    def _distill_general_model(self, ensemble: Dict[str, torch.Tensor], round_idx: int) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = self.config.dataset.batch_size

        soft_all = ensemble["soft_labels"]
        w_all = ensemble["sample_weights"]

        all_imgs: List[torch.Tensor] = []
        for batch in self.distill_loader:
            all_imgs.append(batch[0])
        distill_images = torch.cat(all_imgs, dim=0)
        num_samples = distill_images.size(0)

        optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=float(self.config.federated.distill_lr),
        )
        total_steps = distill_epochs * ((num_samples + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        self.general_model.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            perm = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = perm[start:start + batch_size]
                imgs = distill_images[idx].to(self.device)
                b_soft = soft_all[idx].to(self.device)
                b_w = w_all[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(imgs)

                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                per_kl = F.kl_div(student_log_probs, b_soft, reduction="none").sum(1) * (temperature ** 2)
                loss = (per_kl * b_w).sum() / b_w.sum().clamp_min(1e-8)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_batches += 1

        divisor = max(total_batches, 1)
        return {"total_loss": total_loss / divisor, "kl_loss": total_loss / divisor}

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
        self.clients[client_id].expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            route_features = self._compute_route_features(
                client_id=client_id,
                expert_model=self.clients[client_id].expert_model,
                images=images,
            )
            predictions = route_features["expert_prediction"].clone()
            route_threshold = float(self.client_route_score_thresholds.get(client_id, self._initial_route_score_threshold()))
            fallback_mask = route_features["route_score"] < route_threshold
            invoked_general = int(fallback_mask.sum().item())
            route_types = ["expert"] * images.size(0)
            if fallback_mask.any():
                general_logits = self.general_model(images[fallback_mask])
                predictions[fallback_mask] = torch.argmax(general_logits, dim=1)
                for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_idx] = "general"
            metadata = {
                "route_type": route_types,
                "expert_confidence": route_features["confidence"].detach().cpu().tolist(),
            }
        return predictions, invoked_general, metadata

    def _should_run_temperature_calibration(self, round_idx: int) -> bool:
        if not bool(getattr(self.config.federated, "temperature_calibration_enabled", False)):
            return False
        frequency = max(int(getattr(self.config.federated, "temperature_calibration_frequency", 1)), 1)
        return round_idx <= 0 or (round_idx % frequency) == 0

    def _temperature_candidates(self, previous_temperature: float) -> List[float]:
        min_temperature = max(float(getattr(self.config.federated, "temperature_calibration_min", 0.5)), 1e-3)
        max_temperature = max(
            float(getattr(self.config.federated, "temperature_calibration_max", 5.0)),
            min_temperature,
        )
        num_candidates = max(int(getattr(self.config.federated, "temperature_calibration_candidates", 11)), 2)
        if math.isclose(min_temperature, max_temperature):
            return [min_temperature]

        candidates = torch.logspace(
            math.log10(min_temperature),
            math.log10(max_temperature),
            steps=num_candidates,
        ).tolist()
        candidates.extend([1.0, float(previous_temperature)])
        bounded = {
            min(max(float(candidate), min_temperature), max_temperature)
            for candidate in candidates
        }
        return sorted(bounded)

    def _fit_temperature_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        previous_temperature: float,
    ) -> float:
        if logits.numel() == 0 or targets.numel() == 0:
            return max(float(previous_temperature), 1e-4)

        previous_temperature = max(float(previous_temperature), 1e-4)
        candidates = self._temperature_candidates(previous_temperature)
        criterion = nn.CrossEntropyLoss()

        logits_device = logits.to(self.device, dtype=torch.float32)
        targets_device = targets.to(self.device, dtype=torch.long)
        best_temperature = previous_temperature
        best_loss = float("inf")

        with torch.no_grad():
            for candidate in candidates:
                loss = float(criterion(logits_device / candidate, targets_device).item())
                if loss < best_loss:
                    best_loss = loss
                    best_temperature = float(candidate)

        momentum = min(max(float(getattr(self.config.federated, "temperature_calibration_momentum", 0.0)), 0.0), 0.999)
        return (momentum * previous_temperature) + ((1.0 - momentum) * best_temperature)

    def _estimate_expert_temperature(self, client_id: str, expert_model: nn.Module, round_idx: int) -> float:
        previous_temperature = float(self.client_expert_temperatures.get(client_id, 1.0))
        if not self._should_run_temperature_calibration(round_idx):
            return previous_temperature

        routing_dataset = self.client_routing_datasets.get(client_id)
        if routing_dataset is None or len(routing_dataset) == 0:
            return previous_temperature

        loader = self.data_module.make_loader(routing_dataset, shuffle=False)
        logits_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []
        expert_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                logits_batches.append(expert_model(images).detach().cpu())
                target_batches.append(targets.detach().cpu())
        if not logits_batches or not target_batches:
            return previous_temperature

        return self._fit_temperature_from_logits(
            logits=torch.cat(logits_batches, dim=0),
            targets=torch.cat(target_batches, dim=0),
            previous_temperature=previous_temperature,
        )

    def _compute_client_prototypes(
        self,
        client_id: str,
        expert_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset = self.client_training_datasets.get(client_id)
        num_classes = self.config.model.num_classes
        feature_dim = getattr(expert_model, "feature_dim", None)
        if dataset is None or len(dataset) == 0 or feature_dim is None:
            return (
                torch.zeros(num_classes, int(feature_dim or 1), dtype=torch.float32),
                torch.ones(num_classes, dtype=torch.float32),
                torch.zeros(num_classes, dtype=torch.bool),
            )

        loader = self.data_module.make_loader(dataset, shuffle=False)
        sums = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
        counts = torch.zeros(num_classes, dtype=torch.long)
        expert_model.eval()

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                features = expert_model.forward_features(images).detach().cpu()
                targets_cpu = targets.detach().cpu().long()
                for class_idx in range(num_classes):
                    class_mask = targets_cpu == class_idx
                    if not class_mask.any():
                        continue
                    sums[class_idx] += features[class_mask].sum(dim=0)
                    counts[class_idx] += int(class_mask.sum().item())

        prototypes = torch.zeros_like(sums)
        mask = counts > 0
        if mask.any():
            prototypes[mask] = sums[mask] / counts[mask].unsqueeze(1).to(dtype=torch.float32)

        scales = torch.ones(num_classes, dtype=torch.float32)
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                features = expert_model.forward_features(images).detach().cpu()
                targets_cpu = targets.detach().cpu().long()
                for class_idx in range(num_classes):
                    class_mask = targets_cpu == class_idx
                    if not class_mask.any() or counts[class_idx] <= 0:
                        continue
                    distances = torch.norm(features[class_mask] - prototypes[class_idx].unsqueeze(0), dim=1)
                    scales[class_idx] += distances.sum()
            for class_idx in range(num_classes):
                if counts[class_idx] > 0:
                    scales[class_idx] = max(float(scales[class_idx] / counts[class_idx].to(dtype=torch.float32)), 1e-3)
                else:
                    scales[class_idx] = 1.0
        return prototypes, scales, mask

    def _distance_routing_enabled(self) -> bool:
        warmup_rounds = int(getattr(self.config.inference, "route_warmup_rounds", 0))
        disable_in_warmup = bool(getattr(self.config.inference, "route_disable_distance_during_warmup", False))
        return not (disable_in_warmup and self.current_round <= warmup_rounds)

    def _compute_route_features(
        self,
        client_id: str,
        expert_model: nn.Module,
        images: torch.Tensor,
        temperature: Optional[float] = None,
        prototypes: Optional[torch.Tensor] = None,
        prototype_scales: Optional[torch.Tensor] = None,
        prototype_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = expert_model.forward_features(images)
        logits = expert_model.classify_features(features)
        safe_temperature = max(
            float(
                temperature
                if temperature is not None
                else self.client_expert_temperatures.get(client_id, 1.0)
            ),
            1e-4,
        )
        calibrated_logits = logits / safe_temperature
        probs = torch.softmax(calibrated_logits, dim=1)
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
        confidence = topk.values[:, 0]
        expert_prediction = topk.indices[:, 0]
        if topk.values.size(1) > 1:
            margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            margin = torch.ones_like(confidence)

        distance_penalty = torch.zeros_like(confidence)
        normalized_distance = torch.zeros_like(confidence)
        if self._distance_routing_enabled():
            prototypes = prototypes if prototypes is not None else self.client_prototypes.get(client_id)
            prototype_scales = prototype_scales if prototype_scales is not None else self.client_prototype_scales.get(client_id)
            prototype_mask = prototype_mask if prototype_mask is not None else self.client_prototype_masks.get(client_id)
            if (
                prototypes is not None
                and prototype_scales is not None
                and prototype_mask is not None
                and bool(prototype_mask.any().item())
            ):
                valid_classes = prototype_mask.nonzero(as_tuple=False).flatten()
                device_prototypes = prototypes.to(self.device, dtype=features.dtype)[valid_classes]
                pairwise_distance = torch.cdist(features, device_prototypes, p=2)
                nearest_distance, nearest_positions = pairwise_distance.min(dim=1)
                nearest_classes = valid_classes.to(self.device)[nearest_positions]
                if bool(getattr(self.config.inference, "route_distance_normalization", True)):
                    device_scales = prototype_scales.to(self.device, dtype=features.dtype).clamp_min(1e-6)
                    normalized_distance = nearest_distance / device_scales[nearest_classes]
                else:
                    normalized_distance = nearest_distance
                distance_threshold = max(float(getattr(self.config.inference, "route_distance_threshold", 0.0)), 0.0)
                distance_penalty = (normalized_distance - distance_threshold).clamp_min(0.0)

        confidence_weight = float(getattr(self.config.inference, "route_score_confidence_weight", 0.50))
        margin_weight = float(getattr(self.config.inference, "route_score_margin_weight", 0.35))
        distance_weight = float(getattr(self.config.inference, "route_score_distance_weight", 0.20))
        route_score = (
            (confidence_weight * confidence)
            + (margin_weight * margin)
            - (distance_weight * distance_penalty)
        )
        return {
            "logits": logits,
            "confidence": confidence,
            "expert_prediction": expert_prediction,
            "margin": margin,
            "distance": normalized_distance,
            "distance_penalty": distance_penalty,
            "route_score": route_score,
        }

    def _initial_route_score_threshold(self) -> float:
        confidence_threshold = float(getattr(self.config.inference, "confidence_threshold", 0.5))
        margin_threshold = float(getattr(self.config.inference, "route_margin_threshold", 0.0))
        confidence_weight = float(getattr(self.config.inference, "route_score_confidence_weight", 0.50))
        margin_weight = float(getattr(self.config.inference, "route_score_margin_weight", 0.35))
        return (confidence_weight * confidence_threshold) + (margin_weight * margin_threshold)

    def _target_invocation_rate_for_client(self, client_id: str) -> float:
        if client_id.startswith("complex_"):
            return float(
                getattr(
                    self.config.inference,
                    "complex_target_general_invocation_rate",
                    self.config.inference.target_general_invocation_rate,
                )
            )
        if client_id.startswith("simple_"):
            return float(
                getattr(
                    self.config.inference,
                    "simple_target_general_invocation_rate",
                    self.config.inference.target_general_invocation_rate,
                )
            )
        return float(self.config.inference.target_general_invocation_rate)

    def _collect_client_routing_statistics(
        self,
        client_id: str,
        expert_model: nn.Module,
        temperature: float,
        prototypes: torch.Tensor,
        prototype_scales: torch.Tensor,
        prototype_mask: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        routing_dataset = self.client_routing_datasets.get(client_id)
        if routing_dataset is None or len(routing_dataset) == 0:
            return None

        loader = self.data_module.make_loader(routing_dataset, shuffle=False)
        score_batches: List[torch.Tensor] = []
        expert_correct_batches: List[torch.Tensor] = []
        general_correct_batches: List[torch.Tensor] = []
        confidence_batches: List[torch.Tensor] = []
        margin_batches: List[torch.Tensor] = []
        distance_batches: List[torch.Tensor] = []

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                route_features = self._compute_route_features(
                    client_id=client_id,
                    expert_model=expert_model,
                    images=images,
                    temperature=temperature,
                    prototypes=prototypes,
                    prototype_scales=prototype_scales,
                    prototype_mask=prototype_mask,
                )
                general_predictions = self.general_model(images).argmax(dim=1)
                score_batches.append(route_features["route_score"].detach().cpu())
                confidence_batches.append(route_features["confidence"].detach().cpu())
                margin_batches.append(route_features["margin"].detach().cpu())
                distance_batches.append(route_features["distance"].detach().cpu())
                expert_correct_batches.append(route_features["expert_prediction"].eq(targets_device).detach().cpu())
                general_correct_batches.append(general_predictions.eq(targets_device).detach().cpu())

        if not score_batches:
            return None
        return {
            "scores": torch.cat(score_batches, dim=0),
            "confidence": torch.cat(confidence_batches, dim=0),
            "margin": torch.cat(margin_batches, dim=0),
            "distance": torch.cat(distance_batches, dim=0),
            "expert_correct": torch.cat(expert_correct_batches, dim=0).to(dtype=torch.bool),
            "general_correct": torch.cat(general_correct_batches, dim=0).to(dtype=torch.bool),
        }

    def _select_route_threshold(
        self,
        client_id: str,
        statistics: Dict[str, torch.Tensor],
        previous_threshold: float,
    ) -> Tuple[float, Dict[str, float]]:
        scores = statistics["scores"].float()
        expert_correct = statistics["expert_correct"].to(dtype=torch.float32)
        general_correct = statistics["general_correct"].to(dtype=torch.float32)
        if scores.numel() == 0:
            return previous_threshold, {
                "holdout_routed_accuracy": 0.0,
                "invocation_rate": 0.0,
                "expert_risk": 0.0,
                "threshold": previous_threshold,
            }

        unique_candidates = torch.unique(scores.cpu()).tolist()
        candidates = [float(scores.min().item()) - 1e-6, float(scores.max().item()) + 1e-6, float(previous_threshold)]
        candidates.extend(float(candidate) for candidate in unique_candidates)
        candidates = sorted(set(candidates))

        selection_mode = str(getattr(self.config.inference, "routing_selection_mode", "budget")).lower()
        target_rate = self._target_invocation_rate_for_client(client_id)
        target_expert_risk = max(float(getattr(self.config.inference, "target_expert_risk", 0.25)), 0.0)
        best_threshold = float(previous_threshold)
        best_metrics: Optional[Dict[str, float]] = None
        best_rank: Optional[Tuple[float, float, float, float]] = None

        for threshold in candidates:
            fallback_mask = scores < float(threshold)
            invocation_rate = float(fallback_mask.to(dtype=torch.float32).mean().item())
            routed_correct = torch.where(fallback_mask, general_correct, expert_correct)
            routed_accuracy = float(routed_correct.mean().item())
            expert_keep = ~fallback_mask
            if bool(expert_keep.any().item()):
                expert_risk = float((1.0 - expert_correct[expert_keep].mean()).item())
            else:
                expert_risk = 1.0

            if selection_mode == "risk":
                violation = max(expert_risk - target_expert_risk, 0.0)
                auxiliary_gap = abs(invocation_rate - target_rate)
            else:
                violation = max(invocation_rate - target_rate, 0.0)
                auxiliary_gap = abs(expert_risk - target_expert_risk)

            rank = (violation, -routed_accuracy, auxiliary_gap, abs(invocation_rate - target_rate))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_threshold = float(threshold)
                best_metrics = {
                    "holdout_routed_accuracy": routed_accuracy,
                    "invocation_rate": invocation_rate,
                    "expert_risk": expert_risk,
                    "threshold": float(threshold),
                }

        return best_threshold, best_metrics or {
            "holdout_routed_accuracy": 0.0,
            "invocation_rate": 0.0,
            "expert_risk": 0.0,
            "threshold": best_threshold,
        }

    # ================================================================
    # Routing threshold adaptation
    # ================================================================

    def _update_routing_thresholds(self, updates):
        updated_metrics: List[Dict[str, float]] = []
        for update in updates:
            cid = update.client_id
            client = self.clients[cid]
            temperature = self._estimate_expert_temperature(cid, client.expert_model, self.current_round)
            prototypes, prototype_scales, prototype_mask = self._compute_client_prototypes(cid, client.expert_model)
            self.client_expert_temperatures[cid] = temperature
            self.client_prototypes[cid] = prototypes.detach().cpu()
            self.client_prototype_scales[cid] = prototype_scales.detach().cpu()
            self.client_prototype_masks[cid] = prototype_mask.detach().cpu()

            routing_statistics = self._collect_client_routing_statistics(
                client_id=cid,
                expert_model=client.expert_model,
                temperature=temperature,
                prototypes=prototypes,
                prototype_scales=prototype_scales,
                prototype_mask=prototype_mask,
            )
            previous_threshold = float(
                self.client_route_score_thresholds.get(cid, self._initial_route_score_threshold())
            )
            if routing_statistics is None:
                self.client_route_score_thresholds[cid] = previous_threshold
                continue

            threshold, metrics = self._select_route_threshold(
                client_id=cid,
                statistics=routing_statistics,
                previous_threshold=previous_threshold,
            )
            self.client_route_score_thresholds[cid] = threshold
            metrics["temperature"] = temperature
            self.client_routing_metrics[cid] = metrics
            updated_metrics.append(metrics)

        if updated_metrics:
            mean_threshold = sum(item["threshold"] for item in updated_metrics) / len(updated_metrics)
            mean_temperature = sum(item["temperature"] for item in updated_metrics) / len(updated_metrics)
            mean_holdout_accuracy = sum(item["holdout_routed_accuracy"] for item in updated_metrics) / len(updated_metrics)
            mean_invocation = sum(item["invocation_rate"] for item in updated_metrics) / len(updated_metrics)
            LOGGER.info(
                "fedegsd routing update | round=%d | clients=%d | holdout_acc=%.4f | invoke=%.4f | threshold=%.4f | temp=%.4f",
                self.current_round,
                len(updated_metrics),
                mean_holdout_accuracy,
                mean_invocation,
                mean_threshold,
                mean_temperature,
            )
            if self.writer is not None:
                self._log_compare_scalars(
                    "fedegsd",
                    self.current_round,
                    {
                        "routing_holdout_accuracy": mean_holdout_accuracy,
                        "routing_score_threshold": mean_threshold,
                        "routing_temperature": mean_temperature,
                    },
                )

    def _aggregate_route_effectiveness_metrics(
        self,
        client_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        keys = (
            "general_gain_over_expert",
            "routed_gain_over_expert",
            "invoked_general_accuracy",
            "invoked_expert_accuracy",
            "invoked_general_gain",
            "oracle_route_accuracy",
            "oracle_general_invocation_rate",
            "expert_bad_general_good_rate",
            "routing_regret",
            "expert_general_disagreement_rate",
        )
        if not client_metrics:
            return {key: 0.0 for key in keys}
        return {
            key: sum(float(metrics[key]) for metrics in client_metrics.values()) / max(len(client_metrics), 1)
            for key in keys
        }

    def _evaluate_route_effectiveness_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        expert_macro_accuracy = float(expert_eval["macro"]["accuracy"])
        general_macro_accuracy = float(general_eval["macro"]["accuracy"])
        routed_macro_accuracy = float(routed_eval["macro"]["accuracy"])

        if not getattr(self, "general_enabled", True):
            return {
                "general_gain_over_expert": general_macro_accuracy - expert_macro_accuracy,
                "routed_gain_over_expert": routed_macro_accuracy - expert_macro_accuracy,
                "invoked_general_accuracy": 0.0,
                "invoked_expert_accuracy": 0.0,
                "invoked_general_gain": 0.0,
                "oracle_route_accuracy": routed_macro_accuracy,
                "oracle_general_invocation_rate": 0.0,
                "expert_bad_general_good_rate": 0.0,
                "routing_regret": 0.0,
                "expert_general_disagreement_rate": 0.0,
            }

        client_route_metrics: Dict[str, Dict[str, float]] = {}
        self.general_model.eval()
        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()

                num_samples = 0
                invoked_total = 0
                expert_correct = 0
                general_correct = 0
                routed_correct = 0
                oracle_correct = 0
                oracle_general_invocations = 0
                disagreement_total = 0
                invoked_general_correct = 0
                invoked_expert_correct = 0

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    route_features = self._compute_route_features(
                        client_id=client_id,
                        expert_model=expert_model,
                        images=images,
                    )
                    expert_predictions = route_features["expert_prediction"]
                    route_threshold = float(
                        self.client_route_score_thresholds.get(client_id, self._initial_route_score_threshold())
                    )
                    fallback_mask = route_features["route_score"] < route_threshold
                    general_predictions = self.general_model(images).argmax(dim=1)
                    routed_predictions = expert_predictions.clone()
                    routed_predictions[fallback_mask] = general_predictions[fallback_mask]

                    batch_size = int(targets_device.numel())
                    num_samples += batch_size
                    invoked_total += int(fallback_mask.sum().item())
                    expert_correct += int((expert_predictions == targets_device).sum().item())
                    general_correct += int((general_predictions == targets_device).sum().item())
                    routed_correct += int((routed_predictions == targets_device).sum().item())
                    oracle_correct += int(
                        ((expert_predictions == targets_device) | (general_predictions == targets_device)).sum().item()
                    )
                    oracle_general_invocations += int(
                        ((expert_predictions != targets_device) & (general_predictions == targets_device)).sum().item()
                    )
                    disagreement_total += int((expert_predictions != general_predictions).sum().item())

                    if fallback_mask.any():
                        invoked_general_correct += int(
                            (general_predictions[fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )
                        invoked_expert_correct += int(
                            (expert_predictions[fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )

                expert_accuracy = expert_correct / max(num_samples, 1)
                general_accuracy = general_correct / max(num_samples, 1)
                routed_accuracy = routed_correct / max(num_samples, 1)
                invoked_general_accuracy = invoked_general_correct / max(invoked_total, 1)
                invoked_expert_accuracy = invoked_expert_correct / max(invoked_total, 1)
                oracle_route_accuracy = oracle_correct / max(num_samples, 1)
                client_route_metrics[client_id] = {
                    "general_gain_over_expert": general_accuracy - expert_accuracy,
                    "routed_gain_over_expert": routed_accuracy - expert_accuracy,
                    "invoked_general_accuracy": invoked_general_accuracy,
                    "invoked_expert_accuracy": invoked_expert_accuracy,
                    "invoked_general_gain": invoked_general_accuracy - invoked_expert_accuracy,
                    "oracle_route_accuracy": oracle_route_accuracy,
                    "oracle_general_invocation_rate": oracle_general_invocations / max(num_samples, 1),
                    "expert_bad_general_good_rate": oracle_general_invocations / max(num_samples, 1),
                    "routing_regret": oracle_route_accuracy - routed_accuracy,
                    "expert_general_disagreement_rate": disagreement_total / max(num_samples, 1),
                }

        return self._aggregate_route_effectiveness_metrics(client_route_metrics)

    def _build_round_extra_metrics(
        self,
        round_idx: int,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        if not extra_metrics:
            return ""
        return (
            f" | g_gain={extra_metrics.get('general_gain_over_expert', 0.0):.4f}"
            f" | route_gain={extra_metrics.get('routed_gain_over_expert', 0.0):.4f}"
            f" | invoked_g={extra_metrics.get('invoked_general_gain', 0.0):.4f}"
            f" | oracle={extra_metrics.get('oracle_route_accuracy', 0.0):.4f}"
            f" | oracle_g={extra_metrics.get('oracle_general_invocation_rate', 0.0):.4f}"
            f" | e_bad_g_good={extra_metrics.get('expert_bad_general_good_rate', 0.0):.4f}"
            f" | regret={extra_metrics.get('routing_regret', 0.0):.4f}"
        )

    # ================================================================
    # Snapshots
    # ================================================================

    def _maybe_update_best(self, ri, rm, ea, ga):
        better = self.best_snapshot is None or rm.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not better:
            return False
        self.best_snapshot = {
            "round_idx": ri,
            "routed_accuracy": rm.routed_accuracy,
            "general_accuracy": ga,
            "expert_accuracy": ea,
            "avg_client_loss": rm.avg_client_loss,
            "general_model_state": {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()},
            "client_expert_states": {
                c: {k: v.cpu().clone() for k, v in cl.expert_model.state_dict().items()}
                for c, cl in self.clients.items()
            },
            "client_route_score_thresholds": dict(self.client_route_score_thresholds),
            "client_expert_temperatures": dict(self.client_expert_temperatures),
            "client_prototypes": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototypes.items()
            },
            "client_prototype_scales": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototype_scales.items()
            },
            "client_prototype_masks": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototype_masks.items()
            },
        }
        LOGGER.info(
            "fedegsd best | round=%d | routed=%.4f | general=%.4f | expert=%.4f",
            ri,
            rm.routed_accuracy,
            ga,
            ea,
        )
        return True

    def _restore_best(self):
        if not self.best_snapshot:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for c, st in self.best_snapshot["client_expert_states"].items():
            self.clients[c].expert_model.load_state_dict(st)
        self.client_route_score_thresholds = dict(self.best_snapshot["client_route_score_thresholds"])
        self.client_expert_temperatures = dict(self.best_snapshot["client_expert_temperatures"])
        self.client_prototypes = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototypes"].items()
        }
        self.client_prototype_scales = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototype_scales"].items()
        }
        self.client_prototype_masks = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototype_masks"].items()
        }
        self.current_round = int(self.best_snapshot["round_idx"])
        LOGGER.info("fedegsd restored best from round %d", self.best_snapshot["round_idx"])

    # ================================================================
    # Main loop
    # ================================================================

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        kd_warmup = int(getattr(self.config.federated, "expert_kd_warmup_rounds", 10))
        expert_kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        upload_bytes_total = 0.0

        for ri in range(1, self.config.federated.rounds + 1):
            self._device_synchronize()
            round_start_time = time.perf_counter()
            self.current_round = ri
            sids = self._sample_client_ids()

            use_kd = self.general_enabled and expert_kd_weight > 0.0 and ri > kd_warmup
            teacher = self.general_model if use_kd else None
            if use_kd and ri == kd_warmup + 1:
                LOGGER.info(
                    "fedegsd round %d | KD warmup complete, enabling expert KD from general model",
                    ri,
                )

            LOGGER.info("fedegsd round %d | clients=%s | kd=%s", ri, sids, use_kd)

            updates = [self.clients[c].train_local(general_model=teacher) for c in sids]
            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd-expert")
            if self.general_enabled:
                ensemble = self._extract_ensemble_logits(updates)
                distill_stats = self._distill_general_model(ensemble, ri)
                self._update_routing_thresholds(updates)
                general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd-general")
                routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsd-routed")
            else:
                distill_stats = {"total_loss": 0.0}
                general_eval = self._build_zero_eval_from_template("fedegsd-general-disabled", expert_eval)
                routed_eval = copy.deepcopy(expert_eval)
            extra_metrics = self._build_round_extra_metrics(
                round_idx=ri,
                expert_eval=expert_eval,
                general_eval=general_eval,
                routed_eval=routed_eval,
            )

            avg_loss = sum(u.loss for u in updates) / max(len(updates), 1)
            aggregate = routed_eval["aggregate"]
            macro = routed_eval["macro"]
            compute_mode = "routed" if self.general_enabled else "expert_only"
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate["invocation_rate"],
                compute_mode,
            )
            client_train_profile = self._build_client_training_profile(
                self.expert_flops,
                sids,
                self.client_training_datasets,
            )
            resource_metrics = self._resource_metric_values(
                self.resource_profiles,
                client_train_profile,
                compute_profile,
            )
            round_upload_bytes = float(
                sum(self._estimate_tensor_payload_bytes(update.expert_state_dict) for update in updates)
            )
            upload_bytes_total += round_upload_bytes
            self._device_synchronize()
            round_train_time_seconds = time.perf_counter() - round_start_time

            round_metrics = RoundMetrics(
                round_idx=ri,
                avg_client_loss=avg_loss,
                routed_accuracy=macro["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                inference_latency_ms=aggregate["latency_ms"],
                round_train_time_seconds=round_train_time_seconds,
                upload_bytes_per_round=round_upload_bytes,
                upload_bytes_total=upload_bytes_total,
                extra_metrics=extra_metrics,
                **resource_metrics,
            )
            extra_log = self._format_round_extra_metrics_for_log(extra_metrics)

            LOGGER.info(
                "fedegsd round %d | loss=%.4f | distill=%.4f | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f%s%s",
                ri,
                avg_loss,
                distill_stats["total_loss"],
                macro["accuracy"],
                aggregate["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                extra_log,
                self._format_resource_metrics_for_log(round_metrics),
            )

            if self.writer:
                self._log_compare_scalars(
                    "fedegsd",
                    ri,
                    {
                        "expert_loss": avg_loss,
                        "distill_loss": distill_stats["total_loss"],
                    },
                )
                self._log_auxiliary_accuracy_metrics(
                    "fedegsd",
                    ri,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics("fedegsd", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best(
                ri,
                round_metrics,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best()
        return metrics

    # ================================================================
    # Final eval
    # ================================================================

    def evaluate_baselines(self, test_dataset):
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd_final_expert")
        route_export_path = None
        if self.general_enabled:
            route_export_path = self._build_route_export_path("fedegsd_final_routed")
            routed_eval = self._evaluate_predictor_on_client_tests(
                self._predict_routed,
                "fedegsd_final_routed",
                route_export_path=route_export_path,
            )
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd_final_general")
        else:
            routed_eval = copy.deepcopy(expert_eval)
            general_eval = self._build_zero_eval_from_template("fedegsd_final_general_disabled", expert_eval)
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = 0
        if self.best_snapshot:
            final_loss = float(self.best_snapshot["avg_client_loss"])
            best_round = int(self.best_snapshot["round_idx"])
        compute_mode = "routed" if self.general_enabled else "expert_only"
        routed_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            routed_eval["aggregate"]["invocation_rate"],
            compute_mode,
        )
        expert_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, "expert_only")
        general_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, "general_only")
        final_history = self.last_history[-1] if self.last_history else RoundMetrics(0, 0.0, 0.0, 0.0, 0.0)
        final_client_train = {
            "avg_flops_per_client": final_history.client_train_flops,
            "total_flops": final_history.client_train_flops_total,
            "num_clients": self.config.federated.clients_per_round,
            "num_samples": 0,
        }
        final_resource_metrics = self._resource_metric_values(
            self.resource_profiles,
            final_client_train,
            routed_compute,
        )
        average_round_train_time_seconds = (
            sum(item.round_train_time_seconds for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_train_time_seconds = (
            sum(item.round_train_time_seconds for item in self.last_history)
            if self.last_history
            else 0.0
        )
        average_upload_bytes_per_round = (
            sum(item.upload_bytes_per_round for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_upload_bytes = self.last_history[-1].upload_bytes_total if self.last_history else 0.0
        extra_metrics = self._build_final_extra_metrics(expert_eval, general_eval, routed_eval)
        return {
            "algorithm": "fedegsd",
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
                "final_training_loss": final_loss,
                "best_round": best_round,
                "average_inference_latency_ms": routed_eval["aggregate"]["latency_ms"],
                "average_round_train_time_seconds": average_round_train_time_seconds,
                "total_train_time_seconds": total_train_time_seconds,
                "average_upload_bytes_per_round": average_upload_bytes_per_round,
                "total_upload_bytes": total_upload_bytes,
                **final_resource_metrics,
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
                "client_train": final_client_train,
            },
            "memory_mb": self._resource_memory_table(self.resource_profiles),
            "artifacts": {"route_csv": str(route_export_path) if route_export_path is not None else None},
        }
