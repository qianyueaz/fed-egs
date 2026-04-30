"""
FedEGS-SG: Federated Expert-General System with Stabilized General Distillation.

This variant keeps the four core FedEGS principles intact:
  1. Clients only train a lightweight expert locally.
  2. A stronger general model is maintained and deployed locally for fallback.
  3. Knowledge flows both ways: expert -> general on public anchors, general -> expert on private data.
  4. Routing is expert-first with local general fallback only.

Implementation references:
  - FedDF: server-side AVGLOGITS / KL distillation.
  - FedGO: temporal teacher buffering for a smoother global teacher.
  - Selective routing ideas: confidence + margin + energy-aware fallback.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
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
from fedegs.federated.compression import (
    CompressedStateDict,
    compress_state_dict,
    decompress_state_dict,
    estimate_state_dict_nbytes,
)
from fedegs.models import (
    build_model,
    build_teacher_model,
    estimate_model_flops,
    model_memory_mb,
)


@dataclass
class FedEGSSGClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_logits: torch.Tensor
    external_logits: torch.Tensor
    kd_gate_ratio: float = 0.0
    refresh_gate_ratio: float = 0.0
    raw_upload_bytes: int = 0
    compressed_upload_bytes: int = 0


class GeneralModel(nn.Module):
    """General model wrapper for the ResNet-18 teacher backbone."""

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

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def _normalized_architecture_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def _build_general_model_from_config(config) -> nn.Module:
    architecture = _normalized_architecture_name(getattr(config.model, "general_architecture", "teacher_resnet18"))
    if architecture in {"teacher_resnet18", "teacher_resnet", "resnet18_teacher"}:
        return GeneralModel(
            num_classes=config.model.num_classes,
            pretrained_imagenet=bool(getattr(config.federated, "general_pretrain_imagenet_init", False)),
        )
    return build_model(
        architecture=architecture,
        num_classes=config.model.num_classes,
        width_factor=float(getattr(config.model, "general_width", 1.0)),
        base_channels=int(getattr(config.model, "general_base_channels", 32)),
    )


def _build_expert_model_from_config(config) -> nn.Module:
    architecture = _normalized_architecture_name(getattr(config.model, "expert_architecture", "small_cnn"))
    return build_model(
        architecture=architecture,
        num_classes=config.model.num_classes,
        width_factor=float(getattr(config.model, "expert_width", 1.0)),
        base_channels=int(getattr(config.model, "expert_base_channels", 32)),
    )


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _unpack_loader_batch(batch):
    if not isinstance(batch, (tuple, list)) or not batch:
        raise ValueError("Expected a non-empty tuple/list batch from the dataloader.")
    images = batch[0]
    targets = batch[1] if len(batch) > 1 else None
    indices = batch[2] if len(batch) > 2 else None
    return images, targets, indices


def load_distillation_dataset(
    name: str,
    root: str,
    max_samples: int = 0,
) -> Dataset:
    """Load an external unlabeled dataset for server-side ensemble distillation."""
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    normalized = name.lower().replace("-", "").replace("_", "")

    if normalized == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    elif normalized == "svhn":
        dataset = datasets.SVHN(root=root, split="train", download=True, transform=transform)
    elif normalized in {"imagenet32", "imagenet"}:
        folder = Path(root) / "imagenet32" / "train"
        if not folder.exists():
            raise FileNotFoundError(
                f"ImageNet-32 folder not found at {folder}. Download ImageNet-32 and organize it as ImageFolder."
            )
        dataset = datasets.ImageFolder(str(folder), transform=transform)
    else:
        folder = Path(root) / name
        if folder.exists():
            dataset = datasets.ImageFolder(str(folder), transform=transform)
        else:
            raise ValueError(f"Unsupported distillation dataset: {name}")

    total = len(dataset)
    if 0 < max_samples < total:
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))

    LOGGER.info(
        "fedegssg external distillation dataset '%s': %d samples (max_samples=%d)",
        name,
        len(dataset),
        max_samples,
    )
    return dataset


def _normalized_teacher_aggregation_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _client_holdout_seed(base_seed: int, client_id: str, seed_offset: int) -> int:
    stable_offset = sum((index + 1) * ord(char) for index, char in enumerate(client_id))
    return int(base_seed) + int(seed_offset) + stable_offset


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


class FedEGSSGClient(BaseFederatedClient):
    """Client-side lightweight expert training with gated KD from the deployed general model."""

    def __init__(self, client_id, dataset, num_classes, device, config, data_module):
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.num_classes = num_classes
        self.expert_model = _build_expert_model_from_config(config).to(self.device)

    def train_local(
        self,
        round_idx: int,
        public_batches: Sequence,
        distill_batches: Optional[Sequence] = None,
        general_model: Optional[nn.Module] = None,
        general_model_payload: Optional[CompressedStateDict] = None,
        general_public_accuracy: float = 0.0,
        general_temperature: float = 1.0,
    ) -> FedEGSSGClientUpdate:
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        kd_warmup_rounds = int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0))
        use_kd = general_model is not None or general_model_payload is not None
        use_kd = use_kd and kd_weight > 0.0 and round_idx > kd_warmup_rounds

        teacher_model = general_model
        if teacher_model is None and general_model_payload is not None:
            teacher_model = self._build_general_model_from_payload(general_model_payload)

        refresh_gate_ratio = 0.0
        if use_kd and teacher_model is not None:
            refresh_gate_ratio = self._refresh_expert_from_general(
                public_batches,
                teacher_model,
                general_public_accuracy,
                general_temperature=general_temperature,
            )
            loss, kd_gate_ratio = self._train_with_gated_kd(
                loader,
                teacher_model,
                teacher_temperature=general_temperature,
            )
        else:
            loss = self._optimize_model(
                model=self.expert_model,
                loader=loader,
                epochs=self.config.federated.local_epochs,
                lr=self.config.federated.local_lr,
                momentum=self.config.federated.local_momentum,
                weight_decay=self.config.federated.local_weight_decay,
            )
            kd_gate_ratio = 0.0

        public_logits = self._collect_logits(public_batches)
        external_logits = self._collect_logits(distill_batches or [])
        upload_bytes = (
            int(public_logits.numel() * public_logits.element_size())
            + int(external_logits.numel() * external_logits.element_size())
        )
        return FedEGSSGClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            public_logits=public_logits,
            external_logits=external_logits,
            kd_gate_ratio=kd_gate_ratio,
            refresh_gate_ratio=refresh_gate_ratio,
            raw_upload_bytes=upload_bytes,
            compressed_upload_bytes=upload_bytes,
        )

    def _build_general_model_from_payload(self, payload: CompressedStateDict) -> nn.Module:
        teacher = _build_general_model_from_config(self.config).to(self.device)
        teacher.load_state_dict(decompress_state_dict(payload))
        return teacher

    def _collect_logits(self, batches: Sequence) -> torch.Tensor:
        if not batches:
            return torch.empty((0, self.num_classes), dtype=torch.float32)
        logits: List[torch.Tensor] = []
        self.expert_model.eval()
        with torch.no_grad():
            for batch in batches:
                images, _, _ = _unpack_loader_batch(batch)
                images = images.to(self.device)
                logits.append(self.expert_model(images).detach().cpu())
        return torch.cat(logits, dim=0)

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * weights) / weights.sum().clamp_min(1e-8)

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

    def _align_features(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_dim = min(student_features.size(1), teacher_features.size(1))
        knowledge_dim = max(int(getattr(self.config.model, "knowledge_dim", 0)), 0)
        if knowledge_dim > 0:
            shared_dim = min(shared_dim, knowledge_dim)

        if student_features.size(1) != shared_dim:
            student_features = F.adaptive_avg_pool1d(student_features.unsqueeze(1), shared_dim).squeeze(1)
        if teacher_features.size(1) != shared_dim:
            teacher_features = F.adaptive_avg_pool1d(teacher_features.unsqueeze(1), shared_dim).squeeze(1)

        return F.normalize(student_features, dim=1), F.normalize(teacher_features, dim=1)

    def _weighted_feature_hint_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        aligned_student, aligned_teacher = self._align_features(student_features, teacher_features)
        per_sample_mse = torch.mean((aligned_student - aligned_teacher) ** 2, dim=1)
        return self._weighted_mean(per_sample_mse, sample_weights)

    def _build_refresh_weights(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        hard_weight = max(float(getattr(self.config.federated, "client_hard_weight", 0.0)), 0.0)
        margin_threshold = float(getattr(self.config.federated, "client_hard_margin_threshold", 0.20))
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        uncertainty = 1.0 - teacher_confidence
        low_margin = (teacher_margin < margin_threshold).to(dtype=teacher_confidence.dtype)
        weights = 1.0 + hard_weight * (0.75 * uncertainty + 0.25 * low_margin)
        return weights.clamp_min(1.0)

    def _select_teacher_gate_by_coverage(
        self,
        scores: torch.Tensor,
        target_coverage: float,
    ) -> torch.Tensor:
        target_coverage = min(max(float(target_coverage), 0.0), 1.0)
        if scores.numel() == 0 or target_coverage <= 0.0:
            return scores

        positive_mask = scores > 0
        positive_count = int(positive_mask.sum().item())
        if positive_count <= 0:
            return torch.zeros_like(scores)

        desired_count = min(max(int(math.ceil(scores.numel() * target_coverage)), 1), positive_count)
        selected_indices = torch.topk(scores, k=desired_count).indices
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        selected_mask[selected_indices] = True

        gated_scores = torch.zeros_like(scores)
        gated_scores[selected_mask] = scores[selected_mask]
        return gated_scores

    def _teacher_gate_from_logits(
        self,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        confidence_threshold: float,
        margin_threshold: float,
        student_logits: Optional[torch.Tensor] = None,
        hard_boost: float = 0.0,
        gate_temperature: float = 1.0,
        target_coverage: float = 0.0,
        adaptive_coverage_enabled: bool = False,
    ) -> torch.Tensor:
        safe_gate_temperature = max(float(gate_temperature), 1e-4)
        teacher_probs = F.softmax(teacher_logits.detach() / safe_gate_temperature, dim=1)
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_prediction = teacher_topk.indices[:, 0]
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        teacher_correct = teacher_prediction.eq(targets).to(dtype=teacher_confidence.dtype)
        conf_denominator = max(1.0 - confidence_threshold, 1e-6)
        margin_denominator = max(1.0 - margin_threshold, 1e-6)
        conf_gate = ((teacher_confidence - confidence_threshold) / conf_denominator).clamp(0.0, 1.0)
        margin_gate = ((teacher_margin - margin_threshold) / margin_denominator).clamp(0.0, 1.0)

        teacher_advantage = torch.zeros_like(teacher_confidence)
        student_hard = torch.zeros_like(teacher_confidence)
        student_saturated = torch.zeros_like(teacher_confidence)
        gate_power = max(float(getattr(self.config.federated, "expert_kd_gate_power", 1.0)), 1e-3)

        if student_logits is not None:
            student_probs = F.softmax(student_logits.detach() / safe_gate_temperature, dim=1)
            student_topk = torch.topk(student_probs, k=min(2, student_probs.size(1)), dim=1)
            student_prediction = student_topk.indices[:, 0]
            student_confidence = student_topk.values[:, 0]
            if student_topk.values.size(1) > 1:
                student_margin = student_topk.values[:, 0] - student_topk.values[:, 1]
            else:
                student_margin = torch.ones_like(student_confidence)

            student_correct = student_prediction.eq(targets)
            student_hard = student_correct.logical_not().to(dtype=teacher_confidence.dtype)

            teacher_conf_delta = max(
                float(getattr(self.config.federated, "expert_kd_teacher_confidence_delta", 0.0)),
                0.0,
            )
            teacher_margin_delta = max(
                float(getattr(self.config.federated, "expert_kd_teacher_margin_delta", 0.0)),
                0.0,
            )
            student_conf_ceiling = min(
                max(float(getattr(self.config.federated, "expert_kd_student_confidence_ceiling", 1.0)), 0.0),
                1.0,
            )
            student_margin_ceiling = min(
                max(float(getattr(self.config.federated, "expert_kd_student_margin_ceiling", 1.0)), 0.0),
                1.0,
            )

            conf_advantage = (
                (teacher_confidence - student_confidence - teacher_conf_delta)
                / max(1.0 - teacher_conf_delta, 1e-6)
            ).clamp(0.0, 1.0)
            margin_advantage = (
                (teacher_margin - student_margin - teacher_margin_delta)
                / max(1.0 - teacher_margin_delta, 1e-6)
            ).clamp(0.0, 1.0)
            teacher_advantage = torch.maximum(conf_advantage, margin_advantage)
            student_saturated = (
                student_correct
                & (student_confidence >= student_conf_ceiling)
                & (student_margin >= student_margin_ceiling)
            ).to(dtype=teacher_confidence.dtype)

        if adaptive_coverage_enabled and target_coverage > 0.0:
            gate = teacher_correct * ((0.7 * teacher_confidence) + (0.3 * teacher_margin))
            if student_logits is not None:
                gain_gate = torch.maximum(student_hard, teacher_advantage)
                gate = gate * (0.5 + (0.5 * gain_gate)) * (1.0 - (0.5 * student_saturated))
                if hard_boost > 0.0:
                    gate = gate * (1.0 + hard_boost * student_hard)
            if not math.isclose(gate_power, 1.0):
                gate = gate.pow(gate_power)
            return self._select_teacher_gate_by_coverage(gate, target_coverage)

        gate = teacher_correct * conf_gate * margin_gate
        if student_logits is not None:
            gain_gate = torch.maximum(student_hard, teacher_advantage)
            gate = gate * gain_gate * (1.0 - student_saturated)
            if hard_boost > 0.0:
                gate = gate * (1.0 + hard_boost * student_hard)
        if not math.isclose(gate_power, 1.0):
            gate = gate.pow(gate_power)

        return gate

    def _ensure_min_teacher_gate_coverage(
        self,
        gate: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        min_gate_ratio: float,
        gate_floor: float,
        student_logits: Optional[torch.Tensor] = None,
        gate_temperature: float = 1.0,
    ) -> torch.Tensor:
        min_gate_ratio = min(max(float(min_gate_ratio), 0.0), 1.0)
        if gate.numel() == 0 or min_gate_ratio <= 0.0:
            return gate

        positive_mask = gate > 0
        current_count = int(positive_mask.sum().item())
        desired_count = max(int(math.ceil(gate.numel() * min_gate_ratio)), 1)
        if current_count >= desired_count:
            return gate

        safe_gate_temperature = max(float(gate_temperature), 1e-4)
        teacher_probs = F.softmax(teacher_logits.detach() / safe_gate_temperature, dim=1)
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_prediction = teacher_topk.indices[:, 0]
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        # When the strict gate collapses, backfill with the most reliable
        # teacher-correct samples so the expert still receives some KD signal.
        candidate_scores = teacher_prediction.eq(targets).to(dtype=gate.dtype)
        candidate_scores = candidate_scores * ((0.7 * teacher_confidence) + (0.3 * teacher_margin))

        if student_logits is not None:
            student_probs = F.softmax(student_logits.detach() / safe_gate_temperature, dim=1)
            student_topk = torch.topk(student_probs, k=min(2, student_probs.size(1)), dim=1)
            student_prediction = student_topk.indices[:, 0]
            student_confidence = student_topk.values[:, 0]
            if student_topk.values.size(1) > 1:
                student_margin = student_topk.values[:, 0] - student_topk.values[:, 1]
            else:
                student_margin = torch.ones_like(student_confidence)

            student_hard = student_prediction.ne(targets).to(dtype=gate.dtype)
            teacher_advantage = torch.maximum(
                (teacher_confidence - student_confidence).clamp(0.0, 1.0),
                (teacher_margin - student_margin).clamp(0.0, 1.0),
            )
            candidate_scores = candidate_scores * (0.5 + 0.5 * torch.maximum(student_hard, teacher_advantage))

        candidate_scores = candidate_scores * positive_mask.logical_not().to(dtype=gate.dtype)
        available_count = int(candidate_scores.gt(0).sum().item())
        if available_count <= 0:
            return gate

        missing_count = min(max(desired_count - current_count, 0), available_count)
        if missing_count <= 0:
            return gate

        selected_indices = torch.topk(candidate_scores, k=missing_count).indices
        adjusted_gate = gate.clone()
        adjusted_gate[selected_indices] = torch.maximum(
            adjusted_gate[selected_indices],
            candidate_scores[selected_indices].clamp_min(max(float(gate_floor), 1e-4)),
        )
        return adjusted_gate

    def _refresh_expert_from_general(
        self,
        public_batches: Sequence,
        general_model: nn.Module,
        general_public_accuracy: float,
        general_temperature: float = 1.0,
    ) -> float:
        refresh_epochs = max(int(getattr(self.config.federated, "expert_refresh_epochs", 0)), 0)
        if refresh_epochs <= 0 or not public_batches:
            return 0.0

        refresh_lr_scale = max(float(getattr(self.config.federated, "expert_refresh_lr_scale", 0.5)), 0.0)
        if refresh_lr_scale <= 0.0:
            return 0.0

        min_teacher_accuracy = max(
            float(getattr(self.config.federated, "expert_refresh_min_teacher_accuracy", 0.0)),
            0.0,
        )
        if general_public_accuracy < min_teacher_accuracy:
            return 0.0

        refresh_logit_weight = max(
            float(getattr(self.config.federated, "expert_refresh_logit_weight", 0.0)),
            0.0,
        )
        refresh_feature_hint_weight = max(
            float(getattr(self.config.federated, "expert_refresh_feature_hint_weight", 0.0)),
            0.0,
        )
        if refresh_logit_weight <= 0.0 and refresh_feature_hint_weight <= 0.0:
            return 0.0

        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr * refresh_lr_scale,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        kd_temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))
        refresh_conf_threshold = float(
            getattr(
                self.config.federated,
                "expert_refresh_confidence_threshold",
                getattr(self.config.federated, "expert_kd_confidence_threshold", 0.60),
            )
        )
        refresh_margin_threshold = float(
            getattr(
                self.config.federated,
                "expert_refresh_margin_threshold",
                getattr(self.config.federated, "expert_kd_margin_threshold", 0.08),
            )
        )
        refresh_hard_boost = max(float(getattr(self.config.federated, "expert_refresh_hard_boost", 0.0)), 0.0)
        refresh_target_coverage = max(
            float(getattr(self.config.federated, "expert_refresh_target_coverage", 0.0)),
            0.0,
        )
        adaptive_refresh_gate = bool(
            getattr(self.config.federated, "expert_refresh_adaptive_coverage_enabled", False)
        )
        min_gate_ratio = max(float(getattr(self.config.federated, "expert_refresh_min_gate_ratio", 0.0)), 0.0)

        self.expert_model.train()
        general_model.eval()
        gate_mass = 0.0
        gate_steps = 0
        for _ in range(refresh_epochs):
            for batch in public_batches:
                images, targets, _ = _unpack_loader_batch(batch)
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    teacher_features = general_model.forward_features(images)
                    teacher_logits = general_model.classify_features(teacher_features)
                    teacher_route_probs = torch.softmax(
                        teacher_logits / max(float(general_temperature), 1e-4),
                        dim=1,
                    )
                    teacher_kd_probs = torch.softmax(teacher_logits / kd_temperature, dim=1)
                    refresh_weights = self._build_refresh_weights(teacher_route_probs)

                optimizer.zero_grad(set_to_none=True)
                student_features = self.expert_model.forward_features(images)
                student_logits = self.expert_model.classify_features(student_features)
                refresh_gate = self._teacher_gate_from_logits(
                    teacher_logits,
                    targets,
                    refresh_conf_threshold,
                    refresh_margin_threshold,
                    student_logits=student_logits,
                    hard_boost=refresh_hard_boost,
                    gate_temperature=general_temperature,
                    target_coverage=refresh_target_coverage,
                    adaptive_coverage_enabled=adaptive_refresh_gate,
                )
                refresh_gate = self._ensure_min_teacher_gate_coverage(
                    refresh_gate,
                    teacher_logits,
                    targets,
                    min_gate_ratio,
                    float(getattr(self.config.federated, "expert_refresh_gate_floor", 0.0)),
                    student_logits=student_logits,
                    gate_temperature=general_temperature,
                )
                refresh_weights = refresh_weights * refresh_gate
                gate_ratio = float(refresh_gate.gt(0).to(dtype=torch.float32).mean().item())
                if gate_ratio < min_gate_ratio or float(refresh_weights.sum().detach().cpu().item()) <= 0.0:
                    continue

                kd_loss = torch.zeros((), device=self.device)
                if refresh_logit_weight > 0.0:
                    kd_loss = self._weighted_kd_loss(
                        student_logits,
                        teacher_kd_probs,
                        refresh_weights,
                        kd_temperature,
                    )
                feature_hint_loss = torch.zeros((), device=self.device)
                if refresh_feature_hint_weight > 0.0:
                    feature_hint_loss = self._weighted_feature_hint_loss(
                        student_features,
                        teacher_features,
                        refresh_weights,
                    )
                loss = (refresh_logit_weight * kd_loss) + (refresh_feature_hint_weight * feature_hint_loss)
                loss.backward()
                optimizer.step()
                gate_mass += gate_ratio
                gate_steps += 1

        return gate_mass / max(gate_steps, 1)

    def _train_with_gated_kd(
        self,
        loader: DataLoader,
        general_model: nn.Module,
        teacher_temperature: float = 1.0,
    ) -> Tuple[float, float]:
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))
        kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        feature_hint_weight = max(float(getattr(self.config.federated, "client_feature_hint_weight", 0.0)), 0.0)
        conf_threshold = float(getattr(self.config.federated, "expert_kd_confidence_threshold", 0.60))
        margin_threshold = float(getattr(self.config.federated, "expert_kd_margin_threshold", 0.08))
        hard_boost = max(float(getattr(self.config.federated, "expert_kd_hard_boost", 0.0)), 0.0)
        target_coverage = max(float(getattr(self.config.federated, "expert_kd_target_coverage", 0.0)), 0.0)
        adaptive_gate = bool(getattr(self.config.federated, "expert_kd_adaptive_coverage_enabled", False))
        min_gate_ratio = max(float(getattr(self.config.federated, "expert_kd_min_gate_ratio", 0.0)), 0.0)
        gate_floor = float(getattr(self.config.federated, "expert_kd_gate_floor", 0.0))

        self.expert_model.train()
        general_model.eval()

        total_loss = 0.0
        total_batches = 0
        total_gate_ratio = 0.0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    teacher_features = general_model.forward_features(images)
                    teacher_logits = general_model.classify_features(teacher_features)
                    teacher_kd_probs = F.softmax(teacher_logits / temperature, dim=1)

                optimizer.zero_grad(set_to_none=True)
                student_features = self.expert_model.forward_features(images)
                student_logits = self.expert_model.classify_features(student_features)
                ce_loss = criterion(student_logits, targets)

                gate = self._teacher_gate_from_logits(
                    teacher_logits,
                    targets,
                    conf_threshold,
                    margin_threshold,
                    student_logits=student_logits,
                    hard_boost=hard_boost,
                    gate_temperature=teacher_temperature,
                    target_coverage=target_coverage,
                    adaptive_coverage_enabled=adaptive_gate,
                )
                gate = self._ensure_min_teacher_gate_coverage(
                    gate,
                    teacher_logits,
                    targets,
                    min_gate_ratio,
                    gate_floor,
                    student_logits=student_logits,
                    gate_temperature=teacher_temperature,
                )
                gate_ratio = float(gate.gt(0).to(dtype=torch.float32).mean().item())

                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                per_sample_kl = F.kl_div(
                    student_log_probs,
                    teacher_kd_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)

                if float(gate.sum().detach().cpu().item()) > 0.0:
                    kd_loss = (per_sample_kl * gate).sum() / gate.sum().clamp_min(1e-8)
                else:
                    kd_loss = torch.zeros((), device=self.device)

                feature_hint_loss = torch.zeros((), device=self.device)
                if feature_hint_weight > 0.0 and float(gate.sum().detach().cpu().item()) > 0.0:
                    feature_hint_loss = self._weighted_feature_hint_loss(
                        student_features,
                        teacher_features,
                        gate.clamp_min(1e-8),
                    )

                loss = ce_loss + (kd_weight * kd_loss) + (feature_hint_weight * feature_hint_loss)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
                total_gate_ratio += gate_ratio

        denominator = max(total_batches, 1)
        return total_loss / denominator, total_gate_ratio / denominator


class FedEGSSGServer(BaseFederatedServer):
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
            raise ValueError("fedegssg requires a non-empty public_dataset for expert->general distillation.")
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )

        holdout_ratio = max(float(getattr(config.inference, "routing_holdout_ratio", 0.0)), 0.0)
        holdout_min_samples = max(int(getattr(config.inference, "routing_holdout_min_samples", 0)), 0)
        holdout_max_samples = max(int(getattr(config.inference, "routing_holdout_max_samples", 0)), 0)
        holdout_seed_offset = int(getattr(config.inference, "routing_holdout_seed_offset", 17))
        self.client_routing_datasets: Dict[str, Dataset] = {}
        routed_client_datasets: Dict[str, Dataset] = {}
        for client_id, dataset in client_datasets.items():
            train_dataset, routing_dataset = _split_dataset_for_holdout(
                dataset,
                holdout_ratio=holdout_ratio,
                min_holdout_samples=holdout_min_samples,
                max_holdout_samples=holdout_max_samples,
                seed=_client_holdout_seed(config.federated.seed, client_id, holdout_seed_offset),
            )
            routed_client_datasets[client_id] = train_dataset
            self.client_routing_datasets[client_id] = routing_dataset

        self.client_datasets = routed_client_datasets

        self.general_model = _build_general_model_from_config(config).to(self.device)
        self.deploy_general_model = _build_general_model_from_config(config).to(self.device)
        self.deploy_general_model.load_state_dict(_clone_state_dict(self.general_model.state_dict()))
        self.reference_expert = _build_expert_model_from_config(config).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.deploy_general_model)

        self.clients: Dict[str, FedEGSSGClient] = {
            client_id: FedEGSSGClient(
                client_id,
                dataset,
                config.model.num_classes,
                config.federated.device,
                config,
                data_module,
            )
            for client_id, dataset in self.client_datasets.items()
        }

        self.public_loader = self.data_module.make_loader(public_dataset, shuffle=False)
        self.public_batches = list(self.public_loader)
        self.public_images_cpu, self.public_targets_cpu = self._cache_public_tensors(self.public_batches)
        self.public_size = int(self.public_targets_cpu.size(0))
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
        self.distill_batches = list(self.distill_loader)
        self.distill_images_cpu = self._cache_loader_images(self.distill_batches)
        self.distill_size = int(self.distill_images_cpu.size(0))

        initial_reliability = max(float(getattr(config.federated, "min_client_reliability", 0.05)), 0.25)
        self.client_reliability_scores = {client_id: initial_reliability for client_id in self.client_datasets}
        self.public_teacher_logits_buffer: List[torch.Tensor] = []
        self.external_teacher_logits_buffer: List[torch.Tensor] = []
        self.deploy_general_temperature = 1.0
        self.client_expert_temperatures = {client_id: 1.0 for client_id in self.client_datasets}

        base_confidence = float(config.inference.confidence_threshold)
        base_margin = float(config.inference.route_distance_threshold)
        base_energy = float(getattr(config.inference, "route_energy_threshold", -3.0))
        self.client_confidence_thresholds = {client_id: base_confidence for client_id in self.client_datasets}
        self.client_margin_thresholds = {client_id: base_margin for client_id in self.client_datasets}
        self.client_energy_thresholds = {client_id: base_energy for client_id in self.client_datasets}
        self.client_gain_thresholds = {
            client_id: float(getattr(config.inference, "route_gain_threshold", 0.0))
            for client_id in self.client_datasets
        }

        self.communication_quantization_enabled = bool(
            getattr(config.federated, "communication_quantization_enabled", False)
        )
        self.communication_quantization_bits = int(
            getattr(config.federated, "communication_quantization_bits", 8)
        )
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.current_round = 0

        LOGGER.info(
            "fedegssg routing holdout | ratio=%.3f min=%d max=%d avg_holdout=%.1f avg_train=%.1f",
            holdout_ratio,
            holdout_min_samples,
            holdout_max_samples,
            (
                sum(len(dataset) for dataset in self.client_routing_datasets.values())
                / max(len(self.client_routing_datasets), 1)
            ),
            (
                sum(len(dataset) for dataset in self.client_datasets.values())
                / max(len(self.client_datasets), 1)
            ),
        )

        if bool(getattr(self.config.federated, "general_pretrain_on_public", False)):
            self._pretrain_general_on_public()
            self._sync_deploy_general(hard=True)
        if bool(getattr(self.config.federated, "temperature_calibration_enabled", False)):
            self._update_general_temperature()

    def _cache_public_tensors(self, public_batches: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        images: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for batch in public_batches:
            batch_images, batch_targets, _ = _unpack_loader_batch(batch)
            images.append(batch_images.detach().cpu())
            targets.append(batch_targets.detach().cpu())
        return torch.cat(images, dim=0), torch.cat(targets, dim=0)

    def _cache_loader_images(self, batches: Sequence) -> torch.Tensor:
        images: List[torch.Tensor] = []
        for batch in batches:
            batch_images, _, _ = _unpack_loader_batch(batch)
            images.append(batch_images.detach().cpu())
        if not images:
            return torch.empty((0, 3, 32, 32), dtype=torch.float32)
        return torch.cat(images, dim=0)

    def _evaluate_predictor_on_client_holdouts(
        self,
        predictor,
    ) -> Dict[str, object]:
        client_results: Dict[str, Dict[str, float]] = {}
        for client_id, dataset in self.client_routing_datasets.items():
            if dataset is None or len(dataset) == 0:
                continue
            loader = self.data_module.make_loader(dataset, shuffle=False)
            client_results[client_id] = self._evaluate_predictor_on_loader(
                client_id,
                predictor,
                loader,
                collect_route_records=False,
            )

        weighted = self._aggregate_metrics(client_results, weighted=True)
        macro = self._aggregate_metrics(client_results, weighted=False)
        return {
            "aggregate": weighted,
            "macro": macro,
            "clients": client_results,
        }

    def _teacher_aggregation_mode(self, external: bool = False) -> str:
        attribute_name = "external_teacher_aggregation" if external else "public_teacher_aggregation"
        fallback = getattr(self.config.federated, attribute_name, "avg_logits")
        return _normalized_teacher_aggregation_name(fallback)

    def _aggregate_teacher_logits(
        self,
        teacher_logits: torch.Tensor,
        updates: List[FedEGSSGClientUpdate],
        aggregation_mode: str,
        history_buffer: List[torch.Tensor],
        use_temporal: bool,
        extra_hard_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)
        min_rel = max(float(getattr(self.config.federated, "min_client_reliability", 0.05)), 0.0)
        reliability_alpha = max(float(getattr(self.config.federated, "reliability_alpha", 1.0)), 0.0)
        consistency_weight = max(float(getattr(self.config.federated, "teacher_consistency_weight", 0.0)), 0.0)
        temporal_blend = min(max(float(getattr(self.config.federated, "teacher_temporal_momentum", 0.0)), 0.0), 0.95)
        temporal_buffer_size = max(int(getattr(self.config.federated, "teacher_temporal_buffer_size", 0)), 0)
        hard_subset_ratio = min(max(float(getattr(self.config.federated, "hard_subset_ratio", 0.0)), 0.0), 1.0)
        hard_topk = max(int(getattr(self.config.federated, "hard_sample_topk", 0)), 0)
        hard_weight_power = max(float(getattr(self.config.federated, "hard_sample_weight_power", 1.0)), 1.0)
        hard_distill_boost = max(float(getattr(self.config.federated, "hard_sample_distill_boost", 1.0)), 1.0)

        teacher_probs = F.softmax(teacher_logits / temperature, dim=2)
        log_num_classes = math.log(float(num_classes))
        entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=2) / max(log_num_classes, 1e-8)

        history_logits = None
        history_probs = None
        if use_temporal and history_buffer:
            candidate_history = torch.stack(history_buffer, dim=0).mean(dim=0)
            if candidate_history.shape == teacher_logits.shape[1:]:
                history_logits = candidate_history
                history_probs = F.softmax(history_logits / temperature, dim=1)

        weights: List[torch.Tensor] = []
        for update_idx, update in enumerate(updates):
            reliability = max(float(self.client_reliability_scores.get(update.client_id, min_rel)), min_rel)
            if aggregation_mode in {"avg", "avg_logits", "feddf"}:
                sample_weight = torch.ones_like(entropy[update_idx])
            elif aggregation_mode in {"entropy", "em_entropy_soft", "fedgo_entropy"}:
                sample_weight = torch.exp(-entropy[update_idx])
            elif aggregation_mode in {"variance", "logit_var", "fedgo_variance"}:
                sample_weight = teacher_logits[update_idx].var(dim=1, unbiased=False)
            elif aggregation_mode in {"stabilized", "reliability_entropy", "heuristic"}:
                sample_weight = torch.exp(-entropy[update_idx]) * (reliability ** reliability_alpha)
                if history_probs is not None and consistency_weight > 0.0:
                    divergence = (teacher_probs[update_idx] - history_probs).pow(2).mean(dim=1)
                    sample_weight = sample_weight * torch.exp(-consistency_weight * divergence)
            else:
                raise ValueError(f"Unsupported teacher aggregation mode: {aggregation_mode}")
            weights.append(sample_weight.clamp_min(1e-8))

        stacked_weights = torch.stack(weights, dim=0)
        normalized_weights = stacked_weights / stacked_weights.sum(dim=0, keepdim=True).clamp_min(1e-8)

        predicted_classes = teacher_logits.argmax(dim=2)
        agreement_counts = F.one_hot(predicted_classes, num_classes=num_classes).to(dtype=torch.float32).sum(dim=0)
        disagreement = 1.0 - (agreement_counts.max(dim=1).values / max(float(len(updates)), 1.0))
        mean_entropy = entropy.mean(dim=0)

        hard_mask = torch.zeros_like(mean_entropy, dtype=torch.bool)
        if hard_subset_ratio > 0.0 and hard_mask.numel() > 0:
            hard_scores = disagreement + mean_entropy
            if extra_hard_scores is not None and extra_hard_scores.numel() == hard_scores.numel():
                hard_scores = hard_scores + extra_hard_scores.to(dtype=hard_scores.dtype)
            keep_count = max(int(math.ceil(hard_scores.numel() * hard_subset_ratio)), 1)
            top_indices = torch.topk(hard_scores, k=min(keep_count, hard_scores.numel())).indices
            hard_mask[top_indices] = True

        if bool(hard_mask.any().item()):
            hard_weights = stacked_weights[:, hard_mask].clone()
            if hard_topk > 0 and hard_topk < hard_weights.size(0):
                hard_weights_t = hard_weights.transpose(0, 1)
                topk_indices = torch.topk(hard_weights_t, k=hard_topk, dim=1).indices
                topk_mask = torch.zeros_like(hard_weights_t, dtype=torch.bool)
                topk_mask.scatter_(1, topk_indices, True)
                hard_weights = (hard_weights_t * topk_mask.to(dtype=hard_weights_t.dtype)).transpose(0, 1)
            if hard_weight_power > 1.0:
                hard_weights = hard_weights.pow(hard_weight_power)
            normalized_weights[:, hard_mask] = hard_weights / hard_weights.sum(dim=0, keepdim=True).clamp_min(1e-8)

        avg_logits = (teacher_logits * normalized_weights.unsqueeze(-1)).sum(dim=0)
        if history_logits is not None and temporal_blend > 0.0:
            avg_logits = ((1.0 - temporal_blend) * avg_logits) + (temporal_blend * history_logits)
        if use_temporal and temporal_buffer_size > 0:
            history_buffer.append(avg_logits.detach().cpu())
            if len(history_buffer) > temporal_buffer_size:
                del history_buffer[0]

        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        sample_weights = stacked_weights.mean(dim=0)
        if bool(hard_mask.any().item()) and hard_distill_boost > 1.0:
            sample_weights = sample_weights.clone()
            sample_weights[hard_mask] = sample_weights[hard_mask] * hard_distill_boost

        return {
            "soft_labels": soft_labels,
            "sample_weights": sample_weights.clamp_min(1e-8),
            "hard_mask": hard_mask,
            "hard_ratio": hard_mask.to(dtype=torch.float32).mean(),
            "mean_reliability": torch.tensor(
                sum(float(self.client_reliability_scores.get(update.client_id, min_rel)) for update in updates)
                / max(len(updates), 1),
                dtype=torch.float32,
            ),
        }

    def _batched_model_logits(self, model: nn.Module, images_cpu: torch.Tensor) -> torch.Tensor:
        logits: List[torch.Tensor] = []
        batch_size = self.config.dataset.batch_size
        model.eval()
        with torch.no_grad():
            for start in range(0, images_cpu.size(0), batch_size):
                batch = images_cpu[start:start + batch_size].to(self.device)
                logits.append(model(batch).detach().cpu())
        return torch.cat(logits, dim=0)

    def _apply_logit_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        safe_temperature = max(float(temperature), 1e-4)
        if abs(safe_temperature - 1.0) <= 1e-8:
            return logits
        return logits / safe_temperature

    def _routing_signals_from_logits(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
        predictions = topk.indices[:, 0]
        confidences = topk.values[:, 0]
        if topk.values.size(1) > 1:
            margins = topk.values[:, 0] - topk.values[:, 1]
        else:
            margins = torch.ones_like(confidences)
        energies = self._energy_score_from_logits(logits)
        return predictions, confidences, margins, energies

    def _build_routing_fallback_mask(
        self,
        confidences: torch.Tensor,
        margins: torch.Tensor,
        energies: torch.Tensor,
        confidence_threshold: float,
        margin_threshold: float,
        energy_threshold: float,
    ) -> torch.Tensor:
        return (
            (confidences < confidence_threshold)
            | (margins < margin_threshold)
            | (energies > energy_threshold)
        )

    def _compute_routing_gain_scores(
        self,
        client_id: str,
        confidences: torch.Tensor,
        margins: torch.Tensor,
        energies: torch.Tensor,
        confidence_threshold: float,
        margin_threshold: float,
        energy_threshold: float,
    ) -> torch.Tensor:
        conf_span = max(
            float(self.config.inference.max_confidence_threshold)
            - float(self.config.inference.min_confidence_threshold),
            1e-6,
        )
        margin_span = max(
            float(self.config.inference.max_margin_threshold)
            - float(self.config.inference.min_margin_threshold),
            1e-6,
        )
        energy_span = max(
            float(getattr(self.config.inference, "max_energy_threshold", -1.0))
            - float(getattr(self.config.inference, "min_energy_threshold", -6.0)),
            1e-6,
        )

        conf_weight = max(float(getattr(self.config.inference, "route_gain_confidence_weight", 0.50)), 0.0)
        margin_weight = max(float(getattr(self.config.inference, "route_gain_margin_weight", 0.35)), 0.0)
        energy_weight = max(float(getattr(self.config.inference, "route_gain_energy_weight", 0.25)), 0.0)
        reliability_weight = max(float(getattr(self.config.inference, "route_gain_reliability_weight", 0.15)), 0.0)
        positive_margin = max(float(getattr(self.config.inference, "route_gain_positive_margin", 0.0)), 0.0)

        reliability = float(self.client_reliability_scores.get(client_id, 1.0))
        reliability_center = float(getattr(self.config.inference, "route_reliability_center", 0.60))
        reliability_risk = max(reliability_center - reliability, 0.0)

        confidence_risk = (confidence_threshold - confidences) / conf_span
        margin_risk = (margin_threshold - margins) / margin_span
        energy_risk = (energies - energy_threshold) / energy_span

        scores = (
            (conf_weight * confidence_risk)
            + (margin_weight * margin_risk)
            + (energy_weight * energy_risk)
            + (reliability_weight * reliability_risk)
            - positive_margin
        )
        return scores

    def _gain_threshold_candidates(
        self,
        current_threshold: float,
        gain_scores: torch.Tensor,
    ) -> List[float]:
        min_threshold = float(getattr(self.config.inference, "min_route_gain_threshold", -1.0))
        max_threshold = float(getattr(self.config.inference, "max_route_gain_threshold", 1.0))
        step = max(float(getattr(self.config.inference, "route_gain_threshold_step", 0.05)), 0.0)
        candidates = {min(max(current_threshold, min_threshold), max_threshold)}
        if step > 0.0:
            search_radius = max(int(getattr(self.config.inference, "routing_search_radius", 2)), 0)
            for offset in range(-search_radius, search_radius + 1):
                candidates.add(min(max(current_threshold + (offset * step), min_threshold), max_threshold))
        if gain_scores.numel() > 0:
            quantiles = [0.20, 0.35, 0.50, 0.65, 0.80]
            for quantile in quantiles:
                value = float(torch.quantile(gain_scores, quantile).item())
                candidates.add(min(max(value, min_threshold), max_threshold))
        return sorted(candidates)

    def _should_run_temperature_calibration(self, round_idx: Optional[int] = None) -> bool:
        if not bool(getattr(self.config.federated, "temperature_calibration_enabled", False)):
            return False
        frequency = max(int(getattr(self.config.federated, "temperature_calibration_frequency", 1)), 1)
        if round_idx is None or round_idx <= 0:
            return True
        return (round_idx % frequency) == 0

    def _temperature_candidates(self, previous_temperature: float) -> List[float]:
        min_temperature = max(float(getattr(self.config.federated, "temperature_calibration_min", 0.5)), 1e-3)
        max_temperature = max(
            float(getattr(self.config.federated, "temperature_calibration_max", 5.0)),
            min_temperature,
        )
        num_candidates = max(int(getattr(self.config.federated, "temperature_calibration_candidates", 25)), 2)
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

    def _update_expert_temperatures(self, updates: List[FedEGSSGClientUpdate]) -> None:
        for update in updates:
            previous_temperature = float(self.client_expert_temperatures.get(update.client_id, 1.0))
            fitted_temperature = self._fit_temperature_from_logits(
                update.public_logits.float(),
                self.public_targets_cpu,
                previous_temperature,
            )
            self.client_expert_temperatures[update.client_id] = fitted_temperature

    def _update_general_temperature(self) -> None:
        previous_temperature = float(self.deploy_general_temperature)
        general_logits = self._batched_model_logits(self.deploy_general_model, self.public_images_cpu)
        self.deploy_general_temperature = self._fit_temperature_from_logits(
            general_logits.float(),
            self.public_targets_cpu,
            previous_temperature,
        )

    def _public_accuracy_for_model(self, model: nn.Module) -> float:
        logits = self._batched_model_logits(model, self.public_images_cpu)
        predictions = logits.argmax(dim=1)
        return float(predictions.eq(self.public_targets_cpu).to(dtype=torch.float32).mean().item())

    def _energy_score_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(logits, dim=1)

    def _pretrain_general_on_public(self) -> None:
        epochs = max(int(getattr(self.config.federated, "general_pretrain_epochs", 0)), 0)
        if epochs <= 0:
            return

        optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=float(getattr(self.config.federated, "general_pretrain_lr", 0.01)),
            momentum=0.9,
            weight_decay=float(getattr(self.config.federated, "local_weight_decay", 5e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        criterion = nn.CrossEntropyLoss()

        best_state = _clone_state_dict(self.general_model.state_dict())
        best_accuracy = -1.0
        for epoch_idx in range(epochs):
            self.general_model.train()
            for batch in self.public_batches:
                images, targets, _ = _unpack_loader_batch(batch)
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.general_model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

            predictions = self._batched_model_logits(self.general_model, self.public_images_cpu).argmax(dim=1)
            accuracy = float(predictions.eq(self.public_targets_cpu).to(dtype=torch.float32).mean().item())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = _clone_state_dict(self.general_model.state_dict())
            LOGGER.info(
                "fedegssg pretrain | epoch=%d/%d | public_acc=%.4f",
                epoch_idx + 1,
                epochs,
                accuracy,
            )

        self.general_model.load_state_dict(best_state)

    def _build_quantized_deploy_payload(self) -> CompressedStateDict:
        state = _clone_state_dict(self.deploy_general_model.state_dict())
        return compress_state_dict(state, bits=self.communication_quantization_bits)

    def _sync_deploy_general(self, hard: bool = False) -> None:
        momentum = min(max(float(getattr(self.config.federated, "general_deploy_ema_momentum", 0.90)), 0.0), 0.999)
        warmup_rounds = max(int(getattr(self.config.federated, "general_deploy_warmup_rounds", 0)), 0)
        warmup_momentum = min(
            max(float(getattr(self.config.federated, "general_deploy_warmup_momentum", momentum)), 0.0),
            0.999,
        )
        if not hard and self.current_round > 0 and self.current_round <= warmup_rounds:
            momentum = min(momentum, warmup_momentum)
        general_state = self.general_model.state_dict()
        deploy_state = self.deploy_general_model.state_dict()
        for key, general_tensor in general_state.items():
            deploy_tensor = deploy_state[key]
            source = general_tensor.detach()
            if hard or not deploy_tensor.is_floating_point():
                deploy_tensor.copy_(source)
            else:
                deploy_tensor.mul_(momentum).add_(source, alpha=1.0 - momentum)

    def _update_client_reliability_scores(self, updates: List[FedEGSSGClientUpdate]) -> None:
        ema = min(max(float(getattr(self.config.federated, "reliability_ema_momentum", 0.8)), 0.0), 0.999)
        acc_weight = min(max(float(getattr(self.config.federated, "reliability_accuracy_weight", 0.7)), 0.0), 1.0)
        min_rel = max(float(getattr(self.config.federated, "min_client_reliability", 0.05)), 0.0)
        targets = self.public_targets_cpu

        for update in updates:
            logits = self._apply_logit_temperature(
                update.public_logits.float(),
                self.client_expert_temperatures.get(update.client_id, 1.0),
            )
            probs = F.softmax(logits, dim=1)
            topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
            predictions = topk.indices[:, 0]
            confidences = topk.values[:, 0]
            if topk.values.size(1) > 1:
                margins = topk.values[:, 0] - topk.values[:, 1]
            else:
                margins = torch.ones_like(confidences)
            energies = self._energy_score_from_logits(logits)

            confidence_threshold = self.client_confidence_thresholds.get(
                update.client_id,
                float(self.config.inference.confidence_threshold),
            )
            margin_threshold = self.client_margin_thresholds.get(
                update.client_id,
                float(self.config.inference.route_distance_threshold),
            )
            energy_threshold = self.client_energy_thresholds.get(
                update.client_id,
                float(getattr(self.config.inference, "route_energy_threshold", -3.0)),
            )

            correct = predictions.eq(targets)
            confident_mask = (
                (confidences >= confidence_threshold)
                & (margins >= margin_threshold)
                & (energies <= energy_threshold)
            )
            round_accuracy = float(correct.to(dtype=torch.float32).mean().item())
            if bool(confident_mask.any().item()):
                confident_accuracy = float(correct[confident_mask].to(dtype=torch.float32).mean().item())
            else:
                confident_accuracy = round_accuracy

            round_reliability = (acc_weight * round_accuracy) + ((1.0 - acc_weight) * confident_accuracy)
            previous = float(self.client_reliability_scores.get(update.client_id, round_reliability))
            smoothed = (ema * previous) + ((1.0 - ema) * round_reliability)
            self.client_reliability_scores[update.client_id] = max(min_rel, min(1.0, smoothed))

    def _aggregate_public_teachers(self, updates: List[FedEGSSGClientUpdate]) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("Cannot aggregate public teachers without client updates.")

        teacher_logits = torch.stack(
            [
                self._apply_logit_temperature(
                    update.public_logits.float(),
                    self.client_expert_temperatures.get(update.client_id, 1.0),
                )
                for update in updates
            ],
            dim=0,
        )
        general_predictions = self._batched_model_logits(self.deploy_general_model, self.public_images_cpu).argmax(dim=1)
        general_wrong = general_predictions.ne(self.public_targets_cpu).to(dtype=torch.float32)
        return self._aggregate_teacher_logits(
            teacher_logits=teacher_logits,
            updates=updates,
            aggregation_mode=self._teacher_aggregation_mode(external=False),
            history_buffer=self.public_teacher_logits_buffer,
            use_temporal=bool(getattr(self.config.federated, "public_teacher_use_temporal", False)),
            extra_hard_scores=0.5 * general_wrong,
        )

    def _aggregate_external_teachers(self, updates: List[FedEGSSGClientUpdate]) -> Optional[Dict[str, torch.Tensor]]:
        if self.distill_size <= 0:
            return None
        if not updates:
            raise ValueError("Cannot aggregate external teachers without client updates.")

        teacher_logits = torch.stack(
            [
                self._apply_logit_temperature(
                    update.external_logits.float(),
                    self.client_expert_temperatures.get(update.client_id, 1.0),
                )
                for update in updates
            ],
            dim=0,
        )
        return self._aggregate_teacher_logits(
            teacher_logits=teacher_logits,
            updates=updates,
            aggregation_mode=self._teacher_aggregation_mode(external=True),
            history_buffer=self.external_teacher_logits_buffer,
            use_temporal=bool(getattr(self.config.federated, "external_teacher_use_temporal", False)),
            extra_hard_scores=None,
        )

    def _distill_general_model(
        self,
        public_ensemble: Dict[str, torch.Tensor],
        external_ensemble: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = self.config.dataset.batch_size
        public_ce_weight = max(float(getattr(self.config.federated, "public_ce_weight", 1.0)), 0.0)
        public_kl_weight = max(float(getattr(self.config.federated, "public_logit_align_weight", 1.0)), 0.0)
        external_kl_weight = max(float(getattr(self.config.federated, "external_logit_align_weight", 0.0)), 0.0)
        hard_kl_weight = max(float(getattr(self.config.federated, "hard_subset_kd_boost", 0.0)), 0.0)
        external_kl_warmup_rounds = max(
            int(getattr(self.config.federated, "external_logit_align_warmup_rounds", 0)),
            0,
        )
        external_kl_ramp_rounds = max(
            int(getattr(self.config.federated, "external_logit_align_ramp_rounds", 1)),
            1,
        )
        if self.current_round <= external_kl_warmup_rounds:
            external_kl_weight = 0.0
        elif external_kl_weight > 0.0 and external_kl_ramp_rounds > 0:
            ramp_progress = min(
                max(
                    float(self.current_round - external_kl_warmup_rounds)
                    / float(external_kl_ramp_rounds),
                    0.0,
                ),
                1.0,
            )
            external_kl_weight = external_kl_weight * ramp_progress
        deploy_consistency_weight = max(
            float(
                getattr(
                    self.config.federated,
                    "deploy_consistency_weight",
                    getattr(self.config.federated, "general_anchor_weight", 0.0),
                )
            ),
            0.0,
        )
        deploy_consistency_warmup_rounds = max(
            int(getattr(self.config.federated, "deploy_consistency_warmup_rounds", 0)),
            0,
        )
        deploy_consistency_ramp_rounds = max(
            int(getattr(self.config.federated, "deploy_consistency_ramp_rounds", 1)),
            1,
        )
        if self.current_round <= deploy_consistency_warmup_rounds:
            deploy_consistency_weight = 0.0
        elif deploy_consistency_weight > 0.0 and deploy_consistency_ramp_rounds > 0:
            ramp_progress = min(
                max(
                    float(self.current_round - deploy_consistency_warmup_rounds)
                    / float(deploy_consistency_ramp_rounds),
                    0.0,
                ),
                1.0,
            )
            deploy_consistency_weight = deploy_consistency_weight * ramp_progress

        optimizer = torch.optim.Adam(self.general_model.parameters(), lr=float(self.config.federated.distill_lr))
        public_count = int(self.public_size)
        external_count = int(self.distill_size if external_ensemble is not None else 0)
        public_steps = int(math.ceil(public_count / max(batch_size, 1))) if public_count > 0 else 0
        external_steps = int(math.ceil(external_count / max(batch_size, 1))) if external_count > 0 else 0
        steps_per_epoch = max(public_steps, external_steps, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(distill_epochs * steps_per_epoch, 1),
        )

        public_soft_all = public_ensemble["soft_labels"]
        public_sample_weights_all = public_ensemble["sample_weights"]
        public_hard_mask_all = public_ensemble["hard_mask"]
        external_soft_all = external_ensemble["soft_labels"] if external_ensemble is not None else None
        external_sample_weights_all = external_ensemble["sample_weights"] if external_ensemble is not None else None

        self.general_model.train()
        self.deploy_general_model.eval()

        total_loss = 0.0
        total_public_ce = 0.0
        total_public_kl = 0.0
        total_external_kl = 0.0
        total_hard_kl = 0.0
        total_deploy_consistency = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            public_permutation = torch.randperm(public_count) if public_count > 0 else None
            external_permutation = torch.randperm(external_count) if external_count > 0 else None
            external_ptr = 0
            for step_idx in range(steps_per_epoch):
                public_ce_loss = torch.zeros((), device=self.device)
                public_kl_loss = torch.zeros((), device=self.device)
                external_kl_loss = torch.zeros((), device=self.device)
                hard_kl_loss = torch.zeros((), device=self.device)
                deploy_consistency_loss = torch.zeros((), device=self.device)
                public_anchor_images = None
                public_anchor_log_probs = None
                public_anchor_weights = None

                optimizer.zero_grad(set_to_none=True)

                public_start = step_idx * batch_size
                if public_permutation is not None and public_start < public_count:
                    public_indices = public_permutation[public_start:public_start + batch_size]
                    public_images = self.public_images_cpu[public_indices].to(self.device)
                    public_targets = self.public_targets_cpu[public_indices].to(self.device)
                    public_teacher_probs = public_soft_all[public_indices].to(self.device)
                    public_weights = public_sample_weights_all[public_indices].to(self.device)
                    public_hard_mask = public_hard_mask_all[public_indices].to(self.device)

                    public_logits = self.general_model(public_images)
                    public_log_probs = F.log_softmax(public_logits / temperature, dim=1)
                    public_ce_loss = F.cross_entropy(public_logits, public_targets)
                    per_public_kl = F.kl_div(
                        public_log_probs,
                        public_teacher_probs,
                        reduction="none",
                    ).sum(dim=1) * (temperature ** 2)
                    public_kl_loss = (
                        (per_public_kl * public_weights).sum()
                        / public_weights.sum().clamp_min(1e-8)
                    )
                    if hard_kl_weight > 0.0 and bool(public_hard_mask.any().item()):
                        hard_kl_loss = per_public_kl[public_hard_mask].mean()
                    public_anchor_images = public_images
                    public_anchor_log_probs = public_log_probs
                    public_anchor_weights = public_weights

                if external_count > 0 and external_soft_all is not None and external_sample_weights_all is not None:
                    if external_permutation is None or external_ptr >= external_count:
                        external_permutation = torch.randperm(external_count)
                        external_ptr = 0
                    external_indices = external_permutation[external_ptr:external_ptr + batch_size]
                    external_ptr += int(external_indices.numel())
                    if external_indices.numel() > 0:
                        external_images = self.distill_images_cpu[external_indices].to(self.device)
                        external_teacher_probs = external_soft_all[external_indices].to(self.device)
                        external_weights = external_sample_weights_all[external_indices].to(self.device)
                        external_logits = self.general_model(external_images)
                        external_log_probs = F.log_softmax(external_logits / temperature, dim=1)
                        per_external_kl = F.kl_div(
                            external_log_probs,
                            external_teacher_probs,
                            reduction="none",
                        ).sum(dim=1) * (temperature ** 2)
                        external_kl_loss = (
                            (per_external_kl * external_weights).sum()
                            / external_weights.sum().clamp_min(1e-8)
                        )

                if (
                    deploy_consistency_weight > 0.0
                    and public_anchor_images is not None
                    and public_anchor_log_probs is not None
                    and public_anchor_weights is not None
                ):
                    with torch.no_grad():
                        deploy_logits = self.deploy_general_model(public_anchor_images)
                        deploy_logits = self._apply_logit_temperature(deploy_logits, self.deploy_general_temperature)
                        deploy_probs = F.softmax(deploy_logits / temperature, dim=1)
                    per_deploy_kl = F.kl_div(
                        public_anchor_log_probs,
                        deploy_probs,
                        reduction="none",
                    ).sum(dim=1) * (temperature ** 2)
                    deploy_consistency_loss = (
                        (per_deploy_kl * public_anchor_weights).sum()
                        / public_anchor_weights.sum().clamp_min(1e-8)
                    )

                loss = (
                    (public_ce_weight * public_ce_loss)
                    + (public_kl_weight * public_kl_loss)
                    + (external_kl_weight * external_kl_loss)
                    + (hard_kl_weight * hard_kl_loss)
                    + (deploy_consistency_weight * deploy_consistency_loss)
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.detach().cpu().item())
                total_public_ce += float(public_ce_loss.detach().cpu().item())
                total_public_kl += float(public_kl_loss.detach().cpu().item())
                total_external_kl += float(external_kl_loss.detach().cpu().item())
                total_hard_kl += float(hard_kl_loss.detach().cpu().item())
                total_deploy_consistency += float(deploy_consistency_loss.detach().cpu().item())
                total_batches += 1

        denominator = max(total_batches, 1)
        return {
            "total_loss": total_loss / denominator,
            "public_ce_loss": total_public_ce / denominator,
            "public_kl_loss": total_public_kl / denominator,
            "external_kl_loss": total_external_kl / denominator,
            "hard_kl_loss": total_hard_kl / denominator,
            "deploy_consistency_loss": total_deploy_consistency / denominator,
        }

    def _apply_reliability_to_thresholds(
        self,
        client_id: str,
        confidence_threshold: float,
        margin_threshold: float,
        energy_threshold: float,
    ) -> Tuple[float, float, float]:
        reliability = float(self.client_reliability_scores.get(client_id, 1.0))
        center = float(getattr(self.config.inference, "route_reliability_center", 0.60))
        confidence_scale = max(float(getattr(self.config.inference, "route_reliability_confidence_scale", 0.0)), 0.0)
        margin_scale = max(float(getattr(self.config.inference, "route_reliability_margin_scale", 0.0)), 0.0)
        energy_scale = max(float(getattr(self.config.inference, "route_reliability_energy_scale", 0.0)), 0.0)
        reliability_gap = center - reliability

        adjusted_confidence = min(
            max(
                confidence_threshold + (confidence_scale * reliability_gap),
                float(self.config.inference.min_confidence_threshold),
            ),
            float(self.config.inference.max_confidence_threshold),
        )
        adjusted_margin = min(
            max(
                margin_threshold + (margin_scale * reliability_gap),
                float(self.config.inference.min_margin_threshold),
            ),
            float(self.config.inference.max_margin_threshold),
        )
        adjusted_energy = min(
            max(
                energy_threshold - (energy_scale * reliability_gap),
                float(getattr(self.config.inference, "min_energy_threshold", -6.0)),
            ),
            float(getattr(self.config.inference, "max_energy_threshold", -1.0)),
        )
        return adjusted_confidence, adjusted_margin, adjusted_energy

    def _collect_client_routing_statistics(self, client_id: str) -> Optional[Dict[str, torch.Tensor]]:
        dataset = self.client_routing_datasets.get(client_id)
        if dataset is None or len(dataset) == 0:
            return None

        loader = self.data_module.make_loader(dataset, shuffle=False)
        expert_predictions: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        margins: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        general_predictions: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []
        hard_masks: List[torch.Tensor] = []

        self.clients[client_id].expert_model.eval()
        self.deploy_general_model.eval()
        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                expert_logits = self.clients[client_id].expert_model(images)
                expert_logits = self._apply_logit_temperature(
                    expert_logits,
                    self.client_expert_temperatures.get(client_id, 1.0),
                )
                expert_prediction, confidence, margin, energy = self._routing_signals_from_logits(expert_logits)
                general_logits = self.deploy_general_model(images)
                general_prediction = general_logits.argmax(dim=1)

                expert_predictions.append(expert_prediction.detach().cpu())
                confidences.append(confidence.detach().cpu())
                margins.append(margin.detach().cpu())
                energies.append(energy.detach().cpu())
                general_predictions.append(general_prediction.detach().cpu())
                targets_all.append(targets.detach().cpu())
                hard_masks.append(expert_prediction.ne(targets).detach().cpu())

        if not targets_all:
            return None

        return {
            "expert_predictions": torch.cat(expert_predictions, dim=0),
            "confidences": torch.cat(confidences, dim=0),
            "margins": torch.cat(margins, dim=0),
            "energies": torch.cat(energies, dim=0),
            "general_predictions": torch.cat(general_predictions, dim=0),
            "targets": torch.cat(targets_all, dim=0),
            "hard_mask": torch.cat(hard_masks, dim=0),
        }

    def _predict_general_only(self, client_id, images, indices):
        self.deploy_general_model.eval()
        return self.deploy_general_model(images).argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        return self.clients[client_id].expert_model(images).argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        self.deploy_general_model.eval()

        confidence_threshold = self.client_confidence_thresholds[client_id]
        margin_threshold = self.client_margin_thresholds[client_id]
        energy_threshold = self.client_energy_thresholds[client_id]
        gain_threshold = self.client_gain_thresholds.get(
            client_id,
            float(getattr(self.config.inference, "route_gain_threshold", 0.0)),
        )
        confidence_threshold, margin_threshold, energy_threshold = self._apply_reliability_to_thresholds(
            client_id,
            confidence_threshold,
            margin_threshold,
            energy_threshold,
        )

        if self.current_round <= int(getattr(self.config.inference, "route_warmup_rounds", 0)):
            confidence_threshold = max(
                confidence_threshold,
                float(getattr(self.config.inference, "route_warmup_confidence_threshold", confidence_threshold)),
            )

        gain_search_warmup_rounds = max(
            int(getattr(self.config.inference, "route_gain_search_warmup_rounds", 0)),
            0,
        )

        with torch.no_grad():
            expert_logits = self.clients[client_id].expert_model(images)
            expert_logits = self._apply_logit_temperature(
                expert_logits,
                self.client_expert_temperatures.get(client_id, 1.0),
            )
            expert_prediction, expert_confidence, expert_margin, expert_energy = self._routing_signals_from_logits(
                expert_logits
            )
            base_mask = self._build_routing_fallback_mask(
                expert_confidence,
                expert_margin,
                expert_energy,
                confidence_threshold,
                margin_threshold,
                energy_threshold,
            )
            gain_scores = self._compute_routing_gain_scores(
                client_id,
                expert_confidence,
                expert_margin,
                expert_energy,
                confidence_threshold,
                margin_threshold,
                energy_threshold,
            )
            if self.current_round <= gain_search_warmup_rounds:
                fallback_mask = base_mask
            else:
                fallback_mask = base_mask & (gain_scores >= gain_threshold)

            predictions = expert_prediction.clone()
            invocation_count = int(fallback_mask.sum().item())
            route_types = ["expert"] * images.size(0)
            if invocation_count > 0:
                general_logits = self.deploy_general_model(images[fallback_mask])
                predictions[fallback_mask] = general_logits.argmax(dim=1)
                for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_idx] = "general"

        return predictions, invocation_count, {
            "route_type": route_types,
            "expert_confidence": expert_confidence.detach().cpu().tolist(),
            "gain_score": gain_scores.detach().cpu().tolist(),
        }

    def _target_invocation_rate_for_client(self, client_id: str) -> float:
        if client_id.startswith("simple_"):
            return float(
                getattr(
                    self.config.inference,
                    "simple_target_general_invocation_rate",
                    self.config.inference.target_general_invocation_rate,
                )
            )
        if client_id.startswith("complex_"):
            return float(
                getattr(
                    self.config.inference,
                    "complex_target_general_invocation_rate",
                    self.config.inference.target_general_invocation_rate,
                )
            )
        return float(self.config.inference.target_general_invocation_rate)

    def _minimum_target_invocation_rate(self, target_invocation: float) -> float:
        scale = min(
            max(float(getattr(self.config.inference, "route_min_invocation_rate_scale", 0.0)), 0.0),
            1.0,
        )
        return min(max(target_invocation * scale, 0.0), 1.0)

    def _update_routing_thresholds(self, client_ids: Sequence[str]) -> None:
        if self.current_round <= int(getattr(self.config.inference, "route_warmup_rounds", 0)):
            return

        confidence_step = max(float(self.config.inference.personalized_threshold_step), 0.0)
        margin_step = max(float(self.config.inference.personalized_margin_step), 0.0)
        energy_step = max(float(getattr(self.config.inference, "personalized_energy_step", 0.0)), 0.0)
        hard_tolerance = max(float(getattr(self.config.inference, "route_hard_recall_tolerance", 0.0)), 0.0)
        search_radius = max(int(getattr(self.config.inference, "routing_search_radius", 2)), 0)
        hard_priority_margin = max(float(getattr(self.config.inference, "route_hard_priority_margin", 0.005)), 0.0)
        reliability_center = float(getattr(self.config.inference, "route_reliability_center", 0.60))
        reliability_invocation_scale = max(
            float(getattr(self.config.inference, "route_reliability_invocation_scale", 0.0)),
            0.0,
        )
        reliability_hard_bonus = max(
            float(getattr(self.config.inference, "route_reliability_hard_bonus", 0.0)),
            0.0,
        )
        gain_search_warmup_rounds = max(
            int(getattr(self.config.inference, "route_gain_search_warmup_rounds", 0)),
            0,
        )

        for client_id in client_ids:
            routing_stats = self._collect_client_routing_statistics(client_id)
            if routing_stats is None:
                continue

            expert_predictions_tensor = routing_stats["expert_predictions"]
            confidences_tensor = routing_stats["confidences"]
            margins_tensor = routing_stats["margins"]
            energies_tensor = routing_stats["energies"]
            general_predictions_tensor = routing_stats["general_predictions"]
            targets_tensor = routing_stats["targets"]
            hard_mask_tensor = routing_stats["hard_mask"]

            confidence_threshold = self.client_confidence_thresholds[client_id]
            margin_threshold = self.client_margin_thresholds[client_id]
            energy_threshold = self.client_energy_thresholds[client_id]

            reliability = float(self.client_reliability_scores.get(client_id, 1.0))
            hard_total = int(hard_mask_tensor.to(dtype=torch.int64).sum().item())
            if hard_total > 0:
                target_hard_recall = float(
                    general_predictions_tensor[hard_mask_tensor]
                    .eq(targets_tensor[hard_mask_tensor])
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
            else:
                target_hard_recall = 0.0
            target_hard_recall = max(
                0.0,
                min(
                    1.0,
                    target_hard_recall
                    - hard_tolerance
                    + (max(reliability_center - reliability, 0.0) * reliability_hard_bonus),
                ),
            )
            target_invocation = self._target_invocation_rate_for_client(client_id)
            target_invocation = min(
                1.0,
                max(
                    0.0,
                    target_invocation + ((reliability_center - reliability) * reliability_invocation_scale),
                ),
            )
            minimum_invocation = self._minimum_target_invocation_rate(target_invocation)

            min_confidence = float(self.config.inference.min_confidence_threshold)
            max_confidence = float(self.config.inference.max_confidence_threshold)
            min_margin = float(self.config.inference.min_margin_threshold)
            max_margin = float(self.config.inference.max_margin_threshold)
            min_energy = float(getattr(self.config.inference, "min_energy_threshold", -6.0))
            max_energy = float(getattr(self.config.inference, "max_energy_threshold", -1.0))

            confidence_candidates = (
                {
                    min(max(confidence_threshold + (offset * confidence_step), min_confidence), max_confidence)
                    for offset in range(-search_radius, search_radius + 1)
                }
                if confidence_step > 0.0
                else {min(max(confidence_threshold, min_confidence), max_confidence)}
            )
            margin_candidates = (
                {
                    min(max(margin_threshold + (offset * margin_step), min_margin), max_margin)
                    for offset in range(-search_radius, search_radius + 1)
                }
                if margin_step > 0.0
                else {min(max(margin_threshold, min_margin), max_margin)}
            )
            energy_candidates = (
                {
                    min(max(energy_threshold + (offset * energy_step), min_energy), max_energy)
                    for offset in range(-search_radius, search_radius + 1)
                }
                if energy_step > 0.0
                else {min(max(energy_threshold, min_energy), max_energy)}
            )

            best_feasible = None
            best_relaxed = None
            total_samples = max(int(targets_tensor.numel()), 1)

            for candidate_confidence in sorted(confidence_candidates):
                for candidate_margin in sorted(margin_candidates):
                    for candidate_energy in sorted(energy_candidates):
                        routed_confidence, routed_margin, routed_energy = self._apply_reliability_to_thresholds(
                            client_id,
                            candidate_confidence,
                            candidate_margin,
                            candidate_energy,
                        )
                        fallback_mask_tensor = self._build_routing_fallback_mask(
                            confidences_tensor,
                            margins_tensor,
                            energies_tensor,
                            routed_confidence,
                            routed_margin,
                            routed_energy,
                        )
                        routed_predictions_tensor = torch.where(
                            fallback_mask_tensor,
                            general_predictions_tensor,
                            expert_predictions_tensor,
                        )
                        accuracy = float(
                            routed_predictions_tensor.eq(targets_tensor).to(dtype=torch.float32).mean().item()
                        )
                        invocation = float(
                            fallback_mask_tensor.to(dtype=torch.float32).sum().item() / total_samples
                        )

                        if hard_total > 0:
                            hard_recall = float(
                                routed_predictions_tensor[hard_mask_tensor]
                                .eq(targets_tensor[hard_mask_tensor])
                                .to(dtype=torch.float32)
                                .mean()
                                .item()
                            )
                        else:
                            hard_recall = 0.0

                        candidate = {
                            "confidence": candidate_confidence,
                            "margin": candidate_margin,
                            "energy": candidate_energy,
                            "accuracy": accuracy,
                            "hard_recall": hard_recall,
                            "invocation": invocation,
                            "hard_gap": max(target_hard_recall - hard_recall, 0.0),
                            "under_invocation": max(minimum_invocation - invocation, 0.0),
                        }
                        if hard_recall + 1e-8 >= target_hard_recall:
                            if best_feasible is None:
                                best_feasible = candidate
                            else:
                                lower_under_invocation = (
                                    candidate["under_invocation"] < best_feasible["under_invocation"] - 1e-8
                                )
                                same_under_invocation = (
                                    abs(candidate["under_invocation"] - best_feasible["under_invocation"]) <= 1e-8
                                )
                                lower_invocation = invocation < best_feasible["invocation"] - 1e-8
                                same_invocation = abs(invocation - best_feasible["invocation"]) <= 1e-8
                                higher_accuracy = accuracy > best_feasible["accuracy"] + 1e-8
                                same_accuracy = abs(accuracy - best_feasible["accuracy"]) <= 1e-8
                                higher_hard = hard_recall > best_feasible["hard_recall"] + hard_priority_margin
                                if lower_under_invocation:
                                    best_feasible = candidate
                                elif same_under_invocation:
                                    if candidate["under_invocation"] <= 1e-8 and (
                                        lower_invocation
                                        or (same_invocation and (higher_accuracy or (same_accuracy and higher_hard)))
                                    ):
                                        best_feasible = candidate
                                    elif higher_accuracy or (same_accuracy and higher_hard):
                                        best_feasible = candidate
                        else:
                            smaller_gap = (
                                best_relaxed is None
                                or candidate["hard_gap"] < best_relaxed["hard_gap"] - 1e-8
                            )
                            same_gap = (
                                best_relaxed is not None
                                and abs(candidate["hard_gap"] - best_relaxed["hard_gap"]) <= 1e-8
                            )
                            invocation_penalty = max(invocation - target_invocation, 0.0)
                            best_invocation_penalty = (
                                max(best_relaxed["invocation"] - target_invocation, 0.0)
                                if best_relaxed is not None
                                else float("inf")
                            )
                            better_accuracy = best_relaxed is not None and accuracy > best_relaxed["accuracy"] + 1e-8
                            same_accuracy = (
                                best_relaxed is not None
                                and abs(accuracy - best_relaxed["accuracy"]) <= 1e-8
                            )
                            lower_under_invocation = (
                                best_relaxed is not None
                                and candidate["under_invocation"] < best_relaxed["under_invocation"] - 1e-8
                            )
                            same_under_invocation = (
                                best_relaxed is not None
                                and abs(candidate["under_invocation"] - best_relaxed["under_invocation"]) <= 1e-8
                            )
                            lower_penalty = invocation_penalty < best_invocation_penalty - 1e-8
                            lower_invocation = (
                                best_relaxed is not None
                                and invocation < best_relaxed["invocation"] - 1e-8
                            )
                            if smaller_gap:
                                best_relaxed = candidate
                            elif same_gap:
                                if lower_under_invocation:
                                    best_relaxed = candidate
                                elif same_under_invocation and (
                                    better_accuracy
                                    or (same_accuracy and (lower_penalty or lower_invocation))
                                ):
                                    best_relaxed = candidate

            chosen = best_feasible if best_feasible is not None else best_relaxed
            if chosen is not None:
                self.client_confidence_thresholds[client_id] = float(chosen["confidence"])
                self.client_margin_thresholds[client_id] = float(chosen["margin"])
                self.client_energy_thresholds[client_id] = float(chosen["energy"])

                routed_confidence, routed_margin, routed_energy = self._apply_reliability_to_thresholds(
                    client_id,
                    float(chosen["confidence"]),
                    float(chosen["margin"]),
                    float(chosen["energy"]),
                )
                base_mask_tensor = self._build_routing_fallback_mask(
                    confidences_tensor,
                    margins_tensor,
                    energies_tensor,
                    routed_confidence,
                    routed_margin,
                    routed_energy,
                )
                gain_scores = self._compute_routing_gain_scores(
                    client_id,
                    confidences_tensor,
                    margins_tensor,
                    energies_tensor,
                    routed_confidence,
                    routed_margin,
                    routed_energy,
                )
                general_correct_mask = general_predictions_tensor.eq(targets_tensor)
                expert_correct_mask = expert_predictions_tensor.eq(targets_tensor)
                positive_gain_mask = general_correct_mask & expert_correct_mask.logical_not()
                negative_gain_mask = expert_correct_mask & general_correct_mask.logical_not()

                current_gain_threshold = self.client_gain_thresholds.get(
                    client_id,
                    float(getattr(self.config.inference, "route_gain_threshold", 0.0)),
                )
                if self.current_round <= gain_search_warmup_rounds:
                    self.client_gain_thresholds[client_id] = float(
                        getattr(self.config.inference, "min_route_gain_threshold", -1.0)
                    )
                    continue

                gain_candidates = self._gain_threshold_candidates(current_gain_threshold, gain_scores)
                gain_best_feasible = None
                gain_best_relaxed = None

                for candidate_gain_threshold in gain_candidates:
                    fallback_mask_tensor = base_mask_tensor & (gain_scores >= candidate_gain_threshold)
                    routed_predictions_tensor = torch.where(
                        fallback_mask_tensor,
                        general_predictions_tensor,
                        expert_predictions_tensor,
                    )
                    accuracy = float(
                        routed_predictions_tensor.eq(targets_tensor).to(dtype=torch.float32).mean().item()
                    )
                    invocation = float(
                        fallback_mask_tensor.to(dtype=torch.float32).sum().item() / total_samples
                    )

                    if hard_total > 0:
                        hard_recall = float(
                            routed_predictions_tensor[hard_mask_tensor]
                            .eq(targets_tensor[hard_mask_tensor])
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        )
                    else:
                        hard_recall = 0.0

                    if bool(positive_gain_mask.any().item()):
                        positive_capture = float(
                            fallback_mask_tensor[positive_gain_mask].to(dtype=torch.float32).mean().item()
                        )
                    else:
                        positive_capture = 0.0
                    if bool(negative_gain_mask.any().item()):
                        negative_capture = float(
                            fallback_mask_tensor[negative_gain_mask].to(dtype=torch.float32).mean().item()
                        )
                    else:
                        negative_capture = 0.0

                    candidate = {
                        "gain_threshold": candidate_gain_threshold,
                        "accuracy": accuracy,
                        "hard_recall": hard_recall,
                        "invocation": invocation,
                        "positive_capture": positive_capture,
                        "negative_capture": negative_capture,
                        "hard_gap": max(target_hard_recall - hard_recall, 0.0),
                        "under_invocation": max(minimum_invocation - invocation, 0.0),
                    }
                    if hard_recall + 1e-8 >= target_hard_recall:
                        if gain_best_feasible is None:
                            gain_best_feasible = candidate
                        else:
                            lower_under_invocation = (
                                candidate["under_invocation"] < gain_best_feasible["under_invocation"] - 1e-8
                            )
                            same_under_invocation = (
                                abs(candidate["under_invocation"] - gain_best_feasible["under_invocation"]) <= 1e-8
                            )
                            lower_invocation = invocation < gain_best_feasible["invocation"] - 1e-8
                            same_invocation = abs(invocation - gain_best_feasible["invocation"]) <= 1e-8
                            higher_accuracy = accuracy > gain_best_feasible["accuracy"] + 1e-8
                            higher_gain = (
                                (positive_capture - negative_capture)
                                > (gain_best_feasible["positive_capture"] - gain_best_feasible["negative_capture"])
                                + 1e-8
                            )
                            if lower_under_invocation:
                                gain_best_feasible = candidate
                            elif same_under_invocation:
                                if candidate["under_invocation"] <= 1e-8 and (
                                    lower_invocation or (same_invocation and (higher_accuracy or higher_gain))
                                ):
                                    gain_best_feasible = candidate
                                elif higher_accuracy or higher_gain:
                                    gain_best_feasible = candidate
                    else:
                        gain_balance = positive_capture - negative_capture
                        smaller_gap = (
                            gain_best_relaxed is None
                            or candidate["hard_gap"] < gain_best_relaxed["hard_gap"] - 1e-8
                        )
                        same_gap = (
                            gain_best_relaxed is not None
                            and abs(candidate["hard_gap"] - gain_best_relaxed["hard_gap"]) <= 1e-8
                        )
                        better_accuracy = (
                            gain_best_relaxed is not None
                            and accuracy > gain_best_relaxed["accuracy"] + 1e-8
                        )
                        same_accuracy = (
                            gain_best_relaxed is not None
                            and abs(accuracy - gain_best_relaxed["accuracy"]) <= 1e-8
                        )
                        better_gain = (
                            gain_best_relaxed is not None
                            and gain_balance
                            > (
                                gain_best_relaxed["positive_capture"]
                                - gain_best_relaxed["negative_capture"]
                            )
                            + 1e-8
                        )
                        lower_under_invocation = (
                            gain_best_relaxed is not None
                            and candidate["under_invocation"] < gain_best_relaxed["under_invocation"] - 1e-8
                        )
                        same_under_invocation = (
                            gain_best_relaxed is not None
                            and abs(candidate["under_invocation"] - gain_best_relaxed["under_invocation"]) <= 1e-8
                        )
                        lower_invocation = (
                            gain_best_relaxed is not None
                            and invocation < gain_best_relaxed["invocation"] - 1e-8
                        )
                        if smaller_gap:
                            gain_best_relaxed = candidate
                        elif same_gap:
                            if lower_under_invocation:
                                gain_best_relaxed = candidate
                            elif same_under_invocation and (
                                better_accuracy
                                or (same_accuracy and (better_gain or lower_invocation))
                            ):
                                gain_best_relaxed = candidate

                chosen_gain = gain_best_feasible if gain_best_feasible is not None else gain_best_relaxed
                if chosen_gain is not None:
                    self.client_gain_thresholds[client_id] = float(chosen_gain["gain_threshold"])

    def _maybe_update_best(
        self,
        round_idx: int,
        round_metrics: RoundMetrics,
        expert_accuracy: float,
        general_accuracy: float,
        selection_accuracy: float,
    ) -> bool:
        if self.best_snapshot is None:
            is_better = True
        else:
            previous_routed = float(self.best_snapshot["routed_accuracy"])
            previous_selection = float(self.best_snapshot["selection_accuracy"])
            is_better = round_metrics.routed_accuracy > previous_routed + 1e-8
            if not is_better and abs(round_metrics.routed_accuracy - previous_routed) <= 1e-8:
                is_better = selection_accuracy > previous_selection + 1e-8
        if not is_better:
            return False

        self.best_snapshot = {
            "round_idx": round_idx,
            "selection_accuracy": selection_accuracy,
            "routed_accuracy": round_metrics.routed_accuracy,
            "general_accuracy": general_accuracy,
            "expert_accuracy": expert_accuracy,
            "avg_client_loss": round_metrics.avg_client_loss,
            "general_model_state": _clone_state_dict(self.general_model.state_dict()),
            "deploy_general_model_state": _clone_state_dict(self.deploy_general_model.state_dict()),
            "client_expert_states": {
                client_id: _clone_state_dict(client.expert_model.state_dict())
                for client_id, client in self.clients.items()
            },
            "client_confidence_thresholds": dict(self.client_confidence_thresholds),
            "client_margin_thresholds": dict(self.client_margin_thresholds),
            "client_energy_thresholds": dict(self.client_energy_thresholds),
            "client_gain_thresholds": dict(self.client_gain_thresholds),
            "client_reliability_scores": dict(self.client_reliability_scores),
            "deploy_general_temperature": float(self.deploy_general_temperature),
            "client_expert_temperatures": dict(self.client_expert_temperatures),
        }
        LOGGER.info(
            "fedegssg best | round=%d | holdout=%.4f | routed=%.4f | general=%.4f | expert=%.4f",
            round_idx,
            selection_accuracy,
            round_metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )
        return True

    def _restore_best(self) -> None:
        if self.best_snapshot is None:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        self.deploy_general_model.load_state_dict(self.best_snapshot["deploy_general_model_state"])
        for client_id, state in self.best_snapshot["client_expert_states"].items():
            self.clients[client_id].expert_model.load_state_dict(state)
        self.client_confidence_thresholds = dict(self.best_snapshot["client_confidence_thresholds"])
        self.client_margin_thresholds = dict(self.best_snapshot["client_margin_thresholds"])
        self.client_energy_thresholds = dict(self.best_snapshot["client_energy_thresholds"])
        self.client_gain_thresholds = dict(self.best_snapshot.get("client_gain_thresholds", self.client_gain_thresholds))
        self.client_reliability_scores = dict(self.best_snapshot["client_reliability_scores"])
        self.deploy_general_temperature = float(self.best_snapshot.get("deploy_general_temperature", 1.0))
        self.client_expert_temperatures = dict(self.best_snapshot.get("client_expert_temperatures", {}))
        LOGGER.info("fedegssg restored best from round %d", int(self.best_snapshot["round_idx"]))

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        kd_warmup = int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0))

        for round_idx in range(1, self.config.federated.rounds + 1):
            self.current_round = round_idx
            selected_client_ids = self._sample_client_ids()
            deploy_public_accuracy = self._public_accuracy_for_model(self.deploy_general_model)
            min_general_accuracy = max(
                float(getattr(self.config.federated, "expert_kd_min_general_accuracy", 0.0)),
                0.0,
            )
            use_kd = (round_idx > kd_warmup) and (deploy_public_accuracy >= min_general_accuracy)

            teacher = self.deploy_general_model if use_kd else None
            teacher_payload = None
            teacher_raw_bytes = 0
            teacher_compressed_bytes = 0
            if use_kd and self.communication_quantization_enabled:
                teacher_payload = self._build_quantized_deploy_payload()
                teacher_raw_bytes = teacher_payload.raw_nbytes
                teacher_compressed_bytes = teacher_payload.compressed_nbytes
                teacher = None
            elif use_kd:
                teacher_raw_bytes = estimate_state_dict_nbytes(self.deploy_general_model.state_dict())
                teacher_compressed_bytes = teacher_raw_bytes

            LOGGER.info(
                "fedegssg round %d | clients=%s | kd=%s",
                round_idx,
                selected_client_ids,
                use_kd,
            )

            updates = [
                self.clients[client_id].train_local(
                    round_idx=round_idx,
                    public_batches=self.public_batches,
                    distill_batches=self.distill_batches,
                    general_model=teacher,
                    general_model_payload=teacher_payload,
                    general_public_accuracy=deploy_public_accuracy,
                    general_temperature=self.deploy_general_temperature,
                )
                for client_id in selected_client_ids
            ]
            average_client_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            uplink_raw_bytes = sum(int(update.raw_upload_bytes) for update in updates)
            uplink_compressed_bytes = sum(
                int(update.compressed_upload_bytes or update.raw_upload_bytes)
                for update in updates
            )

            if self._should_run_temperature_calibration(round_idx):
                self._update_expert_temperatures(updates)
            self._update_client_reliability_scores(updates)
            public_ensemble = self._aggregate_public_teachers(updates)
            external_ensemble = self._aggregate_external_teachers(updates)
            distill_stats = self._distill_general_model(public_ensemble, external_ensemble)
            self._sync_deploy_general(hard=False)
            if self._should_run_temperature_calibration(round_idx):
                self._update_general_temperature()
            self._update_routing_thresholds(selected_client_ids)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegssg-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegssg-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegssg-routed")
            extra_metrics = self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)
            holdout_eval = self._evaluate_predictor_on_client_holdouts(self._predict_routed)
            holdout_accuracy = float(holdout_eval["aggregate"]["accuracy"])

            aggregate_metrics = routed_eval["aggregate"]
            macro_metrics = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate_metrics["invocation_rate"],
                mode="routed",
            )
            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=average_client_loss,
                routed_accuracy=macro_metrics["accuracy"],
                hard_accuracy=aggregate_metrics["hard_recall"],
                invocation_rate=aggregate_metrics["invocation_rate"],
                local_accuracy=macro_metrics["accuracy"],
                weighted_accuracy=aggregate_metrics["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                extra_metrics=extra_metrics,
            )

            LOGGER.info(
                "fedegssg round %d | loss=%.4f | distill=%.4f | public_ce=%.4f | public_kl=%.4f | external_kl=%.4f | hard_kl=%.4f | deploy_consistency=%.4f | reliability=%.4f | hard_public=%.4f | holdout=%.4f | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f | uplink=%d/%d | downlink=%d/%d",
                round_idx,
                average_client_loss,
                distill_stats["total_loss"],
                distill_stats["public_ce_loss"],
                distill_stats["public_kl_loss"],
                distill_stats["external_kl_loss"],
                distill_stats["hard_kl_loss"],
                distill_stats["deploy_consistency_loss"],
                float(public_ensemble["mean_reliability"].item()),
                float(public_ensemble["hard_ratio"].item()),
                holdout_accuracy,
                macro_metrics["accuracy"],
                aggregate_metrics["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate_metrics["hard_recall"],
                aggregate_metrics["invocation_rate"],
                uplink_compressed_bytes,
                uplink_raw_bytes,
                teacher_compressed_bytes,
                teacher_raw_bytes,
            )
            LOGGER.info(
                "fedegssg local-kd | round=%d | kd_gate=%.4f | refresh_gate=%.4f | gain_thr=%.4f",
                round_idx,
                sum(float(update.kd_gate_ratio) for update in updates) / max(len(updates), 1),
                sum(float(update.refresh_gate_ratio) for update in updates) / max(len(updates), 1),
                sum(float(self.client_gain_thresholds.get(client_id, 0.0)) for client_id in selected_client_ids)
                / max(len(selected_client_ids), 1),
            )
            LOGGER.info(
                "fedegssg calibration | round=%d | general_temp=%.4f | expert_temp_mean=%.4f | deploy_public_acc=%.4f | kd_active=%s",
                round_idx,
                float(self.deploy_general_temperature),
                sum(float(self.client_expert_temperatures.get(client_id, 1.0)) for client_id in selected_client_ids)
                / max(len(selected_client_ids), 1),
                deploy_public_accuracy,
                use_kd,
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegssg", average_client_loss, round_idx)
                self.writer.add_scalar("distill_loss/fedegssg", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar("distill_public_ce_loss/fedegssg", distill_stats["public_ce_loss"], round_idx)
                self.writer.add_scalar("distill_public_kl_loss/fedegssg", distill_stats["public_kl_loss"], round_idx)
                self.writer.add_scalar("distill_external_kl_loss/fedegssg", distill_stats["external_kl_loss"], round_idx)
                self.writer.add_scalar("distill_hard_kl_loss/fedegssg", distill_stats["hard_kl_loss"], round_idx)
                self.writer.add_scalar(
                    "distill_deploy_consistency_loss/fedegssg",
                    distill_stats["deploy_consistency_loss"],
                    round_idx,
                )
                self.writer.add_scalar(
                    "client_reliability_mean/fedegssg",
                    float(public_ensemble["mean_reliability"].item()),
                    round_idx,
                )
                self.writer.add_scalar("routing_holdout_accuracy/fedegssg", holdout_accuracy, round_idx)
                self.writer.add_scalar(
                    "hard_public_ratio/fedegssg",
                    float(public_ensemble["hard_ratio"].item()),
                    round_idx,
                )
                self.writer.add_scalar(
                    "calibration/general_temperature_fedegssg",
                    float(self.deploy_general_temperature),
                    round_idx,
                )
                self.writer.add_scalar(
                    "calibration/expert_temperature_mean_fedegssg",
                    sum(float(self.client_expert_temperatures.get(client_id, 1.0)) for client_id in selected_client_ids)
                    / max(len(selected_client_ids), 1),
                    round_idx,
                )
                if uplink_raw_bytes > 0:
                    self.writer.add_scalar(
                        "comm/uplink_ratio_fedegssg",
                        uplink_compressed_bytes / uplink_raw_bytes,
                        round_idx,
                    )
                if teacher_raw_bytes > 0:
                    self.writer.add_scalar(
                        "comm/downlink_ratio_fedegssg",
                        teacher_compressed_bytes / teacher_raw_bytes,
                        round_idx,
                    )
                self._log_auxiliary_accuracy_metrics(
                    "fedegssg",
                    round_idx,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics("fedegssg", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best(
                round_idx,
                round_metrics,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                holdout_accuracy,
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best()
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        route_export_path = self._build_route_export_path("fedegssg_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            "fedegssg_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegssg_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegssg_final_general")
        extra_metrics = self._build_final_extra_metrics(expert_eval, general_eval, routed_eval)

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

        return {
            "algorithm": "fedegssg",
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
                "public_dataset_size": self.public_size,
                "mean_client_reliability": sum(self.client_reliability_scores.values()) / max(len(self.client_reliability_scores), 1),
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
            general_route_types=("general",),
        )

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)
