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
  - Lite MKD-style extensions: optional communication quantization, stronger
    general-to-expert KD, and history-anchor regularization for the general model.
  - At inference: expert-first with confidence routing to general model fallback.
"""

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
from fedegs.federated.compression import (
    CompressedStateDict,
    compress_state_dict,
    decompress_state_dict,
    estimate_state_dict_nbytes,
)
from fedegs.models import (
    SmallCNN,
    build_model,
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
    expert_state_dict: Optional[Dict[str, torch.Tensor]] = None
    compressed_expert_state: Optional[CompressedStateDict] = None
    proxy_scope_state_dict: Optional[Dict[str, torch.Tensor]] = None
    compressed_proxy_scope_state: Optional[CompressedStateDict] = None
    expert_raw_upload_bytes: int = 0
    expert_compressed_upload_bytes: int = 0
    proxy_raw_upload_bytes: int = 0
    proxy_compressed_upload_bytes: int = 0
    raw_upload_bytes: int = 0
    compressed_upload_bytes: int = 0


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


def _initialize_feature_adapter(adapter: nn.Linear) -> None:
    with torch.no_grad():
        adapter.weight.zero_()
        diagonal = min(adapter.out_features, adapter.in_features)
        adapter.weight[:diagonal, :diagonal] = torch.eye(diagonal, dtype=adapter.weight.dtype)


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _normalize_scopes(raw_scopes) -> Tuple[str, ...]:
    if raw_scopes is None:
        return tuple()
    if isinstance(raw_scopes, str):
        raw_items = [item.strip() for item in raw_scopes.split(",")]
    else:
        raw_items = [str(item).strip() for item in raw_scopes]
    return tuple(item for item in raw_items if item)


def _matches_scope(name: str, scopes: Tuple[str, ...]) -> bool:
    return any(name == scope or name.startswith(f"{scope}.") for scope in scopes)


def _extract_scoped_state_dict(state_dict: Dict[str, torch.Tensor], scopes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    if not scopes:
        return {}
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if _matches_scope(key, scopes)
    }


def _set_trainable_scopes(model: nn.Module, scopes: Tuple[str, ...]) -> List[nn.Parameter]:
    trainable_parameters: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        should_train = _matches_scope(name, scopes)
        parameter.requires_grad_(should_train)
        if should_train:
            trainable_parameters.append(parameter)
    return trainable_parameters


def _freeze_frozen_batch_norm_stats(model: nn.Module, scopes: Tuple[str, ...]) -> None:
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module_name, module in model.named_modules():
        if not isinstance(module, norm_types):
            continue
        has_trainable_params = False
        for parameter_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if _matches_scope(full_name, scopes):
                has_trainable_params = True
                break
        if not has_trainable_params:
            module.eval()


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
        self.num_classes = num_classes
        self.expert_model = _build_expert_model_from_config(config).to(self.device)
        self.proxy_scopes = _normalize_scopes(
            getattr(self.config.federated, "proxy_trainable_scopes", ("classifier", "fc"))
        )

    def train_local(
        self,
        general_model: nn.Module = None,
        general_model_payload: CompressedStateDict | None = None,
    ) -> FedEGSDClientUpdate:
        """
        Train expert on private data.
        If general_model is provided and kd_weight > 0, add KL distillation loss.
        """
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        kd_weight = float(getattr(self.config.federated, "expert_kd_weight", 0.0))
        kd_temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))
        proxy_enabled = bool(getattr(self.config.federated, "proxy_enabled", False))

        teacher_model = general_model
        if teacher_model is None and general_model_payload is not None:
            teacher_model = self._build_general_model_from_payload(general_model_payload)

        use_proxy = proxy_enabled and teacher_model is not None and len(self.proxy_scopes) > 0
        use_kd = teacher_model is not None and kd_weight > 0.0
        proxy_scope_state = None
        compressed_proxy_scope_state = None
        proxy_raw_upload_bytes = 0
        proxy_compressed_upload_bytes = 0

        if use_proxy:
            proxy_model = self._build_general_model_from_state_dict(teacher_model.state_dict())
            losses = self._train_with_proxy_mkd(loader, proxy_model)
            loss = losses["total_loss"]
            proxy_scope_state = _extract_scoped_state_dict(proxy_model.state_dict(), self.proxy_scopes)
            if proxy_scope_state:
                proxy_raw_upload_bytes = estimate_state_dict_nbytes(proxy_scope_state)
                proxy_compressed_upload_bytes = proxy_raw_upload_bytes
                if bool(getattr(self.config.federated, "communication_quantization_enabled", False)):
                    compressed_proxy_scope_state = compress_state_dict(
                        proxy_scope_state,
                        bits=int(getattr(self.config.federated, "communication_quantization_bits", 8)),
                    )
                    proxy_compressed_upload_bytes = compressed_proxy_scope_state.compressed_nbytes
                    proxy_scope_state = None
        elif not use_kd:
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
            loss = self._train_with_kd(loader, teacher_model, kd_weight, kd_temperature)

        state = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        expert_raw_upload_bytes = estimate_state_dict_nbytes(state)
        compressed_state = None
        expert_compressed_upload_bytes = expert_raw_upload_bytes
        if bool(getattr(self.config.federated, "communication_quantization_enabled", False)):
            compressed_state = compress_state_dict(
                state,
                bits=int(getattr(self.config.federated, "communication_quantization_bits", 8)),
            )
            expert_compressed_upload_bytes = compressed_state.compressed_nbytes
            state = None

        raw_upload_bytes = expert_raw_upload_bytes + proxy_raw_upload_bytes
        compressed_upload_bytes = expert_compressed_upload_bytes + proxy_compressed_upload_bytes

        return FedEGSDClientUpdate(
            client_id=self.client_id, num_samples=len(self.dataset),
            loss=loss,
            expert_state_dict=state,
            compressed_expert_state=compressed_state,
            proxy_scope_state_dict=proxy_scope_state,
            compressed_proxy_scope_state=compressed_proxy_scope_state,
            expert_raw_upload_bytes=expert_raw_upload_bytes,
            expert_compressed_upload_bytes=expert_compressed_upload_bytes,
            proxy_raw_upload_bytes=proxy_raw_upload_bytes,
            proxy_compressed_upload_bytes=proxy_compressed_upload_bytes,
            raw_upload_bytes=raw_upload_bytes,
            compressed_upload_bytes=compressed_upload_bytes,
        )

    def _build_general_model_from_payload(self, payload: CompressedStateDict) -> nn.Module:
        teacher = _build_general_model_from_config(self.config).to(self.device)
        teacher.load_state_dict(decompress_state_dict(payload))
        return teacher

    def _build_general_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        teacher = _build_general_model_from_config(self.config).to(self.device)
        teacher.load_state_dict(_clone_state_dict(state_dict))
        return teacher

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

    def _train_with_proxy_mkd(self, loader: DataLoader, proxy_model: nn.Module) -> Dict[str, float]:
        """Train expert E_k and lightweight proxy P_k with bidirectional KD."""
        expert_optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        trainable_proxy_params = _set_trainable_scopes(proxy_model, self.proxy_scopes)
        if not trainable_proxy_params:
            raise ValueError("Proxy training is enabled but no proxy parameters match proxy_trainable_scopes.")
        proxy_optimizer = torch.optim.SGD(
            trainable_proxy_params,
            lr=float(getattr(self.config.federated, "general_head_lr", self.config.federated.local_lr)),
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        temperature = float(getattr(self.config.federated, "proxy_temperature", 3.0))
        proxy_ce_weight = float(getattr(self.config.federated, "proxy_ce_weight", 1.0))
        proxy_kd_weight = float(getattr(self.config.federated, "proxy_kd_weight", 0.5))
        expert_proxy_kd_weight = float(getattr(self.config.federated, "expert_proxy_kd_weight", 0.5))

        self.expert_model.train()
        proxy_model.train()
        _freeze_frozen_batch_norm_stats(proxy_model, self.proxy_scopes)

        total_loss = 0.0
        total_expert_loss = 0.0
        total_proxy_loss = 0.0
        total_batches = 0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                _freeze_frozen_batch_norm_stats(proxy_model, self.proxy_scopes)
                expert_logits = self.expert_model(images)
                proxy_logits = proxy_model(images)

                expert_loss = criterion(expert_logits, targets)
                if expert_proxy_kd_weight > 0.0:
                    proxy_probs = F.softmax(proxy_logits.detach() / temperature, dim=1)
                    expert_loss = expert_loss + expert_proxy_kd_weight * F.kl_div(
                        F.log_softmax(expert_logits / temperature, dim=1),
                        proxy_probs,
                        reduction="batchmean",
                    ) * (temperature ** 2)

                proxy_loss = proxy_ce_weight * criterion(proxy_logits, targets)
                if proxy_kd_weight > 0.0:
                    expert_probs = F.softmax(expert_logits.detach() / temperature, dim=1)
                    proxy_loss = proxy_loss + proxy_kd_weight * F.kl_div(
                        F.log_softmax(proxy_logits / temperature, dim=1),
                        expert_probs,
                        reduction="batchmean",
                    ) * (temperature ** 2)

                expert_optimizer.zero_grad(set_to_none=True)
                expert_loss.backward()
                expert_optimizer.step()

                proxy_optimizer.zero_grad(set_to_none=True)
                proxy_loss.backward()
                proxy_optimizer.step()

                total_expert_loss += float(expert_loss.detach().cpu().item())
                total_proxy_loss += float(proxy_loss.detach().cpu().item())
                total_loss += float(expert_loss.detach().cpu().item() + proxy_loss.detach().cpu().item())
                total_batches += 1

        divisor = max(total_batches, 1)
        return {
            "total_loss": total_loss / divisor,
            "expert_loss": total_expert_loss / divisor,
            "proxy_loss": total_proxy_loss / divisor,
        }


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
        self.general_model = _build_general_model_from_config(config).to(self.device)

        # FLOPs
        self.reference_expert = _build_expert_model_from_config(config).to(self.device)
        self.feature_adapter = nn.Linear(
            self.reference_expert.feature_dim,
            self.general_model.feature_dim,
            bias=False,
        ).to(self.device)
        _initialize_feature_adapter(self.feature_adapter)
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
        self.communication_quantization_enabled = bool(
            getattr(config.federated, "communication_quantization_enabled", False)
        )
        self.communication_quantization_bits = int(
            getattr(config.federated, "communication_quantization_bits", 8)
        )

    def _materialize_expert_state_dict(self, update: FedEGSDClientUpdate) -> Dict[str, torch.Tensor]:
        if update.expert_state_dict is not None:
            return update.expert_state_dict
        if update.compressed_expert_state is not None:
            return decompress_state_dict(update.compressed_expert_state)
        raise ValueError(f"Client update for {update.client_id} does not contain expert weights.")

    def _materialize_proxy_scope_state_dict(self, update: FedEGSDClientUpdate) -> Optional[Dict[str, torch.Tensor]]:
        if update.proxy_scope_state_dict is not None:
            return update.proxy_scope_state_dict
        if update.compressed_proxy_scope_state is not None:
            return decompress_state_dict(update.compressed_proxy_scope_state)
        return None

    def _aggregate_proxy_scope_state_dict(self, updates: List[FedEGSDClientUpdate]) -> Optional[Dict[str, torch.Tensor]]:
        aggregated: Dict[str, torch.Tensor] = {}
        total_weight = 0.0

        for update in updates:
            proxy_state = self._materialize_proxy_scope_state_dict(update)
            if not proxy_state:
                continue
            weight = float(max(update.num_samples, 1))
            total_weight += weight
            for key, value in proxy_state.items():
                tensor = value.detach().cpu()
                if tensor.is_floating_point():
                    if key not in aggregated:
                        aggregated[key] = tensor.clone() * weight
                    else:
                        aggregated[key] += tensor * weight
                elif key not in aggregated:
                    aggregated[key] = tensor.clone()

        if total_weight <= 0.0 or not aggregated:
            return None

        for key, value in list(aggregated.items()):
            if value.is_floating_point():
                aggregated[key] = value / total_weight
        return aggregated

    def _build_quantized_general_payload(self) -> CompressedStateDict:
        state = {key: value.detach().cpu().clone() for key, value in self.general_model.state_dict().items()}
        return compress_state_dict(state, bits=self.communication_quantization_bits)

    # ================================================================
    # AVGLOGITS extraction (FedDF core)
    # ================================================================

    def _extract_ensemble_logits(self, updates: List[FedEGSDClientUpdate]) -> Dict[str, torch.Tensor]:
        """Average raw logits across experts, THEN softmax (FedDF AVGLOGITS)."""
        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)
        normalize_features = bool(getattr(self.config.federated, "distill_feature_normalize", True))

        all_logits: List[torch.Tensor] = []
        all_features: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        for update in updates:
            expert = _build_expert_model_from_config(self.config).to(self.device)
            expert.load_state_dict(self._materialize_expert_state_dict(update))
            expert.eval()

            batches_logits: List[torch.Tensor] = []
            batches_features: List[torch.Tensor] = []
            with torch.no_grad():
                for batch in self.distill_loader:
                    images = batch[0].to(self.device)
                    expert_features = expert.forward_features(images)
                    expert_logits = expert.classify_features(expert_features)
                    if normalize_features:
                        expert_features = F.normalize(expert_features, dim=1)
                    batches_features.append(expert_features.cpu())
                    batches_logits.append(expert_logits.cpu())

            client_logits = torch.cat(batches_logits, dim=0)  # [N, C]
            client_features = torch.cat(batches_features, dim=0)  # [N, D]

            # Entropy-based uncertainty weight
            probs = F.softmax(client_logits / temperature, dim=1)
            log_C = math.log(float(num_classes))
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(log_C, 1e-8)
            w = torch.exp(-ent)

            all_logits.append(client_logits)
            all_features.append(client_features)
            all_weights.append(w)
            del expert

        stacked_l = torch.stack(all_logits, dim=0)   # [K, N, C]
        stacked_f = torch.stack(all_features, dim=0)  # [K, N, D]
        stacked_w = torch.stack(all_weights, dim=0)   # [K, N]

        norm_w = stacked_w / stacked_w.sum(0, keepdim=True).clamp_min(1e-8)  # [K, N]
        avg_logits = (stacked_l * norm_w.unsqueeze(-1)).sum(0)                # [N, C]
        avg_features = (stacked_f * norm_w.unsqueeze(-1)).sum(0)              # [N, D]
        if normalize_features:
            avg_features = F.normalize(avg_features, dim=1)
        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        sample_w = stacked_w.mean(0)

        return {"soft_labels": soft_labels, "target_features": avg_features, "sample_weights": sample_w}

    # ================================================================
    # Distillation: pure KL + Adam + cosine annealing
    # ================================================================

    def _distill_general_model(
        self,
        ensemble: Dict[str, torch.Tensor],
        round_idx: int,
        proxy_anchor_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        bs = self.config.dataset.batch_size
        anchor_weight = max(float(getattr(self.config.federated, "general_anchor_weight", 0.0)), 0.0)
        proxy_anchor_weight = max(float(getattr(self.config.federated, "server_proxy_anchor_weight", 0.0)), 0.0)
        feature_weight = max(float(getattr(self.config.federated, "distill_feature_weight", 0.0)), 0.0)
        normalize_features = bool(getattr(self.config.federated, "distill_feature_normalize", True))

        soft_all = ensemble["soft_labels"]
        target_features_all = ensemble["target_features"]
        w_all = ensemble["sample_weights"]

        # Collect distillation images (same order as soft_all)
        all_imgs: List[torch.Tensor] = []
        for batch in self.distill_loader:
            all_imgs.append(batch[0])
        distill_images = torch.cat(all_imgs, dim=0)
        N = distill_images.size(0)

        # Fresh optimizer + cosine schedule per round
        optim_params = list(self.general_model.parameters()) + list(self.feature_adapter.parameters())
        optimizer = torch.optim.Adam(optim_params, lr=float(self.config.federated.distill_lr))
        total_steps = distill_epochs * ((N + bs - 1) // bs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        anchor_model = None
        if anchor_weight > 0.0:
            anchor_model = _build_general_model_from_config(self.config).to(self.device)
            anchor_model.load_state_dict(
                {key: value.detach().cpu().clone() for key, value in self.general_model.state_dict().items()}
            )
            anchor_model.eval()
            for parameter in anchor_model.parameters():
                parameter.requires_grad_(False)

        self.general_model.train()
        self.feature_adapter.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_feature_loss = 0.0
        total_anchor_loss = 0.0
        total_proxy_anchor_loss = 0.0
        total_batches = 0
        named_parameters = dict(self.general_model.named_parameters())

        for _ in range(distill_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                imgs = distill_images[idx].to(self.device)
                b_soft = soft_all[idx].to(self.device)
                b_target_features = target_features_all[idx].to(self.device)
                b_w = w_all[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                s_features = self.general_model.forward_features(imgs)
                s_logits = self.general_model.classify_features(s_features)

                s_log_p = F.log_softmax(s_logits / temperature, dim=1)
                per_kl = F.kl_div(s_log_p, b_soft, reduction="none").sum(1) * (temperature ** 2)
                distill_loss = (per_kl * b_w).sum() / b_w.sum().clamp_min(1e-8)

                feature_loss = distill_loss.new_zeros(())
                if feature_weight > 0.0:
                    adapted_target_features = self.feature_adapter(b_target_features)
                    if normalize_features:
                        s_features_cmp = F.normalize(s_features, dim=1)
                        adapted_target_features = F.normalize(adapted_target_features, dim=1)
                    else:
                        s_features_cmp = s_features
                    per_feature = (s_features_cmp - adapted_target_features).pow(2).mean(dim=1)
                    feature_loss = (per_feature * b_w).sum() / b_w.sum().clamp_min(1e-8)

                anchor_loss = distill_loss.new_zeros(())
                if anchor_model is not None:
                    with torch.no_grad():
                        anchor_logits = anchor_model(imgs)
                        anchor_probs = F.softmax(anchor_logits / temperature, dim=1)
                    per_anchor = F.kl_div(s_log_p, anchor_probs, reduction="none").sum(1) * (temperature ** 2)
                    anchor_loss = (per_anchor * b_w).sum() / b_w.sum().clamp_min(1e-8)

                proxy_param_anchor_loss = distill_loss.new_zeros(())
                if proxy_anchor_state and proxy_anchor_weight > 0.0:
                    anchor_terms = []
                    for name, target in proxy_anchor_state.items():
                        parameter = named_parameters.get(name)
                        if parameter is None:
                            continue
                        target_tensor = target.to(device=parameter.device, dtype=parameter.dtype)
                        anchor_terms.append((parameter - target_tensor).pow(2).mean())
                    if anchor_terms:
                        proxy_param_anchor_loss = torch.stack(anchor_terms).mean()

                loss = (
                    distill_loss
                    + feature_weight * feature_loss
                    + anchor_weight * anchor_loss
                    + proxy_anchor_weight * proxy_param_anchor_loss
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_distill_loss += float(distill_loss.detach().cpu().item())
                total_feature_loss += float(feature_loss.detach().cpu().item())
                total_anchor_loss += float(anchor_loss.detach().cpu().item())
                total_proxy_anchor_loss += float(proxy_param_anchor_loss.detach().cpu().item())
                total_batches += 1

        d = max(total_batches, 1)
        return {
            "total_loss": total_loss / d,
            "kl_loss": total_distill_loss / d,
            "feature_loss": total_feature_loss / d,
            "anchor_loss": total_anchor_loss / d,
            "proxy_anchor_loss": total_proxy_anchor_loss / d,
        }

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
            expert = _build_expert_model_from_config(self.config).to(self.device)
            expert.load_state_dict(self._materialize_expert_state_dict(u)); expert.eval()
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
            "feature_adapter_state": {k: v.cpu().clone() for k, v in self.feature_adapter.state_dict().items()},
            "client_expert_states": {c: {k: v.cpu().clone() for k, v in cl.expert_model.state_dict().items()} for c, cl in self.clients.items()},
            "client_confidence_thresholds": dict(self.client_confidence_thresholds),
            "client_margin_thresholds": dict(self.client_margin_thresholds),
        }
        LOGGER.info("fedegsd best | round=%d | routed=%.4f | general=%.4f | expert=%.4f", ri, rm.routed_accuracy, ga, ea)
        return True

    def _restore_best(self):
        if not self.best_snapshot: return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        self.feature_adapter.load_state_dict(self.best_snapshot["feature_adapter_state"])
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
        proxy_enabled = bool(getattr(self.config.federated, "proxy_enabled", False))

        for ri in range(1, self.config.federated.rounds + 1):
            sids = self._sample_client_ids()

            # KD warmup: first N rounds train expert with pure CE (no KD from random general)
            # After warmup, general model has accumulated enough quality to be a useful teacher
            use_kd = ri > kd_warmup
            use_proxy = proxy_enabled and use_kd
            teacher = self.general_model if (use_kd or use_proxy) else None
            teacher_payload = None
            teacher_raw_bytes = 0
            teacher_compressed_bytes = 0
            if (use_kd or use_proxy) and self.communication_quantization_enabled:
                teacher_payload = self._build_quantized_general_payload()
                teacher_raw_bytes = teacher_payload.raw_nbytes
                teacher_compressed_bytes = teacher_payload.compressed_nbytes
                teacher = None
            elif use_kd or use_proxy:
                teacher_raw_bytes = estimate_state_dict_nbytes(self.general_model.state_dict())
                teacher_compressed_bytes = teacher_raw_bytes
            if ri == kd_warmup + 1:
                LOGGER.info("fedegsd round %d | KD warmup complete, enabling general-guided client adaptation", ri)

            LOGGER.info("fedegsd round %d | clients=%s | kd=%s | proxy=%s", ri, sids, use_kd, use_proxy)

            updates = [
                self.clients[c].train_local(
                    general_model=teacher,
                    general_model_payload=teacher_payload,
                )
                for c in sids
            ]
            proxy_anchor_state = self._aggregate_proxy_scope_state_dict(updates)
            ensemble = self._extract_ensemble_logits(updates)
            ds = self._distill_general_model(ensemble, ri, proxy_anchor_state=proxy_anchor_state)
            self._update_routing_thresholds(updates)

            ee = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsd-expert")
            ge = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsd-general")
            re = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsd-routed")

            al = sum(u.loss for u in updates) / max(len(updates), 1)
            uplink_raw_bytes = sum(int(u.raw_upload_bytes) for u in updates)
            uplink_compressed_bytes = sum(int(u.compressed_upload_bytes or u.raw_upload_bytes) for u in updates)
            a = re["aggregate"]; m = re["macro"]
            cp = self._build_compute_profile(self.expert_flops, self.general_flops, a["invocation_rate"], "routed")

            rm = RoundMetrics(round_idx=ri, avg_client_loss=al, routed_accuracy=a["accuracy"],
                              hard_accuracy=a["hard_recall"], invocation_rate=a["invocation_rate"],
                              local_accuracy=m["accuracy"], compute_savings=cp["savings_ratio"])

            LOGGER.info("fedegsd round %d | loss=%.4f | distill=%.4f | feature=%.4f | anchor=%.4f | proxy_anchor=%.4f | routed=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f | uplink=%d/%d | downlink=%d/%d",
                         ri, al, ds["total_loss"], ds["feature_loss"], ds["anchor_loss"], ds["proxy_anchor_loss"], a["accuracy"], ee["aggregate"]["accuracy"],
                         ge["aggregate"]["accuracy"], a["hard_recall"], a["invocation_rate"],
                         uplink_compressed_bytes, uplink_raw_bytes, teacher_compressed_bytes, teacher_raw_bytes)

            if self.writer:
                self.writer.add_scalar("expert_loss/fedegsd", al, ri)
                self.writer.add_scalar("distill_loss/fedegsd", ds["total_loss"], ri)
                self.writer.add_scalar("distill_feature_loss/fedegsd", ds["feature_loss"], ri)
                self.writer.add_scalar("distill_anchor_loss/fedegsd", ds["anchor_loss"], ri)
                self.writer.add_scalar("distill_proxy_anchor_loss/fedegsd", ds["proxy_anchor_loss"], ri)
                if uplink_raw_bytes > 0:
                    self.writer.add_scalar("comm/uplink_ratio_fedegsd", uplink_compressed_bytes / uplink_raw_bytes, ri)
                if teacher_raw_bytes > 0:
                    self.writer.add_scalar("comm/downlink_ratio_fedegsd", teacher_compressed_bytes / teacher_raw_bytes, ri)
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
