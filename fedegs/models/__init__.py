from typing import Optional

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .small_cnn import SmallCNN
from .width_scalable_resnet import (
    WidthScalableResNet,
    apply_expert_delta_to_general,
    average_weighted_deltas,
    estimate_model_flops,
    get_expert_state_dict,
    get_num_expert_blocks,
    load_expert_state_dict,
    model_memory_mb,
)


LOGGER = logging.getLogger(__name__)


def build_model(
    architecture: str,
    num_classes: int,
    width_factor: float = 1.0,
    base_channels: int = 32,
):
    normalized = architecture.lower()
    if normalized in {"width_scalable_resnet18", "width_scalable_resnet", "resnet18_width_scalable"}:
        return WidthScalableResNet(width_factor=width_factor, num_classes=num_classes)
    if normalized in {"small_cnn", "smallcnn"}:
        return SmallCNN(num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def build_baseline_model(config):
    architecture = config.model.baseline_architecture or config.model.architecture
    width_factor: Optional[float] = config.model.baseline_width
    if width_factor is None:
        width_factor = config.model.general_width
    return build_model(
        architecture=architecture,
        num_classes=config.model.num_classes,
        width_factor=width_factor,
        base_channels=config.model.baseline_base_channels,
    )


def build_teacher_model(num_classes: int = 10) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_teacher_checkpoint(model: nn.Module, checkpoint_or_state) -> None:
    state = checkpoint_or_state
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]

    if not isinstance(state, dict):
        raise ValueError("Teacher checkpoint must be a state dict or contain one under 'state_dict' or 'model'.")

    cleaned_state = {}
    for key, value in state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        cleaned_state[normalized_key] = value

    target_state = model.state_dict()
    compatible_state = {}
    for key, target_tensor in target_state.items():
        if key not in cleaned_state:
            continue
        source_tensor = cleaned_state[key]
        if key == "conv1.weight" and source_tensor.ndim == 4 and source_tensor.shape != target_tensor.shape:
            source_tensor = _resize_conv_kernel(source_tensor, target_tensor.shape[-2:])
        if source_tensor.shape == target_tensor.shape:
            compatible_state[key] = source_tensor

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if missing:
        LOGGER.warning("Teacher checkpoint missing keys during compatible load: %s", missing)
    if unexpected:
        LOGGER.warning("Teacher checkpoint had unexpected keys during compatible load: %s", unexpected)


def _resize_conv_kernel(weight: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if weight.shape[-2:] == target_hw:
        return weight
    out_channels, in_channels, _, _ = weight.shape
    resized = F.interpolate(
        weight.reshape(out_channels * in_channels, 1, weight.shape[-2], weight.shape[-1]),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(out_channels, in_channels, target_hw[0], target_hw[1])


__all__ = [
    "SmallCNN",
    "WidthScalableResNet",
    "build_model",
    "build_baseline_model",
    "build_teacher_model",
    "load_teacher_checkpoint",
    "apply_expert_delta_to_general",
    "average_weighted_deltas",
    "estimate_model_flops",
    "get_expert_state_dict",
    "get_num_expert_blocks",
    "load_expert_state_dict",
    "model_memory_mb",
]
