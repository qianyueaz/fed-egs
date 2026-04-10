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
    if bool(config.federated.baseline_match_general_capacity):
        architecture = config.model.architecture
        width_factor = config.model.general_width
        base_channels = config.model.expert_base_channels
    else:
        architecture = config.model.baseline_architecture or config.model.architecture
        width_factor = config.model.baseline_width
        if width_factor is None:
            width_factor = config.model.general_width
        base_channels = config.model.baseline_base_channels
    return build_model(
        architecture=architecture,
        num_classes=config.model.num_classes,
        width_factor=width_factor,
        base_channels=base_channels,
    )


def build_teacher_model(num_classes: int = 10, pretrained_imagenet: bool = False) -> nn.Module:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained_imagenet else None)
    # Adapt for CIFAR-10 small images: 3x3 conv1 (no 7x7), remove maxpool
    original_conv1_weight = model.conv1.weight.data if pretrained_imagenet else None
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained_imagenet and original_conv1_weight is not None:
        # Center-crop the 7x7 ImageNet kernel to 3x3 for CIFAR adaptation
        model.conv1.weight.data.copy_(original_conv1_weight[:, :, 2:5, 2:5])
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

    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    cleaned_state = {}
    for key, value in state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if ".downsample." in normalized_key:
            shortcut_key = normalized_key.replace(".downsample.", ".shortcut.")
            if shortcut_key in target_keys:
                normalized_key = shortcut_key
        cleaned_state[normalized_key] = value

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


def initialize_model_from_teacher(
    model: nn.Module,
    num_classes: int,
    checkpoint_or_state=None,
    pretrained_imagenet: bool = False,
) -> bool:
    if checkpoint_or_state is None and not pretrained_imagenet:
        return False

    if isinstance(model, WidthScalableResNet):
        source_model = WidthScalableResNet(width_factor=1.0, num_classes=num_classes)
        if checkpoint_or_state is not None:
            load_teacher_checkpoint(source_model, checkpoint_or_state)
        else:
            imagenet_teacher = build_teacher_model(num_classes=num_classes, pretrained_imagenet=True)
            load_teacher_checkpoint(source_model, imagenet_teacher.state_dict())

        if abs(float(model.width_factor) - 1.0) < 1e-8:
            model.load_state_dict(source_model.state_dict())
        else:
            model.load_state_dict(get_expert_state_dict(source_model, model, block_index=0))
        return True

    if isinstance(model, SmallCNN):
        LOGGER.warning("Skipping teacher/ImageNet initialization for SmallCNN baseline; no compatible loader is defined.")
        return False

    if checkpoint_or_state is not None:
        load_teacher_checkpoint(model, checkpoint_or_state)
    else:
        imagenet_teacher = build_teacher_model(num_classes=num_classes, pretrained_imagenet=True)
        load_teacher_checkpoint(model, imagenet_teacher.state_dict())
    return True


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
    "initialize_model_from_teacher",
    "load_teacher_checkpoint",
    "apply_expert_delta_to_general",
    "average_weighted_deltas",
    "estimate_model_flops",
    "get_expert_state_dict",
    "get_num_expert_blocks",
    "load_expert_state_dict",
    "model_memory_mb",
]
