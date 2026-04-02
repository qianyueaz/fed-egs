from typing import Optional

from .small_cnn import SmallCNN
from .width_scalable_resnet import (
    WidthScalableResNet,
    apply_expert_delta_to_general,
    average_weighted_deltas,
    get_expert_state_dict,
    get_num_expert_blocks,
    load_expert_state_dict,
    model_memory_mb,
)


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


__all__ = [
    "SmallCNN",
    "WidthScalableResNet",
    "build_model",
    "build_baseline_model",
    "apply_expert_delta_to_general",
    "average_weighted_deltas",
    "get_expert_state_dict",
    "get_num_expert_blocks",
    "load_expert_state_dict",
    "model_memory_mb",
]
