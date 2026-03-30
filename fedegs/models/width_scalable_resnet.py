import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_channels(base_channels: int, width_factor: float) -> int:
    return max(1, int(round(base_channels * width_factor)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class WidthScalableResNet(nn.Module):
    """ResNet18 variant whose channel widths scale with a width factor."""

    def __init__(self, width_factor: float = 1.0, num_classes: int = 10) -> None:
        super().__init__()
        self.width_factor = width_factor
        self.num_classes = num_classes

        widths = [_scaled_channels(c, width_factor) for c in (64, 128, 256, 512)]
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(widths[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(widths[3], blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: List[nn.Module] = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


@dataclass
class SliceSpec:
    name: str
    source_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]


def _slice_tensor(source: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:
    return source[slices].clone()


def get_num_expert_blocks(general_model: nn.Module, expert_model: nn.Module) -> int:
    general_out = general_model.conv1.weight.shape[0]
    expert_out = expert_model.conv1.weight.shape[0]
    if expert_out == 0 or general_out % expert_out != 0:
        raise ValueError("Expert width must divide general width into uniform channel blocks.")
    return general_out // expert_out


def _tensor_slices(general_tensor: torch.Tensor, expert_tensor: torch.Tensor, block_index: int) -> Tuple[slice, ...]:
    slices: List[slice] = []
    for general_dim, expert_dim in zip(general_tensor.shape, expert_tensor.shape):
        if general_dim == expert_dim:
            slices.append(slice(0, expert_dim))
            continue

        if expert_dim == 0 or general_dim % expert_dim != 0:
            raise ValueError(
                f"Cannot build expert block slice for tensor shape {tuple(general_tensor.shape)} -> {tuple(expert_tensor.shape)}"
            )

        num_blocks = general_dim // expert_dim
        effective_block = block_index % num_blocks
        start = effective_block * expert_dim
        slices.append(slice(start, start + expert_dim))
    return tuple(slices)


def get_expert_state_dict(
    general_model: nn.Module,
    expert_model: nn.Module,
    block_index: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract a subnet state dict by slicing a contiguous channel block from the supernet.

    Unlike the prefix-only variant, different clients can request different channel
    blocks. If the expert width is 0.25 and the general width is 1.0, block 0 maps
    to channels [0:25%), block 1 to [25%:50%), and so on.
    """
    general_state = general_model.state_dict()
    expert_state = expert_model.state_dict()
    sliced_state: Dict[str, torch.Tensor] = {}

    for key, expert_tensor in expert_state.items():
        general_tensor = general_state[key]
        if general_tensor.shape == expert_tensor.shape:
            sliced_state[key] = general_tensor.clone()
        else:
            slices = _tensor_slices(general_tensor, expert_tensor, block_index)
            sliced_state[key] = _slice_tensor(general_tensor, slices)

    return sliced_state


def load_expert_state_dict(general_model: nn.Module, expert_model: nn.Module, block_index: int = 0) -> None:
    expert_model.load_state_dict(get_expert_state_dict(general_model, expert_model, block_index=block_index))


def apply_expert_delta_to_general(
    general_model: nn.Module,
    aggregated_delta: Dict[str, torch.Tensor],
    expert_model: nn.Module,
    block_index: int = 0,
) -> None:
    """
    Apply an aggregated expert delta back onto the corresponding channel block.

    Each client expert updates a disjoint contiguous block of the general model.
    Different clients can therefore cover different quarters (or more generally,
    different blocks) of the supernet over time.
    """
    general_state = general_model.state_dict()
    expert_state = expert_model.state_dict()
    updated_state = copy.deepcopy(general_state)

    for key, delta in aggregated_delta.items():
        if key not in updated_state:
            continue
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue

        general_tensor = updated_state[key]
        expert_tensor = expert_state[key]
        delta = delta.to(device=general_tensor.device, dtype=general_tensor.dtype)

        if general_tensor.shape == delta.shape:
            updated_state[key] = general_tensor + delta
            continue

        slices = _tensor_slices(general_tensor, expert_tensor, block_index)
        general_tensor = general_tensor.clone()
        general_tensor[slices] = general_tensor[slices] + delta
        updated_state[key] = general_tensor

    general_model.load_state_dict(updated_state)


def state_dict_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta: Dict[str, torch.Tensor] = {}
    for key, tensor in new_state.items():
        if tensor.dtype.is_floating_point:
            delta[key] = tensor.detach().cpu() - old_state[key].detach().cpu()
    return delta


def average_weighted_deltas(weighted_deltas: Iterable[Tuple[float, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    total_weight = 0.0
    accumulator: Dict[str, torch.Tensor] = {}

    for weight, delta in weighted_deltas:
        total_weight += weight
        for key, tensor in delta.items():
            if key not in accumulator:
                accumulator[key] = tensor.clone() * weight
            else:
                accumulator[key] += tensor * weight

    if total_weight == 0:
        return accumulator

    for key in accumulator:
        accumulator[key] /= total_weight
    return accumulator


def model_memory_mb(model: nn.Module) -> float:
    params = sum(p.numel() for p in model.parameters())
    return params * 4 / (1024 ** 2)
