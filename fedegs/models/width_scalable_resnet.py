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
        self.feature_dim = widths[3]

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: List[nn.Module] = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        return features, self.classify_features(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


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


def _normalize_input_shape(input_shape: Tuple[int, int, int, int], batch_size: int) -> Tuple[int, int, int, int]:
    _, channels, height, width = input_shape
    return batch_size, channels, height, width


def _tensor_collection_numel(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_collection_numel(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_collection_numel(item) for item in value.values())
    return 0


def estimate_activation_memory_mb(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
) -> float:
    activation_numel = 0
    hooks = []

    def activation_hook(module: nn.Module, inputs, output) -> None:
        nonlocal activation_numel
        activation_numel += _tensor_collection_numel(output)

    for module in model.modules():
        if module is model:
            continue
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(activation_hook))

    was_training = model.training
    try:
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32
    dummy = torch.zeros(input_shape, device=device, dtype=dtype)
    with torch.no_grad():
        model.eval()
        model(dummy)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return activation_numel * 4 / (1024 ** 2)


def estimate_inference_memory_mb(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
) -> float:
    single_sample_shape = _normalize_input_shape(input_shape, 1)
    input_memory_mb = ((torch.zeros(single_sample_shape).numel() * 4) / (1024 ** 2)) * batch_size
    activation_memory_mb = estimate_activation_memory_mb(model, input_shape=single_sample_shape) * batch_size
    return model_memory_mb(model) + input_memory_mb + activation_memory_mb


def estimate_training_memory_mb(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    optimizer_name: str = "sgd",
) -> float:
    optimizer_multiplier = {
        "sgd": 0.0,
        "sgd_momentum": 1.0,
        "momentum": 1.0,
        "adam": 2.0,
        "adamw": 2.0,
    }.get(str(optimizer_name).lower(), 0.0)
    parameter_memory_mb = model_memory_mb(model)
    inference_memory_mb = estimate_inference_memory_mb(model, batch_size=batch_size, input_shape=input_shape)
    activation_and_input_mb = max(inference_memory_mb - parameter_memory_mb, 0.0)
    gradient_memory_mb = parameter_memory_mb
    optimizer_memory_mb = parameter_memory_mb * optimizer_multiplier
    return parameter_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_and_input_mb


def measure_peak_memory_mb(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    mode: str = "inference",
    optimizer_name: str = "sgd",
) -> float:
    try:
        first_param = next(model.parameters())
    except StopIteration:
        return 0.0

    device = first_param.device
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0

    effective_shape = _normalize_input_shape(input_shape, batch_size)
    dummy = torch.zeros(effective_shape, device=device, dtype=first_param.dtype)
    baseline_allocated = torch.cuda.memory_allocated(device)
    was_training = model.training
    state_backup = None
    optimizer = None

    def _optimizer_kwargs(name: str) -> Dict[str, float]:
        normalized = str(name).lower()
        if normalized in {"sgd_momentum", "momentum"}:
            return {"lr": 1e-3, "momentum": 0.9}
        if normalized in {"adam", "adamw"}:
            return {"lr": 1e-3}
        return {"lr": 1e-3}

    try:
        if mode == "train":
            state_backup = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        if mode == "train":
            model.train()
            if str(optimizer_name).lower() in {"adam", "adamw"}:
                optimizer = torch.optim.Adam(model.parameters(), **_optimizer_kwargs(optimizer_name))
            else:
                optimizer = torch.optim.SGD(model.parameters(), **_optimizer_kwargs(optimizer_name))
            optimizer.zero_grad(set_to_none=True)
            logits = model(dummy)
            targets = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                model(dummy)

        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        delta_mb = max(float(peak_allocated - baseline_allocated), 0.0) / (1024 ** 2)
        return model_memory_mb(model) + delta_mb
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if state_backup is not None:
            model.load_state_dict(state_backup)
        if was_training:
            model.train()
        else:
            model.eval()


def estimate_client_training_flops(
    model_flops: float,
    num_samples: int,
    local_epochs: int,
    backward_factor: float = 3.0,
) -> float:
    return float(model_flops) * max(float(backward_factor), 0.0) * max(int(num_samples), 0) * max(int(local_epochs), 0)


def estimate_model_flops(model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> float:
    flops = 0.0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal flops
        batch_size = inputs[0].shape[0]
        output_height, output_width = output.shape[-2:]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        flops += float(batch_size * output_height * output_width * module.out_channels * kernel_ops)

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal flops
        batch_size = inputs[0].shape[0]
        flops += float(batch_size * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model.eval()
        model(dummy)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return flops / max(float(input_shape[0]), 1.0)
