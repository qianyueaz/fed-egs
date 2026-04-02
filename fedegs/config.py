from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data"
    partition_strategy: str = "difficulty_skewed"
    num_clients: int = 20
    simple_clients: int = 10
    complex_clients: int = 10
    hard_ratio: float = 0.30
    simple_easy_ratio: float = 0.90
    complex_easy_ratio: float = 0.30
    batch_size: int = 128
    num_workers: int = 0
    difficulty_checkpoint: Optional[str] = None
    cache_dir: str = "./artifacts/cache"
    partition_cache_name: str = "client_partitions.json"
    public_dataset_size: int = 1000
    public_split_strategy: str = "random"


@dataclass
class FederatedConfig:
    server_algorithm: str = "fedegs"
    client_algorithm: str = "sgd_expert_local"
    compare_algorithms: List[str] = field(default_factory=lambda: ["fedavg", "fedprox"])
    sequential_simulation: bool = True
    rounds: int = 10
    clients_per_round: int = 5
    local_epochs: int = 1
    local_lr: float = 0.01
    local_momentum: float = 0.9
    local_weight_decay: float = 5e-4
    prox_mu: float = 0.001
    distill_epochs: int = 1
    distill_lr: float = 0.001
    distill_temperature: float = 2.0
    device: str = "cuda"
    seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "width_scalable_resnet18"
    num_classes: int = 10
    general_width: float = 1.0
    expert_width: float = 0.25
    expert_base_channels: int = 32
    baseline_architecture: str = "width_scalable_resnet18"
    baseline_width: Optional[float] = None
    baseline_base_channels: int = 32


@dataclass
class InferenceConfig:
    routing_policy: str = "dual_threshold_general_fallback"
    high_threshold: float = 0.85
    low_threshold: float = 0.60


@dataclass
class ExperimentConfig:
    experiment_name: str = "fedegs_cifar10_default"
    output_dir: str = "./artifacts"
    run_name: str = ""
    run_timestamp: str = ""
    log_dir: str = "./artifacts/logs"
    tensorboard_dir: str = "./artifacts/tensorboard"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dataset.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
        return _merge_dataclass(cls(), payload)

    def dump_yaml(self, path: str) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=False), encoding="utf-8")


def apply_cli_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    for key, value in overrides.items():
        if value is None:
            continue
        target, attr = _resolve_override_target(config, key)
        setattr(target, attr, value)
    return config


def build_runtime_paths(config: ExperimentConfig, run_name: str, run_timestamp: str) -> ExperimentConfig:
    base = Path(config.output_dir)
    config.run_name = run_name
    config.run_timestamp = run_timestamp
    config.log_dir = str(base / "logs" / run_timestamp[:8])
    config.tensorboard_dir = str(base / "tensorboard" / config.experiment_name / run_name)
    config.dataset.cache_dir = str(base / "cache")
    return config


def _resolve_override_target(config: ExperimentConfig, key: str):
    mapping = {
        "data_root": (config.dataset, "root"),
        "rounds": (config.federated, "rounds"),
        "clients_per_round": (config.federated, "clients_per_round"),
        "local_epochs": (config.federated, "local_epochs"),
        "batch_size": (config.dataset, "batch_size"),
        "device": (config.federated, "device"),
        "difficulty_checkpoint": (config.dataset, "difficulty_checkpoint"),
        "high_threshold": (config.inference, "high_threshold"),
        "low_threshold": (config.inference, "low_threshold"),
        "num_workers": (config.dataset, "num_workers"),
        "output_dir": (config, "output_dir"),
        "experiment_name": (config, "experiment_name"),
        "prox_mu": (config.federated, "prox_mu"),
    }
    return mapping[key]


def _merge_dataclass(instance: T, updates: Dict[str, Any]) -> T:
    for field_info in fields(instance):
        if field_info.name not in updates:
            continue
        current_value = getattr(instance, field_info.name)
        new_value = updates[field_info.name]
        if is_dataclass(current_value) and isinstance(new_value, dict):
            setattr(instance, field_info.name, _merge_dataclass(current_value, new_value))
        else:
            setattr(instance, field_info.name, new_value)
    return instance
