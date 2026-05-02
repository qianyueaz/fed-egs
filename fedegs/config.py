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
    public_per_class_ratio: float = 0.0
    # Dirichlet partition
    dirichlet_alpha: float = 0.5
    dirichlet_min_client_size: int = 32
    quantity_skew_sigma: float = 1.0
    quantity_min_size: int = 32
    # Long-tail partition
    longtail_major_classes: int = 3
    longtail_major_ratio: float = 0.9


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
    fedala_rand_percent: float = 0.8
    fedala_layer_idx: int = 0
    fedala_eta: float = 1.0
    fedala_init_alpha: float = 0.5
    fedala_convergence_threshold: float = 0.01
    fedala_start_phase_epochs: int = 5
    fedala_adaptation_epochs: int = 1
    confree_alpha: float = 0.5
    confree_solver_iterations: int = 80
    confree_solver_lr: float = 0.05
    pfedfda_beta: float = 0.5
    pfedfda_beta_search: bool = True
    pfedfda_beta_candidates: int = 11
    pfedfda_local_beta: bool = False
    pfedfda_eps: float = 0.0001
    pfedfda_covariance_ridge: float = 0.0001
    pfedfda_min_class_samples: int = 2
    pfedfda_refresh_eval_stats: bool = True
    distill_epochs: int = 1
    distill_lr: float = 0.001
    distill_temperature: float = 2.0
    public_distill_epochs: int = 1
    distill_feature_weight: float = 0.0
    distill_feature_normalize: bool = True
    enable_public_in_domain_anchor: bool = True
    public_ce_weight: float = 1.0
    public_logit_align_weight: float = 0.8
    external_logit_align_weight: float = 1.0
    external_logit_align_warmup_rounds: int = 0
    external_logit_align_ramp_rounds: int = 1
    public_teacher_aggregation: str = "avg_logits"
    external_teacher_aggregation: str = "avg_logits"
    public_teacher_confidence_threshold: float = 0.60
    public_teacher_weight_power: float = 2.0
    public_teacher_topk: int = 3
    public_teacher_use_temporal: bool = False
    external_teacher_use_temporal: bool = False
    public_teacher_use_hard_mining: bool = False
    client_kd_weight: float = 0.5
    client_kd_temperature: float = 3.0
    client_feature_hint_weight: float = 0.3
    client_hard_weight: float = 1.0
    client_hard_focus_power: float = 1.5
    client_hard_margin_threshold: float = 0.20
    client_hard_extra_weight: float = 0.6
    hard_subset_ratio: float = 0.25
    hard_subset_kd_boost: float = 0.5
    hard_subset_hint_boost: float = 0.5
    expert_refresh_epochs: int = 1
    expert_refresh_lr_scale: float = 0.5
    expert_refresh_gain_threshold: float = 0.0
    freeze_expert_refresh_epochs: int = 2
    freeze_client_kd_weight: float = 1.3
    freeze_client_feature_hint_weight: float = 0.6
    freeze_local_lr_scale: float = 0.7
    feature_align_weight: float = 1.0
    logit_align_weight: float = 1.0
    relation_align_weight: float = 0.0
    prototype_align_weight: float = 0.0
    route_prior_align_weight: float = 0.0
    feature_noise_std: float = 0.01
    min_uncertainty_weight: float = 0.2
    min_client_reliability: float = 0.05
    prototype_momentum: float = 0.7
    general_anchor_weight: float = 0.7
    general_warmup_rounds: int = 20
    general_distill_ramp_rounds: int = 20
    general_distill_max_scale: float = 0.25
    general_freeze_patience: int = 15
    general_init_from_teacher: bool = False
    general_pretrain_on_public: bool = False
    general_pretrain_epochs: int = 10
    general_pretrain_lr: float = 0.01
    general_pretrain_imagenet_init: bool = False
    general_enabled: bool = True
    communication_quantization_enabled: bool = False
    communication_quantization_bits: int = 8
    proxy_enabled: bool = False
    proxy_trainable_scopes: List[str] = field(default_factory=lambda: ["classifier", "fc", "backbone.layer4.1", "layer4.1"])
    proxy_temperature: float = 3.0
    proxy_ce_weight: float = 1.0
    proxy_kd_weight: float = 0.5
    expert_proxy_kd_weight: float = 0.5
    proxy_fisher_enabled: bool = False
    proxy_fisher_estimation_batches: int = 0
    server_proxy_anchor_weight: float = 0.0
    server_fisher_anchor_weight: float = 0.0
    general_distill_mode: str = "route_aware"
    fallback_weight_max: float = 3.0
    fallback_weight_confidence: float = 0.45
    fallback_weight_route_score: float = 0.35
    fallback_weight_disagreement: float = 0.20
    teacher_topk: int = 4
    proxy_fallback_ratio_mode: str = "holdout_advantage"
    min_proxy_fallback_ratio: float = 0.01
    max_proxy_fallback_ratio: float = 0.20
    proxy_fallback_real_invocation_ema_weight: float = 0.7
    proxy_fallback_invocation_ema_decay: float = 0.8
    enable_fallback_oversampling: bool = False
    distill_dataset: str = "cifar100"
    distill_dataset_root: str = "./data"
    distill_max_samples: int = 0
    uncertainty_alpha_min: float = 0.2
    uncertainty_alpha_max: float = 0.8
    calibration_ratio: float = 0.1
    calibration_min_samples: int = 16
    calibration_max_samples: int = 0
    router_validation_ratio: float = 0.5
    distill_gradient_clip_norm: float = 1.0
    teacher_bank_staleness_decay: float = 0.99
    teacher_bank_max_staleness: int = 0
    teacher_selection_mode: str = "mean_confidence"
    enable_general_ema_anchor: bool = True
    dkdr_reliability_center: float = 0.5
    dkdr_beta_temperature: float = 0.15
    dkdr_mu: float = 0.5
    rad_teacher_reliability_beta: float = 2.0
    rad_route_demand_mode: str = "soft"
    rad_route_focus_alpha: float = 1.0
    rad_focus_max: float = 3.0
    rad_label_correction_gamma: float = 0.5
    rad_label_correction_max: float = 0.7
    risk_predictor_epochs: int = 40
    risk_predictor_lr: float = 0.05
    risk_predictor_weight_decay: float = 0.0
    risk_predictor_hidden_dim: int = 32
    risk_predictor_dropout: float = 0.1
    risk_predictor_retrain_on_load: bool = False
    risk_predictor_hard_negative_enabled: bool = False
    risk_predictor_hard_negative_quantile: float = 0.9
    risk_predictor_hard_negative_weight: float = 1.25
    risk_predictor_hard_negative_warmup_epochs: int = 40
    risk_predictor_tta_enabled: bool = False
    route_min_gain: float = 0.0
    route_gain_filter_min_invoked: int = 10
    route_gain_filter_require_positive_net: bool = True
    route_max_invocation_when_general_worse: float = 0.0
    route_disable_when_no_gain: bool = True
    router_group_mode: str = "none"
    router_group_filter_strategy: str = "blocklist"
    router_group_min_support: int = 20
    router_group_min_invoked: int = 5
    router_group_require_positive_net: bool = True
    router_group_fallback_to_client: bool = True
    router_group_threshold_mode: str = "none"
    router_group_threshold_min_support: int = 20
    router_group_threshold_min_errors: int = 3
    router_group_threshold_min_predicted_positive: int = 3
    router_group_threshold_target_fpr: float = 0.10
    router_group_threshold_max_false_positive: int = 2
    router_group_threshold_boost: float = 0.0
    router_group_boost_min_support: int = 20
    router_group_boost_min_invoked: int = 5
    router_group_boost_max_net_rescue: float = 1.0
    router_group_boost_min_harm: int = 1
    fedasym_best_metric: str = "auto"
    general_deploy_ema_momentum: float = 0.90
    general_deploy_warmup_rounds: int = 80
    general_deploy_warmup_momentum: float = 0.35
    deploy_consistency_weight: float = 0.10
    deploy_consistency_warmup_rounds: int = 0
    deploy_consistency_ramp_rounds: int = 1
    expert_kd_weight: float = 0.0
    expert_kd_temperature: float = 3.0
    expert_kd_warmup_rounds: int = 10
    expert_kd_min_general_accuracy: float = 0.0
    expert_kd_confidence_threshold: float = 0.60
    expert_kd_margin_threshold: float = 0.08
    expert_kd_hard_boost: float = 0.40
    expert_kd_teacher_confidence_delta: float = 0.0
    expert_kd_teacher_margin_delta: float = 0.0
    expert_kd_student_confidence_ceiling: float = 1.0
    expert_kd_student_margin_ceiling: float = 1.0
    expert_kd_gate_power: float = 1.0
    expert_kd_target_coverage: float = 0.0
    expert_kd_adaptive_coverage_enabled: bool = False
    expert_kd_min_gate_ratio: float = 0.0
    expert_kd_gate_floor: float = 0.0
    expert_refresh_min_teacher_accuracy: float = 0.0
    expert_refresh_confidence_threshold: float = 0.60
    expert_refresh_margin_threshold: float = 0.08
    expert_refresh_logit_weight: float = 0.0
    expert_refresh_feature_hint_weight: float = 0.0
    expert_refresh_hard_boost: float = 0.0
    expert_refresh_target_coverage: float = 0.0
    expert_refresh_adaptive_coverage_enabled: bool = False
    expert_refresh_min_gate_ratio: float = 0.0
    expert_refresh_gate_floor: float = 0.0
    expert_personalization_weight: float = 0.0
    drel_alpha: float = 1.0
    drel_beta: float = 8.0
    lambda_ge: float = 1.0
    lambda_eg: float = 0.5
    general_head_lr: float = 0.001
    drel_confidence_threshold: float = 0.6
    reliability_ema_momentum: float = 0.8
    reliability_accuracy_weight: float = 0.7
    reliability_alpha: float = 1.5
    teacher_consistency_weight: float = 8.0
    teacher_temporal_buffer_size: int = 5
    teacher_temporal_momentum: float = 0.35
    selective_proxy_ratio: float = 1.0
    general_light_update_scope: str = "head_last_block"
    general_full_refresh_interval: int = 10
    general_light_distill_epochs: int = 1
    general_teacher_ema_momentum: float = 0.90
    temperature_calibration_enabled: bool = False
    temperature_calibration_frequency: int = 1
    temperature_calibration_min: float = 0.5
    temperature_calibration_max: float = 5.0
    temperature_calibration_candidates: int = 25
    temperature_calibration_momentum: float = 0.5
    hard_sample_topk: int = 3
    hard_sample_disagreement_threshold: float = 0.30
    hard_sample_entropy_threshold: float = 0.55
    hard_sample_weight_power: float = 1.5
    hard_sample_distill_boost: float = 1.25
    restore_best_checkpoint: bool = True
    save_best_checkpoint: bool = True
    best_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    eval_only_from_checkpoint: bool = False
    recalibrate_route_thresholds_on_load: bool = True
    device: str = "cuda"
    seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "width_scalable_resnet18"
    general_architecture: str = "teacher_resnet18"
    expert_architecture: str = "small_cnn"
    num_classes: int = 10
    general_width: float = 1.0
    general_base_channels: int = 32
    expert_width: float = 0.25
    expert_base_channels: int = 32
    knowledge_dim: int = 512
    baseline_architecture: str = "width_scalable_resnet18"
    baseline_width: Optional[float] = None
    baseline_base_channels: int = 32


@dataclass
class InferenceConfig:
    routing_policy: str = "dual_threshold_general_fallback"
    routing_selection_mode: str = "budget"
    confidence_threshold: float = 0.18
    high_threshold: float = 0.85
    low_threshold: float = 0.60
    route_margin_threshold: float = 0.03
    route_distance_threshold: float = 0.01
    target_expert_risk: float = 0.25
    route_score_confidence_weight: float = 0.50
    route_score_margin_weight: float = 0.35
    route_score_distance_weight: float = 0.20
    route_distance_normalization: bool = True
    min_confidence_threshold: float = 0.05
    max_confidence_threshold: float = 0.30
    min_margin_threshold: float = 0.00
    max_margin_threshold: float = 0.04
    personalized_threshold_step: float = 0.01
    personalized_margin_step: float = 0.005
    expert_priority_accuracy_floor: float = 0.55
    expert_priority_accuracy_target: float = 0.70
    expert_priority_reliability_floor: float = 0.35
    expert_priority_reliability_target: float = 0.55
    target_general_invocation_rate: float = 0.20
    simple_target_general_invocation_rate: float = 0.36
    complex_target_general_invocation_rate: float = 0.56
    simple_client_fallback_floor: float = 0.02
    complex_client_fallback_floor: float = 0.08
    public_teacher_gap_guard: float = 0.03
    route_hard_recall_tolerance: float = 0.005
    routing_search_radius: int = 3
    route_hard_priority_margin: float = 0.005
    route_hard_confidence_delta: float = 0.03
    route_hard_margin_delta: float = 0.02
    route_distance_std_multiplier: float = 1.5
    route_energy_threshold: float = -3.0
    min_energy_threshold: float = -6.0
    max_energy_threshold: float = -1.0
    personalized_energy_step: float = 0.20
    route_warmup_rounds: int = 0
    route_warmup_confidence_threshold: float = 0.40
    route_disable_distance_during_warmup: bool = True
    route_reliability_center: float = 0.60
    route_reliability_confidence_scale: float = 0.08
    route_reliability_margin_scale: float = 0.015
    route_reliability_energy_scale: float = 0.25
    route_reliability_invocation_scale: float = 0.12
    route_reliability_hard_bonus: float = 0.02
    route_hard_energy_delta: float = 0.15
    route_gain_threshold: float = 0.0
    min_route_gain_threshold: float = -1.0
    max_route_gain_threshold: float = 1.0
    route_gain_threshold_step: float = 0.05
    route_gain_search_warmup_rounds: int = 120
    route_min_invocation_rate_scale: float = 0.50
    route_gain_confidence_weight: float = 0.50
    route_gain_margin_weight: float = 0.35
    route_gain_energy_weight: float = 0.25
    route_gain_reliability_weight: float = 0.15
    route_gain_distance_weight: float = 0.20
    route_gain_prior_weight: float = 0.20
    route_gain_positive_margin: float = 0.0
    fusion_band: float = 0.0
    routing_holdout_ratio: float = 0.10
    routing_holdout_min_samples: int = 64
    routing_holdout_max_samples: int = 512
    routing_holdout_seed_offset: int = 17
    error_predictor_threshold: float = 0.5
    error_predictor_threshold_mode: str = "fixed"
    error_predictor_target_precision: float = 0.80
    error_predictor_target_fpr: float = 0.01
    error_predictor_max_false_positive: int = 5
    error_predictor_min_predicted_positive: int = 3
    error_predictor_disable_on_precision_fail: bool = True
    error_predictor_high_confidence_guard: float = 1.0
    error_predictor_use_wilson_lower_bound: bool = False
    error_predictor_wilson_z: float = 1.96
    router_diagnostics_enabled: bool = False
    router_diagnostics_min_samples: int = 20
    router_diagnostics_include_classes: bool = True
    router_diagnostics_confidence_bins: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 0.95])
    router_regret_diagnostics_enabled: bool = False
    router_candidate_diagnostics_enabled: bool = False
    router_candidate_rates: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    router_candidate_rate: float = 0.10
    router_candidate_tta_weight: float = 0.10
    router_candidate_min_score: float = 0.0
    router_candidate_disable_high_confidence_guard: bool = True
    router_candidate_confidence_bins_enabled: bool = False
    router_candidate_confidence_bin_rates: List[float] = field(default_factory=list)
    route_verifier_hidden_dim: int = 32
    route_verifier_dropout: float = 0.10
    route_verifier_epochs: int = 60
    route_verifier_lr: float = 0.001
    route_verifier_weight_decay: float = 0.0
    route_verifier_negative_weight: float = 2.0
    route_verifier_neutral_weight: float = 0.0
    route_verifier_threshold_mode: str = "harm_constrained"
    route_verifier_threshold: float = 0.5
    route_verifier_min_adopt_threshold: float = 0.0
    route_verifier_low_threshold_train_harm_ratio: float = 0.0
    route_verifier_low_threshold_train_harm_min_candidates: int = 10
    route_verifier_harm_lambda: float = 1.0
    route_verifier_adoption_mode: str = "fusion"
    route_verifier_confidence_bin_thresholds_enabled: bool = False
    route_verifier_bin_min_validation_adopted: int = 2
    route_fusion_alpha_source: str = "verifier"
    route_fusion_alpha_min: float = 0.35
    route_fusion_alpha_max: float = 0.85
    route_fusion_alpha_fixed: float = 0.50
    route_fusion_confidence_alpha_enabled: bool = False
    route_fusion_confidence_alpha_max_values: List[float] = field(default_factory=list)
    router_max_harm_rate: float = 0.01
    router_min_rescue_harm_ratio: float = 1.0
    router_min_adopted: int = 3
    router_min_rescue: int = 1
    route_verifier_min_validation_adopted: int = 0
    route_verifier_min_validation_net: float = 0.0
    route_verifier_min_validation_rescue_harm_ratio: float = 1.0
    route_verifier_disable_on_validation_fail: bool = True
    routing_error_min_threshold: float = 0.05
    routing_error_max_threshold: float = 0.95
    client_force_general_gap: float = 0.12
    general_reliability_threshold: float = 0.55
    general_reliability_confidence_weight: float = 0.45
    general_reliability_margin_weight: float = 0.25
    general_reliability_entropy_weight: float = 0.30
    routing_general_reliability_candidates: int = 9
    routing_general_reliability_min: float = 0.30
    routing_general_reliability_max: float = 0.90
    routing_veto_gain_threshold: float = 0.0
    routing_veto_reliability_threshold: float = 0.55
    routing_veto_min_holdout_samples: int = 64
    routing_veto_advantage_ema_decay: float = 0.7
    routing_veto_reliability_ema_decay: float = 0.7
    gain_confidence_bins: int = 6
    gain_margin_bins: int = 6
    gain_min_bucket_support: int = 8
    gain_min_class_support: int = 16


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
        config = _merge_dataclass(cls(), payload)
        if config.federated.compare_algorithms is None:
            config.federated.compare_algorithms = []
        return config

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
        "routing_threshold": (config.inference, "confidence_threshold"),
        "routing_policy": (config.inference, "routing_policy"),
        "high_threshold": (config.inference, "high_threshold"),
        "low_threshold": (config.inference, "low_threshold"),
        "route_distance_threshold": (config.inference, "route_distance_threshold"),
        "error_predictor_threshold": (config.inference, "error_predictor_threshold"),
        "error_predictor_threshold_mode": (config.inference, "error_predictor_threshold_mode"),
        "error_predictor_target_precision": (config.inference, "error_predictor_target_precision"),
        "error_predictor_target_fpr": (config.inference, "error_predictor_target_fpr"),
        "error_predictor_max_false_positive": (config.inference, "error_predictor_max_false_positive"),
        "error_predictor_min_predicted_positive": (config.inference, "error_predictor_min_predicted_positive"),
        "error_predictor_disable_on_precision_fail": (config.inference, "error_predictor_disable_on_precision_fail"),
        "error_predictor_high_confidence_guard": (config.inference, "error_predictor_high_confidence_guard"),
        "error_predictor_use_wilson_lower_bound": (config.inference, "error_predictor_use_wilson_lower_bound"),
        "error_predictor_wilson_z": (config.inference, "error_predictor_wilson_z"),
        "router_diagnostics_enabled": (config.inference, "router_diagnostics_enabled"),
        "router_diagnostics_min_samples": (config.inference, "router_diagnostics_min_samples"),
        "router_diagnostics_include_classes": (config.inference, "router_diagnostics_include_classes"),
        "router_regret_diagnostics_enabled": (config.inference, "router_regret_diagnostics_enabled"),
        "router_candidate_diagnostics_enabled": (config.inference, "router_candidate_diagnostics_enabled"),
        "router_candidate_rates": (config.inference, "router_candidate_rates"),
        "router_candidate_rate": (config.inference, "router_candidate_rate"),
        "router_candidate_tta_weight": (config.inference, "router_candidate_tta_weight"),
        "router_candidate_min_score": (config.inference, "router_candidate_min_score"),
        "router_candidate_disable_high_confidence_guard": (
            config.inference,
            "router_candidate_disable_high_confidence_guard",
        ),
        "router_candidate_confidence_bins_enabled": (
            config.inference,
            "router_candidate_confidence_bins_enabled",
        ),
        "router_candidate_confidence_bin_rates": (
            config.inference,
            "router_candidate_confidence_bin_rates",
        ),
        "route_verifier_hidden_dim": (config.inference, "route_verifier_hidden_dim"),
        "route_verifier_dropout": (config.inference, "route_verifier_dropout"),
        "route_verifier_epochs": (config.inference, "route_verifier_epochs"),
        "route_verifier_lr": (config.inference, "route_verifier_lr"),
        "route_verifier_weight_decay": (config.inference, "route_verifier_weight_decay"),
        "route_verifier_negative_weight": (config.inference, "route_verifier_negative_weight"),
        "route_verifier_neutral_weight": (config.inference, "route_verifier_neutral_weight"),
        "route_verifier_threshold_mode": (config.inference, "route_verifier_threshold_mode"),
        "route_verifier_threshold": (config.inference, "route_verifier_threshold"),
        "route_verifier_min_adopt_threshold": (config.inference, "route_verifier_min_adopt_threshold"),
        "route_verifier_low_threshold_train_harm_ratio": (
            config.inference,
            "route_verifier_low_threshold_train_harm_ratio",
        ),
        "route_verifier_low_threshold_train_harm_min_candidates": (
            config.inference,
            "route_verifier_low_threshold_train_harm_min_candidates",
        ),
        "route_verifier_harm_lambda": (config.inference, "route_verifier_harm_lambda"),
        "route_verifier_adoption_mode": (config.inference, "route_verifier_adoption_mode"),
        "route_verifier_confidence_bin_thresholds_enabled": (
            config.inference,
            "route_verifier_confidence_bin_thresholds_enabled",
        ),
        "route_verifier_bin_min_validation_adopted": (
            config.inference,
            "route_verifier_bin_min_validation_adopted",
        ),
        "route_fusion_alpha_source": (config.inference, "route_fusion_alpha_source"),
        "route_fusion_alpha_min": (config.inference, "route_fusion_alpha_min"),
        "route_fusion_alpha_max": (config.inference, "route_fusion_alpha_max"),
        "route_fusion_alpha_fixed": (config.inference, "route_fusion_alpha_fixed"),
        "route_fusion_confidence_alpha_enabled": (
            config.inference,
            "route_fusion_confidence_alpha_enabled",
        ),
        "route_fusion_confidence_alpha_max_values": (
            config.inference,
            "route_fusion_confidence_alpha_max_values",
        ),
        "router_max_harm_rate": (config.inference, "router_max_harm_rate"),
        "router_min_rescue_harm_ratio": (config.inference, "router_min_rescue_harm_ratio"),
        "router_min_adopted": (config.inference, "router_min_adopted"),
        "router_min_rescue": (config.inference, "router_min_rescue"),
        "route_verifier_min_validation_adopted": (
            config.inference,
            "route_verifier_min_validation_adopted",
        ),
        "route_verifier_min_validation_net": (config.inference, "route_verifier_min_validation_net"),
        "route_verifier_min_validation_rescue_harm_ratio": (
            config.inference,
            "route_verifier_min_validation_rescue_harm_ratio",
        ),
        "route_verifier_disable_on_validation_fail": (
            config.inference,
            "route_verifier_disable_on_validation_fail",
        ),
        "calibration_ratio": (config.federated, "calibration_ratio"),
        "calibration_max_samples": (config.federated, "calibration_max_samples"),
        "router_validation_ratio": (config.federated, "router_validation_ratio"),
        "risk_predictor_epochs": (config.federated, "risk_predictor_epochs"),
        "risk_predictor_lr": (config.federated, "risk_predictor_lr"),
        "risk_predictor_hidden_dim": (config.federated, "risk_predictor_hidden_dim"),
        "risk_predictor_dropout": (config.federated, "risk_predictor_dropout"),
        "risk_predictor_retrain_on_load": (config.federated, "risk_predictor_retrain_on_load"),
        "risk_predictor_hard_negative_enabled": (
            config.federated,
            "risk_predictor_hard_negative_enabled",
        ),
        "risk_predictor_hard_negative_quantile": (
            config.federated,
            "risk_predictor_hard_negative_quantile",
        ),
        "risk_predictor_hard_negative_weight": (
            config.federated,
            "risk_predictor_hard_negative_weight",
        ),
        "risk_predictor_hard_negative_warmup_epochs": (
            config.federated,
            "risk_predictor_hard_negative_warmup_epochs",
        ),
        "risk_predictor_tta_enabled": (config.federated, "risk_predictor_tta_enabled"),
        "route_min_gain": (config.federated, "route_min_gain"),
        "route_gain_filter_min_invoked": (config.federated, "route_gain_filter_min_invoked"),
        "route_gain_filter_require_positive_net": (config.federated, "route_gain_filter_require_positive_net"),
        "route_disable_when_no_gain": (config.federated, "route_disable_when_no_gain"),
        "router_group_mode": (config.federated, "router_group_mode"),
        "router_group_filter_strategy": (config.federated, "router_group_filter_strategy"),
        "router_group_min_support": (config.federated, "router_group_min_support"),
        "router_group_min_invoked": (config.federated, "router_group_min_invoked"),
        "router_group_require_positive_net": (config.federated, "router_group_require_positive_net"),
        "router_group_fallback_to_client": (config.federated, "router_group_fallback_to_client"),
        "router_group_threshold_mode": (config.federated, "router_group_threshold_mode"),
        "router_group_threshold_min_support": (config.federated, "router_group_threshold_min_support"),
        "router_group_threshold_min_errors": (config.federated, "router_group_threshold_min_errors"),
        "router_group_threshold_min_predicted_positive": (
            config.federated,
            "router_group_threshold_min_predicted_positive",
        ),
        "router_group_threshold_target_fpr": (config.federated, "router_group_threshold_target_fpr"),
        "router_group_threshold_max_false_positive": (
            config.federated,
            "router_group_threshold_max_false_positive",
        ),
        "router_group_threshold_boost": (config.federated, "router_group_threshold_boost"),
        "router_group_boost_min_support": (config.federated, "router_group_boost_min_support"),
        "router_group_boost_min_invoked": (config.federated, "router_group_boost_min_invoked"),
        "router_group_boost_max_net_rescue": (config.federated, "router_group_boost_max_net_rescue"),
        "router_group_boost_min_harm": (config.federated, "router_group_boost_min_harm"),
        "best_checkpoint_path": (config.federated, "best_checkpoint_path"),
        "load_checkpoint_path": (config.federated, "load_checkpoint_path"),
        "eval_only_from_checkpoint": (config.federated, "eval_only_from_checkpoint"),
        "recalibrate_route_thresholds_on_load": (config.federated, "recalibrate_route_thresholds_on_load"),
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
