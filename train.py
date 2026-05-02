import argparse
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fedegs.config import ExperimentConfig, apply_cli_overrides, build_runtime_paths
from fedegs.data import CIFAR10FederatedDataModule
from fedegs.evaluation import format_memory_table, save_metrics
from fedegs.experiment import run_experiment_suite
from fedegs.tensorboard import FederatedSummaryWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fed-EGS CIFAR-10 simulation")
    parser.add_argument("--config", default="configs/fedegs_cifar10.yaml")
    parser.add_argument("--experiment-name", dest="experiment_name", default=None)
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--difficulty-checkpoint", default=None)
    parser.add_argument("--routing-threshold", dest="routing_threshold", type=float, default=None)
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--route-distance-threshold", dest="route_distance_threshold", type=float, default=None)
    parser.add_argument("--error-predictor-threshold", dest="error_predictor_threshold", type=float, default=None)
    parser.add_argument("--error-predictor-threshold-mode", dest="error_predictor_threshold_mode", default=None)
    parser.add_argument("--error-predictor-target-precision", dest="error_predictor_target_precision", type=float, default=None)
    parser.add_argument("--error-predictor-target-fpr", dest="error_predictor_target_fpr", type=float, default=None)
    parser.add_argument("--error-predictor-max-false-positive", dest="error_predictor_max_false_positive", type=int, default=None)
    parser.add_argument("--error-predictor-min-predicted-positive", dest="error_predictor_min_predicted_positive", type=int, default=None)
    parser.add_argument("--allow-error-predictor-precision-fail", dest="error_predictor_disable_on_precision_fail", action="store_false")
    parser.add_argument("--error-predictor-high-confidence-guard", dest="error_predictor_high_confidence_guard", type=float, default=None)
    parser.add_argument("--error-predictor-use-wilson", dest="error_predictor_use_wilson_lower_bound", action="store_true")
    parser.add_argument("--error-predictor-wilson-z", dest="error_predictor_wilson_z", type=float, default=None)
    parser.add_argument("--router-diagnostics", dest="router_diagnostics_enabled", action="store_true")
    parser.add_argument("--router-diagnostics-min-samples", dest="router_diagnostics_min_samples", type=int, default=None)
    parser.add_argument("--disable-router-diagnostics-classes", dest="router_diagnostics_include_classes", action="store_false")
    parser.add_argument("--calibration-ratio", dest="calibration_ratio", type=float, default=None)
    parser.add_argument("--calibration-max-samples", dest="calibration_max_samples", type=int, default=None)
    parser.add_argument("--router-validation-ratio", dest="router_validation_ratio", type=float, default=None)
    parser.add_argument("--risk-predictor-epochs", dest="risk_predictor_epochs", type=int, default=None)
    parser.add_argument("--risk-predictor-lr", dest="risk_predictor_lr", type=float, default=None)
    parser.add_argument("--risk-predictor-hidden-dim", dest="risk_predictor_hidden_dim", type=int, default=None)
    parser.add_argument("--risk-predictor-dropout", dest="risk_predictor_dropout", type=float, default=None)
    parser.add_argument("--risk-predictor-hard-negative", dest="risk_predictor_hard_negative_enabled", action="store_true")
    parser.add_argument(
        "--disable-risk-predictor-hard-negative",
        dest="risk_predictor_hard_negative_enabled",
        action="store_false",
    )
    parser.add_argument(
        "--risk-predictor-hard-negative-quantile",
        dest="risk_predictor_hard_negative_quantile",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--risk-predictor-hard-negative-weight",
        dest="risk_predictor_hard_negative_weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--risk-predictor-hard-negative-warmup-epochs",
        dest="risk_predictor_hard_negative_warmup_epochs",
        type=int,
        default=None,
    )
    parser.add_argument("--route-min-gain", dest="route_min_gain", type=float, default=None)
    parser.add_argument("--route-gain-filter-min-invoked", dest="route_gain_filter_min_invoked", type=int, default=None)
    parser.add_argument("--allow-route-gain-filter-nonpositive-net", dest="route_gain_filter_require_positive_net", action="store_false")
    parser.add_argument("--disable-route-gain-filter", dest="route_disable_when_no_gain", action="store_false")
    parser.add_argument("--router-group-mode", dest="router_group_mode", default=None)
    parser.add_argument("--router-group-filter-strategy", dest="router_group_filter_strategy", default=None)
    parser.add_argument("--router-group-min-support", dest="router_group_min_support", type=int, default=None)
    parser.add_argument("--router-group-min-invoked", dest="router_group_min_invoked", type=int, default=None)
    parser.add_argument("--router-group-threshold-mode", dest="router_group_threshold_mode", default=None)
    parser.add_argument("--router-group-threshold-min-support", dest="router_group_threshold_min_support", type=int, default=None)
    parser.add_argument("--router-group-threshold-min-errors", dest="router_group_threshold_min_errors", type=int, default=None)
    parser.add_argument(
        "--router-group-threshold-min-predicted-positive",
        dest="router_group_threshold_min_predicted_positive",
        type=int,
        default=None,
    )
    parser.add_argument("--router-group-threshold-target-fpr", dest="router_group_threshold_target_fpr", type=float, default=None)
    parser.add_argument(
        "--router-group-threshold-max-false-positive",
        dest="router_group_threshold_max_false_positive",
        type=int,
        default=None,
    )
    parser.add_argument("--router-group-threshold-boost", dest="router_group_threshold_boost", type=float, default=None)
    parser.add_argument("--router-group-boost-min-support", dest="router_group_boost_min_support", type=int, default=None)
    parser.add_argument("--router-group-boost-min-invoked", dest="router_group_boost_min_invoked", type=int, default=None)
    parser.add_argument("--router-group-boost-max-net-rescue", dest="router_group_boost_max_net_rescue", type=float, default=None)
    parser.add_argument("--router-group-boost-min-harm", dest="router_group_boost_min_harm", type=int, default=None)
    parser.add_argument("--allow-router-group-nonpositive-net", dest="router_group_require_positive_net", action="store_false")
    parser.add_argument("--disable-router-group-fallback", dest="router_group_fallback_to_client", action="store_false")
    parser.add_argument("--retrain-router-from-checkpoint", dest="risk_predictor_retrain_on_load", action="store_true")
    parser.add_argument("--best-checkpoint-path", dest="best_checkpoint_path", default=None)
    parser.add_argument("--load-checkpoint-path", dest="load_checkpoint_path", default=None)
    parser.add_argument("--eval-only-from-checkpoint", dest="eval_only_from_checkpoint", action="store_true")
    parser.add_argument("--no-recalibrate-route-thresholds-on-load", dest="recalibrate_route_thresholds_on_load", action="store_false")
    parser.set_defaults(recalibrate_route_thresholds_on_load=None)
    parser.set_defaults(error_predictor_disable_on_precision_fail=None)
    parser.set_defaults(route_disable_when_no_gain=None)
    parser.set_defaults(route_gain_filter_require_positive_net=None)
    parser.set_defaults(router_group_require_positive_net=None)
    parser.set_defaults(router_group_fallback_to_client=None)
    parser.set_defaults(router_diagnostics_enabled=None)
    parser.set_defaults(router_diagnostics_include_classes=None)
    parser.set_defaults(risk_predictor_hard_negative_enabled=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prox-mu", dest="prox_mu", type=float, default=None)
    return parser


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "experiment"


def build_run_identity(experiment_name: str, algorithm_name: str) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_experiment = sanitize_name(experiment_name)
    safe_algorithm = sanitize_name(algorithm_name)
    return f"{safe_experiment}_{safe_algorithm}_{timestamp}", timestamp


def configure_logging(log_dir: str, run_name: str) -> Path:
    log_path = Path(log_dir) / f"{run_name}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_path


def _log_effective_config_summary(config: ExperimentConfig) -> None:
    dataset = {
        "name": config.dataset.name,
        "root": config.dataset.root,
        "partition_strategy": config.dataset.partition_strategy,
        "num_clients": config.dataset.num_clients,
        "batch_size": config.dataset.batch_size,
        "num_workers": config.dataset.num_workers,
        "public_dataset_size": config.dataset.public_dataset_size,
        "public_split_strategy": config.dataset.public_split_strategy,
        "public_per_class_ratio": config.dataset.public_per_class_ratio,
        "dirichlet_alpha": config.dataset.dirichlet_alpha,
        "dirichlet_min_client_size": config.dataset.dirichlet_min_client_size,
        "longtail_major_classes": config.dataset.longtail_major_classes,
        "longtail_major_ratio": config.dataset.longtail_major_ratio,
    }
    federated = {
        "server_algorithm": config.federated.server_algorithm,
        "client_algorithm": config.federated.client_algorithm,
        "compare_algorithms": config.federated.compare_algorithms,
        "rounds": config.federated.rounds,
        "clients_per_round": config.federated.clients_per_round,
        "local_epochs": config.federated.local_epochs,
        "local_lr": config.federated.local_lr,
        "local_momentum": config.federated.local_momentum,
        "local_weight_decay": config.federated.local_weight_decay,
        "prox_mu": config.federated.prox_mu,
        "distill_epochs": config.federated.distill_epochs,
        "distill_lr": config.federated.distill_lr,
        "distill_temperature": config.federated.distill_temperature,
        "public_distill_epochs": config.federated.public_distill_epochs,
        "expert_kd_temperature": config.federated.expert_kd_temperature,
        "dkdr_mu": config.federated.dkdr_mu,
        "device": config.federated.device,
        "seed": config.federated.seed,
    }
    router = {
        "calibration_ratio": config.federated.calibration_ratio,
        "calibration_min_samples": config.federated.calibration_min_samples,
        "calibration_max_samples": config.federated.calibration_max_samples,
        "router_validation_ratio": config.federated.router_validation_ratio,
        "risk_predictor_epochs": config.federated.risk_predictor_epochs,
        "risk_predictor_lr": config.federated.risk_predictor_lr,
        "risk_predictor_weight_decay": config.federated.risk_predictor_weight_decay,
        "risk_predictor_hidden_dim": config.federated.risk_predictor_hidden_dim,
        "risk_predictor_dropout": config.federated.risk_predictor_dropout,
        "risk_predictor_retrain_on_load": config.federated.risk_predictor_retrain_on_load,
        "risk_predictor_hard_negative_enabled": config.federated.risk_predictor_hard_negative_enabled,
        "risk_predictor_hard_negative_quantile": config.federated.risk_predictor_hard_negative_quantile,
        "risk_predictor_hard_negative_weight": config.federated.risk_predictor_hard_negative_weight,
        "risk_predictor_hard_negative_warmup_epochs": config.federated.risk_predictor_hard_negative_warmup_epochs,
        "route_disable_when_no_gain": config.federated.route_disable_when_no_gain,
        "route_min_gain": config.federated.route_min_gain,
        "route_gain_filter_min_invoked": config.federated.route_gain_filter_min_invoked,
        "route_gain_filter_require_positive_net": config.federated.route_gain_filter_require_positive_net,
        "router_group_mode": config.federated.router_group_mode,
        "router_group_filter_strategy": config.federated.router_group_filter_strategy,
        "router_group_min_support": config.federated.router_group_min_support,
        "router_group_min_invoked": config.federated.router_group_min_invoked,
        "router_group_require_positive_net": config.federated.router_group_require_positive_net,
        "router_group_fallback_to_client": config.federated.router_group_fallback_to_client,
        "router_group_threshold_mode": config.federated.router_group_threshold_mode,
        "router_group_threshold_min_support": config.federated.router_group_threshold_min_support,
        "router_group_threshold_min_errors": config.federated.router_group_threshold_min_errors,
        "router_group_threshold_min_predicted_positive": config.federated.router_group_threshold_min_predicted_positive,
        "router_group_threshold_target_fpr": config.federated.router_group_threshold_target_fpr,
        "router_group_threshold_max_false_positive": config.federated.router_group_threshold_max_false_positive,
        "router_group_threshold_boost": config.federated.router_group_threshold_boost,
        "router_group_boost_min_support": config.federated.router_group_boost_min_support,
        "router_group_boost_min_invoked": config.federated.router_group_boost_min_invoked,
        "router_group_boost_max_net_rescue": config.federated.router_group_boost_max_net_rescue,
        "router_group_boost_min_harm": config.federated.router_group_boost_min_harm,
        "routing_policy": config.inference.routing_policy,
        "error_predictor_threshold": config.inference.error_predictor_threshold,
        "error_predictor_threshold_mode": config.inference.error_predictor_threshold_mode,
        "error_predictor_target_precision": config.inference.error_predictor_target_precision,
        "error_predictor_target_fpr": config.inference.error_predictor_target_fpr,
        "error_predictor_max_false_positive": config.inference.error_predictor_max_false_positive,
        "error_predictor_min_predicted_positive": config.inference.error_predictor_min_predicted_positive,
        "error_predictor_disable_on_precision_fail": config.inference.error_predictor_disable_on_precision_fail,
        "error_predictor_use_wilson_lower_bound": config.inference.error_predictor_use_wilson_lower_bound,
        "error_predictor_wilson_z": config.inference.error_predictor_wilson_z,
        "error_predictor_high_confidence_guard": config.inference.error_predictor_high_confidence_guard,
        "router_diagnostics_enabled": config.inference.router_diagnostics_enabled,
        "router_diagnostics_min_samples": config.inference.router_diagnostics_min_samples,
        "router_diagnostics_include_classes": config.inference.router_diagnostics_include_classes,
    }
    model = {
        "general_architecture": config.model.general_architecture,
        "expert_architecture": config.model.expert_architecture,
        "num_classes": config.model.num_classes,
        "general_width": config.model.general_width,
        "general_base_channels": config.model.general_base_channels,
        "expert_width": config.model.expert_width,
        "expert_base_channels": config.model.expert_base_channels,
        "baseline_architecture": config.model.baseline_architecture,
        "baseline_width": config.model.baseline_width,
        "baseline_base_channels": config.model.baseline_base_channels,
    }
    rad = {
        "teacher_bank_staleness_decay": config.federated.teacher_bank_staleness_decay,
        "teacher_bank_max_staleness": config.federated.teacher_bank_max_staleness,
        "rad_teacher_reliability_beta": config.federated.rad_teacher_reliability_beta,
        "rad_route_demand_mode": config.federated.rad_route_demand_mode,
        "rad_route_focus_alpha": config.federated.rad_route_focus_alpha,
        "rad_focus_max": config.federated.rad_focus_max,
        "rad_label_correction_gamma": config.federated.rad_label_correction_gamma,
        "rad_label_correction_max": config.federated.rad_label_correction_max,
    }

    logging.info("Effective config summary begins")
    for section_name, section in (
        ("dataset", dataset),
        ("federated", federated),
        ("router", router),
        ("model", model),
        ("rad", rad),
    ):
        logging.info("Effective config | %s=%s", section_name, json.dumps(section, sort_keys=True))
    logging.info("Effective config summary ends")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_file(args.config)
    overrides = {
        "experiment_name": args.experiment_name,
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "rounds": args.rounds,
        "clients_per_round": args.clients_per_round,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "difficulty_checkpoint": args.difficulty_checkpoint,
        "routing_threshold": args.routing_threshold,
        "high_threshold": args.high_threshold,
        "low_threshold": args.low_threshold,
        "route_distance_threshold": args.route_distance_threshold,
        "error_predictor_threshold": args.error_predictor_threshold,
        "error_predictor_threshold_mode": args.error_predictor_threshold_mode,
        "error_predictor_target_precision": args.error_predictor_target_precision,
        "error_predictor_target_fpr": args.error_predictor_target_fpr,
        "error_predictor_max_false_positive": args.error_predictor_max_false_positive,
        "error_predictor_min_predicted_positive": args.error_predictor_min_predicted_positive,
        "error_predictor_disable_on_precision_fail": args.error_predictor_disable_on_precision_fail,
        "error_predictor_high_confidence_guard": args.error_predictor_high_confidence_guard,
        "error_predictor_use_wilson_lower_bound": (
            args.error_predictor_use_wilson_lower_bound if args.error_predictor_use_wilson_lower_bound else None
        ),
        "error_predictor_wilson_z": args.error_predictor_wilson_z,
        "router_diagnostics_enabled": args.router_diagnostics_enabled,
        "router_diagnostics_min_samples": args.router_diagnostics_min_samples,
        "router_diagnostics_include_classes": args.router_diagnostics_include_classes,
        "calibration_ratio": args.calibration_ratio,
        "calibration_max_samples": args.calibration_max_samples,
        "router_validation_ratio": args.router_validation_ratio,
        "risk_predictor_epochs": args.risk_predictor_epochs,
        "risk_predictor_lr": args.risk_predictor_lr,
        "risk_predictor_hidden_dim": args.risk_predictor_hidden_dim,
        "risk_predictor_dropout": args.risk_predictor_dropout,
        "risk_predictor_hard_negative_enabled": args.risk_predictor_hard_negative_enabled,
        "risk_predictor_hard_negative_quantile": args.risk_predictor_hard_negative_quantile,
        "risk_predictor_hard_negative_weight": args.risk_predictor_hard_negative_weight,
        "risk_predictor_hard_negative_warmup_epochs": args.risk_predictor_hard_negative_warmup_epochs,
        "route_min_gain": args.route_min_gain,
        "route_gain_filter_min_invoked": args.route_gain_filter_min_invoked,
        "route_gain_filter_require_positive_net": args.route_gain_filter_require_positive_net,
        "route_disable_when_no_gain": args.route_disable_when_no_gain,
        "router_group_mode": args.router_group_mode,
        "router_group_filter_strategy": args.router_group_filter_strategy,
        "router_group_min_support": args.router_group_min_support,
        "router_group_min_invoked": args.router_group_min_invoked,
        "router_group_threshold_mode": args.router_group_threshold_mode,
        "router_group_threshold_min_support": args.router_group_threshold_min_support,
        "router_group_threshold_min_errors": args.router_group_threshold_min_errors,
        "router_group_threshold_min_predicted_positive": args.router_group_threshold_min_predicted_positive,
        "router_group_threshold_target_fpr": args.router_group_threshold_target_fpr,
        "router_group_threshold_max_false_positive": args.router_group_threshold_max_false_positive,
        "router_group_threshold_boost": args.router_group_threshold_boost,
        "router_group_boost_min_support": args.router_group_boost_min_support,
        "router_group_boost_min_invoked": args.router_group_boost_min_invoked,
        "router_group_boost_max_net_rescue": args.router_group_boost_max_net_rescue,
        "router_group_boost_min_harm": args.router_group_boost_min_harm,
        "router_group_require_positive_net": args.router_group_require_positive_net,
        "router_group_fallback_to_client": args.router_group_fallback_to_client,
        "risk_predictor_retrain_on_load": args.risk_predictor_retrain_on_load if args.risk_predictor_retrain_on_load else None,
        "best_checkpoint_path": args.best_checkpoint_path,
        "load_checkpoint_path": args.load_checkpoint_path,
        "eval_only_from_checkpoint": args.eval_only_from_checkpoint if args.eval_only_from_checkpoint else None,
        "recalibrate_route_thresholds_on_load": args.recalibrate_route_thresholds_on_load,
        "num_workers": args.num_workers,
        "prox_mu": args.prox_mu,
    }
    config = apply_cli_overrides(config, overrides)
    run_name, run_timestamp = build_run_identity(config.experiment_name, config.federated.server_algorithm)
    config = build_runtime_paths(config, run_name=run_name, run_timestamp=run_timestamp)
    config.ensure_dirs()
    return config


def main() -> None:
    args = build_argparser().parse_args()
    config = build_config(args)

    log_path = configure_logging(config.log_dir, config.run_name)
    set_seed(config.federated.seed)

    logging.info("Experiment: %s", config.experiment_name)
    logging.info("Run name: %s", config.run_name)
    logging.info("Run timestamp: %s", config.run_timestamp)
    logging.info("Config file: %s", args.config)
    logging.info("Dataset=%s | partition_strategy=%s", config.dataset.name, config.dataset.partition_strategy)
    logging.info(
        "Primary algorithm=%s | comparison algorithms=%s | prox_mu=%.6f",
        config.federated.server_algorithm,
        config.federated.compare_algorithms,
        config.federated.prox_mu,
    )
    logging.info("Single-process federated simulation enabled. Clients are trained sequentially on one device.")
    logging.info("Personalized evaluation uses local client test sets with per-client macro-averaged accuracy.")
    logging.info("Log file: %s", log_path)
    logging.info("TensorBoard run dir: %s", config.tensorboard_dir)
    _log_effective_config_summary(config)

    writer = FederatedSummaryWriter(log_dir=config.tensorboard_dir)
    writer.add_text("run/name", config.run_name, 0)
    writer.add_text("run/config_yaml", Path(args.config).read_text(encoding="utf-8"), 0)
    writer.add_text("run/config_effective", str(config.to_dict()), 0)

    data_module = CIFAR10FederatedDataModule(
        config=config.dataset,
        device=config.federated.device,
        seed=config.federated.seed,
    )
    data_bundle = data_module.build()

    primary_history, suite_results = run_experiment_suite(
        config=config,
        data_bundle=data_bundle,
        data_module=data_module,
        writer=writer,
    )
    save_metrics(config.output_dir, primary_history, suite_results)
    writer.flush()
    writer.close()

    primary = suite_results["primary"]
    print("Primary algorithm result:")
    print(primary)
    print()
    print("Approximate memory table:")
    print(format_memory_table(primary.get("memory_mb", {})))
    print()
    print(f"Run name: {config.run_name}")
    print(f"Log file: {log_path}")
    print(f"TensorBoard dir: {config.tensorboard_dir}")


if __name__ == "__main__":
    main()
