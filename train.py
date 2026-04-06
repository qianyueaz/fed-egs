import argparse
import logging
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fedegs.config import ExperimentConfig, apply_cli_overrides, build_runtime_paths
from fedegs.data import CIFAR10FederatedDataModule
from fedegs.evaluation import format_memory_table, save_metrics
from fedegs.experiment import run_experiment_suite


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
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prox-mu", dest="prox_mu", type=float, default=None)
    return parser


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "experiment"


def _run_name_conflicts(output_dir: str, experiment_name: str, run_name: str, run_timestamp: str) -> bool:
    base_output = Path(output_dir)
    log_path = base_output / "logs" / run_timestamp[:8] / f"{run_name}.log"
    tensorboard_dir = base_output / "tensorboard" / experiment_name / run_name
    route_dir = base_output / "routes" / run_name
    return log_path.exists() or tensorboard_dir.exists() or route_dir.exists()


def build_run_identity(experiment_name: str, algorithm_name: str, output_dir: str) -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_experiment = sanitize_name(experiment_name)
    safe_algorithm = sanitize_name(algorithm_name)
    base_name = f"{safe_experiment}_{safe_algorithm}_{timestamp}"
    run_name = base_name
    suffix = 1

    while _run_name_conflicts(output_dir, experiment_name, run_name, timestamp):
        run_name = f"{base_name}_{suffix:02d}"
        suffix += 1

    return run_name, timestamp


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
        "high_threshold": args.high_threshold,
        "low_threshold": args.low_threshold,
        "num_workers": args.num_workers,
        "prox_mu": args.prox_mu,
    }
    config = apply_cli_overrides(config, overrides)
    run_name, run_timestamp = build_run_identity(
        config.experiment_name,
        config.federated.server_algorithm,
        config.output_dir,
    )
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
    logging.info("Personalized evaluation uses local client test sets with sample-weighted aggregation.")
    logging.info("Log file: %s", log_path)
    logging.info("TensorBoard run dir: %s", config.tensorboard_dir)

    writer = SummaryWriter(log_dir=config.tensorboard_dir)
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
