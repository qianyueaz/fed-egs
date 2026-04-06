import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fedegs.federated import create_federated_server

LOGGER = logging.getLogger(__name__)


def run_experiment_suite(config, data_bundle, data_module, writer=None) -> Tuple[List[object], Dict[str, object]]:
    """Run the primary algorithm and all comparison algorithms.

    Each algorithm gets its own ``SummaryWriter`` sub-directory so that
    TensorBoard treats them as separate *runs* that can be overlaid in one
    chart without overwriting each other's scalars.

    The caller-provided *writer* is used for the primary algorithm.  For
    every comparison algorithm a new writer is created under a sibling
    directory (same parent, algorithm name appended).
    """
    primary_history, primary_result = run_single_algorithm(
        algorithm_name=config.federated.server_algorithm,
        config=config,
        data_bundle=data_bundle,
        data_module=data_module,
        writer=writer,
    )

    comparison_results: Dict[str, object] = {}
    seen = {config.federated.server_algorithm.lower()}
    for algorithm_name in config.federated.compare_algorithms:
        normalized = algorithm_name.lower()
        if normalized in seen:
            continue

        # Create a dedicated writer for each comparison algorithm
        comparison_writer = _make_comparison_writer(config, algorithm_name, writer)

        _, result = run_single_algorithm(
            algorithm_name=algorithm_name,
            config=config,
            data_bundle=data_bundle,
            data_module=data_module,
            writer=comparison_writer,
        )
        comparison_results[normalized] = result
        seen.add(normalized)

        # Flush and close the per-algorithm writer
        if comparison_writer is not None and comparison_writer is not writer:
            comparison_writer.flush()
            comparison_writer.close()

    suite = {
        "primary": primary_result,
        "comparisons": comparison_results,
    }
    return primary_history, suite


def run_single_algorithm(algorithm_name: str, config, data_bundle, data_module, writer=None):
    run_config = copy.deepcopy(config)
    run_config.federated.server_algorithm = algorithm_name
    server = create_federated_server(
        algorithm_name,
        config=run_config,
        client_datasets=data_bundle["client_datasets"],
        client_test_datasets=data_bundle["client_test_datasets"],
        data_module=data_module,
        test_hard_indices=data_bundle["test_hard_indices"],
        writer=writer,
        public_dataset=data_bundle.get("public_dataset"),
    )
    history = server.train(data_bundle["test_dataset"])
    result = server.evaluate_baselines(data_bundle["test_dataset"])
    result["history"] = [
        {
            "round": item.round_idx,
            "avg_client_loss": item.avg_client_loss,
            "routed_accuracy": item.routed_accuracy,
            "hard_accuracy": item.hard_accuracy,
            "invocation_rate": item.invocation_rate,
        }
        for item in history
    ]

    if writer is not None:
        metrics = result.get("metrics", {})
        summary_step = 0
        if "routed_accuracy" in metrics:
            writer.add_scalar("summary/routed_accuracy", metrics["routed_accuracy"], summary_step)
        elif "accuracy" in metrics:
            writer.add_scalar("summary/routed_accuracy", metrics["accuracy"], summary_step)
        if "routed_hard_accuracy" in metrics:
            writer.add_scalar("summary/hard_accuracy", metrics["routed_hard_accuracy"], summary_step)
        elif "hard_accuracy" in metrics:
            writer.add_scalar("summary/hard_accuracy", metrics["hard_accuracy"], summary_step)
        if "general_invocation_rate" in metrics:
            writer.add_scalar("summary/invocation_rate", metrics["general_invocation_rate"], summary_step)
        elif "invocation_rate" in metrics:
            writer.add_scalar("summary/invocation_rate", metrics["invocation_rate"], summary_step)
        if "expert_only_accuracy" in metrics:
            writer.add_scalar("summary/expert_accuracy", metrics["expert_only_accuracy"], summary_step)
        if "general_only_accuracy" in metrics:
            writer.add_scalar("summary/general_accuracy", metrics["general_only_accuracy"], summary_step)
        if "final_training_loss" in metrics:
            writer.add_scalar("summary/final_training_loss", metrics["final_training_loss"], summary_step)
    return history, result


def _make_comparison_writer(config, algorithm_name: str, primary_writer) -> Optional[object]:
    """Create a SummaryWriter in a sibling directory for a comparison algorithm.

    Directory layout (example)::

        artifacts/tensorboard/fedegs3_cifar10_default/
            fedegs3_cifar10_default_fedegs3_20260405_120000/   <- primary
            fedegs3_cifar10_default_fedavg_20260405_120000/    <- comparison
            fedegs3_cifar10_default_fedprox_20260405_120000/   <- comparison

    TensorBoard ``--logdir artifacts/tensorboard/fedegs3_cifar10_default``
    will discover all three as separate runs and overlay them in the same
    charts automatically.
    """
    if primary_writer is None:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None

    # Derive the comparison log directory from the primary writer's path
    primary_dir = Path(primary_writer.log_dir)
    parent_dir = primary_dir.parent

    # Build a run name that replaces the algorithm portion
    # Primary run_name pattern: {experiment}_{algorithm}_{timestamp}
    # We swap the algorithm segment
    primary_name = primary_dir.name
    primary_algo = config.federated.server_algorithm

    # Replace only the trailing "_<algorithm>_" segment. The experiment name
    # itself often includes the primary algorithm token, so replace(..., 1)
    # corrupts the run name and produces misleading TensorBoard run labels.
    marker = f"_{primary_algo}_"
    if marker in primary_name:
        base_name, suffix = primary_name.rsplit(marker, 1)
        comparison_name = f"{base_name}_{algorithm_name}_{suffix}"
    else:
        comparison_name = f"{primary_name}_{algorithm_name}"

    comparison_dir = parent_dir / comparison_name
    LOGGER.info(
        "Creating TensorBoard run for %s at %s", algorithm_name, comparison_dir,
    )
    return SummaryWriter(log_dir=str(comparison_dir))
