import copy
from typing import Dict, List, Tuple

from fedegs.federated import create_federated_server


def run_experiment_suite(config, data_bundle, data_module, writer=None) -> Tuple[List[object], Dict[str, object]]:
    primary_history, primary_result = _run_single_algorithm(
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
        _, result = _run_single_algorithm(
            algorithm_name=algorithm_name,
            config=config,
            data_bundle=data_bundle,
            data_module=data_module,
            writer=writer,
        )
        comparison_results[normalized] = result
        seen.add(normalized)

    suite = {
        "primary": primary_result,
        "comparisons": comparison_results,
    }
    return primary_history, suite


def _run_single_algorithm(algorithm_name: str, config, data_bundle, data_module, writer=None):
    run_config = copy.deepcopy(config)
    normalized = algorithm_name.lower()
    run_config.federated.server_algorithm = algorithm_name
    if normalized in {"ideal", "ideal_upper_bound", "fat_client"}:
        run_config.federated.server_algorithm = "fedavg"
        run_config.model.baseline_architecture = run_config.model.architecture
        run_config.model.baseline_width = run_config.model.general_width
        run_config.model.baseline_base_channels = run_config.model.expert_base_channels
    server = create_federated_server(
        run_config.federated.server_algorithm,
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
    if normalized in {"ideal", "ideal_upper_bound", "fat_client"}:
        result["algorithm"] = "ideal"
    result["history"] = [
        {
            "round": item.round_idx,
            "avg_client_loss": item.avg_client_loss,
            "routed_accuracy": item.routed_accuracy,
            "hard_accuracy": item.hard_accuracy,
            "invocation_rate": item.invocation_rate,
            "local_accuracy": item.local_accuracy,
            "compute_savings": item.compute_savings,
        }
        for item in history
    ]

    if writer is not None:
        metrics = result.get("metrics", {})
        summary_step = 0
        summary_prefix = result.get("algorithm", normalized)
        if "routed_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/routed_accuracy", metrics["routed_accuracy"], summary_step)
        elif "accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/routed_accuracy", metrics["accuracy"], summary_step)
        if "local_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/local_accuracy", metrics["local_accuracy"], summary_step)
        if "routed_hard_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/hard_accuracy", metrics["routed_hard_accuracy"], summary_step)
        elif "hard_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/hard_accuracy", metrics["hard_accuracy"], summary_step)
        elif "hard_sample_recall" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/hard_accuracy", metrics["hard_sample_recall"], summary_step)
        if "general_invocation_rate" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/invocation_rate", metrics["general_invocation_rate"], summary_step)
        elif "invocation_rate" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/invocation_rate", metrics["invocation_rate"], summary_step)
        if "compute_savings" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/compute_savings", metrics["compute_savings"], summary_step)
        if "expert_only_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/expert_accuracy", metrics["expert_only_accuracy"], summary_step)
        if "general_only_accuracy" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/general_accuracy", metrics["general_only_accuracy"], summary_step)
        if "final_training_loss" in metrics:
            writer.add_scalar(f"summary/{summary_prefix}/final_training_loss", metrics["final_training_loss"], summary_step)
    return history, result
