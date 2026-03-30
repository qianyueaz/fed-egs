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
            writer.add_scalars("summary/routed_accuracy", {algorithm_name: metrics["routed_accuracy"]}, summary_step)
        elif "accuracy" in metrics:
            writer.add_scalars("summary/routed_accuracy", {algorithm_name: metrics["accuracy"]}, summary_step)
        if "routed_hard_accuracy" in metrics:
            writer.add_scalars("summary/hard_accuracy", {algorithm_name: metrics["routed_hard_accuracy"]}, summary_step)
        elif "hard_accuracy" in metrics:
            writer.add_scalars("summary/hard_accuracy", {algorithm_name: metrics["hard_accuracy"]}, summary_step)
        if "general_invocation_rate" in metrics:
            writer.add_scalars("summary/invocation_rate", {algorithm_name: metrics["general_invocation_rate"]}, summary_step)
        elif "invocation_rate" in metrics:
            writer.add_scalars("summary/invocation_rate", {algorithm_name: metrics["invocation_rate"]}, summary_step)
        if "final_training_loss" in metrics:
            writer.add_scalars("summary/final_training_loss", {algorithm_name: metrics["final_training_loss"]}, summary_step)
    return history, result
