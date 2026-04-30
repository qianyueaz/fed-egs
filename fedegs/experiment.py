import copy
from dataclasses import asdict
from typing import Dict, List, Tuple

from fedegs.federated import create_federated_server


SUMMARY_METRIC_ALIASES = [
    ("accuracy", ("routed_accuracy", "accuracy")),
    ("routed_accuracy", ("routed_accuracy", "accuracy")),
    ("weighted_accuracy", ("weighted_accuracy",)),
    ("hard_accuracy", ("routed_hard_accuracy", "hard_accuracy", "hard_sample_recall")),
    ("invocation_rate", ("general_invocation_rate", "invocation_rate")),
    ("compute_savings", ("compute_savings",)),
    ("client_train_flops", ("client_train_flops",)),
    ("client_train_flops_total", ("client_train_flops_total",)),
    ("expert_infer_flops", ("expert_infer_flops",)),
    ("general_infer_flops", ("general_infer_flops",)),
    ("routed_infer_flops", ("routed_infer_flops",)),
    ("expert_train_memory_mb", ("expert_train_memory_mb",)),
    ("expert_infer_memory_mb", ("expert_infer_memory_mb",)),
    ("general_train_memory_mb", ("general_train_memory_mb",)),
    ("general_infer_memory_mb", ("general_infer_memory_mb",)),
    ("expert_train_peak_memory_mb", ("expert_train_peak_memory_mb",)),
    ("expert_infer_peak_memory_mb", ("expert_infer_peak_memory_mb",)),
    ("general_train_peak_memory_mb", ("general_train_peak_memory_mb",)),
    ("general_infer_peak_memory_mb", ("general_infer_peak_memory_mb",)),
    ("inference_latency_ms", ("average_inference_latency_ms",)),
    ("average_round_train_time_seconds", ("average_round_train_time_seconds",)),
    ("total_train_time_seconds", ("total_train_time_seconds",)),
    ("average_upload_bytes_per_round", ("average_upload_bytes_per_round",)),
    ("total_upload_bytes", ("total_upload_bytes",)),
    ("expert_only_accuracy", ("expert_only_accuracy",)),
    ("general_only_accuracy", ("general_only_accuracy",)),
    ("final_training_loss", ("final_training_loss",)),
    ("distill_loss", ("distill_loss", "total_loss")),
    ("distill_kd_loss", ("distill_kd_loss", "kd_loss")),
    ("distill_ce_loss", ("distill_ce_loss", "ce_loss")),
    ("distill_alpha_mean", ("distill_alpha_mean", "alpha_mean")),
    ("dkdr_forward_kl", ("dkdr_forward_kl",)),
    ("dkdr_reverse_kl", ("dkdr_reverse_kl",)),
    ("dkdr_gamma_forward", ("dkdr_gamma_forward",)),
    ("dkdr_gamma_reverse", ("dkdr_gamma_reverse",)),
    ("distill_beta_mean", ("distill_beta_mean",)),
    ("teacher_reliability_mean", ("teacher_reliability_mean",)),
    ("teacher_error_mean", ("teacher_error_mean",)),
    ("teacher_bank_size", ("teacher_bank_size",)),
    ("teacher_bank_avg_staleness", ("teacher_bank_avg_staleness",)),
    ("teacher_bank_max_staleness", ("teacher_bank_max_staleness",)),
    ("teacher_bank_memory_mb", ("teacher_bank_memory_mb",)),
    ("pfedfda_beta_mean", ("pfedfda_beta_mean",)),
    ("teacher_bank_effective_size", ("teacher_bank_effective_size",)),
    ("teacher_weight_max_share", ("teacher_weight_max_share",)),
    ("teacher_confidence_mean", ("teacher_confidence_mean",)),
    ("teacher_entropy_mean", ("teacher_entropy_mean",)),
    ("selected_teacher_count_mean", ("selected_teacher_count_mean",)),
    ("selected_teacher_coverage", ("selected_teacher_coverage",)),
    ("teacher_topk_selected_mean", ("teacher_topk_selected_mean",)),
    ("teacher_selected_coverage", ("teacher_selected_coverage", "selected_teacher_coverage")),
    ("ema_kd_loss", ("ema_kd_loss",)),
    ("fallback_weight_mean", ("fallback_weight_mean",)),
    ("fallback_weight_invoked_mean", ("fallback_weight_invoked_mean",)),
    ("fallback_weight_noninvoked_mean", ("fallback_weight_noninvoked_mean",)),
    ("proxy_fallback_ratio", ("proxy_fallback_ratio",)),
    ("proxy_fallback_target_ratio", ("proxy_fallback_target_ratio",)),
    ("proxy_teacher_disagreement_mean", ("proxy_teacher_disagreement_mean",)),
    ("proxy_route_score_mean", ("proxy_route_score_mean",)),
    ("proxy_route_score_invoked_mean", ("proxy_route_score_invoked_mean",)),
    ("proxy_expert_confidence_invoked_mean", ("proxy_expert_confidence_invoked_mean",)),
    ("routing_real_invocation_ema", ("routing_real_invocation_ema",)),
    ("routing_general_reliability_threshold", ("routing_general_reliability_threshold",)),
    ("routing_general_route_enabled_rate", ("routing_general_route_enabled_rate",)),
    ("routing_holdout_general_reliability", ("routing_holdout_general_reliability",)),
    ("routing_invoked_general_reliability", ("routing_invoked_general_reliability",)),
    ("routing_error_threshold_mean", ("routing_error_threshold_mean",)),
    ("latest_real_invocation_rate", ("latest_real_invocation_rate",)),
    ("general_gain_over_expert", ("general_gain_over_expert",)),
    ("routed_gain_over_expert", ("routed_gain_over_expert",)),
    ("invoked_general_accuracy", ("invoked_general_accuracy",)),
    ("invoked_expert_accuracy", ("invoked_expert_accuracy",)),
    ("invoked_general_gain", ("invoked_general_gain",)),
    ("oracle_route_accuracy", ("oracle_route_accuracy",)),
    ("oracle_general_invocation_rate", ("oracle_general_invocation_rate",)),
    ("expert_bad_general_good_rate", ("expert_bad_general_good_rate",)),
    ("routing_regret", ("routing_regret",)),
    ("expert_general_disagreement_rate", ("expert_general_disagreement_rate",)),
]


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
    compare_algorithms = config.federated.compare_algorithms or []
    for algorithm_name in compare_algorithms:
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
    result["history"] = [{"round": item.round_idx, **asdict(item)} for item in history]

    if writer is not None:
        metrics = result.get("metrics", {})
        summary_step = 0
        summary_prefix = result.get("algorithm", normalized)
        for summary_metric, aliases in SUMMARY_METRIC_ALIASES:
            for alias in aliases:
                if alias not in metrics:
                    continue
                value = metrics[alias]
                if hasattr(writer, "add_summary_scalar"):
                    writer.add_summary_scalar(summary_prefix, summary_metric, value, summary_step)
                else:
                    writer.add_scalar(f"summary/{summary_metric}/{summary_prefix}", value, summary_step)
                break
    return history, result
