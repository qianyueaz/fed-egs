import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter


KNOWN_ALGORITHMS = {
    "fedavg",
    "fedprox",
    "fedegs",
    "fedegs2",
    "fedegs3",
    "fedegsba",
    "fedegsbg",
    "fedasym",
    "fedasym_gain",
    "fedasym_rad",
    "fedegsd",
    "fedegsd_s",
    "fedegse",
    "fedegseu",
    "fedegss",
    "fedegssl",
    "fedegssg",
    "ideal",
}


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "algorithm"


def _split_algorithm_tag(tag: str) -> Tuple[Optional[str], str]:
    normalized_tag = str(tag)
    if "/" in normalized_tag:
        metric_tag, tail = normalized_tag.rsplit("/", 1)
        sanitized_tail = _sanitize_run_name(tail)
        if sanitized_tail in KNOWN_ALGORITHMS:
            return sanitized_tail, metric_tag

    final_segment = normalized_tag.rsplit("/", 1)[-1]
    for algorithm in sorted(KNOWN_ALGORITHMS, key=len, reverse=True):
        suffix = f"_{algorithm}"
        if final_segment.endswith(suffix):
            metric_segment = final_segment[: -len(suffix)]
            if "/" in normalized_tag:
                parent = normalized_tag.rsplit("/", 1)[0]
                metric_tag = f"{parent}/{metric_segment}" if metric_segment else parent
            else:
                metric_tag = metric_segment
            return algorithm, metric_tag
    return None, normalized_tag


class FederatedSummaryWriter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.root_writer = SummaryWriter(log_dir=str(self.log_dir))
        self._algorithm_writers: Dict[str, SummaryWriter] = {}
        self._register_custom_scalar_layout()

    def _register_custom_scalar_layout(self) -> None:
        compare_metrics = [
            "loss",
            "accuracy",
            "routed_accuracy",
            "weighted_accuracy",
            "hard_accuracy",
            "invocation_rate",
            "compute_savings",
            "routing_holdout_accuracy",
            "routing_score_threshold",
            "routing_temperature",
            "routing_beneficial_invocation_rate",
            "routing_effective_target_rate",
            "routing_post_veto_target_rate",
            "routing_veto_client_rate",
            "routing_veto_client_count",
            "routing_holdout_invoked_general_gain",
            "routing_real_invocation_ema",
            "routing_general_reliability_threshold",
            "routing_general_route_enabled_rate",
            "routing_holdout_general_reliability",
            "routing_invoked_general_reliability",
            "latest_real_invocation_rate",
            "client_train_flops",
            "client_train_flops_total",
            "expert_infer_flops",
            "general_infer_flops",
            "routed_infer_flops",
            "expert_train_memory_mb",
            "expert_infer_memory_mb",
            "general_train_memory_mb",
            "general_infer_memory_mb",
            "expert_train_peak_memory_mb",
            "expert_infer_peak_memory_mb",
            "general_train_peak_memory_mb",
            "general_infer_peak_memory_mb",
            "inference_latency_ms",
            "round_train_time_seconds",
            "upload_bytes_per_round",
            "upload_bytes_total",
            "expert_only_accuracy",
            "general_only_accuracy",
            "expert_loss",
            "distill_loss",
            "distill_kd_loss",
            "distill_ce_loss",
            "error_predictor_precision",
            "error_predictor_recall",
            "error_predictor_f1",
            "error_predictor_auprc",
            "error_predictor_false_positive_rate",
            "error_predictor_predicted_positive_rate",
            "rad_route_demand",
            "rad_label_ce_weight",
            "rad_focus_weight",
            "distill_alpha_mean",
            "teacher_bank_size",
            "teacher_bank_avg_staleness",
            "teacher_bank_max_staleness",
            "teacher_bank_memory_mb",
            "teacher_bank_effective_size",
            "teacher_weight_max_share",
            "teacher_confidence_mean",
            "teacher_entropy_mean",
            "selected_teacher_count_mean",
            "selected_teacher_coverage",
            "teacher_topk_selected_mean",
            "teacher_selected_coverage",
            "ema_kd_loss",
            "fallback_weight_mean",
            "fallback_weight_invoked_mean",
            "fallback_weight_noninvoked_mean",
            "proxy_fallback_ratio",
            "proxy_fallback_target_ratio",
            "proxy_teacher_disagreement_mean",
            "proxy_route_score_mean",
            "proxy_route_score_invoked_mean",
            "proxy_expert_confidence_invoked_mean",
            "routing_real_invocation_ema",
            "routing_general_reliability_threshold",
            "routing_general_route_enabled_rate",
            "routing_holdout_general_reliability",
            "routing_invoked_general_reliability",
            "latest_real_invocation_rate",
            "general_gain_over_expert",
            "routed_gain_over_expert",
            "invoked_general_accuracy",
            "invoked_expert_accuracy",
            "invoked_general_gain",
            "oracle_route_accuracy",
            "oracle_general_invocation_rate",
            "expert_bad_general_good_rate",
            "routing_regret",
            "expert_general_disagreement_rate",
        ]
        layout = {
            "Compare": {
                metric_name: [
                    "Multiline",
                    [f"compare_group/{metric_name}/{algorithm}" for algorithm in sorted(KNOWN_ALGORITHMS)],
                ]
                for metric_name in compare_metrics
            },
        }
        self.root_writer.add_custom_scalars(layout)

    def _get_algorithm_writer(self, algorithm: str) -> SummaryWriter:
        normalized = _sanitize_run_name(algorithm)
        writer = self._algorithm_writers.get(normalized)
        if writer is None:
            writer = SummaryWriter(log_dir=str(self.log_dir / "algorithms" / normalized))
            self._algorithm_writers[normalized] = writer
        return writer

    def add_text(self, *args, **kwargs) -> None:
        self.root_writer.add_text(*args, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs) -> None:
        algorithm, metric_tag = _split_algorithm_tag(str(tag))
        if algorithm is not None:
            self._get_algorithm_writer(algorithm).add_scalar(metric_tag, scalar_value, global_step, *args, **kwargs)
            return
        self.root_writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)

    def add_compare_scalar(self, algorithm: str, metric_name: str, value: float, step: int) -> None:
        self._get_algorithm_writer(algorithm).add_scalar(f"compare/{metric_name}", value, step)
        self.root_writer.add_scalar(f"compare_group/{metric_name}/{_sanitize_run_name(algorithm)}", value, step)

    def add_algorithm_scalar(self, algorithm: str, metric_tag: str, value: float, step: int) -> None:
        self._get_algorithm_writer(algorithm).add_scalar(metric_tag, value, step)

    def flush(self) -> None:
        self.root_writer.flush()
        for writer in self._algorithm_writers.values():
            writer.flush()

    def close(self) -> None:
        self.flush()
        self.root_writer.close()
        for writer in self._algorithm_writers.values():
            writer.close()
