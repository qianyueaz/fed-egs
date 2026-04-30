"""
FedAsym-Gain: expert-only gain lookup routing for FedAsym.

This variant keeps FedAsym training and distillation unchanged, but replaces
the inference router with a calibration-time gain table that uses only expert
logits. The router decides fallback from client-local statistics of whether
general outperforms expert under the same expert prediction pattern.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from fedegs.federated.algorithms.fedasym import (
    FedAsymServer,
    _average_precision_score,
    _clone_tensor_dict,
)


def _clone_gain_table(table: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in table.items()}


def _expert_route_features(
    expert_logits: torch.Tensor,
    confidence_bins: int,
    margin_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_probs = torch.softmax(expert_logits, dim=1)
    topk = torch.topk(expert_probs, k=min(2, expert_probs.size(1)), dim=1)
    confidence = topk.values[:, 0]
    if topk.values.size(1) > 1:
        margin = topk.values[:, 0] - topk.values[:, 1]
    else:
        margin = torch.ones_like(confidence)
    predicted_class = expert_logits.argmax(dim=1)
    confidence_bin = torch.clamp((confidence * confidence_bins).long(), min=0, max=confidence_bins - 1)
    margin_bin = torch.clamp((margin * margin_bins).long(), min=0, max=margin_bins - 1)
    return predicted_class, confidence, confidence_bin, margin_bin


class FedAsymGainServer(FedAsymServer):
    def __init__(
        self,
        config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices,
        writer=None,
        public_dataset: Dataset | None = None,
    ) -> None:
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer=writer,
            public_dataset=public_dataset,
        )
        self.algorithm_name = "fedasym_gain"
        self.gain_confidence_bins = max(int(getattr(config.inference, "gain_confidence_bins", 6)), 2)
        self.gain_margin_bins = max(int(getattr(config.inference, "gain_margin_bins", 6)), 2)
        self.gain_min_bucket_support = max(int(getattr(config.inference, "gain_min_bucket_support", 8)), 1)
        self.gain_min_class_support = max(int(getattr(config.inference, "gain_min_class_support", 16)), 1)
        self.gain_high_confidence_keep = float(getattr(config.inference, "high_threshold", 0.85))
        self.gain_route_threshold = float(getattr(config.inference, "route_gain_threshold", 0.0))
        self.route_disable_when_no_gain = bool(getattr(config.federated, "route_disable_when_no_gain", True))
        self.client_gain_tables: Dict[str, Dict[str, torch.Tensor]] = {
            client_id: self._default_gain_table() for client_id in self.clients
        }
        self.client_gain_thresholds: Dict[str, float] = {
            client_id: self.gain_route_threshold for client_id in self.clients
        }

    def _default_gain_table(self) -> Dict[str, torch.Tensor]:
        num_classes = int(self.config.model.num_classes)
        return {
            "bucket_gain": torch.zeros(
                num_classes,
                self.gain_confidence_bins,
                self.gain_margin_bins,
                dtype=torch.float32,
            ),
            "bucket_support": torch.zeros(
                num_classes,
                self.gain_confidence_bins,
                self.gain_margin_bins,
                dtype=torch.float32,
            ),
            "class_gain": torch.zeros(num_classes, dtype=torch.float32),
            "class_support": torch.zeros(num_classes, dtype=torch.float32),
            "client_enabled": torch.zeros(1, dtype=torch.float32),
            "calibration_invocation_rate": torch.zeros(1, dtype=torch.float32),
            "calibration_route_gain": torch.zeros(1, dtype=torch.float32),
        }

    def _distill_general_model(
        self,
        public_knowledge: Dict[str, torch.Tensor | float],
        round_idx: int,
    ) -> Dict[str, float]:
        stats = super()._distill_general_model(public_knowledge, round_idx)
        self._refresh_client_gain_tables()
        return stats

    def _refresh_client_gain_tables(self) -> None:
        for client_id, calibration_dataset in self.client_calibration_datasets.items():
            self.client_gain_tables[client_id] = self._fit_client_gain_table(client_id, calibration_dataset)
            self.client_gain_thresholds[client_id] = self.gain_route_threshold

    def _fit_client_gain_table(self, client_id: str, calibration_dataset: Dataset) -> Dict[str, torch.Tensor]:
        if calibration_dataset is None or len(calibration_dataset) == 0:
            return self._default_gain_table()

        table = self._default_gain_table()
        bucket_gain_sum = torch.zeros_like(table["bucket_gain"])
        bucket_support = torch.zeros_like(table["bucket_support"])
        class_gain_sum = torch.zeros_like(table["class_gain"])
        class_support = torch.zeros_like(table["class_support"])

        client = self.clients[client_id]
        loader = self.data_module.make_loader(calibration_dataset, shuffle=False)
        client.expert_model.eval()
        self.general_model.eval()

        predicted_gains: List[torch.Tensor] = []
        realized_gains: List[torch.Tensor] = []
        fallback_masks: List[torch.Tensor] = []

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                expert_logits = client.expert_model(images)
                general_logits = self.general_model(images)
                expert_predictions = expert_logits.argmax(dim=1)
                general_predictions = general_logits.argmax(dim=1)
                predicted_class, confidence, confidence_bin, margin_bin = _expert_route_features(
                    expert_logits,
                    self.gain_confidence_bins,
                    self.gain_margin_bins,
                )

                expert_correct = expert_predictions.eq(targets_device)
                general_correct = general_predictions.eq(targets_device)
                gain = general_correct.to(torch.float32) - expert_correct.to(torch.float32)

                for sample_idx in range(images.size(0)):
                    pred_class = int(predicted_class[sample_idx].item())
                    conf_idx = int(confidence_bin[sample_idx].item())
                    margin_idx = int(margin_bin[sample_idx].item())
                    bucket_gain_sum[pred_class, conf_idx, margin_idx] += gain[sample_idx].cpu()
                    bucket_support[pred_class, conf_idx, margin_idx] += 1.0
                    class_gain_sum[pred_class] += gain[sample_idx].cpu()
                    class_support[pred_class] += 1.0

        bucket_gain = torch.where(
            bucket_support > 0.0,
            bucket_gain_sum / bucket_support.clamp_min(1.0),
            torch.zeros_like(bucket_gain_sum),
        )
        class_gain = torch.where(
            class_support > 0.0,
            class_gain_sum / class_support.clamp_min(1.0),
            torch.zeros_like(class_gain_sum),
        )
        table["bucket_gain"] = bucket_gain
        table["bucket_support"] = bucket_support
        table["class_gain"] = class_gain
        table["class_support"] = class_support

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                expert_logits = client.expert_model(images)
                general_logits = self.general_model(images)
                expert_predictions = expert_logits.argmax(dim=1)
                general_predictions = general_logits.argmax(dim=1)
                gain_scores, _, confidence = self._lookup_gain_scores(client_id, expert_logits, table=table)
                fallback_mask = (gain_scores > self.gain_route_threshold) & (confidence < self.gain_high_confidence_keep)
                realized_gain = (
                    general_predictions.eq(targets_device).to(torch.float32)
                    - expert_predictions.eq(targets_device).to(torch.float32)
                )
                predicted_gains.append(gain_scores.detach().cpu())
                realized_gains.append(realized_gain.detach().cpu())
                fallback_masks.append(fallback_mask.detach().cpu())

        if predicted_gains:
            realized_gain_all = torch.cat(realized_gains, dim=0)
            fallback_mask_all = torch.cat(fallback_masks, dim=0)
            policy_gain = float(realized_gain_all[fallback_mask_all].sum().item()) if fallback_mask_all.any() else 0.0
            invocation_rate = float(fallback_mask_all.to(torch.float32).mean().item())
        else:
            policy_gain = 0.0
            invocation_rate = 0.0

        enabled = 1.0
        if self.route_disable_when_no_gain and policy_gain <= 0.0:
            enabled = 0.0
        table["client_enabled"] = torch.tensor([enabled], dtype=torch.float32)
        table["calibration_invocation_rate"] = torch.tensor([invocation_rate], dtype=torch.float32)
        table["calibration_route_gain"] = torch.tensor([policy_gain], dtype=torch.float32)
        return table

    def _lookup_gain_scores(
        self,
        client_id: str,
        expert_logits: torch.Tensor,
        table: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gain_table = table if table is not None else self.client_gain_tables[client_id]
        predicted_class, confidence, confidence_bin, margin_bin = _expert_route_features(
            expert_logits,
            self.gain_confidence_bins,
            self.gain_margin_bins,
        )

        bucket_gain = gain_table["bucket_gain"].to(expert_logits.device)
        bucket_support = gain_table["bucket_support"].to(expert_logits.device)
        class_gain = gain_table["class_gain"].to(expert_logits.device)
        class_support = gain_table["class_support"].to(expert_logits.device)

        per_bucket_gain = bucket_gain[predicted_class, confidence_bin, margin_bin]
        per_bucket_support = bucket_support[predicted_class, confidence_bin, margin_bin]
        per_class_gain = class_gain[predicted_class]
        per_class_support = class_support[predicted_class]

        use_bucket = per_bucket_support >= float(self.gain_min_bucket_support)
        use_class = (~use_bucket) & (per_class_support >= float(self.gain_min_class_support))
        selected_support = torch.where(use_bucket, per_bucket_support, torch.where(use_class, per_class_support, torch.zeros_like(per_bucket_support)))
        selected_gain = torch.where(use_bucket, per_bucket_gain, torch.where(use_class, per_class_gain, torch.full_like(per_bucket_gain, -1.0)))
        return selected_gain, selected_support, confidence

    def _predict_routed(self, client_id, images, indices):
        expert_model = self.clients[client_id].expert_model
        gain_table = self.client_gain_tables[client_id]
        threshold = float(self.client_gain_thresholds.get(client_id, self.gain_route_threshold))
        client_enabled = bool(float(gain_table["client_enabled"].item()) > 0.5)

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            expert_logits = expert_model(images)
            expert_prediction = expert_logits.argmax(dim=1)
            gain_scores, selected_support, confidence = self._lookup_gain_scores(client_id, expert_logits)

            fallback_mask = (
                client_enabled
                & (selected_support > 0.0)
                & (gain_scores > threshold)
                & (confidence < self.gain_high_confidence_keep)
            )

            predictions = expert_prediction.clone()
            invoked_general = int(fallback_mask.sum().item())
            route_types = ["expert"] * images.size(0)
            if fallback_mask.any():
                general_logits = self.general_model(images[fallback_mask])
                predictions[fallback_mask] = general_logits.argmax(dim=1)
                for sample_index in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_index] = "general"

            metadata = {
                "route_type": route_types,
                "expert_confidence": confidence.detach().cpu().tolist(),
            }
            return predictions, invoked_general, metadata

    def _evaluate_error_predictor_metrics(self) -> Dict[str, float]:
        beneficial_labels: List[int] = []
        gain_scores: List[float] = []
        predicted_positive = 0
        true_positive = 0
        actual_positive = 0

        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
                client = self.clients[client_id]
                client.expert_model.eval()
                self.general_model.eval()
                gain_table = self.client_gain_tables[client_id]
                threshold = float(self.client_gain_thresholds.get(client_id, self.gain_route_threshold))
                client_enabled = bool(float(gain_table["client_enabled"].item()) > 0.5)

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    expert_logits = client.expert_model(images)
                    general_logits = self.general_model(images)
                    gain_score_batch, selected_support, confidence = self._lookup_gain_scores(client_id, expert_logits)
                    expert_predictions = expert_logits.argmax(dim=1)
                    general_predictions = general_logits.argmax(dim=1)

                    beneficial = general_predictions.eq(targets_device) & expert_predictions.ne(targets_device)
                    fallback_mask = (
                        client_enabled
                        & (selected_support > 0.0)
                        & (gain_score_batch > threshold)
                        & (confidence < self.gain_high_confidence_keep)
                    )

                    predicted_positive += int(fallback_mask.sum().item())
                    true_positive += int((fallback_mask & beneficial).sum().item())
                    actual_positive += int(beneficial.sum().item())
                    beneficial_labels.extend(beneficial.detach().cpu().to(torch.int64).tolist())
                    gain_scores.extend(gain_score_batch.detach().cpu().to(torch.float32).tolist())

        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(actual_positive, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        auprc = _average_precision_score(beneficial_labels, gain_scores)
        return {
            "gain_router_precision": precision,
            "gain_router_recall": recall,
            "gain_router_f1": f1,
            "gain_router_auprc": auprc,
        }

    def _build_round_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        metrics = self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)
        router_metrics = self._evaluate_error_predictor_metrics()
        threshold_mean = (
            sum(self.client_gain_thresholds.values()) / max(len(self.client_gain_thresholds), 1)
            if self.client_gain_thresholds
            else 0.0
        )
        enabled_rate = (
            sum(float(table["client_enabled"].item()) for table in self.client_gain_tables.values())
            / max(len(self.client_gain_tables), 1)
            if self.client_gain_tables
            else 0.0
        )
        calibration_invocation_rate = (
            sum(float(table["calibration_invocation_rate"].item()) for table in self.client_gain_tables.values())
            / max(len(self.client_gain_tables), 1)
            if self.client_gain_tables
            else 0.0
        )
        calibration_route_gain = (
            sum(float(table["calibration_route_gain"].item()) for table in self.client_gain_tables.values())
            / max(len(self.client_gain_tables), 1)
            if self.client_gain_tables
            else 0.0
        )
        metrics.update(
            {
                **router_metrics,
                "distill_loss": self.latest_distill_stats.get("total_loss", 0.0),
                "dkdr_forward_kl": self.latest_distill_stats.get("forward_kl", 0.0),
                "dkdr_reverse_kl": self.latest_distill_stats.get("reverse_kl", 0.0),
                "dkdr_gamma_forward": self.latest_distill_stats.get("gamma_forward", 0.5),
                "dkdr_gamma_reverse": self.latest_distill_stats.get("gamma_reverse", 0.5),
                "teacher_reliability_mean": self.latest_distill_stats.get("teacher_reliability_mean", 0.0),
                "teacher_error_mean": self.latest_distill_stats.get("teacher_error_mean", 0.0),
                "routing_gain_threshold_mean": threshold_mean,
                "gain_router_enabled_client_rate": enabled_rate,
                "gain_router_calibration_invocation_rate": calibration_invocation_rate,
                "gain_router_calibration_route_gain": calibration_route_gain,
            }
        )
        return metrics

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        if not extra_metrics:
            return ""
        return (
            f" | g_gain={extra_metrics.get('general_gain_over_expert', 0.0):.4f}"
            f" | route_gain={extra_metrics.get('routed_gain_over_expert', 0.0):.4f}"
            f" | invoked_g={extra_metrics.get('invoked_general_gain', 0.0):.4f}"
            f" | gp_p={extra_metrics.get('gain_router_precision', 0.0):.4f}"
            f" | gp_r={extra_metrics.get('gain_router_recall', 0.0):.4f}"
            f" | gp_f1={extra_metrics.get('gain_router_f1', 0.0):.4f}"
            f" | gp_auprc={extra_metrics.get('gain_router_auprc', 0.0):.4f}"
            f" | gain_cal_inv={extra_metrics.get('gain_router_calibration_invocation_rate', 0.0):.4f}"
            f" | gain_cal={extra_metrics.get('gain_router_calibration_route_gain', 0.0):.4f}"
            f" | enabled={extra_metrics.get('gain_router_enabled_client_rate', 0.0):.4f}"
            f" | dkdr_f={extra_metrics.get('dkdr_forward_kl', 0.0):.4f}"
            f" | dkdr_r={extra_metrics.get('dkdr_reverse_kl', 0.0):.4f}"
            f" | gam_f={extra_metrics.get('dkdr_gamma_forward', 0.5):.4f}"
            f" | gam_r={extra_metrics.get('dkdr_gamma_reverse', 0.5):.4f}"
            f" | thr={extra_metrics.get('routing_gain_threshold_mean', 0.0):.4f}"
        )

    def _maybe_update_best(self, round_idx: int, metrics, expert_accuracy: float, general_accuracy: float) -> None:
        previous_snapshot = self.best_snapshot
        super()._maybe_update_best(round_idx, metrics, expert_accuracy, general_accuracy)
        if self.best_snapshot is previous_snapshot:
            return
        self.best_snapshot["client_gain_tables"] = {
            client_id: _clone_gain_table(table) for client_id, table in self.client_gain_tables.items()
        }
        self.best_snapshot["client_gain_thresholds"] = dict(self.client_gain_thresholds)

    def _restore_best(self) -> None:
        super()._restore_best()
        if not self.best_snapshot:
            return
        stored_tables = self.best_snapshot.get("client_gain_tables")
        if isinstance(stored_tables, dict):
            self.client_gain_tables = {
                client_id: _clone_gain_table(table) for client_id, table in stored_tables.items()
            }
        else:
            self._refresh_client_gain_tables()
        self.client_gain_thresholds = dict(
            self.best_snapshot.get("client_gain_thresholds", self.client_gain_thresholds)
        )
