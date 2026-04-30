"""
FedEGS-E: uncertainty-aware teacher-bank distillation for general enhancement.

Design goals:
  - Keep the current FedEGS-S client path unchanged:
    clients only train local experts and upload expert weights.
  - Keep the current FedEGS-D routed inference path unchanged:
    temperature calibration, prototype-aware route features, per-client
    threshold adaptation, and expert-first fallback to general are all reused
    through the FedEGS-S -> FedEGS-D inheritance chain.
  - Strengthen the server-side general model by borrowing the main idea from
    "Stragglers Can Contribute More: Uncertainty-Aware Distillation for
    Asynchronous Federated Learning":
      1. maintain a teacher bank over the proxy dataset,
      2. decay stale teachers instead of discarding them immediately,
      3. distill with uncertainty-aware KL/CE mixing,
      4. clip distillation gradients for stability.
"""

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from fedegs.federated.common import LOGGER
from fedegs.federated.algorithms.fedegss import FedEGSSServer
from fedegs.models import SmallCNN


class FedEGSEServer(FedEGSSServer):
    def __init__(
        self,
        config,
        client_datasets,
        client_test_datasets,
        data_module,
        test_hard_indices,
        writer=None,
        public_dataset=None,
    ) -> None:
        self.algorithm_name = "fedegse"
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer=writer,
            public_dataset=public_dataset,
        )
        self.algorithm_name = "fedegse"
        self.teacher_bank: Dict[str, Dict[str, object]] = {}
        self.latest_teacher_bank_stats: Dict[str, float] = {}
        self.latest_distill_stats: Dict[str, float] = {}
        self.routing_real_invocation_ema: float | None = None
        self.latest_real_invocation_rate: float = 0.0
        self.teacher_selection_mode = str(getattr(config.federated, "teacher_selection_mode", "mean_confidence"))
        self.general_distill_mode = str(getattr(config.federated, "general_distill_mode", "route_aware")).lower()
        self.enable_general_ema_anchor = bool(getattr(config.federated, "enable_general_ema_anchor", True))
        self.general_ema_teacher = copy.deepcopy(self.general_model).to(self.device)
        self.general_ema_teacher.eval()
        for parameter in self.general_ema_teacher.parameters():
            parameter.requires_grad_(False)
        LOGGER.info(
            "%s routing mode | reusing fedegsd route stack: calibrated expert-first dynamic routing",
            self.algorithm_name,
        )

    def _infer_client_proxy_logits(self, update) -> torch.Tensor:
        num_classes = self.config.model.num_classes
        expert = SmallCNN(
            num_classes=num_classes,
            base_channels=self.config.model.expert_base_channels,
        ).to(self.device)
        expert.load_state_dict(update.expert_state_dict)
        expert.eval()

        logits_batches: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in self.distill_loader:
                images = batch[0].to(self.device)
                logits_batches.append(expert(images).cpu())
        del expert
        return torch.cat(logits_batches, dim=0)

    def _update_general_ema_teacher(self) -> None:
        if not self.enable_general_ema_anchor:
            self.general_ema_teacher.load_state_dict(self.general_model.state_dict())
            return

        momentum = 0.99
        with torch.no_grad():
            for ema_param, param in zip(self.general_ema_teacher.parameters(), self.general_model.parameters()):
                ema_param.mul_(momentum).add_(param.detach(), alpha=1.0 - momentum)
            for ema_buffer, buffer in zip(self.general_ema_teacher.buffers(), self.general_model.buffers()):
                ema_buffer.copy_(buffer.detach())
        self.general_ema_teacher.eval()

    def _refresh_teacher_bank(self, updates) -> None:
        temperature = max(float(self.config.federated.distill_temperature), 1e-8)
        num_classes = self.config.model.num_classes
        log_c = math.log(float(num_classes))
        for update in updates:
            logits = self._infer_client_proxy_logits(update)
            probs = F.softmax(logits / temperature, dim=1)
            mean_confidence = float(probs.max(dim=1).values.mean().item())
            normalized_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(log_c, 1e-8)
            self.teacher_bank[update.client_id] = {
                "logits": logits,
                "last_round": int(self.current_round),
                "num_samples": int(update.num_samples),
                "mean_confidence": mean_confidence,
                "mean_normalized_entropy": float(normalized_entropy.mean().item()),
            }

        max_staleness = max(int(getattr(self.config.federated, "teacher_bank_max_staleness", 0)), 0)
        if max_staleness > 0:
            expired_client_ids = [
                client_id
                for client_id, entry in self.teacher_bank.items()
                if int(self.current_round) - int(entry["last_round"]) > max_staleness
            ]
            for client_id in expired_client_ids:
                self.teacher_bank.pop(client_id, None)

    def _summarize_teacher_bank(self) -> Dict[str, float]:
        if not self.teacher_bank:
            return {
                "teacher_bank_size": 0.0,
                "teacher_bank_avg_staleness": 0.0,
                "teacher_bank_max_staleness": 0.0,
                "teacher_bank_memory_mb": 0.0,
            }
        staleness_values = [
            max(int(self.current_round) - int(entry["last_round"]), 0)
            for entry in self.teacher_bank.values()
        ]
        memory_bytes = sum(
            int(entry["logits"].numel() * entry["logits"].element_size())
            for entry in self.teacher_bank.values()
        )
        return {
            "teacher_bank_size": float(len(self.teacher_bank)),
            "teacher_bank_avg_staleness": float(sum(staleness_values) / max(len(staleness_values), 1)),
            "teacher_bank_max_staleness": float(max(staleness_values)),
            "teacher_bank_memory_mb": float(memory_bytes / (1024 ** 2)),
        }

    def _select_teacher_mask(self, teacher_confidence: torch.Tensor) -> torch.Tensor:
        if self.teacher_selection_mode == "topk_confidence":
            topk = max(int(getattr(self.config.federated, "teacher_topk", 4)), 1)
            topk = min(topk, teacher_confidence.size(0))
            top_teacher_indices = torch.topk(teacher_confidence, k=topk, dim=0).indices
            selection_mask = torch.zeros_like(teacher_confidence, dtype=torch.bool)
            selection_mask.scatter_(0, top_teacher_indices, True)
        elif self.teacher_selection_mode == "mean_confidence":
            selection_mask = teacher_confidence >= teacher_confidence.mean(dim=0, keepdim=True)
        else:
            selection_mask = torch.ones_like(teacher_confidence, dtype=torch.bool)

        missing_mask = ~selection_mask.any(dim=0)
        if missing_mask.any():
            top_teachers = teacher_confidence.argmax(dim=0)
            selection_mask[top_teachers[missing_mask], missing_mask] = True
        return selection_mask

    def _current_proxy_fallback_ratio(self) -> float:
        default_ratio = float(getattr(self.config.inference, "target_general_invocation_rate", 0.0))
        default_ratio = min(max(default_ratio, 0.0), 1.0)
        ratio_mode = str(getattr(self.config.federated, "proxy_fallback_ratio_mode", "holdout_advantage")).lower()
        holdout_ratio = default_ratio
        if self.client_routing_metrics:
            if ratio_mode == "holdout_invocation":
                values = [
                    float(metrics.get("invocation_rate", default_ratio))
                    for metrics in self.client_routing_metrics.values()
                ]
            elif ratio_mode == "post_veto_target":
                values = [
                    float(metrics.get("post_veto_target_rate", metrics.get("effective_target_rate", default_ratio)))
                    for metrics in self.client_routing_metrics.values()
                ]
            else:
                values = [
                    float(
                        metrics.get(
                            "post_veto_target_rate",
                            metrics.get(
                                "beneficial_invocation_rate",
                                metrics.get("invocation_rate", default_ratio),
                            ),
                        )
                    )
                    for metrics in self.client_routing_metrics.values()
                ]
            if values:
                holdout_ratio = sum(values) / max(len(values), 1)

        ema_weight = min(
            max(float(getattr(self.config.federated, "proxy_fallback_real_invocation_ema_weight", 0.7)), 0.0),
            1.0,
        )
        if ratio_mode in {"real_invocation_ema", "ema_invocation"} and self.routing_real_invocation_ema is not None:
            mean_ratio = float(self.routing_real_invocation_ema)
        else:
            mean_ratio = holdout_ratio
            if self.routing_real_invocation_ema is not None:
                mean_ratio = (
                    (ema_weight * float(self.routing_real_invocation_ema))
                    + ((1.0 - ema_weight) * holdout_ratio)
                )
        min_ratio = max(float(getattr(self.config.federated, "min_proxy_fallback_ratio", 0.01)), 0.0)
        max_ratio = min(
            max(float(getattr(self.config.federated, "max_proxy_fallback_ratio", default_ratio)), min_ratio),
            1.0,
        )
        return min(max(mean_ratio, min_ratio), max_ratio)

    def _update_routing_real_invocation_ema(self, invocation_rate: float) -> None:
        value = min(max(float(invocation_rate), 0.0), 1.0)
        self.latest_real_invocation_rate = value
        decay = min(
            max(float(getattr(self.config.federated, "proxy_fallback_invocation_ema_decay", 0.8)), 0.0),
            1.0,
        )
        if self.routing_real_invocation_ema is None:
            self.routing_real_invocation_ema = value
            return
        self.routing_real_invocation_ema = (
            (decay * float(self.routing_real_invocation_ema))
            + ((1.0 - decay) * value)
        )

    def _build_proxy_route_aware_weights(
        self,
        soft_labels: torch.Tensor,
        sample_teacher_weights: torch.Tensor,
        teacher_probs: torch.Tensor,
        normalized_sample_entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_samples = soft_labels.size(0)
        if num_samples == 0:
            empty = torch.empty(0, dtype=soft_labels.dtype)
            return empty, {
                "fallback_weight_mean": 1.0,
                "fallback_weight_invoked_mean": 1.0,
                "fallback_weight_noninvoked_mean": 1.0,
                "proxy_fallback_ratio": 0.0,
                "proxy_teacher_disagreement_mean": 0.0,
                "proxy_route_score_mean": 0.0,
                "proxy_route_score_invoked_mean": 0.0,
                "proxy_expert_confidence_invoked_mean": 0.0,
                "proxy_fallback_target_ratio": 0.0,
                "routing_real_invocation_ema": float(self.routing_real_invocation_ema or 0.0),
                "latest_real_invocation_rate": float(self.latest_real_invocation_rate),
            }

        fallback_weight_max = max(float(getattr(self.config.federated, "fallback_weight_max", 3.0)), 0.0)
        confidence_weight = max(float(getattr(self.config.federated, "fallback_weight_confidence", 0.45)), 0.0)
        route_weight = max(float(getattr(self.config.federated, "fallback_weight_route_score", 0.35)), 0.0)
        disagreement_weight = max(float(getattr(self.config.federated, "fallback_weight_disagreement", 0.20)), 0.0)
        weight_denominator = max(confidence_weight + route_weight + disagreement_weight, 1e-8)

        topk = torch.topk(soft_labels, k=min(2, soft_labels.size(1)), dim=1)
        proxy_confidence = topk.values[:, 0]
        if topk.values.size(1) > 1:
            proxy_margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            proxy_margin = torch.ones_like(proxy_confidence)

        route_confidence_weight = max(float(getattr(self.config.inference, "route_score_confidence_weight", 0.55)), 0.0)
        route_margin_weight = max(float(getattr(self.config.inference, "route_score_margin_weight", 0.30)), 0.0)
        route_weight_denominator = max(route_confidence_weight + route_margin_weight, 1e-8)
        proxy_route_score = (
            (route_confidence_weight * proxy_confidence)
            + (route_margin_weight * proxy_margin)
        ) / route_weight_denominator
        proxy_route_score = proxy_route_score.clamp(0.0, 1.0)

        teacher_predictions = teacher_probs.argmax(dim=2)
        weighted_vote = (
            F.one_hot(teacher_predictions, num_classes=self.config.model.num_classes).to(dtype=teacher_probs.dtype)
            * sample_teacher_weights.unsqueeze(-1)
        ).sum(dim=0)
        proxy_teacher_disagreement = (1.0 - weighted_vote.max(dim=1).values).clamp(0.0, 1.0)

        fallback_weights = torch.ones_like(proxy_confidence)
        target_ratio = 0.0
        if self.general_distill_mode == "route_aware":
            fallback_score = (
                (confidence_weight * (1.0 - proxy_confidence))
                + (route_weight * (1.0 - proxy_route_score))
                + (disagreement_weight * proxy_teacher_disagreement)
            ) / weight_denominator
            fallback_score = fallback_score.clamp(0.0, 1.0)
        else:
            fallback_score = torch.zeros_like(proxy_confidence)

        proxy_fallback_mask = torch.zeros_like(fallback_score, dtype=torch.bool)
        if self.general_distill_mode == "route_aware":
            target_ratio = self._current_proxy_fallback_ratio()
            if target_ratio >= 1.0:
                proxy_fallback_mask[:] = True
            elif target_ratio > 0.0 and fallback_score.numel() > 0:
                top_count = min(max(int(round(fallback_score.numel() * target_ratio)), 1), fallback_score.numel())
                hardest_indices = torch.topk(fallback_score, k=top_count, dim=0).indices
                proxy_fallback_mask[hardest_indices] = True
            if bool(proxy_fallback_mask.any().item()):
                hard_scores = fallback_score[proxy_fallback_mask]
                hard_scale = hard_scores.max().clamp_min(1e-8)
                fallback_weights[proxy_fallback_mask] = 1.0 + (fallback_weight_max * (hard_scores / hard_scale))

        if bool(proxy_fallback_mask.any().item()):
            invoked_weight_mean = float(fallback_weights[proxy_fallback_mask].mean().item())
            invoked_route_score_mean = float(proxy_route_score[proxy_fallback_mask].mean().item())
            invoked_confidence_mean = float(proxy_confidence[proxy_fallback_mask].mean().item())
        else:
            invoked_weight_mean = float(fallback_weights.mean().item())
            invoked_route_score_mean = float(proxy_route_score.mean().item())
            invoked_confidence_mean = float(proxy_confidence.mean().item())

        noninvoked_mask = ~proxy_fallback_mask
        if bool(noninvoked_mask.any().item()):
            noninvoked_weight_mean = float(fallback_weights[noninvoked_mask].mean().item())
        else:
            noninvoked_weight_mean = float(fallback_weights.mean().item())

        stats = {
            "fallback_weight_mean": float(fallback_weights.mean().item()),
            "fallback_weight_invoked_mean": invoked_weight_mean,
            "fallback_weight_noninvoked_mean": noninvoked_weight_mean,
            "proxy_fallback_ratio": float(proxy_fallback_mask.float().mean().item()),
            "proxy_teacher_disagreement_mean": float(proxy_teacher_disagreement.mean().item()),
            "proxy_route_score_mean": float(proxy_route_score.mean().item()),
            "proxy_route_score_invoked_mean": invoked_route_score_mean,
            "proxy_expert_confidence_invoked_mean": invoked_confidence_mean,
            "proxy_teacher_entropy_mean": float(normalized_sample_entropy.mean().item()),
            "proxy_fallback_target_ratio": float(target_ratio),
            "routing_real_invocation_ema": float(self.routing_real_invocation_ema or 0.0),
            "latest_real_invocation_rate": float(self.latest_real_invocation_rate),
        }
        return fallback_weights, stats

    def _select_route_threshold(
        self,
        client_id: str,
        statistics: Dict[str, torch.Tensor],
        previous_threshold: float,
    ) -> Tuple[float, Dict[str, float]]:
        scores = statistics["scores"].float()
        expert_correct = statistics["expert_correct"].to(dtype=torch.float32)
        general_correct = statistics["general_correct"].to(dtype=torch.float32)
        if scores.numel() == 0:
            return previous_threshold, {
                "holdout_routed_accuracy": 0.0,
                "invocation_rate": 0.0,
                "expert_risk": 0.0,
                "threshold": previous_threshold,
                "beneficial_invocation_rate": 0.0,
                "invoked_general_gain_holdout": 0.0,
                "invoked_general_accuracy_holdout": 0.0,
                "invoked_expert_accuracy_holdout": 0.0,
                "effective_target_rate": 0.0,
                "post_veto_target_rate": 0.0,
                "pre_veto_invocation_rate": 0.0,
                "veto_applied": 0.0,
                "holdout_samples": 0.0,
            }

        holdout_samples = int(scores.numel())
        unique_candidates = torch.unique(scores.cpu()).tolist()
        candidates = [float(scores.min().item()) - 1e-6, float(scores.max().item()) + 1e-6, float(previous_threshold)]
        candidates.extend(float(candidate) for candidate in unique_candidates)
        candidates = sorted(set(candidates))

        selection_mode = str(getattr(self.config.inference, "routing_selection_mode", "budget")).lower()
        target_rate = self._target_invocation_rate_for_client(client_id)
        target_expert_risk = max(float(getattr(self.config.inference, "target_expert_risk", 0.25)), 0.0)
        beneficial_rate = float((general_correct > expert_correct).to(dtype=torch.float32).mean().item())
        effective_target_rate = min(target_rate, beneficial_rate)

        best_threshold = float(previous_threshold)
        best_metrics: Dict[str, float] | None = None
        best_rank: Tuple[float, float, float, float, float] | None = None

        for threshold in candidates:
            fallback_mask = scores < float(threshold)
            invocation_rate = float(fallback_mask.to(dtype=torch.float32).mean().item())
            routed_correct = torch.where(fallback_mask, general_correct, expert_correct)
            routed_accuracy = float(routed_correct.mean().item())
            expert_keep = ~fallback_mask
            if bool(expert_keep.any().item()):
                expert_risk = float((1.0 - expert_correct[expert_keep].mean()).item())
            else:
                expert_risk = 1.0

            if bool(fallback_mask.any().item()):
                invoked_general_accuracy = float(general_correct[fallback_mask].mean().item())
                invoked_expert_accuracy = float(expert_correct[fallback_mask].mean().item())
            else:
                invoked_general_accuracy = 0.0
                invoked_expert_accuracy = 0.0
            invoked_general_gain = invoked_general_accuracy - invoked_expert_accuracy

            if selection_mode == "risk":
                violation = max(expert_risk - target_expert_risk, 0.0)
                auxiliary_gap = abs(invocation_rate - effective_target_rate)
            else:
                violation = max(invocation_rate - effective_target_rate, 0.0)
                auxiliary_gap = abs(expert_risk - target_expert_risk)

            gain_penalty = max(-invoked_general_gain, 0.0)
            rank = (
                violation,
                gain_penalty,
                -routed_accuracy,
                auxiliary_gap,
                abs(invocation_rate - beneficial_rate),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_threshold = float(threshold)
                best_metrics = {
                    "holdout_routed_accuracy": routed_accuracy,
                    "invocation_rate": invocation_rate,
                    "expert_risk": expert_risk,
                    "threshold": float(threshold),
                    "beneficial_invocation_rate": beneficial_rate,
                    "invoked_general_gain_holdout": invoked_general_gain,
                    "invoked_general_accuracy_holdout": invoked_general_accuracy,
                    "invoked_expert_accuracy_holdout": invoked_expert_accuracy,
                    "effective_target_rate": effective_target_rate,
                }

        best_metrics = best_metrics or {
            "holdout_routed_accuracy": 0.0,
            "invocation_rate": 0.0,
            "expert_risk": 0.0,
            "threshold": best_threshold,
            "beneficial_invocation_rate": beneficial_rate,
            "invoked_general_gain_holdout": 0.0,
            "invoked_general_accuracy_holdout": 0.0,
            "invoked_expert_accuracy_holdout": 0.0,
            "effective_target_rate": effective_target_rate,
        }
        best_metrics["pre_veto_invocation_rate"] = float(best_metrics.get("invocation_rate", 0.0))
        best_metrics["holdout_samples"] = float(holdout_samples)

        veto_gain_threshold = float(getattr(self.config.inference, "routing_veto_gain_threshold", 0.0))
        veto_min_holdout_samples = max(int(getattr(self.config.inference, "routing_veto_min_holdout_samples", 0)), 0)
        veto_applied = (
            holdout_samples >= veto_min_holdout_samples
            and float(best_metrics.get("invoked_general_gain_holdout", 0.0)) <= veto_gain_threshold
        )
        if veto_applied:
            no_fallback_threshold = float(scores.min().item()) - 1e-6
            expert_only_accuracy = float(expert_correct.mean().item())
            expert_only_risk = float((1.0 - expert_correct.mean()).item())
            best_threshold = no_fallback_threshold
            best_metrics["holdout_routed_accuracy"] = expert_only_accuracy
            best_metrics["invocation_rate"] = 0.0
            best_metrics["expert_risk"] = expert_only_risk
            best_metrics["threshold"] = no_fallback_threshold

        best_metrics["post_veto_target_rate"] = 0.0 if veto_applied else float(
            best_metrics.get("effective_target_rate", effective_target_rate)
        )
        best_metrics["veto_applied"] = 1.0 if veto_applied else 0.0
        return best_threshold, best_metrics

    def _update_routing_thresholds(self, updates):
        super()._update_routing_thresholds(updates)
        if not self.client_routing_metrics:
            return

        updated_client_ids = [update.client_id for update in updates]
        metrics = [
            self.client_routing_metrics[client_id]
            for client_id in updated_client_ids
            if client_id in self.client_routing_metrics
        ]
        if not metrics:
            return

        vetoed_client_ids = [
            client_id
            for client_id in updated_client_ids
            if float(self.client_routing_metrics.get(client_id, {}).get("veto_applied", 0.0)) > 0.5
        ]
        mean_advantage_rate = sum(float(item.get("beneficial_invocation_rate", 0.0)) for item in metrics) / len(metrics)
        mean_effective_target = sum(float(item.get("effective_target_rate", 0.0)) for item in metrics) / len(metrics)
        mean_post_veto_target = sum(float(item.get("post_veto_target_rate", 0.0)) for item in metrics) / len(metrics)
        mean_holdout_gain = sum(float(item.get("invoked_general_gain_holdout", 0.0)) for item in metrics) / len(metrics)
        veto_client_rate = len(vetoed_client_ids) / len(metrics)
        LOGGER.info(
            "%s routing veto | round=%d | vetoed=%d/%d | veto_rate=%.4f | post_target=%.4f | veto_clients=%s",
            self.algorithm_name,
            self.current_round,
            len(vetoed_client_ids),
            len(metrics),
            veto_client_rate,
            mean_post_veto_target,
            vetoed_client_ids,
        )
        for client_id in updated_client_ids:
            client_metrics = self.client_routing_metrics.get(client_id)
            if client_metrics is None:
                continue
            LOGGER.info(
                "%s routing client=%s | veto=%s | holdout_samples=%d | pre_invoke=%.4f | post_invoke=%.4f | target=%.4f | holdout_gain=%.4f | beneficial=%.4f | threshold=%.4f",
                self.algorithm_name,
                client_id,
                bool(float(client_metrics.get("veto_applied", 0.0)) > 0.5),
                int(client_metrics.get("holdout_samples", 0.0)),
                float(client_metrics.get("pre_veto_invocation_rate", client_metrics.get("invocation_rate", 0.0))),
                float(client_metrics.get("invocation_rate", 0.0)),
                float(client_metrics.get("post_veto_target_rate", 0.0)),
                float(client_metrics.get("invoked_general_gain_holdout", 0.0)),
                float(client_metrics.get("beneficial_invocation_rate", 0.0)),
                float(client_metrics.get("threshold", 0.0)),
            )
        LOGGER.info(
            "%s routing advantage | round=%d | beneficial=%.4f | effective_target=%.4f | invoked_gain=%.4f",
            self.algorithm_name,
            self.current_round,
            mean_advantage_rate,
            mean_effective_target,
            mean_holdout_gain,
        )
        if self.writer is not None:
            self._log_compare_scalars(
                self.algorithm_name,
                self.current_round,
                {
                    "routing_beneficial_invocation_rate": mean_advantage_rate,
                    "routing_effective_target_rate": mean_effective_target,
                    "routing_post_veto_target_rate": mean_post_veto_target,
                    "routing_veto_client_rate": veto_client_rate,
                    "routing_veto_client_count": float(len(vetoed_client_ids)),
                    "routing_holdout_invoked_general_gain": mean_holdout_gain,
                },
            )

    def _evaluate_route_link_metrics(self) -> Dict[str, float]:
        total_samples = 0
        invoked_total = 0
        routed_correct = 0
        oracle_correct = 0
        oracle_general_invocations = 0
        invoked_general_correct = 0
        invoked_expert_correct = 0
        disagreement_total = 0

        self.general_model.eval()
        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()
                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    route_features = self._compute_route_features(
                        client_id=client_id,
                        expert_model=expert_model,
                        images=images,
                    )
                    expert_predictions = route_features["expert_prediction"]
                    route_threshold = float(
                        self.client_route_score_thresholds.get(client_id, self._initial_route_score_threshold())
                    )
                    fallback_mask = route_features["route_score"] < route_threshold
                    general_predictions = self.general_model(images).argmax(dim=1)
                    routed_predictions = expert_predictions.clone()
                    routed_predictions[fallback_mask] = general_predictions[fallback_mask]

                    total_samples += int(targets_device.numel())
                    invoked_total += int(fallback_mask.sum().item())
                    routed_correct += int((routed_predictions == targets_device).sum().item())
                    oracle_correct += int(
                        ((expert_predictions == targets_device) | (general_predictions == targets_device)).sum().item()
                    )
                    oracle_general_invocations += int(
                        ((expert_predictions != targets_device) & (general_predictions == targets_device)).sum().item()
                    )
                    disagreement_total += int((expert_predictions != general_predictions).sum().item())

                    if fallback_mask.any():
                        invoked_general_correct += int(
                            (general_predictions[fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )
                        invoked_expert_correct += int(
                            (expert_predictions[fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )

        invoked_general_accuracy = invoked_general_correct / max(invoked_total, 1)
        invoked_expert_accuracy = invoked_expert_correct / max(invoked_total, 1)
        oracle_route_accuracy = oracle_correct / max(total_samples, 1)
        routed_accuracy = routed_correct / max(total_samples, 1)
        return {
            "invoked_general_accuracy": invoked_general_accuracy,
            "invoked_expert_accuracy": invoked_expert_accuracy,
            "invoked_general_gain": invoked_general_accuracy - invoked_expert_accuracy,
            "oracle_route_accuracy": oracle_route_accuracy,
            "oracle_general_invocation_rate": oracle_general_invocations / max(total_samples, 1),
            "expert_bad_general_good_rate": oracle_general_invocations / max(total_samples, 1),
            "routing_regret": oracle_route_accuracy - routed_accuracy,
            "expert_general_disagreement_rate": disagreement_total / max(total_samples, 1),
        }

    def _build_round_extra_metrics(
        self,
        round_idx: int,
        distill_stats: Dict[str, float],
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        base_metrics = super()._build_round_extra_metrics(
            round_idx=round_idx,
            distill_stats=distill_stats,
            expert_eval=expert_eval,
            general_eval=general_eval,
            routed_eval=routed_eval,
        )
        self._update_routing_real_invocation_ema(float(routed_eval["aggregate"]["invocation_rate"]))
        base_metrics["routing_real_invocation_ema"] = float(self.routing_real_invocation_ema or 0.0)
        base_metrics["latest_real_invocation_rate"] = float(self.latest_real_invocation_rate)
        return base_metrics

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        base_metrics = super()._build_final_extra_metrics(
            expert_eval=expert_eval,
            general_eval=general_eval,
            routed_eval=routed_eval,
        )
        return {
            "distill_loss": self.latest_distill_stats.get("total_loss", 0.0),
            "kd_loss": self.latest_distill_stats.get("kd_loss", 0.0),
            "ce_loss": self.latest_distill_stats.get("ce_loss", 0.0),
            "alpha_mean": self.latest_distill_stats.get("alpha_mean", 0.0),
            "distill_kd_loss": self.latest_distill_stats.get("kd_loss", 0.0),
            "distill_ce_loss": self.latest_distill_stats.get("ce_loss", 0.0),
            "distill_alpha_mean": self.latest_distill_stats.get("alpha_mean", 0.0),
            **base_metrics,
            **self.latest_distill_stats,
            "routing_real_invocation_ema": float(self.routing_real_invocation_ema or 0.0),
            "latest_real_invocation_rate": float(self.latest_real_invocation_rate),
        }

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        route_text = (
            f" | g_gain={extra_metrics.get('general_gain_over_expert', 0.0):.4f}"
            f" | route_gain={extra_metrics.get('routed_gain_over_expert', 0.0):.4f}"
            f" | invoked_g={extra_metrics.get('invoked_general_gain', 0.0):.4f}"
            f" | oracle={extra_metrics.get('oracle_route_accuracy', 0.0):.4f}"
            f" | oracle_g={extra_metrics.get('oracle_general_invocation_rate', 0.0):.4f}"
            f" | e_bad_g_good={extra_metrics.get('expert_bad_general_good_rate', 0.0):.4f}"
            f" | regret={extra_metrics.get('routing_regret', 0.0):.4f}"
        )
        if not self.latest_distill_stats:
            return route_text if extra_metrics else ""
        return (
            route_text
            + f" | fb_w={self.latest_distill_stats.get('fallback_weight_mean', 0.0):.4f}"
            + f" | proxy_fb={self.latest_distill_stats.get('proxy_fallback_ratio', 0.0):.4f}"
            + f" | proxy_tgt={self.latest_distill_stats.get('proxy_fallback_target_ratio', 0.0):.4f}"
            + f" | real_inv_ema={extra_metrics.get('routing_real_invocation_ema', 0.0):.4f}"
            + f" | proxy_dis={self.latest_distill_stats.get('proxy_teacher_disagreement_mean', 0.0):.4f}"
        )

    def _extract_ensemble_logits(self, updates) -> Dict[str, torch.Tensor]:
        self._refresh_teacher_bank(updates)

        if not self.teacher_bank:
            raise RuntimeError("Teacher bank is empty; cannot distill the general model.")

        temperature = float(self.config.federated.distill_temperature)
        staleness_decay = min(
            max(float(getattr(self.config.federated, "teacher_bank_staleness_decay", 0.99)), 0.0),
            1.0,
        )

        all_logits: List[torch.Tensor] = []
        teacher_weights: List[float] = []

        for entry in self.teacher_bank.values():
            client_logits = entry["logits"]
            stale_rounds_count = max(int(self.current_round) - int(entry["last_round"]), 0)
            all_logits.append(client_logits)
            freshness_weight = float(staleness_decay ** stale_rounds_count)
            sample_weight = math.sqrt(max(int(entry["num_samples"]), 1))
            confidence_weight = max(1.0 - float(entry.get("mean_normalized_entropy", 0.0)), 1e-3)
            teacher_weights.append(freshness_weight * sample_weight * confidence_weight)

        stacked_logits = torch.stack(all_logits, dim=0)
        normalized_teacher_weights = torch.tensor(
            teacher_weights,
            dtype=stacked_logits.dtype,
        )
        normalized_teacher_weights = normalized_teacher_weights / normalized_teacher_weights.sum().clamp_min(1e-8)
        teacher_probs = F.softmax(stacked_logits / temperature, dim=2)
        teacher_confidence = teacher_probs.max(dim=2).values
        selection_mask = self._select_teacher_mask(teacher_confidence)
        sample_teacher_weights = normalized_teacher_weights.view(-1, 1) * selection_mask.to(dtype=stacked_logits.dtype)
        sample_teacher_weights = sample_teacher_weights / sample_teacher_weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        avg_logits = (stacked_logits * sample_teacher_weights.unsqueeze(-1)).sum(0)
        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        hard_labels = avg_logits.argmax(dim=1)

        num_classes = self.config.model.num_classes
        log_c = math.log(float(num_classes))
        sample_entropy = -(soft_labels * soft_labels.clamp_min(1e-8).log()).sum(1)
        normalized_sample_entropy = sample_entropy / max(log_c, 1e-8)
        fallback_weights, proxy_stats = self._build_proxy_route_aware_weights(
            soft_labels=soft_labels,
            sample_teacher_weights=sample_teacher_weights,
            teacher_probs=teacher_probs,
            normalized_sample_entropy=normalized_sample_entropy,
        )
        effective_size = float(1.0 / normalized_teacher_weights.pow(2).sum().clamp_min(1e-8).item())
        selected_teacher_count_mean = float(selection_mask.float().sum(dim=0).mean().item())
        selected_teacher_coverage = float(selection_mask.any(dim=1).float().mean().item())
        self.latest_teacher_bank_stats = {
            **self._summarize_teacher_bank(),
            "teacher_bank_effective_size": effective_size,
            "teacher_weight_max_share": float(normalized_teacher_weights.max().item()),
            "teacher_confidence_mean": float(teacher_confidence.mean().item()),
            "teacher_entropy_mean": float(normalized_sample_entropy.mean().item()),
            "selected_teacher_count_mean": selected_teacher_count_mean,
            "selected_teacher_coverage": selected_teacher_coverage,
            "teacher_topk_selected_mean": selected_teacher_count_mean,
            "teacher_selected_coverage": selected_teacher_coverage,
            **proxy_stats,
        }

        LOGGER.info(
            "%s teacher bank | round=%d | teachers=%d | avg_staleness=%.2f | max_staleness=%.2f | memory=%.3f MB | eff=%.2f | max_share=%.4f | conf=%.4f | entropy=%.4f | selected=%.2f | coverage=%.4f | fallback_w=%.4f | fallback_ratio=%.4f | proxy_disagree=%.4f",
            self.algorithm_name,
            self.current_round,
            int(self.latest_teacher_bank_stats["teacher_bank_size"]),
            self.latest_teacher_bank_stats["teacher_bank_avg_staleness"],
            self.latest_teacher_bank_stats["teacher_bank_max_staleness"],
            self.latest_teacher_bank_stats["teacher_bank_memory_mb"],
            self.latest_teacher_bank_stats["teacher_bank_effective_size"],
            self.latest_teacher_bank_stats["teacher_weight_max_share"],
            self.latest_teacher_bank_stats["teacher_confidence_mean"],
            self.latest_teacher_bank_stats["teacher_entropy_mean"],
            self.latest_teacher_bank_stats["selected_teacher_count_mean"],
            self.latest_teacher_bank_stats["selected_teacher_coverage"],
            self.latest_teacher_bank_stats["fallback_weight_mean"],
            self.latest_teacher_bank_stats["proxy_fallback_ratio"],
            self.latest_teacher_bank_stats["proxy_teacher_disagreement_mean"],
        )

        return {
            "soft_labels": soft_labels,
            "hard_labels": hard_labels,
            "normalized_sample_entropy": normalized_sample_entropy,
            "fallback_weights": fallback_weights,
        }

    def _distill_general_model(self, ensemble: Dict[str, torch.Tensor], round_idx: int) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = self.config.dataset.batch_size
        grad_clip_norm = max(float(getattr(self.config.federated, "distill_gradient_clip_norm", 1.0)), 0.0)
        alpha_min = min(max(float(getattr(self.config.federated, "uncertainty_alpha_min", 0.2)), 0.0), 1.0)
        alpha_max = min(max(float(getattr(self.config.federated, "uncertainty_alpha_max", 0.8)), alpha_min), 1.0)

        soft_all = ensemble["soft_labels"]
        hard_all = ensemble["hard_labels"]
        normalized_entropy_all = ensemble["normalized_sample_entropy"]
        fallback_weights_all = ensemble.get("fallback_weights")

        all_images: List[torch.Tensor] = []
        for batch in self.distill_loader:
            all_images.append(batch[0])
        distill_images = torch.cat(all_images, dim=0)
        num_samples = distill_images.size(0)

        optimizer = torch.optim.Adam(
            self.general_model.parameters(),
            lr=float(self.config.federated.distill_lr),
        )
        total_steps = distill_epochs * ((num_samples + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        self.general_model.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        total_alpha = 0.0
        total_ema_kd_loss = 0.0
        total_batches = 0
        ema_kd_weight = 0.1

        for _ in range(distill_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = permutation[start:start + batch_size]
                images = distill_images[idx].to(self.device)
                batch_soft = soft_all[idx].to(self.device)
                batch_hard = hard_all[idx].to(self.device)
                batch_normalized_entropy = normalized_entropy_all[idx].to(self.device)
                batch_fallback_weights = (
                    fallback_weights_all[idx].to(self.device)
                    if fallback_weights_all is not None
                    else torch.ones_like(batch_normalized_entropy)
                )
                batch_entropy = float(batch_normalized_entropy.mean().item())
                batch_alpha = alpha_min + ((alpha_max - alpha_min) * batch_entropy)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                per_sample_kd = F.kl_div(
                    student_log_probs,
                    batch_soft,
                    reduction="none",
                ).sum(1) * (temperature ** 2)
                per_sample_ce = F.cross_entropy(student_logits, batch_hard, reduction="none")
                combined = (batch_alpha * per_sample_kd) + ((1.0 - batch_alpha) * per_sample_ce)
                ema_kd_loss = torch.zeros(1, device=self.device, dtype=student_logits.dtype).squeeze(0)
                if self.enable_general_ema_anchor:
                    ema_mask = batch_normalized_entropy > batch_entropy
                    if ema_mask.any():
                        with torch.no_grad():
                            ema_logits = self.general_ema_teacher(images[ema_mask])
                            ema_probs = F.softmax(ema_logits / temperature, dim=1)
                        ema_student_log_probs = F.log_softmax(student_logits[ema_mask] / temperature, dim=1)
                        ema_kd_loss = (
                            F.kl_div(ema_student_log_probs, ema_probs, reduction="none").sum(1) * (temperature ** 2)
                        ).mean()
                weighted_combined = combined * batch_fallback_weights
                loss = (
                    weighted_combined.sum() / batch_fallback_weights.sum().clamp_min(1e-8)
                ) + (ema_kd_weight * ema_kd_loss)
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.general_model.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.detach().cpu().item())
                total_kd_loss += float(per_sample_kd.mean().detach().cpu().item())
                total_ce_loss += float(per_sample_ce.mean().detach().cpu().item())
                total_alpha += batch_alpha
                total_ema_kd_loss += float(ema_kd_loss.detach().cpu().item())
                total_batches += 1

        self._update_general_ema_teacher()
        stats = {
            "total_loss": total_loss / max(total_batches, 1),
            "kd_loss": total_kd_loss / max(total_batches, 1),
            "ce_loss": total_ce_loss / max(total_batches, 1),
            "alpha_mean": total_alpha / max(total_batches, 1),
            "ema_kd_loss": total_ema_kd_loss / max(total_batches, 1),
            **self.latest_teacher_bank_stats,
        }
        self.latest_distill_stats = dict(stats)
        return stats

    def _log_algorithm_process_metrics(
        self,
        algorithm_name: str,
        round_idx: int,
        avg_loss: float,
        distill_stats: Dict[str, float],
    ) -> None:
        super()._log_algorithm_process_metrics(algorithm_name, round_idx, avg_loss, distill_stats)
        if self.writer is None:
            return
        self._log_compare_scalars(
            algorithm_name,
            round_idx,
            {
                "fallback_weight_mean": distill_stats.get("fallback_weight_mean"),
                "fallback_weight_invoked_mean": distill_stats.get("fallback_weight_invoked_mean"),
                "fallback_weight_noninvoked_mean": distill_stats.get("fallback_weight_noninvoked_mean"),
                "proxy_fallback_ratio": distill_stats.get("proxy_fallback_ratio"),
                "proxy_fallback_target_ratio": distill_stats.get("proxy_fallback_target_ratio"),
                "teacher_topk_selected_mean": distill_stats.get("teacher_topk_selected_mean"),
                "teacher_selected_coverage": distill_stats.get("teacher_selected_coverage"),
                "proxy_teacher_disagreement_mean": distill_stats.get("proxy_teacher_disagreement_mean"),
                "proxy_route_score_mean": distill_stats.get("proxy_route_score_mean"),
                "proxy_route_score_invoked_mean": distill_stats.get("proxy_route_score_invoked_mean"),
                "proxy_expert_confidence_invoked_mean": distill_stats.get("proxy_expert_confidence_invoked_mean"),
            },
        )
