"""
FedEGS-EU: FedEGSE with uncertainty-aware dual-gated routing.

Design:
  - Keep the current FedEGSE server-side general distillation pipeline.
  - Replace the routed inference policy with a joint gate:
      1. expert route score must indicate fallback is needed,
      2. general reliability must indicate the fallback is trustworthy.
  - Keep per-client routing adaptation, but extend it to learn both:
      1. a route-score threshold,
      2. a general-reliability threshold.
  - Apply a client-level veto when holdout evidence suggests the general path
    is not reliably beneficial for that client.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from fedegs.federated.common import LOGGER
from fedegs.federated.algorithms.fedegse import FedEGSEServer


class FedEGSEUServer(FedEGSEServer):
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
        self.algorithm_name = "fedegseu"
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer=writer,
            public_dataset=public_dataset,
        )
        self.algorithm_name = "fedegseu"
        initial_reliability_threshold = self._initial_general_reliability_threshold()
        self.client_general_reliability_thresholds = {
            client_id: initial_reliability_threshold for client_id in client_datasets
        }
        self.client_general_route_enabled = {client_id: True for client_id in client_datasets}
        self.client_general_advantage_ema = {client_id: 0.0 for client_id in client_datasets}
        self.client_general_reliability_ema = {
            client_id: initial_reliability_threshold for client_id in client_datasets
        }
        LOGGER.info(
            "%s routing mode | uncertainty-aware dual gate with expert-risk + general-reliability",
            self.algorithm_name,
        )

    def _initial_general_reliability_threshold(self) -> float:
        return float(getattr(self.config.inference, "general_reliability_threshold", 0.55))

    def _general_reliability_candidate_values(
        self,
        reliabilities: torch.Tensor,
        previous_threshold: float,
    ) -> List[float]:
        rel_min = float(getattr(self.config.inference, "routing_general_reliability_min", 0.30))
        rel_max = float(getattr(self.config.inference, "routing_general_reliability_max", 0.90))
        rel_min = min(max(rel_min, 0.0), 1.0)
        rel_max = min(max(rel_max, rel_min), 1.0)
        num_candidates = max(int(getattr(self.config.inference, "routing_general_reliability_candidates", 9)), 2)
        if reliabilities.numel() == 0:
            return [min(max(previous_threshold, rel_min), rel_max)]

        quantiles = torch.linspace(0.0, 1.0, steps=num_candidates)
        quantile_values = torch.quantile(reliabilities.cpu(), quantiles).tolist()
        candidates = {min(max(float(value), rel_min), rel_max) for value in quantile_values}
        candidates.add(min(max(float(previous_threshold), rel_min), rel_max))
        candidates.add(rel_min)
        candidates.add(rel_max)
        return sorted(candidates)

    def _compute_general_reliability_features(self, general_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = torch.softmax(general_logits, dim=1)
        topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
        confidence = topk.values[:, 0]
        prediction = topk.indices[:, 0]
        if topk.values.size(1) > 1:
            margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            margin = torch.ones_like(confidence)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / max(math.log(float(probs.size(1))), 1e-8)

        confidence_weight = max(
            float(getattr(self.config.inference, "general_reliability_confidence_weight", 0.45)),
            0.0,
        )
        margin_weight = max(
            float(getattr(self.config.inference, "general_reliability_margin_weight", 0.25)),
            0.0,
        )
        entropy_weight = max(
            float(getattr(self.config.inference, "general_reliability_entropy_weight", 0.30)),
            0.0,
        )
        denominator = max(confidence_weight + margin_weight + entropy_weight, 1e-8)
        reliability = (
            (confidence_weight * confidence)
            + (margin_weight * margin)
            + (entropy_weight * (1.0 - entropy))
        ) / denominator
        reliability = reliability.clamp(0.0, 1.0)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "reliability": reliability,
            "logits": general_logits,
        }

    def _joint_fallback_mask(
        self,
        client_id: str,
        route_score: torch.Tensor,
        general_reliability: torch.Tensor,
    ) -> torch.Tensor:
        if not self.client_general_route_enabled.get(client_id, True):
            return torch.zeros_like(route_score, dtype=torch.bool)
        route_threshold = float(
            self.client_route_score_thresholds.get(client_id, self._initial_route_score_threshold())
        )
        reliability_threshold = float(
            self.client_general_reliability_thresholds.get(
                client_id,
                self._initial_general_reliability_threshold(),
            )
        )
        return (route_score < route_threshold) & (general_reliability >= reliability_threshold)

    def _predict_routed(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            route_features = self._compute_route_features(
                client_id=client_id,
                expert_model=self.clients[client_id].expert_model,
                images=images,
            )
            predictions = route_features["expert_prediction"].clone()
            route_types = ["expert"] * images.size(0)

            if self.client_general_route_enabled.get(client_id, True):
                general_logits = self.general_model(images)
                general_features = self._compute_general_reliability_features(general_logits)
                fallback_mask = self._joint_fallback_mask(
                    client_id,
                    route_features["route_score"],
                    general_features["reliability"],
                )
                if fallback_mask.any():
                    predictions[fallback_mask] = general_features["prediction"][fallback_mask]
                    for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                        route_types[sample_idx] = "general"
                invoked_general = int(fallback_mask.sum().item())
                metadata = {
                    "route_type": route_types,
                    "expert_confidence": route_features["confidence"].detach().cpu().tolist(),
                    "general_reliability": general_features["reliability"].detach().cpu().tolist(),
                }
                return predictions, invoked_general, metadata

            metadata = {
                "route_type": route_types,
                "expert_confidence": route_features["confidence"].detach().cpu().tolist(),
                "general_reliability": [0.0] * images.size(0),
            }
        return predictions, 0, metadata

    def _collect_client_routing_statistics(
        self,
        client_id: str,
        expert_model,
        temperature: float,
        prototypes,
        prototype_scales,
        prototype_mask,
    ) -> Optional[Dict[str, torch.Tensor]]:
        routing_dataset = self.client_routing_datasets.get(client_id)
        if routing_dataset is None or len(routing_dataset) == 0:
            return None

        loader = self.data_module.make_loader(routing_dataset, shuffle=False)
        score_batches: List[torch.Tensor] = []
        expert_correct_batches: List[torch.Tensor] = []
        general_correct_batches: List[torch.Tensor] = []
        general_reliability_batches: List[torch.Tensor] = []

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                route_features = self._compute_route_features(
                    client_id=client_id,
                    expert_model=expert_model,
                    images=images,
                    temperature=temperature,
                    prototypes=prototypes,
                    prototype_scales=prototype_scales,
                    prototype_mask=prototype_mask,
                )
                general_logits = self.general_model(images)
                general_features = self._compute_general_reliability_features(general_logits)
                score_batches.append(route_features["route_score"].detach().cpu())
                expert_correct_batches.append(route_features["expert_prediction"].eq(targets_device).detach().cpu())
                general_correct_batches.append(general_features["prediction"].eq(targets_device).detach().cpu())
                general_reliability_batches.append(general_features["reliability"].detach().cpu())

        if not score_batches:
            return None
        return {
            "scores": torch.cat(score_batches, dim=0),
            "expert_correct": torch.cat(expert_correct_batches, dim=0).to(dtype=torch.bool),
            "general_correct": torch.cat(general_correct_batches, dim=0).to(dtype=torch.bool),
            "general_reliability": torch.cat(general_reliability_batches, dim=0),
        }

    def _select_route_threshold(
        self,
        client_id: str,
        statistics: Dict[str, torch.Tensor],
        previous_threshold: float,
        previous_reliability_threshold: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        scores = statistics["scores"].float()
        expert_correct = statistics["expert_correct"].to(dtype=torch.float32)
        general_correct = statistics["general_correct"].to(dtype=torch.float32)
        general_reliability = statistics["general_reliability"].float().clamp(0.0, 1.0)
        holdout_samples = int(scores.numel())
        if holdout_samples == 0:
            return previous_threshold, previous_reliability_threshold, {
                "holdout_routed_accuracy": 0.0,
                "invocation_rate": 0.0,
                "expert_risk": 0.0,
                "threshold": previous_threshold,
                "general_reliability_threshold": previous_reliability_threshold,
                "beneficial_invocation_rate": 0.0,
                "invoked_general_gain_holdout": 0.0,
                "invoked_general_accuracy_holdout": 0.0,
                "invoked_expert_accuracy_holdout": 0.0,
                "effective_target_rate": 0.0,
                "post_veto_target_rate": 0.0,
                "pre_veto_invocation_rate": 0.0,
                "veto_applied": 0.0,
                "holdout_samples": 0.0,
                "general_reliability_mean_holdout": 0.0,
                "invoked_general_reliability_holdout": 0.0,
                "client_general_advantage_ema": 0.0,
                "client_general_reliability_ema": 0.0,
                "general_route_enabled": 0.0,
            }

        route_candidates = [float(scores.min().item()) - 1e-6, float(scores.max().item()) + 1e-6, float(previous_threshold)]
        route_candidates.extend(float(candidate) for candidate in torch.unique(scores.cpu()).tolist())
        route_candidates = sorted(set(route_candidates))
        reliability_candidates = self._general_reliability_candidate_values(
            general_reliability,
            previous_reliability_threshold,
        )

        selection_mode = str(getattr(self.config.inference, "routing_selection_mode", "budget")).lower()
        target_rate = self._target_invocation_rate_for_client(client_id)
        target_expert_risk = max(float(getattr(self.config.inference, "target_expert_risk", 0.25)), 0.0)
        best_threshold = float(previous_threshold)
        best_reliability_threshold = float(previous_reliability_threshold)
        best_metrics: Optional[Dict[str, float]] = None
        best_rank: Optional[Tuple[float, float, float, float, float, float]] = None
        overall_reliability_mean = float(general_reliability.mean().item())

        for reliability_threshold in reliability_candidates:
            reliable_mask = general_reliability >= float(reliability_threshold)
            beneficial_feasible_rate = float(
                ((general_correct > expert_correct) & reliable_mask).to(dtype=torch.float32).mean().item()
            )
            effective_target_rate = min(target_rate, beneficial_feasible_rate)

            for threshold in route_candidates:
                fallback_mask = (scores < float(threshold)) & reliable_mask
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
                    invoked_general_reliability = float(general_reliability[fallback_mask].mean().item())
                else:
                    invoked_general_accuracy = 0.0
                    invoked_expert_accuracy = 0.0
                    invoked_general_reliability = 0.0
                invoked_general_gain = invoked_general_accuracy - invoked_expert_accuracy

                if selection_mode == "risk":
                    violation = max(expert_risk - target_expert_risk, 0.0)
                    auxiliary_gap = abs(invocation_rate - effective_target_rate)
                else:
                    violation = max(invocation_rate - effective_target_rate, 0.0)
                    auxiliary_gap = abs(expert_risk - target_expert_risk)

                gain_penalty = max(-invoked_general_gain, 0.0)
                reliability_penalty = max(
                    float(getattr(self.config.inference, "routing_veto_reliability_threshold", 0.55))
                    - invoked_general_reliability,
                    0.0,
                )
                rank = (
                    violation,
                    gain_penalty,
                    reliability_penalty,
                    -routed_accuracy,
                    auxiliary_gap,
                    abs(invocation_rate - beneficial_feasible_rate),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_threshold = float(threshold)
                    best_reliability_threshold = float(reliability_threshold)
                    best_metrics = {
                        "holdout_routed_accuracy": routed_accuracy,
                        "invocation_rate": invocation_rate,
                        "expert_risk": expert_risk,
                        "threshold": float(threshold),
                        "general_reliability_threshold": float(reliability_threshold),
                        "beneficial_invocation_rate": beneficial_feasible_rate,
                        "invoked_general_gain_holdout": invoked_general_gain,
                        "invoked_general_accuracy_holdout": invoked_general_accuracy,
                        "invoked_expert_accuracy_holdout": invoked_expert_accuracy,
                        "effective_target_rate": effective_target_rate,
                        "general_reliability_mean_holdout": overall_reliability_mean,
                        "invoked_general_reliability_holdout": invoked_general_reliability,
                    }

        best_metrics = best_metrics or {
            "holdout_routed_accuracy": 0.0,
            "invocation_rate": 0.0,
            "expert_risk": 0.0,
            "threshold": best_threshold,
            "general_reliability_threshold": best_reliability_threshold,
            "beneficial_invocation_rate": 0.0,
            "invoked_general_gain_holdout": 0.0,
            "invoked_general_accuracy_holdout": 0.0,
            "invoked_expert_accuracy_holdout": 0.0,
            "effective_target_rate": 0.0,
            "general_reliability_mean_holdout": overall_reliability_mean,
            "invoked_general_reliability_holdout": 0.0,
        }
        best_metrics["pre_veto_invocation_rate"] = float(best_metrics.get("invocation_rate", 0.0))
        best_metrics["holdout_samples"] = float(holdout_samples)

        advantage_decay = min(
            max(float(getattr(self.config.inference, "routing_veto_advantage_ema_decay", 0.7)), 0.0),
            1.0,
        )
        reliability_decay = min(
            max(float(getattr(self.config.inference, "routing_veto_reliability_ema_decay", 0.7)), 0.0),
            1.0,
        )
        current_advantage = float(best_metrics.get("invoked_general_gain_holdout", 0.0))
        current_reliability = float(best_metrics.get("invoked_general_reliability_holdout", 0.0))
        advantage_ema = (
            (advantage_decay * float(self.client_general_advantage_ema.get(client_id, 0.0)))
            + ((1.0 - advantage_decay) * current_advantage)
        )
        reliability_ema = (
            (reliability_decay * float(self.client_general_reliability_ema.get(client_id, self._initial_general_reliability_threshold())))
            + ((1.0 - reliability_decay) * current_reliability)
        )
        best_metrics["client_general_advantage_ema"] = advantage_ema
        best_metrics["client_general_reliability_ema"] = reliability_ema

        veto_gain_threshold = float(getattr(self.config.inference, "routing_veto_gain_threshold", 0.0))
        veto_reliability_threshold = float(getattr(self.config.inference, "routing_veto_reliability_threshold", 0.55))
        veto_min_holdout_samples = max(int(getattr(self.config.inference, "routing_veto_min_holdout_samples", 0)), 0)
        veto_applied = (
            holdout_samples >= veto_min_holdout_samples
            and (
                advantage_ema <= veto_gain_threshold
                or reliability_ema <= veto_reliability_threshold
            )
        )
        if veto_applied:
            no_fallback_threshold = float(scores.min().item()) - 1e-6
            expert_only_accuracy = float(expert_correct.mean().item())
            expert_only_risk = float((1.0 - expert_correct.mean()).item())
            best_threshold = no_fallback_threshold
            best_reliability_threshold = 1.0
            best_metrics["holdout_routed_accuracy"] = expert_only_accuracy
            best_metrics["invocation_rate"] = 0.0
            best_metrics["expert_risk"] = expert_only_risk
            best_metrics["threshold"] = no_fallback_threshold
            best_metrics["general_reliability_threshold"] = 1.0

        best_metrics["post_veto_target_rate"] = 0.0 if veto_applied else float(
            best_metrics.get("effective_target_rate", 0.0)
        )
        best_metrics["veto_applied"] = 1.0 if veto_applied else 0.0
        best_metrics["general_route_enabled"] = 0.0 if veto_applied else 1.0
        return best_threshold, best_reliability_threshold, best_metrics

    def _update_routing_thresholds(self, updates):
        updated_metrics: List[Dict[str, float]] = []
        for update in updates:
            client_id = update.client_id
            client = self.clients[client_id]
            temperature = self._estimate_expert_temperature(client_id, client.expert_model, self.current_round)
            prototypes, prototype_scales, prototype_mask = self._compute_client_prototypes(client_id, client.expert_model)
            self.client_expert_temperatures[client_id] = temperature
            self.client_prototypes[client_id] = prototypes.detach().cpu()
            self.client_prototype_scales[client_id] = prototype_scales.detach().cpu()
            self.client_prototype_masks[client_id] = prototype_mask.detach().cpu()

            routing_statistics = self._collect_client_routing_statistics(
                client_id=client_id,
                expert_model=client.expert_model,
                temperature=temperature,
                prototypes=prototypes,
                prototype_scales=prototype_scales,
                prototype_mask=prototype_mask,
            )
            previous_threshold = float(
                self.client_route_score_thresholds.get(client_id, self._initial_route_score_threshold())
            )
            previous_reliability_threshold = float(
                self.client_general_reliability_thresholds.get(
                    client_id,
                    self._initial_general_reliability_threshold(),
                )
            )
            if routing_statistics is None:
                self.client_route_score_thresholds[client_id] = previous_threshold
                self.client_general_reliability_thresholds[client_id] = previous_reliability_threshold
                continue

            threshold, reliability_threshold, metrics = self._select_route_threshold(
                client_id=client_id,
                statistics=routing_statistics,
                previous_threshold=previous_threshold,
                previous_reliability_threshold=previous_reliability_threshold,
            )
            self.client_route_score_thresholds[client_id] = threshold
            self.client_general_reliability_thresholds[client_id] = reliability_threshold
            self.client_general_route_enabled[client_id] = bool(float(metrics.get("general_route_enabled", 1.0)) > 0.5)
            self.client_general_advantage_ema[client_id] = float(metrics.get("client_general_advantage_ema", 0.0))
            self.client_general_reliability_ema[client_id] = float(
                metrics.get("client_general_reliability_ema", self._initial_general_reliability_threshold())
            )
            metrics["temperature"] = temperature
            self.client_routing_metrics[client_id] = metrics
            updated_metrics.append(metrics)

        if not updated_metrics:
            return

        mean_threshold = sum(item["threshold"] for item in updated_metrics) / len(updated_metrics)
        mean_temperature = sum(item["temperature"] for item in updated_metrics) / len(updated_metrics)
        mean_holdout_accuracy = sum(item["holdout_routed_accuracy"] for item in updated_metrics) / len(updated_metrics)
        mean_invocation = sum(item["invocation_rate"] for item in updated_metrics) / len(updated_metrics)
        mean_reliability_threshold = sum(item["general_reliability_threshold"] for item in updated_metrics) / len(updated_metrics)
        mean_invoked_reliability = sum(item["invoked_general_reliability_holdout"] for item in updated_metrics) / len(updated_metrics)
        enabled_rate = sum(item["general_route_enabled"] for item in updated_metrics) / len(updated_metrics)
        veto_rate = 1.0 - enabled_rate
        LOGGER.info(
            "%s routing update | round=%d | clients=%d | holdout_acc=%.4f | invoke=%.4f | threshold=%.4f | rel_thr=%.4f | rel_invoked=%.4f | enabled=%.4f | temp=%.4f",
            self.algorithm_name,
            self.current_round,
            len(updated_metrics),
            mean_holdout_accuracy,
            mean_invocation,
            mean_threshold,
            mean_reliability_threshold,
            mean_invoked_reliability,
            enabled_rate,
            mean_temperature,
        )
        if self.writer is not None:
            self._log_compare_scalars(
                self.algorithm_name,
                self.current_round,
                {
                    "routing_holdout_accuracy": mean_holdout_accuracy,
                    "routing_score_threshold": mean_threshold,
                    "routing_temperature": mean_temperature,
                    "routing_general_reliability_threshold": mean_reliability_threshold,
                    "routing_general_route_enabled_rate": enabled_rate,
                    "routing_veto_client_rate": veto_rate,
                    "routing_holdout_general_reliability": mean_invoked_reliability,
                },
            )

    def _evaluate_route_effectiveness_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        expert_macro_accuracy = float(expert_eval["macro"]["accuracy"])
        general_macro_accuracy = float(general_eval["macro"]["accuracy"])
        routed_macro_accuracy = float(routed_eval["macro"]["accuracy"])

        client_metrics: Dict[str, Dict[str, float]] = {}
        self.general_model.eval()
        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()

                num_samples = 0
                invoked_total = 0
                expert_correct = 0
                general_correct = 0
                routed_correct = 0
                oracle_correct = 0
                oracle_general_invocations = 0
                disagreement_total = 0
                invoked_general_correct = 0
                invoked_expert_correct = 0
                invoked_general_reliability = 0.0

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    route_features = self._compute_route_features(
                        client_id=client_id,
                        expert_model=expert_model,
                        images=images,
                    )
                    general_logits = self.general_model(images)
                    general_features = self._compute_general_reliability_features(general_logits)
                    fallback_mask = self._joint_fallback_mask(
                        client_id,
                        route_features["route_score"],
                        general_features["reliability"],
                    )
                    routed_predictions = route_features["expert_prediction"].clone()
                    routed_predictions[fallback_mask] = general_features["prediction"][fallback_mask]

                    batch_size = int(targets_device.numel())
                    num_samples += batch_size
                    invoked_total += int(fallback_mask.sum().item())
                    expert_correct += int((route_features["expert_prediction"] == targets_device).sum().item())
                    general_correct += int((general_features["prediction"] == targets_device).sum().item())
                    routed_correct += int((routed_predictions == targets_device).sum().item())
                    oracle_correct += int(
                        ((route_features["expert_prediction"] == targets_device) | (general_features["prediction"] == targets_device)).sum().item()
                    )
                    oracle_general_invocations += int(
                        ((route_features["expert_prediction"] != targets_device) & (general_features["prediction"] == targets_device)).sum().item()
                    )
                    disagreement_total += int(
                        (route_features["expert_prediction"] != general_features["prediction"]).sum().item()
                    )

                    if fallback_mask.any():
                        invoked_general_correct += int(
                            (general_features["prediction"][fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )
                        invoked_expert_correct += int(
                            (route_features["expert_prediction"][fallback_mask] == targets_device[fallback_mask]).sum().item()
                        )
                        invoked_general_reliability += float(
                            general_features["reliability"][fallback_mask].sum().item()
                        )

                expert_accuracy = expert_correct / max(num_samples, 1)
                general_accuracy = general_correct / max(num_samples, 1)
                routed_accuracy = routed_correct / max(num_samples, 1)
                invoked_general_accuracy = invoked_general_correct / max(invoked_total, 1)
                invoked_expert_accuracy = invoked_expert_correct / max(invoked_total, 1)
                oracle_route_accuracy = oracle_correct / max(num_samples, 1)
                client_metrics[client_id] = {
                    "general_gain_over_expert": general_accuracy - expert_accuracy,
                    "routed_gain_over_expert": routed_accuracy - expert_accuracy,
                    "invoked_general_accuracy": invoked_general_accuracy,
                    "invoked_expert_accuracy": invoked_expert_accuracy,
                    "invoked_general_gain": invoked_general_accuracy - invoked_expert_accuracy,
                    "oracle_route_accuracy": oracle_route_accuracy,
                    "oracle_general_invocation_rate": oracle_general_invocations / max(num_samples, 1),
                    "expert_bad_general_good_rate": oracle_general_invocations / max(num_samples, 1),
                    "routing_regret": oracle_route_accuracy - routed_accuracy,
                    "expert_general_disagreement_rate": disagreement_total / max(num_samples, 1),
                    "routing_general_route_enabled_rate": 1.0 if self.client_general_route_enabled.get(client_id, True) else 0.0,
                    "routing_general_reliability_threshold": float(
                        self.client_general_reliability_thresholds.get(
                            client_id,
                            self._initial_general_reliability_threshold(),
                        )
                    ),
                    "routing_invoked_general_reliability": invoked_general_reliability / max(invoked_total, 1),
                }

        keys = (
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
            "routing_general_route_enabled_rate",
            "routing_general_reliability_threshold",
            "routing_invoked_general_reliability",
        )
        if not client_metrics:
            return {
                "general_gain_over_expert": general_macro_accuracy - expert_macro_accuracy,
                "routed_gain_over_expert": routed_macro_accuracy - expert_macro_accuracy,
                **{key: 0.0 for key in keys if key not in {"general_gain_over_expert", "routed_gain_over_expert"}},
            }
        return {
            key: sum(float(metrics[key]) for metrics in client_metrics.values()) / max(len(client_metrics), 1)
            for key in keys
        } | {
            "general_gain_over_expert": general_macro_accuracy - expert_macro_accuracy,
            "routed_gain_over_expert": routed_macro_accuracy - expert_macro_accuracy,
        }

    def _build_round_extra_metrics(
        self,
        round_idx: int,
        distill_stats: Dict[str, float],
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        base_metrics = self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)
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
        base_metrics = self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)
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
