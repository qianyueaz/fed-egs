"""
FedAsym-RAD: router-aware label-corrected server distillation.

This variant keeps FedAsym's client training, explicit expert-error predictor,
and inference router unchanged. It only replaces the server-side public
distillation target:
  - cached public logits form a cross-client teacher bank,
  - 1 - R_k weights reliable expert teachers,
  - R_k defines where the general model should focus,
  - public labels correct the target when experts are unreliable.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from fedegs.federated.algorithms.fedasym import (
    FedAsymClientUpdate,
    FedAsymServer,
    _build_error_fallback_mask,
    _clone_tensor_dict,
    _predict_error_probabilities,
    _predictor_features_from_logits,
)


class FedAsymRADServer(FedAsymServer):
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
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )
        self.algorithm_name = "fedasym_rad"
        self.public_targets_cpu = torch.cat([batch[1] for batch in self.public_batches], dim=0).to(torch.long)
        self.public_teacher_bank: Dict[str, Dict[str, object]] = {}
        self.latest_teacher_bank_stats: Dict[str, float] = {
            "teacher_bank_size": 0.0,
            "teacher_bank_avg_staleness": 0.0,
            "teacher_bank_max_staleness": 0.0,
            "teacher_bank_effective_size": 0.0,
            "teacher_bank_memory_mb": 0.0,
        }

    def _empty_public_knowledge(self) -> Dict[str, torch.Tensor | float]:
        return {
            "teacher_probs": torch.empty((0, self.config.model.num_classes), dtype=torch.float32),
            "sample_reliability": torch.empty((0,), dtype=torch.float32),
            "route_demand": torch.empty((0,), dtype=torch.float32),
            "teacher_reliability_mean": 0.0,
            "teacher_error_mean": 0.0,
            "route_demand_mean": 0.0,
            **self.latest_teacher_bank_stats,
        }

    def _update_public_teacher_bank(self, updates: List[FedAsymClientUpdate]) -> None:
        current_round = int(getattr(self, "current_round", 0))
        for update in updates:
            self.public_teacher_bank[update.client_id] = {
                "public_logits": update.public_logits.detach().cpu().clone(),
                "predictor_state": _clone_tensor_dict(update.predictor_state),
                "error_threshold": float(update.error_threshold),
                "round_idx": current_round,
            }

    @staticmethod
    def _teacher_entry_memory_mb(entry: Dict[str, object]) -> float:
        total_elements = 0
        public_logits = entry.get("public_logits")
        if torch.is_tensor(public_logits):
            total_elements += int(public_logits.numel())
        predictor_state = entry.get("predictor_state")
        if isinstance(predictor_state, dict):
            total_elements += sum(int(value.numel()) for value in predictor_state.values() if torch.is_tensor(value))
        return total_elements * 4.0 / (1024.0 ** 2)

    def _active_public_teacher_entries(self) -> List[tuple[str, Dict[str, object], int, float]]:
        current_round = int(getattr(self, "current_round", 0))
        max_staleness = max(int(getattr(self.config.federated, "teacher_bank_max_staleness", 0)), 0)
        staleness_decay = min(max(float(getattr(self.config.federated, "teacher_bank_staleness_decay", 0.99)), 0.0), 1.0)
        active_entries: List[tuple[str, Dict[str, object], int, float]] = []
        stale_client_ids: List[str] = []

        for client_id, entry in self.public_teacher_bank.items():
            round_idx = int(entry.get("round_idx", current_round))
            staleness = max(current_round - round_idx, 0)
            if max_staleness > 0 and staleness > max_staleness:
                stale_client_ids.append(client_id)
                continue

            public_logits = entry.get("public_logits")
            if not torch.is_tensor(public_logits) or public_logits.size(0) != self.public_images_cpu.size(0):
                stale_client_ids.append(client_id)
                continue

            freshness_weight = staleness_decay ** staleness
            if freshness_weight <= 0.0:
                continue
            active_entries.append((client_id, entry, staleness, freshness_weight))

        for client_id in stale_client_ids:
            self.public_teacher_bank.pop(client_id, None)

        if active_entries:
            staleness_values = [float(staleness) for _, _, staleness, _ in active_entries]
            freshness_values = [float(freshness) for _, _, _, freshness in active_entries]
            self.latest_teacher_bank_stats = {
                "teacher_bank_size": float(len(active_entries)),
                "teacher_bank_avg_staleness": sum(staleness_values) / len(staleness_values),
                "teacher_bank_max_staleness": max(staleness_values),
                "teacher_bank_effective_size": sum(freshness_values),
                "teacher_bank_memory_mb": sum(self._teacher_entry_memory_mb(entry) for _, entry, _, _ in active_entries),
            }
        else:
            self.latest_teacher_bank_stats = {
                "teacher_bank_size": 0.0,
                "teacher_bank_avg_staleness": 0.0,
                "teacher_bank_max_staleness": 0.0,
                "teacher_bank_effective_size": 0.0,
                "teacher_bank_memory_mb": 0.0,
            }
        return active_entries

    def _aggregate_public_knowledge(self, updates: List[FedAsymClientUpdate]) -> Dict[str, torch.Tensor | float]:
        self._update_public_teacher_bank(updates)
        teacher_entries = self._active_public_teacher_entries()
        if not teacher_entries:
            return self._empty_public_knowledge()

        temperature = float(self.config.federated.distill_temperature)
        reliability_beta = max(float(getattr(self.config.federated, "rad_teacher_reliability_beta", 2.0)), 0.0)
        demand_mode = str(getattr(self.config.federated, "rad_route_demand_mode", "soft")).lower()

        teacher_probs: List[torch.Tensor] = []
        teacher_weights: List[torch.Tensor] = []
        error_probs_all: List[torch.Tensor] = []
        route_masks: List[torch.Tensor] = []
        freshness_weights: List[torch.Tensor] = []

        for _, entry, _, freshness_weight in teacher_entries:
            public_logits = entry["public_logits"]
            predictor_state = entry["predictor_state"]
            features = _predictor_features_from_logits(public_logits)
            error_probs = _predict_error_probabilities(predictor_state, features).detach().cpu().clamp(0.0, 1.0)
            reliability = (1.0 - error_probs).clamp(1e-4, 1.0)
            weight = reliability.pow(reliability_beta) if reliability_beta != 1.0 else reliability
            weight = weight * float(freshness_weight)

            teacher_probs.append(F.softmax(public_logits / temperature, dim=1).detach().cpu())
            teacher_weights.append(weight)
            error_probs_all.append(error_probs)
            freshness_weights.append(torch.full_like(error_probs, float(freshness_weight)))

            if demand_mode == "hard":
                route_masks.append(
                    _build_error_fallback_mask(
                        error_probs,
                        features,
                        float(entry["error_threshold"]),
                        self.config,
                    )
                    .to(torch.float32)
                    .cpu()
                )

        stacked_teacher_probs = torch.stack(teacher_probs, dim=0)
        stacked_weights = torch.stack(teacher_weights, dim=0)
        normalized_weights = stacked_weights / stacked_weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        fused_teacher_probs = (stacked_teacher_probs * normalized_weights.unsqueeze(-1)).sum(dim=0)
        fused_teacher_probs = fused_teacher_probs.clamp_min(1e-8)
        fused_teacher_probs = fused_teacher_probs / fused_teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        stacked_error_probs = torch.stack(error_probs_all, dim=0)
        stacked_freshness = torch.stack(freshness_weights, dim=0)
        freshness_denominator = stacked_freshness.sum(dim=0).clamp_min(1e-8)
        route_source = torch.stack(route_masks, dim=0) if demand_mode == "hard" and route_masks else stacked_error_probs
        route_demand = ((route_source * stacked_freshness).sum(dim=0) / freshness_denominator).clamp(0.0, 1.0)
        sample_reliability = (
            (((1.0 - stacked_error_probs) * stacked_freshness).sum(dim=0) / freshness_denominator)
            .clamp(0.0, 1.0)
        )

        return {
            "teacher_probs": fused_teacher_probs,
            "sample_reliability": sample_reliability,
            "route_demand": route_demand,
            "teacher_reliability_mean": float(sample_reliability.mean().item()),
            "teacher_error_mean": float((1.0 - sample_reliability).mean().item()),
            "route_demand_mean": float(route_demand.mean().item()),
            **self.latest_teacher_bank_stats,
        }

    def _distill_general_model(
        self,
        public_knowledge: Dict[str, torch.Tensor | float],
        round_idx: int,
    ) -> Dict[str, float]:
        teacher_probs_all = public_knowledge["teacher_probs"]
        if not torch.is_tensor(teacher_probs_all) or teacher_probs_all.numel() == 0:
            self.latest_distill_stats = {
                "total_loss": 0.0,
                "kd_loss": 0.0,
                "label_ce_loss": 0.0,
                "forward_kl": 0.0,
                "reverse_kl": 0.0,
                "gamma_forward": 0.5,
                "gamma_reverse": 0.5,
                "teacher_reliability_mean": float(public_knowledge.get("teacher_reliability_mean", 0.0)),
                "teacher_error_mean": float(public_knowledge.get("teacher_error_mean", 0.0)),
                "route_demand_mean": float(public_knowledge.get("route_demand_mean", 0.0)),
                "label_ce_weight": 0.0,
                "focus_weight": 1.0,
                "teacher_bank_size": float(public_knowledge.get("teacher_bank_size", 0.0)),
                "teacher_bank_avg_staleness": float(public_knowledge.get("teacher_bank_avg_staleness", 0.0)),
                "teacher_bank_max_staleness": float(public_knowledge.get("teacher_bank_max_staleness", 0.0)),
                "teacher_bank_effective_size": float(public_knowledge.get("teacher_bank_effective_size", 0.0)),
                "teacher_bank_memory_mb": float(public_knowledge.get("teacher_bank_memory_mb", 0.0)),
            }
            return dict(self.latest_distill_stats)

        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = self.config.dataset.batch_size
        label_gamma = max(float(getattr(self.config.federated, "rad_label_correction_gamma", 0.5)), 0.0)
        label_max = min(max(float(getattr(self.config.federated, "rad_label_correction_max", 0.7)), 0.0), 1.0)
        focus_alpha = max(float(getattr(self.config.federated, "rad_route_focus_alpha", 1.0)), 0.0)
        focus_max = max(float(getattr(self.config.federated, "rad_focus_max", 3.0)), 0.0)

        sample_reliability_all = public_knowledge.get("sample_reliability")
        if not torch.is_tensor(sample_reliability_all) or sample_reliability_all.numel() != teacher_probs_all.size(0):
            fallback_reliability = float(public_knowledge.get("teacher_reliability_mean", 0.5))
            sample_reliability_all = torch.full(
                (teacher_probs_all.size(0),),
                min(max(fallback_reliability, 0.0), 1.0),
                dtype=torch.float32,
            )

        route_demand_all = public_knowledge.get("route_demand")
        if not torch.is_tensor(route_demand_all) or route_demand_all.numel() != teacher_probs_all.size(0):
            fallback_demand = float(public_knowledge.get("route_demand_mean", 1.0 - float(sample_reliability_all.mean().item())))
            route_demand_all = torch.full(
                (teacher_probs_all.size(0),),
                min(max(fallback_demand, 0.0), 1.0),
                dtype=torch.float32,
            )

        has_public_targets = self.public_targets_cpu.numel() == teacher_probs_all.size(0)
        optimizer = torch.optim.Adam(self.general_model.parameters(), lr=float(self.config.federated.distill_lr))
        total_steps = distill_epochs * ((self.public_images_cpu.size(0) + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        self.general_model.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_label_ce_loss = 0.0
        total_forward_kl = 0.0
        total_reverse_kl = 0.0
        total_gamma_forward = 0.0
        total_gamma_reverse = 0.0
        total_label_ce_weight = 0.0
        total_focus_weight = 0.0
        total_route_demand = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            permutation = torch.randperm(self.public_images_cpu.size(0))
            for start in range(0, self.public_images_cpu.size(0), batch_size):
                indices = permutation[start:start + batch_size]
                images = self.public_images_cpu[indices].to(self.device)
                teacher_probs = teacher_probs_all[indices].to(self.device).clamp_min(1e-8)
                teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                sample_reliability = sample_reliability_all[indices].to(self.device).clamp(0.0, 1.0)
                route_demand = route_demand_all[indices].to(self.device).clamp(0.0, 1.0)

                gamma_forward = 1.0 - sample_reliability
                gamma_reverse = sample_reliability
                label_ce_weight = (label_gamma * route_demand).clamp(0.0, label_max)
                kd_weight = 1.0 - label_ce_weight
                focus_weight = 1.0 + (focus_alpha * route_demand)
                if focus_max > 0.0:
                    focus_weight = focus_weight.clamp(max=focus_max)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)
                teacher_log_probs = teacher_probs.log()
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                student_probs = F.softmax(student_logits / temperature, dim=1)

                forward_kl_per_sample = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                reverse_kl_per_sample = F.kl_div(
                    teacher_log_probs,
                    student_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                kd_loss_per_sample = (gamma_forward * forward_kl_per_sample) + (gamma_reverse * reverse_kl_per_sample)

                if has_public_targets:
                    targets = self.public_targets_cpu[indices].to(self.device)
                    label_ce_per_sample = F.cross_entropy(student_logits, targets, reduction="none")
                else:
                    label_ce_per_sample = torch.zeros_like(kd_loss_per_sample)
                    label_ce_weight = torch.zeros_like(label_ce_weight)

                loss_per_sample = focus_weight * (
                    (kd_weight * kd_loss_per_sample) + (label_ce_weight * label_ce_per_sample)
                )
                loss = loss_per_sample.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.detach().cpu().item())
                total_kd_loss += float(kd_loss_per_sample.detach().mean().cpu().item())
                total_label_ce_loss += float(label_ce_per_sample.detach().mean().cpu().item())
                total_forward_kl += float(forward_kl_per_sample.detach().mean().cpu().item())
                total_reverse_kl += float(reverse_kl_per_sample.detach().mean().cpu().item())
                total_gamma_forward += float(gamma_forward.detach().mean().cpu().item())
                total_gamma_reverse += float(gamma_reverse.detach().mean().cpu().item())
                total_label_ce_weight += float(label_ce_weight.detach().mean().cpu().item())
                total_focus_weight += float(focus_weight.detach().mean().cpu().item())
                total_route_demand += float(route_demand.detach().mean().cpu().item())
                total_batches += 1

        divisor = max(total_batches, 1)
        self.latest_distill_stats = {
            "total_loss": total_loss / divisor,
            "kd_loss": total_kd_loss / divisor,
            "label_ce_loss": total_label_ce_loss / divisor,
            "forward_kl": total_forward_kl / divisor,
            "reverse_kl": total_reverse_kl / divisor,
            "gamma_forward": total_gamma_forward / divisor,
            "gamma_reverse": total_gamma_reverse / divisor,
            "teacher_reliability_mean": float(public_knowledge.get("teacher_reliability_mean", 0.0)),
            "teacher_error_mean": float(public_knowledge.get("teacher_error_mean", 0.0)),
            "route_demand_mean": total_route_demand / divisor,
            "label_ce_weight": total_label_ce_weight / divisor,
            "focus_weight": total_focus_weight / divisor,
            "teacher_bank_size": float(public_knowledge.get("teacher_bank_size", 0.0)),
            "teacher_bank_avg_staleness": float(public_knowledge.get("teacher_bank_avg_staleness", 0.0)),
            "teacher_bank_max_staleness": float(public_knowledge.get("teacher_bank_max_staleness", 0.0)),
            "teacher_bank_effective_size": float(public_knowledge.get("teacher_bank_effective_size", 0.0)),
            "teacher_bank_memory_mb": float(public_knowledge.get("teacher_bank_memory_mb", 0.0)),
        }
        return dict(self.latest_distill_stats)

    def _build_round_extra_metrics(self, expert_eval, general_eval, routed_eval) -> Dict[str, float]:
        metrics = super()._build_round_extra_metrics(expert_eval, general_eval, routed_eval)
        metrics.update(
            {
                "rad_route_demand": self.latest_distill_stats.get("route_demand_mean", 0.0),
                "rad_label_ce_weight": self.latest_distill_stats.get("label_ce_weight", 0.0),
                "rad_focus_weight": self.latest_distill_stats.get("focus_weight", 1.0),
                "distill_kd_loss": self.latest_distill_stats.get("kd_loss", 0.0),
                "distill_ce_loss": self.latest_distill_stats.get("label_ce_loss", 0.0),
                "teacher_bank_size": self.latest_distill_stats.get("teacher_bank_size", 0.0),
                "teacher_bank_avg_staleness": self.latest_distill_stats.get("teacher_bank_avg_staleness", 0.0),
                "teacher_bank_max_staleness": self.latest_distill_stats.get("teacher_bank_max_staleness", 0.0),
                "teacher_bank_effective_size": self.latest_distill_stats.get("teacher_bank_effective_size", 0.0),
                "teacher_bank_memory_mb": self.latest_distill_stats.get("teacher_bank_memory_mb", 0.0),
            }
        )
        return metrics

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        base = super()._format_round_extra_metrics_for_log(extra_metrics)
        return (
            f"{base}"
            f" | rad_q={extra_metrics.get('rad_route_demand', 0.0):.4f}"
            f" | rad_cew={extra_metrics.get('rad_label_ce_weight', 0.0):.4f}"
            f" | rad_fw={extra_metrics.get('rad_focus_weight', 1.0):.4f}"
            f" | rad_ce={extra_metrics.get('distill_ce_loss', 0.0):.4f}"
            f" | bank={extra_metrics.get('teacher_bank_size', 0.0):.0f}"
            f" | stale={extra_metrics.get('teacher_bank_avg_staleness', 0.0):.2f}"
        )
