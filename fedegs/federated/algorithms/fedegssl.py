"""
FedEGSS-L: light server-side general enhancement.

Minimal-change variant over FedEGS-S:
  - clients remain unchanged and still train only the local expert
  - server selects only a high-value proxy subset in most rounds
  - server updates only the general head / last block in light rounds
  - server blends current ensemble targets with an EMA temporal teacher
"""

import copy
import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fedegs.federated.common import LOGGER, RoundMetrics
from fedegs.federated.algorithms.fedegss import FedEGSSServer
class FedEGSSLServer(FedEGSSServer):
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
        self.algorithm_name = "fedegssl"
        self.temporal_teacher_model = copy.deepcopy(self.general_model).to(self.device)
        self.temporal_teacher_model.eval()

    def _is_full_refresh_round(self, round_idx: int) -> bool:
        interval = max(int(getattr(self.config.federated, "general_full_refresh_interval", 10)), 1)
        return round_idx <= 1 or (round_idx % interval) == 0

    def _build_subset_loader(self, indices: Sequence[int] | None) -> DataLoader:
        if indices is None:
            dataset = self.distill_dataset
        else:
            dataset = Subset(self.distill_dataset, list(indices))
        return DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
        )

    def _select_proxy_indices(self, round_idx: int) -> Dict[str, object]:
        total = len(self.distill_dataset)
        if total <= 0:
            return {"indices": [], "selected_ratio": 0.0, "selection_score": 0.0, "full_refresh": True}

        if self._is_full_refresh_round(round_idx):
            return {
                "indices": list(range(total)),
                "selected_ratio": 1.0,
                "selection_score": 0.0,
                "full_refresh": True,
            }

        ratio = float(getattr(self.config.federated, "selective_proxy_ratio", 1.0))
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio >= 1.0:
            return {
                "indices": list(range(total)),
                "selected_ratio": 1.0,
                "selection_score": 0.0,
                "full_refresh": False,
            }

        num_classes = max(int(self.config.model.num_classes), 2)
        normalization = math.log(float(num_classes))
        scores: List[torch.Tensor] = []

        self.general_model.eval()
        with torch.no_grad():
            for batch in self.distill_loader:
                images = batch[0]
                images = images.to(self.device)
                logits = self.general_model(images)
                probs = torch.softmax(logits / max(float(self.config.federated.distill_temperature), 1e-4), dim=1)
                topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
                confidence = topk.values[:, 0]
                if topk.values.size(1) > 1:
                    margin = topk.values[:, 0] - topk.values[:, 1]
                else:
                    margin = torch.ones_like(confidence)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / max(normalization, 1e-8)
                score = entropy + (1.0 - confidence) + (1.0 - margin)
                scores.append(score.detach().cpu())

        all_scores = torch.cat(scores, dim=0)
        keep = max(1, int(round(total * ratio)))
        selected_positions = torch.topk(all_scores, k=keep, largest=True).indices.tolist()
        selected_positions.sort()
        selected_score = float(all_scores[selected_positions].mean().item()) if selected_positions else 0.0
        return {
            "indices": selected_positions,
            "selected_ratio": keep / max(total, 1),
            "selection_score": selected_score,
            "full_refresh": False,
        }

    def _extract_ensemble_logits(self, updates, indices: Sequence[int] | None = None) -> Dict[str, torch.Tensor]:
        num_classes = self.config.model.num_classes
        temperature = float(self.config.federated.distill_temperature)
        distill_loader = self._build_subset_loader(indices)

        all_logits: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        for update in updates:
            expert = copy.deepcopy(self.reference_expert).to(self.device)
            expert.load_state_dict(update.expert_state_dict)
            expert.eval()

            batches: List[torch.Tensor] = []
            with torch.no_grad():
                for batch in distill_loader:
                    images = batch[0].to(self.device)
                    batches.append(expert(images).cpu())

            client_logits = torch.cat(batches, dim=0)
            probs = F.softmax(client_logits / temperature, dim=1)
            log_c = math.log(float(num_classes))
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(log_c, 1e-8)
            weight = torch.exp(-ent)

            all_logits.append(client_logits)
            all_weights.append(weight)
            del expert

        stacked_logits = torch.stack(all_logits, dim=0)
        stacked_weights = torch.stack(all_weights, dim=0)
        normalized_weights = stacked_weights / stacked_weights.sum(0, keepdim=True).clamp_min(1e-8)
        avg_logits = (stacked_logits * normalized_weights.unsqueeze(-1)).sum(0)
        soft_labels = F.softmax(avg_logits / temperature, dim=1)
        sample_weights = stacked_weights.mean(0)
        return {
            "soft_labels": soft_labels,
            "sample_weights": sample_weights,
            "indices": list(indices) if indices is not None else list(range(len(self.distill_dataset))),
        }

    def _set_general_trainable_scope(self, full_refresh: bool) -> List[nn.Parameter]:
        for parameter in self.general_model.parameters():
            parameter.requires_grad_(False)

        if full_refresh:
            self.general_model.train()
            for parameter in self.general_model.parameters():
                parameter.requires_grad_(True)
            return list(self.general_model.parameters())

        scope = str(getattr(self.config.federated, "general_light_update_scope", "head_last_block")).lower()
        self.general_model.eval()
        self.general_model.classifier.train()

        trainable: List[nn.Parameter] = []
        for parameter in self.general_model.classifier.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)

        if scope == "head_last_block":
            self.general_model.backbone.layer4.train()
            for parameter in self.general_model.backbone.layer4.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
        elif scope != "head":
            raise ValueError(f"Unsupported general_light_update_scope: {scope}")

        return trainable

    def _update_temporal_teacher(self) -> None:
        momentum = float(getattr(self.config.federated, "general_teacher_ema_momentum", 0.90))
        momentum = min(max(momentum, 0.0), 0.9999)
        self.temporal_teacher_model.eval()
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(
                self.temporal_teacher_model.parameters(),
                self.general_model.parameters(),
            ):
                teacher_parameter.data.mul_(momentum).add_(student_parameter.data, alpha=1.0 - momentum)

    def _distill_general_model(self, ensemble: Dict[str, torch.Tensor], round_idx: int) -> Dict[str, float]:
        temperature = float(self.config.federated.distill_temperature)
        full_refresh = self._is_full_refresh_round(round_idx)
        epochs = (
            max(int(self.config.federated.distill_epochs), 1)
            if full_refresh
            else max(int(getattr(self.config.federated, "general_light_distill_epochs", 1)), 1)
        )

        indices = ensemble.get("indices", [])
        distill_loader = self._build_subset_loader(indices)
        all_images = [batch[0] for batch in distill_loader]
        distill_images = torch.cat(all_images, dim=0)
        num_samples = distill_images.size(0)
        batch_size = self.config.dataset.batch_size

        soft_targets = ensemble["soft_labels"]
        sample_weights = ensemble["sample_weights"]

        self.temporal_teacher_model.eval()
        teacher_batches: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                images = distill_images[start:start + batch_size].to(self.device)
                teacher_logits = self.temporal_teacher_model(images)
                teacher_batches.append(torch.softmax(teacher_logits / temperature, dim=1).cpu())
        teacher_probs = torch.cat(teacher_batches, dim=0)
        blended_targets = 0.5 * (soft_targets + teacher_probs)
        blended_targets = blended_targets / blended_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)

        trainable_parameters = self._set_general_trainable_scope(full_refresh=full_refresh)
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=float(self.config.federated.distill_lr),
        )
        total_steps = epochs * ((num_samples + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        total_loss = 0.0
        total_batches = 0
        for _ in range(epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = permutation[start:start + batch_size]
                images = distill_images[idx].to(self.device)
                target_probs = blended_targets[idx].to(self.device)
                weight = sample_weights[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                per_kl = F.kl_div(student_log_probs, target_probs, reduction="none").sum(dim=1) * (temperature ** 2)
                loss = (per_kl * weight).sum() / weight.sum().clamp_min(1e-8)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.item())
                total_batches += 1

        for parameter in self.general_model.parameters():
            parameter.requires_grad_(True)
        self._update_temporal_teacher()

        divisor = max(total_batches, 1)
        return {
            "total_loss": total_loss / divisor,
            "kl_loss": total_loss / divisor,
            "selected_ratio": float(len(indices) / max(len(self.distill_dataset), 1)),
            "full_refresh": float(full_refresh),
        }

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        algorithm_name = self.algorithm_name
        metrics: List[RoundMetrics] = []

        for round_idx in range(1, self.config.federated.rounds + 1):
            self.current_round = round_idx
            sampled_client_ids = self._sample_client_ids()
            LOGGER.info("%s round %d | clients=%s", algorithm_name, round_idx, sampled_client_ids)

            updates = [self.clients[client_id].train_local() for client_id in sampled_client_ids]
            proxy_bundle = self._select_proxy_indices(round_idx)
            ensemble = self._extract_ensemble_logits(updates, indices=proxy_bundle["indices"])
            ensemble.update(proxy_bundle)
            distill_stats = self._distill_general_model(ensemble, round_idx)
            self._update_routing_thresholds(updates)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, f"{algorithm_name}-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, f"{algorithm_name}-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, f"{algorithm_name}-routed")

            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            aggregate = routed_eval["aggregate"]
            macro = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate["invocation_rate"],
                "routed",
            )
            client_train_profile = self._build_client_training_profile(
                self.expert_flops,
                sampled_client_ids,
                self.client_training_datasets,
            )
            resource_metrics = self._resource_metric_values(
                self.resource_profiles,
                client_train_profile,
                compute_profile,
            )

            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=avg_loss,
                routed_accuracy=macro["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                **resource_metrics,
            )

            LOGGER.info(
                "%s round %d | loss=%.4f | distill=%.4f | proxy=%.3f | refresh=%s | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f%s",
                algorithm_name,
                round_idx,
                avg_loss,
                distill_stats["total_loss"],
                distill_stats["selected_ratio"],
                bool(distill_stats["full_refresh"]),
                macro["accuracy"],
                aggregate["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                self._format_resource_metrics_for_log(round_metrics),
            )

            if self.writer:
                self.writer.add_scalar(f"expert_loss/{algorithm_name}", avg_loss, round_idx)
                self.writer.add_scalar(f"distill_loss/{algorithm_name}", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar(f"distill_selected_ratio/{algorithm_name}", distill_stats["selected_ratio"], round_idx)
                self.writer.add_scalar(f"distill_full_refresh/{algorithm_name}", distill_stats["full_refresh"], round_idx)
                self._log_auxiliary_accuracy_metrics(
                    algorithm_name,
                    round_idx,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics(algorithm_name, round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best(
                round_idx,
                round_metrics,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best()
        return metrics

    def evaluate_baselines(self, test_dataset):
        result = super().evaluate_baselines(test_dataset)
        result["algorithm"] = self.algorithm_name
        return result
