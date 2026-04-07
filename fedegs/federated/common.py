import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.config import ExperimentConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class HyperKnowledge:
    features: Dict[int, torch.Tensor] = field(default_factory=dict)
    soft_predictions: Dict[int, torch.Tensor] = field(default_factory=dict)
    counts: Dict[int, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.counts


@dataclass
class ClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    delta: Dict[str, torch.Tensor]
    hyper_knowledge: Optional[HyperKnowledge] = None


@dataclass
class RoundMetrics:
    round_idx: int
    avg_client_loss: float
    routed_accuracy: float
    hard_accuracy: float
    invocation_rate: float
    local_accuracy: float = 0.0
    compute_savings: float = 0.0

    @property
    def global_accuracy(self) -> float:
        return self.routed_accuracy

    @property
    def hard_recall(self) -> float:
        return self.hard_accuracy


class BaseFederatedClient:
    def __init__(self, client_id: str, dataset: Dataset, device: str) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _optimize_model(
        self,
        model: nn.Module,
        loader: DataLoader,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        reference_model: Optional[nn.Module] = None,
        prox_mu: float = 0.0,
    ) -> float:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        if reference_model is not None:
            reference_model.eval()

        total_loss = 0.0
        total_batches = 0
        for _ in range(epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                if reference_model is not None and prox_mu > 0:
                    loss = loss + 0.5 * prox_mu * self._proximal_penalty(model, reference_model)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
        return total_loss / max(total_batches, 1)

    def _proximal_penalty(self, model: nn.Module, reference_model: nn.Module) -> torch.Tensor:
        penalty = torch.zeros(1, device=self.device)
        for current_param, reference_param in zip(model.parameters(), reference_model.parameters()):
            penalty = penalty + torch.sum((current_param - reference_param.detach()) ** 2)
        return penalty.squeeze()


class BaseFederatedServer:
    def __init__(
        self,
        config: ExperimentConfig,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices: Sequence[int],
        writer: Optional[object] = None,
        public_dataset: Optional[Dataset] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.federated.device if torch.cuda.is_available() else "cpu")
        self.random = random.Random(config.federated.seed)
        self.data_module = data_module
        self.test_hard_indices = set(test_hard_indices)
        self.writer = writer
        self.client_datasets = client_datasets
        self.client_test_datasets = client_test_datasets
        self.public_dataset = public_dataset
        self.routing_threshold = (
            config.inference.confidence_threshold
            if config.inference.confidence_threshold is not None
            else config.inference.high_threshold
        )

    def _confidence_route(
        self,
        expert_model: nn.Module,
        general_model: nn.Module,
        images: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, object]]:
        threshold = self.routing_threshold if confidence_threshold is None else max(float(confidence_threshold), 0.0)
        margin_threshold = (
            max(float(self.config.inference.route_distance_threshold), 0.0)
            if margin_threshold is None
            else max(float(margin_threshold), 0.0)
        )
        expert_model.eval()
        general_model.eval()

        expert_logits = expert_model(images)
        expert_probs = torch.softmax(expert_logits, dim=1)
        topk = torch.topk(expert_probs, k=min(2, expert_probs.size(1)), dim=1)
        expert_confidence = topk.values[:, 0]
        expert_prediction = topk.indices[:, 0]
        if topk.values.size(1) > 1:
            expert_margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            expert_margin = torch.ones_like(expert_confidence)

        fallback_mask = self._build_fallback_mask(
            expert_confidence=expert_confidence,
            expert_margin=expert_margin,
            confidence_threshold=threshold,
            margin_threshold=margin_threshold,
        )
        predictions = expert_prediction.clone()
        invoked_general = int(fallback_mask.sum().item())
        route_types = ["expert"] * images.size(0)
        if fallback_mask.any():
            general_logits = general_model(images[fallback_mask])
            predictions[fallback_mask] = torch.argmax(general_logits, dim=1)
            for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                route_types[sample_idx] = "general"

        metadata = {
            "route_type": route_types,
            "expert_confidence": expert_confidence.detach().cpu().tolist(),
        }
        return predictions, invoked_general, metadata

    def _build_fallback_mask(
        self,
        expert_confidence: torch.Tensor,
        expert_margin: torch.Tensor,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> torch.Tensor:
        confidence_delta = max(float(self.config.inference.route_hard_confidence_delta), 0.0)
        margin_delta = max(float(self.config.inference.route_hard_margin_delta), 0.0)
        base_mask = (expert_confidence < confidence_threshold) | (expert_margin < margin_threshold)
        hard_proxy_mask = (
            (expert_confidence < confidence_threshold + confidence_delta)
            & (expert_margin < margin_threshold + margin_delta)
        )
        return base_mask | hard_proxy_mask

    def _sample_client_ids(self) -> List[str]:
        return self.random.sample(
            list(self.client_datasets.keys()),
            k=min(self.config.federated.clients_per_round, len(self.client_datasets)),
        )

    def _build_route_export_path(self, prefix: str) -> Path:
        run_name = self.config.run_name or "manual_run"
        return Path(self.config.output_dir) / "routes" / run_name / f"{prefix}.csv"

    def _write_route_records_csv(self, path: Path, route_records: List[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "client_id",
                    "sample_index",
                    "true_label",
                    "pred_label",
                    "route_type",
                    "expert_confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(route_records)
        LOGGER.info("Saved route records to %s | rows=%d", path, len(route_records))

    def _evaluate_predictor_on_client_tests(
        self,
        predictor: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        prefix: str,
        route_export_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        client_results: Dict[str, Dict[str, float]] = {}
        route_records: List[Dict[str, object]] = []
        for client_id, dataset in self.client_test_datasets.items():
            loader = self.data_module.make_loader(dataset, shuffle=False)
            client_metrics = self._evaluate_predictor_on_loader(
                client_id,
                predictor,
                loader,
                collect_route_records=route_export_path is not None,
            )
            route_records.extend(client_metrics.pop("_route_records", []))
            client_results[client_id] = client_metrics

        weighted = self._aggregate_metrics(client_results, weighted=True)
        macro = self._aggregate_metrics(client_results, weighted=False)
        groups = self._aggregate_group_metrics(client_results)
        self._log_client_metrics_table(prefix, client_results, weighted, macro, groups)
        if route_export_path is not None:
            self._write_route_records_csv(route_export_path, route_records)
        return {
            "aggregate": weighted,
            "macro": macro,
            "groups": groups,
            "clients": client_results,
        }

    def _aggregate_metrics(self, client_results: Dict[str, Dict[str, float]], weighted: bool) -> Dict[str, float]:
        metric_keys = ("accuracy", "hard_recall", "precision_macro", "recall_macro", "f1_macro", "invocation_rate")
        if not client_results:
            return {
                **{key: 0.0 for key in metric_keys},
                "hard_accuracy": 0.0,
                "num_clients": 0,
                "num_samples": 0,
                "num_hard_samples": 0,
            }

        accumulator = {key: 0.0 for key in metric_keys}
        total_weight = 0.0
        total_samples = 0
        total_hard_samples = 0

        for metrics in client_results.values():
            weight = float(metrics["num_samples"] if weighted else 1.0)
            total_weight += weight
            total_samples += int(metrics["num_samples"])
            total_hard_samples += int(metrics["num_hard_samples"])
            for key in metric_keys:
                accumulator[key] += metrics[key] * weight

        aggregated = {key: accumulator[key] / max(total_weight, 1.0) for key in metric_keys}
        aggregated["hard_accuracy"] = aggregated["hard_recall"]
        aggregated["num_clients"] = len(client_results)
        aggregated["num_samples"] = total_samples
        aggregated["num_hard_samples"] = total_hard_samples
        return aggregated

    def _aggregate_group_metrics(self, client_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
        for client_id, metrics in client_results.items():
            if client_id.startswith("simple_"):
                group_name = "simple"
            elif client_id.startswith("complex_"):
                group_name = "complex"
            else:
                group_name = "other"
            grouped.setdefault(group_name, {})[client_id] = metrics
        return {
            group_name: self._aggregate_metrics(metrics_map, weighted=True)
            for group_name, metrics_map in grouped.items()
        }

    def _evaluate_predictor_on_loader(
        self,
        client_id: str,
        predictor: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        loader: DataLoader,
        collect_route_records: bool = False,
    ) -> Dict[str, float]:
        all_predictions: List[int] = []
        all_targets: List[int] = []
        hard_predictions: List[int] = []
        hard_targets: List[int] = []
        invoked_general = 0
        route_records: List[Dict[str, object]] = []

        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                predictor_output = predictor(client_id, images, indices.tolist())
                route_metadata: Optional[Dict[str, object]] = None
                if len(predictor_output) == 3:
                    predictions, batch_invocations, route_metadata = predictor_output
                else:
                    predictions, batch_invocations = predictor_output
                invoked_general += batch_invocations

                batch_predictions = predictions.detach().cpu().tolist()
                batch_targets = targets.tolist()
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)

                if collect_route_records and route_metadata is not None:
                    batch_route_types = route_metadata.get("route_type")
                    batch_confidences = route_metadata.get("expert_confidence")
                    if batch_route_types is not None and batch_confidences is not None:
                        for sample_index, target, pred, route_type, expert_confidence in zip(
                            indices.tolist(),
                            batch_targets,
                            batch_predictions,
                            batch_route_types,
                            batch_confidences,
                        ):
                            route_records.append(
                                {
                                    "client_id": client_id,
                                    "sample_index": sample_index,
                                    "true_label": target,
                                    "pred_label": pred,
                                    "route_type": route_type,
                                    "expert_confidence": f"{float(expert_confidence):.6f}",
                                }
                            )

                for sample_index, pred, target in zip(indices.tolist(), batch_predictions, batch_targets):
                    if sample_index in self.test_hard_indices:
                        hard_predictions.append(pred)
                        hard_targets.append(target)

        metrics = self._classification_metrics(all_predictions, all_targets)
        hard_metrics = self._classification_metrics(hard_predictions, hard_targets) if hard_targets else {"accuracy": 0.0}
        metrics["hard_recall"] = hard_metrics["accuracy"]
        metrics["hard_accuracy"] = hard_metrics["accuracy"]
        metrics["invocation_rate"] = invoked_general / max(len(all_targets), 1)
        metrics["num_hard_samples"] = len(hard_targets)
        if collect_route_records:
            metrics["_route_records"] = route_records
        return metrics

    def _classification_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        num_classes = self.config.model.num_classes
        if not targets:
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "num_samples": 0,
            }

        confusion = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for pred, target in zip(predictions, targets):
            confusion[target, pred] += 1.0

        total = confusion.sum().item()
        accuracy = confusion.diag().sum().item() / max(total, 1.0)
        precision_values = []
        recall_values = []
        f1_values = []

        for class_idx in range(num_classes):
            tp = confusion[class_idx, class_idx].item()
            fp = confusion[:, class_idx].sum().item() - tp
            fn = confusion[class_idx, :].sum().item() - tp
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        return {
            "accuracy": accuracy,
            "precision_macro": sum(precision_values) / num_classes,
            "recall_macro": sum(recall_values) / num_classes,
            "f1_macro": sum(f1_values) / num_classes,
            "num_samples": int(total),
        }

    def _log_client_metrics_table(
        self,
        prefix: str,
        client_results: Dict[str, Dict[str, float]],
        weighted: Dict[str, float],
        macro: Dict[str, float],
        groups: Dict[str, Dict[str, float]],
    ) -> None:
        LOGGER.info(
            "%s weighted | clients=%d | samples=%d | acc=%.4f | local_acc=%.4f | hard_recall=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f",
            prefix,
            weighted["num_clients"],
            weighted["num_samples"],
            weighted["accuracy"],
            macro["accuracy"],
            weighted["hard_recall"],
            weighted["precision_macro"],
            weighted["recall_macro"],
            weighted["f1_macro"],
            weighted["invocation_rate"],
        )
        for group_name, metrics in sorted(groups.items()):
            LOGGER.info(
                "%s group=%s | samples=%d | acc=%.4f | hard_recall=%.4f | invocation=%.4f",
                prefix,
                group_name,
                metrics["num_samples"],
                metrics["accuracy"],
                metrics["hard_recall"],
                metrics["invocation_rate"],
            )
        for client_id, metrics in sorted(client_results.items()):
            LOGGER.info(
                "%s client=%s | n=%d | acc=%.4f | hard_recall=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f",
                prefix,
                client_id,
                metrics["num_samples"],
                metrics["accuracy"],
                metrics["hard_recall"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
                metrics["invocation_rate"],
            )

    def _log_round_metrics(self, prefix: str, metrics: RoundMetrics) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(f"loss/{prefix}", metrics.avg_client_loss, metrics.round_idx)
        self.writer.add_scalar(f"accuracy/{prefix}", metrics.routed_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"local_accuracy/{prefix}", metrics.local_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"hard_accuracy/{prefix}", metrics.hard_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"invocation_rate/{prefix}", metrics.invocation_rate, metrics.round_idx)
        self.writer.add_scalar(f"compute_savings/{prefix}", metrics.compute_savings, metrics.round_idx)

        self.writer.add_scalar(f"compare/{prefix}/loss", metrics.avg_client_loss, metrics.round_idx)
        self.writer.add_scalar(f"compare/{prefix}/accuracy", metrics.routed_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"compare/{prefix}/local_accuracy", metrics.local_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"compare/{prefix}/hard_accuracy", metrics.hard_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"compare/{prefix}/invocation_rate", metrics.invocation_rate, metrics.round_idx)
        self.writer.add_scalar(f"compare/{prefix}/compute_savings", metrics.compute_savings, metrics.round_idx)

    def _log_auxiliary_accuracy_metrics(
        self,
        prefix: str,
        round_idx: int,
        expert_accuracy: float,
        general_accuracy: float,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(f"expert_accuracy/{prefix}", expert_accuracy, round_idx)
        self.writer.add_scalar(f"general_accuracy/{prefix}", general_accuracy, round_idx)
        self.writer.add_scalar(f"compare/{prefix}/expert_accuracy", expert_accuracy, round_idx)
        self.writer.add_scalar(f"compare/{prefix}/general_accuracy", general_accuracy, round_idx)

    def _build_compute_profile(
        self,
        expert_flops: float,
        general_flops: float,
        invocation_rate: float,
        mode: str = "routed",
    ) -> Dict[str, float]:
        if mode == "general_only":
            average_flops = general_flops
        elif mode == "expert_only":
            average_flops = expert_flops
        else:
            average_flops = expert_flops + invocation_rate * general_flops

        savings_ratio = 1.0 - (average_flops / max(general_flops, 1e-8))
        return {
            "expert_flops": expert_flops,
            "general_flops": general_flops,
            "average_flops": average_flops,
            "invocation_rate": invocation_rate,
            "savings_ratio": savings_ratio,
        }
