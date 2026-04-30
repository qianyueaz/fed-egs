import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.config import ExperimentConfig
from fedegs.models import (
    estimate_client_training_flops,
    estimate_inference_memory_mb,
    estimate_training_memory_mb,
    measure_peak_memory_mb,
)


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
    weighted_accuracy: float = 0.0
    compute_savings: float = 0.0
    client_train_flops: float = 0.0
    client_train_flops_total: float = 0.0
    expert_infer_flops: float = 0.0
    general_infer_flops: float = 0.0
    routed_infer_flops: float = 0.0
    expert_train_memory_mb: float = 0.0
    expert_infer_memory_mb: float = 0.0
    general_train_memory_mb: float = 0.0
    general_infer_memory_mb: float = 0.0
    expert_train_peak_memory_mb: float = 0.0
    expert_infer_peak_memory_mb: float = 0.0
    general_train_peak_memory_mb: float = 0.0
    general_infer_peak_memory_mb: float = 0.0
    inference_latency_ms: float = 0.0
    round_train_time_seconds: float = 0.0
    upload_bytes_per_round: float = 0.0
    upload_bytes_total: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)

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

    def _device_synchronize(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _model_input_shape(self, batch_size: int = 1) -> Tuple[int, int, int, int]:
        dataset_name = str(self.config.dataset.name).lower()
        if dataset_name in {"cifar10", "cifar100", "svhn"}:
            return batch_size, 3, 32, 32
        return batch_size, 3, 32, 32

    def _client_optimizer_name(self) -> str:
        return "sgd_momentum" if float(getattr(self.config.federated, "local_momentum", 0.0)) > 0.0 else "sgd"

    def _build_model_resource_profile(self, model: nn.Module, flops: float) -> Dict[str, float]:
        return {
            "flops": float(flops),
            "train_memory_mb": estimate_training_memory_mb(
                model,
                batch_size=self.config.dataset.batch_size,
                input_shape=self._model_input_shape(),
                optimizer_name=self._client_optimizer_name(),
            ),
            "infer_memory_mb": estimate_inference_memory_mb(
                model,
                batch_size=self.config.dataset.batch_size,
                input_shape=self._model_input_shape(),
            ),
            "train_peak_memory_mb": measure_peak_memory_mb(
                model,
                batch_size=self.config.dataset.batch_size,
                input_shape=self._model_input_shape(),
                mode="train",
                optimizer_name=self._client_optimizer_name(),
            ),
            "infer_peak_memory_mb": measure_peak_memory_mb(
                model,
                batch_size=self.config.dataset.batch_size,
                input_shape=self._model_input_shape(),
                mode="inference",
            ),
        }

    def _build_dual_model_resource_profiles(
        self,
        expert_model: nn.Module,
        general_model: nn.Module,
        expert_flops: float,
        general_flops: float,
    ) -> Dict[str, Dict[str, float]]:
        return {
            "expert": self._build_model_resource_profile(expert_model, expert_flops),
            "general": self._build_model_resource_profile(general_model, general_flops),
        }

    def _build_client_training_profile(
        self,
        expert_flops: float,
        client_ids: Sequence[str],
        client_datasets: Optional[Dict[str, Dataset]] = None,
    ) -> Dict[str, float]:
        datasets = self.client_datasets if client_datasets is None else client_datasets
        client_flops: List[float] = []
        total_samples = 0
        for client_id in client_ids:
            dataset = datasets.get(client_id)
            num_samples = len(dataset) if dataset is not None else 0
            total_samples += num_samples
            client_flops.append(
                estimate_client_training_flops(
                    expert_flops,
                    num_samples=num_samples,
                    local_epochs=self.config.federated.local_epochs,
                )
            )
        total_flops = float(sum(client_flops))
        avg_flops = total_flops / max(len(client_flops), 1)
        return {
            "avg_flops_per_client": avg_flops,
            "total_flops": total_flops,
            "num_clients": len(client_flops),
            "num_samples": total_samples,
        }

    def _resource_metric_values(
        self,
        resource_profiles: Dict[str, Dict[str, float]],
        client_training_profile: Dict[str, float],
        routed_compute: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        routed_average_flops = 0.0 if routed_compute is None else float(routed_compute.get("average_flops", 0.0))
        return {
            "client_train_flops": float(client_training_profile.get("avg_flops_per_client", 0.0)),
            "client_train_flops_total": float(client_training_profile.get("total_flops", 0.0)),
            "expert_infer_flops": float(resource_profiles["expert"]["flops"]),
            "general_infer_flops": float(resource_profiles["general"]["flops"]),
            "routed_infer_flops": routed_average_flops,
            "expert_train_memory_mb": float(resource_profiles["expert"]["train_memory_mb"]),
            "expert_infer_memory_mb": float(resource_profiles["expert"]["infer_memory_mb"]),
            "general_train_memory_mb": float(resource_profiles["general"]["train_memory_mb"]),
            "general_infer_memory_mb": float(resource_profiles["general"]["infer_memory_mb"]),
            "expert_train_peak_memory_mb": float(resource_profiles["expert"]["train_peak_memory_mb"]),
            "expert_infer_peak_memory_mb": float(resource_profiles["expert"]["infer_peak_memory_mb"]),
            "general_train_peak_memory_mb": float(resource_profiles["general"]["train_peak_memory_mb"]),
            "general_infer_peak_memory_mb": float(resource_profiles["general"]["infer_peak_memory_mb"]),
        }

    def _resource_memory_table(self, resource_profiles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {
            "expert": {
                "train": float(resource_profiles["expert"]["train_memory_mb"]),
                "infer": float(resource_profiles["expert"]["infer_memory_mb"]),
                "train_peak": float(resource_profiles["expert"]["train_peak_memory_mb"]),
                "infer_peak": float(resource_profiles["expert"]["infer_peak_memory_mb"]),
            },
            "general": {
                "train": float(resource_profiles["general"]["train_memory_mb"]),
                "infer": float(resource_profiles["general"]["infer_memory_mb"]),
                "train_peak": float(resource_profiles["general"]["train_peak_memory_mb"]),
                "infer_peak": float(resource_profiles["general"]["infer_peak_memory_mb"]),
            },
        }

    def _format_flops_value(self, value: float) -> str:
        absolute_value = abs(float(value))
        if absolute_value >= 1e12:
            return f"{value / 1e12:.3f} TFLOPs"
        if absolute_value >= 1e9:
            return f"{value / 1e9:.3f} GFLOPs"
        if absolute_value >= 1e6:
            return f"{value / 1e6:.3f} MFLOPs"
        return f"{value:.1f} FLOPs"

    def _format_bytes_value(self, value: float) -> str:
        absolute_value = abs(float(value))
        if absolute_value >= 1024 ** 3:
            return f"{value / (1024 ** 3):.3f} GB"
        if absolute_value >= 1024 ** 2:
            return f"{value / (1024 ** 2):.3f} MB"
        if absolute_value >= 1024:
            return f"{value / 1024:.3f} KB"
        return f"{value:.1f} B"

    def _estimate_tensor_payload_bytes(self, payload: object) -> int:
        if torch.is_tensor(payload):
            return int(payload.numel() * payload.element_size())
        if isinstance(payload, dict):
            return sum(self._estimate_tensor_payload_bytes(value) for value in payload.values())
        if isinstance(payload, (list, tuple)):
            return sum(self._estimate_tensor_payload_bytes(value) for value in payload)
        return 0

    def _format_resource_metrics_for_log(self, metrics: RoundMetrics) -> str:
        return (
            " | client_train="
            f"{self._format_flops_value(metrics.client_train_flops)}"
            " | client_train_total="
            f"{self._format_flops_value(metrics.client_train_flops_total)}"
            " | expert_infer="
            f"{self._format_flops_value(metrics.expert_infer_flops)}"
            " | general_infer="
            f"{self._format_flops_value(metrics.general_infer_flops)}"
            " | routed_infer="
            f"{self._format_flops_value(metrics.routed_infer_flops)}"
            " | expert_mem_theory(train/infer)="
            f"{metrics.expert_train_memory_mb:.1f}/{metrics.expert_infer_memory_mb:.1f} MB"
            " | general_mem_theory(train/infer)="
            f"{metrics.general_train_memory_mb:.1f}/{metrics.general_infer_memory_mb:.1f} MB"
            " | expert_mem_peak(train/infer)="
            f"{metrics.expert_train_peak_memory_mb:.1f}/{metrics.expert_infer_peak_memory_mb:.1f} MB"
            " | general_mem_peak(train/infer)="
            f"{metrics.general_train_peak_memory_mb:.1f}/{metrics.general_infer_peak_memory_mb:.1f} MB"
            " | infer_latency="
            f"{metrics.inference_latency_ms:.3f} ms/sample"
            " | round_time="
            f"{metrics.round_train_time_seconds:.3f} s"
            " | upload_round="
            f"{self._format_bytes_value(metrics.upload_bytes_per_round)}"
            " | upload_total="
            f"{self._format_bytes_value(metrics.upload_bytes_total)}"
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
        base_fields = ["client_id", "sample_index", "true_label", "pred_label", "route_type", "expert_confidence"]
        extra_fields = sorted(
            {
                key
                for record in route_records
                for key in record.keys()
                if key not in base_fields
            }
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=base_fields + extra_fields,
            )
            writer.writeheader()
            writer.writerows(route_records)
        LOGGER.info("Saved route records to %s | rows=%d", path, len(route_records))

    def _format_route_record_value(self, value: object) -> object:
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.detach().cpu().item()
            else:
                value = value.detach().cpu().tolist()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

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
        metric_keys = ("accuracy", "hard_recall", "precision_macro", "recall_macro", "f1_macro", "invocation_rate", "latency_ms")
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
        total_predictor_seconds = 0.0
        route_records: List[Dict[str, object]] = []

        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                self._device_synchronize()
                predictor_start = time.perf_counter()
                predictor_output = predictor(client_id, images, indices.tolist())
                self._device_synchronize()
                total_predictor_seconds += time.perf_counter() - predictor_start
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
                    metadata_columns = {
                        key: value
                        for key, value in route_metadata.items()
                        if isinstance(value, (list, tuple)) and len(value) == len(batch_targets)
                    }
                    if metadata_columns:
                        for row_index, (sample_index, target, pred) in enumerate(
                            zip(indices.tolist(), batch_targets, batch_predictions)
                        ):
                            record = {
                                "client_id": client_id,
                                "sample_index": sample_index,
                                "true_label": target,
                                "pred_label": pred,
                            }
                            for key, values in metadata_columns.items():
                                record[key] = self._format_route_record_value(values[row_index])
                            if "expert_pred" in record:
                                record["expert_correct"] = int(int(record["expert_pred"]) == int(target))
                            if "general_pred" in record:
                                record["general_correct"] = int(int(record["general_pred"]) == int(target))
                            route_records.append(record)

                for sample_index, pred, target in zip(indices.tolist(), batch_predictions, batch_targets):
                    if sample_index in self.test_hard_indices:
                        hard_predictions.append(pred)
                        hard_targets.append(target)

        metrics = self._classification_metrics(all_predictions, all_targets)
        hard_metrics = self._classification_metrics(hard_predictions, hard_targets) if hard_targets else {"accuracy": 0.0}
        metrics["hard_recall"] = hard_metrics["accuracy"]
        metrics["hard_accuracy"] = hard_metrics["accuracy"]
        metrics["invocation_rate"] = invoked_general / max(len(all_targets), 1)
        metrics["latency_ms"] = (total_predictor_seconds / max(len(all_targets), 1)) * 1000.0
        metrics["num_hard_samples"] = len(hard_targets)
        if collect_route_records:
            metrics["_route_records"] = route_records
        return metrics

    def _route_effectiveness_metric_keys(self) -> Tuple[str, ...]:
        return (
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
        )

    def _evaluate_route_effectiveness_metrics_from_predictors(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
        predictor_expert: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        predictor_general: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        predictor_routed: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        general_route_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        expert_macro_accuracy = float(expert_eval["macro"]["accuracy"])
        general_macro_accuracy = float(general_eval["macro"]["accuracy"])
        routed_macro_accuracy = float(routed_eval["macro"]["accuracy"])
        metric_keys = self._route_effectiveness_metric_keys()

        route_type_set = {
            str(route_type)
            for route_type in (general_route_types if general_route_types is not None else ("general",))
        }
        client_metrics: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
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

                for images, targets, indices in loader:
                    images = images.to(self.device)
                    batch_indices = indices.tolist()
                    targets_device = targets.to(self.device)

                    expert_output = predictor_expert(client_id, images, batch_indices)
                    general_output = predictor_general(client_id, images, batch_indices)
                    routed_output = predictor_routed(client_id, images, batch_indices)

                    expert_predictions = expert_output[0]
                    general_predictions = general_output[0]
                    if len(routed_output) == 3:
                        routed_predictions, _, route_metadata = routed_output
                    else:
                        routed_predictions, _ = routed_output
                        route_metadata = None

                    if route_metadata is not None:
                        route_types = route_metadata.get("route_type")
                    else:
                        route_types = None
                    if route_types is not None and len(route_types) == targets_device.numel():
                        invoked_mask = torch.tensor(
                            [str(route_type) in route_type_set for route_type in route_types],
                            device=targets_device.device,
                            dtype=torch.bool,
                        )
                    else:
                        invoked_mask = (
                            routed_predictions.eq(general_predictions)
                            & routed_predictions.ne(expert_predictions)
                        )

                    batch_size = int(targets_device.numel())
                    num_samples += batch_size
                    invoked_total += int(invoked_mask.sum().item())
                    expert_correct += int(expert_predictions.eq(targets_device).sum().item())
                    general_correct += int(general_predictions.eq(targets_device).sum().item())
                    routed_correct += int(routed_predictions.eq(targets_device).sum().item())
                    oracle_correct += int(
                        (expert_predictions.eq(targets_device) | general_predictions.eq(targets_device)).sum().item()
                    )
                    oracle_general_invocations += int(
                        (expert_predictions.ne(targets_device) & general_predictions.eq(targets_device)).sum().item()
                    )
                    disagreement_total += int(expert_predictions.ne(general_predictions).sum().item())

                    if invoked_mask.any():
                        invoked_general_correct += int(
                            general_predictions[invoked_mask].eq(targets_device[invoked_mask]).sum().item()
                        )
                        invoked_expert_correct += int(
                            expert_predictions[invoked_mask].eq(targets_device[invoked_mask]).sum().item()
                        )

                if num_samples == 0:
                    continue

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
                }

        if not client_metrics:
            return {
                "general_gain_over_expert": general_macro_accuracy - expert_macro_accuracy,
                "routed_gain_over_expert": routed_macro_accuracy - expert_macro_accuracy,
                **{
                    key: 0.0
                    for key in metric_keys
                    if key not in {"general_gain_over_expert", "routed_gain_over_expert"}
                },
            }

        aggregated = {
            key: sum(float(metrics[key]) for metrics in client_metrics.values()) / max(len(client_metrics), 1)
            for key in metric_keys
        }
        aggregated["general_gain_over_expert"] = general_macro_accuracy - expert_macro_accuracy
        aggregated["routed_gain_over_expert"] = routed_macro_accuracy - expert_macro_accuracy
        return aggregated

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
            "%s summary | clients=%d | samples=%d | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f | latency_ms=%.4f",
            prefix,
            weighted["num_clients"],
            weighted["num_samples"],
            macro["accuracy"],
            weighted["accuracy"],
            weighted["hard_recall"],
            weighted["precision_macro"],
            weighted["recall_macro"],
            weighted["f1_macro"],
            weighted["invocation_rate"],
            weighted["latency_ms"],
        )
        for group_name, metrics in sorted(groups.items()):
            LOGGER.info(
                "%s group=%s | samples=%d | acc=%.4f | hard_recall=%.4f | invocation=%.4f | latency_ms=%.4f",
                prefix,
                group_name,
                metrics["num_samples"],
                metrics["accuracy"],
                metrics["hard_recall"],
                metrics["invocation_rate"],
                metrics["latency_ms"],
            )
        for client_id, metrics in sorted(client_results.items()):
            LOGGER.info(
                "%s client=%s | n=%d | acc=%.4f | hard_recall=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f | latency_ms=%.4f",
                prefix,
                client_id,
                metrics["num_samples"],
                metrics["accuracy"],
                metrics["hard_recall"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
                metrics["invocation_rate"],
                metrics["latency_ms"],
            )

    def _log_round_metrics(self, prefix: str, metrics: RoundMetrics) -> None:
        if self.writer is None:
            return
        self._log_compare_scalars(
            prefix,
            metrics.round_idx,
            {
                "loss": metrics.avg_client_loss,
                "accuracy": metrics.routed_accuracy,
                "routed_accuracy": metrics.routed_accuracy,
                "weighted_accuracy": metrics.weighted_accuracy,
                "hard_accuracy": metrics.hard_accuracy,
                "invocation_rate": metrics.invocation_rate,
                "compute_savings": metrics.compute_savings,
                "client_train_flops": metrics.client_train_flops,
                "client_train_flops_total": metrics.client_train_flops_total,
                "expert_infer_flops": metrics.expert_infer_flops,
                "general_infer_flops": metrics.general_infer_flops,
                "routed_infer_flops": metrics.routed_infer_flops,
                "expert_train_memory_mb": metrics.expert_train_memory_mb,
                "expert_infer_memory_mb": metrics.expert_infer_memory_mb,
                "general_train_memory_mb": metrics.general_train_memory_mb,
                "general_infer_memory_mb": metrics.general_infer_memory_mb,
                "expert_train_peak_memory_mb": metrics.expert_train_peak_memory_mb,
                "expert_infer_peak_memory_mb": metrics.expert_infer_peak_memory_mb,
                "general_train_peak_memory_mb": metrics.general_train_peak_memory_mb,
                "general_infer_peak_memory_mb": metrics.general_infer_peak_memory_mb,
                "inference_latency_ms": metrics.inference_latency_ms,
                "round_train_time_seconds": metrics.round_train_time_seconds,
                "upload_bytes_per_round": metrics.upload_bytes_per_round,
                "upload_bytes_total": metrics.upload_bytes_total,
                **metrics.extra_metrics,
            },
        )

    def _log_auxiliary_accuracy_metrics(
        self,
        prefix: str,
        round_idx: int,
        expert_accuracy: Optional[float],
        general_accuracy: Optional[float],
    ) -> None:
        if self.writer is None:
            return
        if hasattr(self.writer, "add_compare_scalar"):
            if expert_accuracy is not None:
                self.writer.add_compare_scalar(prefix, "expert_only_accuracy", expert_accuracy, round_idx)
            if general_accuracy is not None:
                self.writer.add_compare_scalar(prefix, "general_only_accuracy", general_accuracy, round_idx)
            return
        if expert_accuracy is not None:
            self.writer.add_scalar(f"compare/expert_only_accuracy/{prefix}", expert_accuracy, round_idx)
        if general_accuracy is not None:
            self.writer.add_scalar(f"compare/general_only_accuracy/{prefix}", general_accuracy, round_idx)

    def _log_compare_scalars(self, prefix: str, round_idx: int, scalar_map: Dict[str, Optional[float]]) -> None:
        if self.writer is None:
            return
        if hasattr(self.writer, "add_compare_scalar"):
            for metric_name, value in scalar_map.items():
                if value is None:
                    continue
                self.writer.add_compare_scalar(prefix, metric_name, float(value), round_idx)
            return
        for metric_name, value in scalar_map.items():
            if value is None:
                continue
            self.writer.add_scalar(f"compare/{metric_name}/{prefix}", float(value), round_idx)

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
