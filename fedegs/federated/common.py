import logging
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.config import ExperimentConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    delta: Dict[str, torch.Tensor]


@dataclass
class RoundMetrics:
    round_idx: int
    avg_client_loss: float
    routed_accuracy: float
    hard_accuracy: float
    invocation_rate: float


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

    def _sample_client_ids(self) -> List[str]:
        return self.random.sample(
            list(self.client_datasets.keys()),
            k=min(self.config.federated.clients_per_round, len(self.client_datasets)),
        )

    def _evaluate_predictor_on_client_tests(
        self,
        predictor: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        prefix: str,
    ) -> Dict[str, object]:
        client_results: Dict[str, Dict[str, float]] = {}
        weighted_sums = {
            "accuracy": 0.0,
            "hard_accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "invocation_rate": 0.0,
        }
        total_samples = 0

        for client_id, dataset in self.client_test_datasets.items():
            loader = self.data_module.make_loader(dataset, shuffle=False)
            client_metrics = self._evaluate_predictor_on_loader(client_id, predictor, loader)
            client_results[client_id] = client_metrics
            sample_count = client_metrics["num_samples"]
            total_samples += sample_count
            for key in weighted_sums:
                weighted_sums[key] += client_metrics[key] * sample_count

        aggregated = {key: weighted_sums[key] / max(total_samples, 1) for key in weighted_sums}
        aggregated["num_clients"] = len(client_results)
        aggregated["num_samples"] = total_samples

        self._log_client_metrics_table(prefix, client_results, aggregated)
        return {"aggregate": aggregated, "clients": client_results}

    def _evaluate_predictor_on_loader(
        self,
        client_id: str,
        predictor: Callable[[str, torch.Tensor, List[int]], Tuple[torch.Tensor, int]],
        loader: DataLoader,
    ) -> Dict[str, float]:
        all_predictions: List[int] = []
        all_targets: List[int] = []
        hard_predictions: List[int] = []
        hard_targets: List[int] = []
        invoked_general = 0

        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                predictions, batch_invocations = predictor(client_id, images, indices.tolist())
                invoked_general += batch_invocations

                batch_predictions = predictions.detach().cpu().tolist()
                batch_targets = targets.tolist()
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)

                for sample_index, pred, target in zip(indices.tolist(), batch_predictions, batch_targets):
                    if sample_index in self.test_hard_indices:
                        hard_predictions.append(pred)
                        hard_targets.append(target)

        metrics = self._classification_metrics(all_predictions, all_targets)
        hard_metrics = self._classification_metrics(hard_predictions, hard_targets) if hard_targets else {"accuracy": 0.0}
        metrics["hard_accuracy"] = hard_metrics["accuracy"]
        metrics["invocation_rate"] = invoked_general / max(len(all_targets), 1)
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

    def _log_client_metrics_table(self, prefix: str, client_results: Dict[str, Dict[str, float]], aggregated: Dict[str, float]) -> None:
        LOGGER.info(
            "%s personalized aggregate | clients=%d | samples=%d | acc=%.4f | hard_acc=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f",
            prefix,
            aggregated["num_clients"],
            aggregated["num_samples"],
            aggregated["accuracy"],
            aggregated["hard_accuracy"],
            aggregated["precision_macro"],
            aggregated["recall_macro"],
            aggregated["f1_macro"],
            aggregated["invocation_rate"],
        )
        for client_id, metrics in sorted(client_results.items()):
            LOGGER.info(
                "%s client=%s | n=%d | acc=%.4f | hard_acc=%.4f | prec=%.4f | recall=%.4f | f1=%.4f | invocation=%.4f",
                prefix,
                client_id,
                metrics["num_samples"],
                metrics["accuracy"],
                metrics["hard_accuracy"],
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
        self.writer.add_scalar(f"hard_accuracy/{prefix}", metrics.hard_accuracy, metrics.round_idx)
        self.writer.add_scalar(f"invocation_rate/{prefix}", metrics.invocation_rate, metrics.round_idx)

        # Also log grouped comparison panes so multiple algorithms appear in one TensorBoard chart.
        self.writer.add_scalars("compare/loss", {prefix: metrics.avg_client_loss}, metrics.round_idx)
        self.writer.add_scalars("compare/accuracy", {prefix: metrics.routed_accuracy}, metrics.round_idx)
        self.writer.add_scalars("compare/hard_accuracy", {prefix: metrics.hard_accuracy}, metrics.round_idx)
        self.writer.add_scalars("compare/invocation_rate", {prefix: metrics.invocation_rate}, metrics.round_idx)
