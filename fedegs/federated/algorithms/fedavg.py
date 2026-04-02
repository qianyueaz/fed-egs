import copy
from typing import Dict, List

from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, ClientUpdate, RoundMetrics, LOGGER
from fedegs.models import average_weighted_deltas, build_baseline_model, model_memory_mb
from fedegs.models.width_scalable_resnet import state_dict_delta


class FedAvgClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, device: str, config) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config

    def train(self, global_model: nn.Module, loader: DataLoader) -> ClientUpdate:
        local_model = copy.deepcopy(global_model).to(self.device)
        before_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
        loss = self._optimize_model(
            model=local_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        after_state = local_model.state_dict()
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            delta=state_dict_delta(after_state, before_state),
        )


class FedAvgServer(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.global_model = build_baseline_model(config).to(self.device)
        self.clients = {
            client_id: FedAvgClient(client_id, dataset, config.federated.device, config)
            for client_id, dataset in client_datasets.items()
        }
        self.last_history: List[RoundMetrics] = []

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            updates: List[ClientUpdate] = []
            for client_id in sampled_ids:
                loader = self.data_module.make_loader(self.clients[client_id].dataset, shuffle=True)
                updates.append(self.clients[client_id].train(self.global_model, loader))

            aggregated_delta = average_weighted_deltas((update.num_samples, update.delta) for update in updates)
            updated_state = self.global_model.state_dict()
            for key, delta in aggregated_delta.items():
                if key in updated_state:
                    updated_state[key] = updated_state[key] + delta.to(updated_state[key].device)
            self.global_model.load_state_dict(updated_state)

            personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="fedavg")
            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            aggregate = personalized_eval["aggregate"]
            round_metrics = RoundMetrics(round_idx, avg_loss, aggregate["accuracy"], aggregate["hard_accuracy"], aggregate["invocation_rate"])
            LOGGER.info("fedavg round %d | loss=%.4f | acc=%.4f | hard_acc=%.4f", round_idx, avg_loss, aggregate["accuracy"], aggregate["hard_accuracy"])
            self._log_round_metrics("fedavg", round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="fedavg_final")
        aggregate = personalized_eval["aggregate"]
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        return {
            "algorithm": "fedavg",
            "metrics": {
                **aggregate,
                "final_training_loss": final_loss,
            },
            "client_metrics": personalized_eval["clients"],
            "memory_mb": {
                "expert": model_memory_mb(self.global_model),
                "general": model_memory_mb(self.global_model),
            },
        }

    def _predict_with_global_model(self, client_id, images, indices):
        logits = self.global_model(images)
        return logits.argmax(dim=1), 0
