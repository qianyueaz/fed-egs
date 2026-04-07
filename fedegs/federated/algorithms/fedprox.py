import copy
from typing import Dict, List

from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, ClientUpdate, RoundMetrics, LOGGER
from fedegs.models import SmallCNN, average_weighted_deltas, build_model, estimate_model_flops, model_memory_mb
from fedegs.models.width_scalable_resnet import state_dict_delta


class FedProxClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, device: str, config) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config

    def train(self, global_model: nn.Module, loader: DataLoader) -> ClientUpdate:
        local_model = copy.deepcopy(global_model).to(self.device)
        reference_model = copy.deepcopy(global_model).to(self.device)
        before_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
        loss = self._optimize_model(
            model=local_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
            reference_model=reference_model,
            prox_mu=self.config.federated.prox_mu,
        )
        after_state = local_model.state_dict()
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            delta=state_dict_delta(after_state, before_state),
        )


class FedProxServer(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.global_model = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
            knowledge_dim=config.model.knowledge_dim,
        ).to(self.device)
        reference_general_model = build_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            width_factor=config.model.general_width,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.model_flops = estimate_model_flops(self.global_model)
        self.general_flops = estimate_model_flops(reference_general_model)
        self.clients = {
            client_id: FedProxClient(client_id, dataset, config.federated.device, config)
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

            personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="fedprox")
            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            aggregate = personalized_eval["aggregate"]
            macro = personalized_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.model_flops,
                self.general_flops,
                aggregate["invocation_rate"],
                mode="expert_only" if self.model_flops < self.general_flops else "general_only",
            )
            round_metrics = RoundMetrics(
                round_idx,
                avg_loss,
                aggregate["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
            )
            LOGGER.info(
                "fedprox round %d | loss=%.4f | global_acc=%.4f | local_acc=%.4f | hard_recall=%.4f | savings=%.4f",
                round_idx,
                avg_loss,
                aggregate["accuracy"],
                macro["accuracy"],
                aggregate["hard_recall"],
                compute_profile["savings_ratio"],
            )
            self._log_round_metrics("fedprox", round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="fedprox_final")
        aggregate = personalized_eval["aggregate"]
        macro = personalized_eval["macro"]
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        compute_profile = self._build_compute_profile(
            self.model_flops,
            self.general_flops,
            aggregate["invocation_rate"],
            mode="expert_only" if self.model_flops < self.general_flops else "general_only",
        )
        return {
            "algorithm": "fedprox",
            "metrics": {
                **aggregate,
                "global_accuracy": aggregate["accuracy"],
                "local_accuracy": macro["accuracy"],
                "hard_sample_recall": aggregate["hard_recall"],
                "routed_accuracy": aggregate["accuracy"],
                "routed_hard_accuracy": aggregate["hard_recall"],
                "compute_savings": compute_profile["savings_ratio"],
                "final_training_loss": final_loss,
            },
            "client_metrics": personalized_eval["clients"],
            "group_metrics": personalized_eval["groups"],
            "compute": {
                "model": compute_profile,
            },
            "memory_mb": {
                "expert": model_memory_mb(self.global_model),
                "general": model_memory_mb(
                    build_model(
                        architecture=self.config.model.architecture,
                        num_classes=self.config.model.num_classes,
                        width_factor=self.config.model.general_width,
                        base_channels=self.config.model.expert_base_channels,
                    )
                ),
            },
        }

    def _predict_with_global_model(self, client_id, images, indices):
        logits = self.global_model(images)
        return logits.argmax(dim=1), 0
