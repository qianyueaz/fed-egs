import copy
import time
from typing import Dict, List

from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, ClientUpdate, RoundMetrics, LOGGER
from fedegs.models import (
    average_weighted_deltas,
    build_baseline_model,
    build_model,
    estimate_model_flops,
)
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
        self.global_model = build_baseline_model(config).to(self.device)
        reference_general_model = build_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            width_factor=config.model.general_width,
            base_channels=config.model.general_base_channels,
        ).to(self.device)
        self.model_flops = estimate_model_flops(self.global_model)
        self.general_flops = estimate_model_flops(reference_general_model)
        self.resource_profiles = self._build_dual_model_resource_profiles(
            self.global_model,
            reference_general_model,
            self.model_flops,
            self.general_flops,
        )
        self.clients = {
            client_id: FedProxClient(client_id, dataset, config.federated.device, config)
            for client_id, dataset in client_datasets.items()
        }
        self.last_history: List[RoundMetrics] = []

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        upload_bytes_total = 0.0
        for round_idx in range(1, self.config.federated.rounds + 1):
            self._device_synchronize()
            round_start_time = time.perf_counter()
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
            client_train_profile = self._build_client_training_profile(
                self.model_flops,
                sampled_ids,
                self.client_datasets,
            )
            resource_metrics = self._resource_metric_values(
                self.resource_profiles,
                client_train_profile,
                compute_profile,
            )
            round_upload_bytes = float(sum(self._estimate_tensor_payload_bytes(update.delta) for update in updates))
            upload_bytes_total += round_upload_bytes
            self._device_synchronize()
            round_train_time_seconds = time.perf_counter() - round_start_time
            round_metrics = RoundMetrics(
                round_idx,
                avg_loss,
                macro["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                inference_latency_ms=aggregate["latency_ms"],
                round_train_time_seconds=round_train_time_seconds,
                upload_bytes_per_round=round_upload_bytes,
                upload_bytes_total=upload_bytes_total,
                **resource_metrics,
            )
            LOGGER.info(
                "fedprox round %d | loss=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | savings=%.4f%s",
                round_idx,
                avg_loss,
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                compute_profile["savings_ratio"],
                self._format_resource_metrics_for_log(round_metrics),
            )
            self._log_round_metrics("fedprox", round_metrics)
            self._log_auxiliary_accuracy_metrics("fedprox", round_idx, expert_accuracy=None, general_accuracy=macro["accuracy"])
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
        general_reference_profile = self._build_compute_profile(
            self.model_flops,
            self.general_flops,
            1.0,
            mode="general_only",
        )
        final_history = self.last_history[-1] if self.last_history else RoundMetrics(0, 0.0, 0.0, 0.0, 0.0)
        final_client_train = {
            "avg_flops_per_client": final_history.client_train_flops,
            "total_flops": final_history.client_train_flops_total,
            "num_clients": self.config.federated.clients_per_round,
            "num_samples": 0,
        }
        final_resource_metrics = self._resource_metric_values(
            self.resource_profiles,
            final_client_train,
            compute_profile,
        )
        average_round_train_time_seconds = (
            sum(item.round_train_time_seconds for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_train_time_seconds = (
            sum(item.round_train_time_seconds for item in self.last_history)
            if self.last_history
            else 0.0
        )
        average_upload_bytes_per_round = (
            sum(item.upload_bytes_per_round for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_upload_bytes = self.last_history[-1].upload_bytes_total if self.last_history else 0.0
        return {
            "algorithm": "fedprox",
            "metrics": {
                **aggregate,
                "accuracy": macro["accuracy"],
                "personalized_accuracy": macro["accuracy"],
                "weighted_accuracy": aggregate["accuracy"],
                "global_accuracy": macro["accuracy"],
                "local_accuracy": macro["accuracy"],
                "hard_sample_recall": aggregate["hard_recall"],
                "routed_accuracy": macro["accuracy"],
                "routed_hard_accuracy": aggregate["hard_recall"],
                "general_only_accuracy": macro["accuracy"],
                "compute_savings": compute_profile["savings_ratio"],
                "final_training_loss": final_loss,
                "average_inference_latency_ms": aggregate["latency_ms"],
                "average_round_train_time_seconds": average_round_train_time_seconds,
                "total_train_time_seconds": total_train_time_seconds,
                "average_upload_bytes_per_round": average_upload_bytes_per_round,
                "total_upload_bytes": total_upload_bytes,
                **final_resource_metrics,
            },
            "client_metrics": personalized_eval["clients"],
            "group_metrics": personalized_eval["groups"],
            "compute": {
                "model": compute_profile,
                "general_reference": general_reference_profile,
                "client_train": final_client_train,
            },
            "memory_mb": self._resource_memory_table(self.resource_profiles),
        }

    def _predict_with_global_model(self, client_id, images, indices):
        logits = self.global_model(images)
        return logits.argmax(dim=1), 0
