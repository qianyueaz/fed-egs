import random
import statistics
import time
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, ClientUpdate, LOGGER, RoundMetrics
from fedegs.models import (
    average_weighted_deltas,
    build_baseline_model,
    build_model,
    estimate_model_flops,
)
from fedegs.models.width_scalable_resnet import state_dict_delta


class FedALAClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, device: str, config, data_module) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.personal_state: Optional[Dict[str, torch.Tensor]] = None
        self._temp_model = build_baseline_model(config).to(self.device)
        self._ala_start_phase = True
        self._ala_calls = 0

    def train(self, global_model: nn.Module, loader: DataLoader) -> ClientUpdate:
        local_model = build_baseline_model(self.config).to(self.device)
        global_state_cpu = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}

        if self.personal_state is None:
            local_model.load_state_dict(global_state_cpu)
        else:
            local_model.load_state_dict(self.personal_state, strict=True)
            self._adaptive_local_aggregation(global_model, local_model)

        loss = self._optimize_model(
            model=local_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        after_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
        self.personal_state = after_state
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            delta=state_dict_delta(after_state, global_state_cpu),
        )

    def export_personal_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.personal_state is None:
            return None
        return {key: value.detach().cpu().clone() for key, value in self.personal_state.items()}

    def _adaptive_local_aggregation(self, global_model: nn.Module, local_model: nn.Module) -> None:
        local_params = list(local_model.parameters())
        global_params = list(global_model.parameters())
        if not local_params:
            return

        layer_count = int(getattr(self.config.federated, "fedala_layer_idx", 0))
        if layer_count <= 0 or layer_count > len(local_params):
            layer_count = len(local_params)
        preserve_count = len(local_params) - layer_count

        for local_param, global_param in zip(local_params[:preserve_count], global_params[:preserve_count]):
            local_param.data.copy_(global_param.data)

        selected_local = local_params[preserve_count:]
        selected_global = global_params[preserve_count:]
        if not selected_local:
            return

        if not any(not torch.equal(local_param.data, global_param.data) for local_param, global_param in zip(selected_local, selected_global)):
            return

        subset_loader = self._build_ala_subset_loader()
        if subset_loader is None:
            return

        temp_model = self._temp_model
        temp_model.load_state_dict(local_model.state_dict(), strict=True)
        temp_params = list(temp_model.parameters())
        selected_temp = temp_params[preserve_count:]

        for temp_param in temp_params:
            temp_param.requires_grad_(True)
        for temp_param in temp_params[:preserve_count]:
            temp_param.requires_grad_(False)

        init_alpha = float(getattr(self.config.federated, "fedala_init_alpha", 0.5))
        init_alpha = min(max(init_alpha, 0.0), 1.0)
        for local_param, temp_param, global_param in zip(selected_local, selected_temp, selected_global):
            temp_param.data.copy_(global_param.data + (local_param.data - global_param.data) * init_alpha)

        criterion = torch.nn.CrossEntropyLoss()
        weights = [torch.ones_like(temp_param.data, device=self.device) for temp_param in selected_temp]
        eta = float(getattr(self.config.federated, "fedala_eta", 1.0))
        threshold = max(float(getattr(self.config.federated, "fedala_convergence_threshold", 0.01)), 0.0)
        start_phase_epochs = max(int(getattr(self.config.federated, "fedala_start_phase_epochs", 5)), 1)
        adaptation_epochs = max(int(getattr(self.config.federated, "fedala_adaptation_epochs", 1)), 1)
        max_epochs = start_phase_epochs if self._ala_start_phase else adaptation_epochs
        recent_losses: List[float] = []

        for _ in range(max_epochs):
            for images, targets, _ in subset_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                temp_model.zero_grad(set_to_none=True)
                outputs = temp_model(images)
                loss = criterion(outputs, targets)
                loss.backward()

                for local_param, temp_param, global_param, weight in zip(selected_local, selected_temp, selected_global, weights):
                    if temp_param.grad is None:
                        continue
                    weight.data = torch.clamp(
                        weight.data - eta * temp_param.grad * (global_param.data - local_param.data),
                        0.0,
                        1.0,
                    )
                    temp_param.data.copy_(global_param.data + (local_param.data - global_param.data) * weight.data)

                recent_losses.append(float(loss.detach().cpu().item()))
                if len(recent_losses) > 5:
                    recent_losses.pop(0)

            if not self._ala_start_phase:
                break
            if len(recent_losses) >= 2 and statistics.pstdev(recent_losses) < threshold:
                break

        for local_param, temp_param in zip(selected_local, selected_temp):
            local_param.data.copy_(temp_param.data)

        self._ala_start_phase = False
        self._ala_calls += 1

    def _build_ala_subset_loader(self) -> Optional[DataLoader]:
        dataset_size = len(self.dataset)
        if dataset_size == 0:
            return None

        rand_percent = float(getattr(self.config.federated, "fedala_rand_percent", 0.8))
        rand_percent = min(max(rand_percent, 0.0), 1.0)
        subset_size = max(1, int(round(dataset_size * rand_percent)))
        subset_size = min(subset_size, dataset_size)

        if subset_size == dataset_size:
            subset_dataset = self.dataset
        else:
            rng = random.Random(self._stable_ala_seed())
            sampled_indices = sorted(rng.sample(range(dataset_size), subset_size))
            subset_dataset = Subset(self.dataset, sampled_indices)
        return self.data_module.make_loader(subset_dataset, shuffle=False)

    def _stable_ala_seed(self) -> int:
        client_hash = sum((index + 1) * ord(char) for index, char in enumerate(self.client_id))
        return int(self.config.federated.seed) + client_hash + self._ala_calls * 9973


class FedALAServer(BaseFederatedServer):
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
            client_id: FedALAClient(client_id, dataset, config.federated.device, config, data_module)
            for client_id, dataset in client_datasets.items()
        }
        self.last_history: List[RoundMetrics] = []
        self._personalized_eval_model = build_baseline_model(config).to(self.device)
        self._cached_eval_client_id: Optional[str] = None

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

            self._cached_eval_client_id = None
            personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_personalized_model, prefix="fedala")
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
                "fedala round %d | loss=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | savings=%.4f%s",
                round_idx,
                avg_loss,
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                compute_profile["savings_ratio"],
                self._format_resource_metrics_for_log(round_metrics),
            )
            self._log_round_metrics("fedala", round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        self._cached_eval_client_id = None
        personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_personalized_model, prefix="fedala_final")
        global_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="fedala_global_final")
        aggregate = personalized_eval["aggregate"]
        macro = personalized_eval["macro"]
        global_macro = global_eval["macro"]
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
            "algorithm": "fedala",
            "metrics": {
                **aggregate,
                "accuracy": macro["accuracy"],
                "personalized_accuracy": macro["accuracy"],
                "weighted_accuracy": aggregate["accuracy"],
                "global_accuracy": global_macro["accuracy"],
                "local_accuracy": macro["accuracy"],
                "hard_sample_recall": aggregate["hard_recall"],
                "routed_accuracy": macro["accuracy"],
                "routed_hard_accuracy": aggregate["hard_recall"],
                "general_only_accuracy": global_macro["accuracy"],
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

    def _predict_with_personalized_model(self, client_id, images, indices):
        model = self._load_personalized_eval_model(client_id)
        logits = model(images)
        return logits.argmax(dim=1), 0

    def _load_personalized_eval_model(self, client_id: str) -> nn.Module:
        if self._cached_eval_client_id != client_id:
            client_state = self.clients[client_id].export_personal_state()
            if client_state is None:
                client_state = {key: value.detach().cpu().clone() for key, value in self.global_model.state_dict().items()}
            self._personalized_eval_model.load_state_dict(client_state, strict=True)
            self._personalized_eval_model.to(self.device)
            self._personalized_eval_model.eval()
            self._cached_eval_client_id = client_id
        return self._personalized_eval_model
