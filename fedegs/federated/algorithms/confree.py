import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fedegs.federated.algorithms.fedala import FedALAClient
from fedegs.federated.common import BaseFederatedServer, ClientUpdate, LOGGER, RoundMetrics
from fedegs.models import build_baseline_model, build_model, estimate_model_flops


class ConFREEAggregator:
    def __init__(self, alpha: float, seed: int, solver_iterations: int = 80, solver_lr: float = 0.05) -> None:
        self.alpha = max(float(alpha), 0.0)
        self.solver_iterations = max(int(solver_iterations), 1)
        self.solver_lr = max(float(solver_lr), 1e-6)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, updates: List[ClientUpdate], conflict_keys: Optional[set[str]] = None) -> Dict[str, torch.Tensor]:
        if not updates:
            return {}

        keys = list(updates[0].delta.keys())
        aggregated: Dict[str, torch.Tensor] = {}
        for key in keys:
            client_updates = [update.delta[key].detach().cpu() for update in updates if key in update.delta]
            if not client_updates:
                continue
            if conflict_keys is not None and key not in conflict_keys:
                total_samples = sum(update.num_samples for update in updates if key in update.delta)
                weighted_sum = sum(
                    update.delta[key].detach().cpu() * update.num_samples
                    for update in updates
                    if key in update.delta
                )
                aggregated[key] = weighted_sum / max(total_samples, 1)
                continue
            if len(client_updates) == 1:
                aggregated[key] = client_updates[0].clone()
                continue

            shape = client_updates[0].shape
            flattened = torch.stack([tensor.reshape(-1) for tensor in client_updates], dim=1).float()
            aggregated[key] = self._resolve_conflict(flattened).reshape(shape).to(dtype=client_updates[0].dtype)
        return aggregated

    def _resolve_conflict(self, client_updates: torch.Tensor) -> torch.Tensor:
        guidance = self._build_guidance_vector(client_updates)
        guidance_norm = guidance.norm()
        constraint_radius = self.alpha * guidance_norm + 1e-8
        optimum = self._solve_simplex_objective(client_updates, guidance, constraint_radius)
        weighted_update = (client_updates * optimum.view(1, -1)).sum(dim=1)
        scale = constraint_radius / (weighted_update.norm() + 1e-8)
        return (guidance + scale * weighted_update) / (1.0 + self.alpha**2)

    def _build_guidance_vector(self, client_updates: torch.Tensor) -> torch.Tensor:
        grad_vec = client_updates.t()
        num_clients = client_updates.shape[1]
        if num_clients <= 1:
            return grad_vec.mean(dim=0)

        shuffled_indices = np.zeros((num_clients, num_clients - 1), dtype=int)
        for client_idx in range(num_clients):
            candidates = np.arange(num_clients)
            candidates[client_idx] = candidates[-1]
            shuffled_indices[client_idx] = candidates[:-1]
            self.rng.shuffle(shuffled_indices[client_idx])

        modified = grad_vec.clone()
        for task_indices in shuffled_indices.T:
            shuffled = grad_vec[task_indices]
            dot = (modified * shuffled).sum(dim=1, keepdim=True)
            modified = modified - torch.clamp_max(dot, 0.0) * shuffled
        return modified.mean(dim=0)

    def _solve_simplex_objective(
        self,
        client_updates: torch.Tensor,
        guidance: torch.Tensor,
        constraint_radius: torch.Tensor,
    ) -> torch.Tensor:
        scipy_solution = self._solve_with_scipy(client_updates, guidance, constraint_radius)
        if scipy_solution is not None:
            return scipy_solution
        return self._solve_with_torch(client_updates, guidance, constraint_radius)

    def _solve_with_scipy(
        self,
        client_updates: torch.Tensor,
        guidance: torch.Tensor,
        constraint_radius: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        try:
            from scipy.optimize import minimize
        except Exception:
            return None

        gram = client_updates.t().mm(client_updates).numpy()
        guidance_column = guidance.view(-1, 1)
        update_guidance = client_updates.t().mm(guidance_column).numpy()
        radius = float(constraint_radius.item())
        num_clients = client_updates.shape[1]
        start = np.ones(num_clients, dtype=np.float64) / num_clients

        def objective(weights):
            linear_term = float(np.dot(weights, update_guidance).item())
            quadratic_term = float(radius * np.sqrt(weights.dot(gram).dot(weights) + 1e-8))
            return linear_term + quadratic_term

        result = minimize(
            objective,
            start,
            bounds=tuple((0.0, 1.0) for _ in range(num_clients)),
            constraints={"type": "eq", "fun": lambda weights: 1.0 - np.sum(weights)},
        )
        if not result.success:
            return None
        return torch.tensor(result.x, dtype=client_updates.dtype)

    def _solve_with_torch(
        self,
        client_updates: torch.Tensor,
        guidance: torch.Tensor,
        constraint_radius: torch.Tensor,
    ) -> torch.Tensor:
        num_clients = client_updates.shape[1]
        logits = torch.zeros(num_clients, dtype=client_updates.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=self.solver_lr)
        guidance_projection = client_updates.t().mv(guidance)
        gram = client_updates.t().mm(client_updates)

        for _ in range(self.solver_iterations):
            weights = torch.softmax(logits, dim=0)
            objective = weights.dot(guidance_projection) + constraint_radius * torch.sqrt(weights.dot(gram.mv(weights)) + 1e-8)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        return torch.softmax(logits.detach(), dim=0)


class ConFREEServer(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.algorithm_name = "confree"
        self.global_model = build_baseline_model(config).to(self.device)
        self.conflict_keys = {name for name, _ in self.global_model.named_parameters()}
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
        self.aggregator = ConFREEAggregator(
            alpha=float(getattr(config.federated, "confree_alpha", 0.5)),
            seed=int(config.federated.seed),
            solver_iterations=int(getattr(config.federated, "confree_solver_iterations", 80)),
            solver_lr=float(getattr(config.federated, "confree_solver_lr", 0.05)),
        )
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

            aggregated_delta = self.aggregator.aggregate(updates, conflict_keys=self.conflict_keys)
            updated_state = self.global_model.state_dict()
            for key, delta in aggregated_delta.items():
                if key in updated_state:
                    updated_state[key] = updated_state[key] + delta.to(updated_state[key].device)
            self.global_model.load_state_dict(updated_state)

            self._cached_eval_client_id = None
            personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_personalized_model, prefix=self.algorithm_name)
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
                extra_metrics={"confree_alpha": self.aggregator.alpha},
                **resource_metrics,
            )
            LOGGER.info(
                "confree round %d | loss=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | savings=%.4f%s",
                round_idx,
                avg_loss,
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                compute_profile["savings_ratio"],
                self._format_resource_metrics_for_log(round_metrics),
            )
            self._log_round_metrics(self.algorithm_name, round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        self._cached_eval_client_id = None
        personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_personalized_model, prefix="confree_final")
        global_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="confree_global_final")
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
            "algorithm": "confree",
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
                "confree_alpha": self.aggregator.alpha,
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
