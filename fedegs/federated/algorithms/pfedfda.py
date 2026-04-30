import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.models import average_weighted_deltas, build_baseline_model, build_model, estimate_model_flops
from fedegs.models.width_scalable_resnet import state_dict_delta


@dataclass
class PFedFDAClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    delta: Dict[str, torch.Tensor]
    adaptive_means: torch.Tensor
    adaptive_covariance: torch.Tensor
    means_beta: float
    covariance_beta: float


def _unpack_batch(batch):
    if len(batch) == 3:
        images, targets, _ = batch
    else:
        images, targets = batch
    return images, targets


def _forward_with_features(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_with_features"):
        return model.forward_with_features(images)
    raise ValueError(f"Model {type(model).__name__} must expose forward_with_features for pFedFDA.")


def _classifier(model: nn.Module) -> nn.Module:
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise ValueError(f"Model {type(model).__name__} must expose classifier or fc for pFedFDA.")


class PFedFDAClient(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, device: str, config, data_module) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.num_classes = int(config.model.num_classes)
        self.eps = float(getattr(config.federated, "pfedfda_eps", 1e-4))
        self.ridge = float(getattr(config.federated, "pfedfda_covariance_ridge", 1e-4))
        self.min_class_samples = max(int(getattr(config.federated, "pfedfda_min_class_samples", 2)), 1)
        reference_model = build_baseline_model(config)
        self.feature_dim = int(getattr(reference_model, "feature_dim"))
        self.means = torch.randn(self.num_classes, self.feature_dim)
        self.covariance = torch.eye(self.feature_dim)
        self.global_means = self.means.clone()
        self.global_covariance = self.covariance.clone()
        self.adaptive_means = self.means.clone()
        self.adaptive_covariance = self.covariance.clone()
        self.means_beta = float(getattr(config.federated, "pfedfda_beta", 0.5))
        self.covariance_beta = self.means_beta
        self.priors = self._estimate_priors()

    def train(self, global_model: nn.Module, loader: DataLoader) -> PFedFDAClientUpdate:
        local_model = build_baseline_model(self.config).to(self.device)
        global_state_cpu = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
        local_model.load_state_dict(global_state_cpu, strict=True)
        self._move_statistics(self.device)
        self._set_lda_classifier(local_model, self.global_means, self.global_covariance, self.priors)
        self._set_classifier_requires_grad(local_model, False)

        trainable_params = [param for param in local_model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        local_model.train()
        total_loss = 0.0
        total_batches = 0
        feature_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []

        for _ in range(self.config.federated.local_epochs):
            for batch in loader:
                images, targets = _unpack_batch(batch)
                images = images.to(self.device)
                targets = targets.to(self.device)
                features, logits = _forward_with_features(local_model, images)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
                feature_batches.append(features.detach())
                label_batches.append(targets.detach())

        if feature_batches:
            features = torch.cat(feature_batches, dim=0)
            labels = torch.cat(label_batches, dim=0)
            means, covariance, _ = self._compute_mle_statistics(features, labels)
            self._update_adaptive_statistics(features, labels, means, covariance)

        after_state = {key: value.detach().cpu().clone() for key, value in local_model.state_dict().items()}
        self._move_statistics(torch.device("cpu"))
        return PFedFDAClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=total_loss / max(total_batches, 1),
            delta=state_dict_delta(after_state, global_state_cpu),
            adaptive_means=self.adaptive_means.detach().cpu().clone(),
            adaptive_covariance=self.adaptive_covariance.detach().cpu().clone(),
            means_beta=float(self.means_beta),
            covariance_beta=float(self.covariance_beta),
        )

    def receive_global_statistics(self, means: torch.Tensor, covariance: torch.Tensor) -> None:
        self.global_means = means.detach().cpu().clone()
        self.global_covariance = covariance.detach().cpu().clone()

    def refresh_statistics(self, global_model: nn.Module) -> None:
        self._move_statistics(self.device)
        features, labels = self._compute_features(global_model)
        if features.numel() > 0:
            means, covariance, _ = self._compute_mle_statistics(features, labels)
            self._update_adaptive_statistics(features, labels, means, covariance)
        self._move_statistics(torch.device("cpu"))

    def export_personal_statistics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.adaptive_means.detach().cpu().clone(),
            self.adaptive_covariance.detach().cpu().clone(),
            self.priors.detach().cpu().clone(),
        )

    def _estimate_priors(self) -> torch.Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        loader = self.data_module.make_loader(self.dataset, shuffle=False)
        for batch in loader:
            _, targets = _unpack_batch(batch)
            counts += torch.bincount(targets.cpu(), minlength=self.num_classes).float()
        counts = counts + self.eps
        return counts / counts.sum().clamp_min(self.eps)

    def _compute_features(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = self.data_module.make_loader(self.dataset, shuffle=False)
        eval_model = build_baseline_model(self.config).to(self.device)
        eval_model.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, strict=True)
        eval_model.eval()
        feature_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                images, targets = _unpack_batch(batch)
                images = images.to(self.device)
                features, _ = _forward_with_features(eval_model, images)
                feature_batches.append(features.detach())
                label_batches.append(targets.to(self.device))
        if not feature_batches:
            return torch.empty(0, self.feature_dim, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
        return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)

    def _compute_mle_statistics(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(labels.detach().cpu(), minlength=self.num_classes).to(device=self.device, dtype=torch.float32)
        means: List[torch.Tensor] = []
        centered_batches: List[torch.Tensor] = []
        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            if int(mask.sum().item()) >= self.min_class_samples:
                class_mean = features[mask].mean(dim=0)
            else:
                class_mean = self.global_means[class_idx].to(self.device)
            means.append(class_mean)
            if mask.any():
                centered_batches.append(features[mask] - class_mean)
        means_tensor = torch.stack(means, dim=0)
        if centered_batches:
            centered = torch.cat(centered_batches, dim=0)
            scatter = centered.t().mm(centered)
            denominator = max(int(labels.numel()) - 1, 1)
            covariance = scatter / denominator
        else:
            covariance = self.global_covariance.to(self.device).clone()
        covariance = self._regularize_covariance(covariance)
        return means_tensor, covariance, counts

    def _regularize_covariance(self, covariance: torch.Tensor) -> torch.Tensor:
        covariance = 0.5 * (covariance + covariance.t())
        eye = torch.eye(self.feature_dim, device=covariance.device, dtype=covariance.dtype)
        return covariance + self.ridge * eye

    def _update_adaptive_statistics(self, features: torch.Tensor, labels: torch.Tensor, means: torch.Tensor, covariance: torch.Tensor) -> None:
        beta = self._solve_beta(features, labels, means, covariance)
        self.means_beta = beta
        self.covariance_beta = beta
        self.means = means.detach()
        self.covariance = covariance.detach()
        global_means = self.global_means.to(self.device)
        global_covariance = self.global_covariance.to(self.device)
        self.adaptive_means = beta * self.means + (1.0 - beta) * global_means
        self.adaptive_covariance = beta * self.covariance + (1.0 - beta) * global_covariance

    def _solve_beta(self, features: torch.Tensor, labels: torch.Tensor, local_means: torch.Tensor, local_covariance: torch.Tensor) -> float:
        if bool(getattr(self.config.federated, "pfedfda_local_beta", False)):
            return 1.0
        if not bool(getattr(self.config.federated, "pfedfda_beta_search", True)):
            return min(max(float(getattr(self.config.federated, "pfedfda_beta", 0.5)), 0.0), 1.0)

        candidates = max(int(getattr(self.config.federated, "pfedfda_beta_candidates", 11)), 2)
        best_beta = min(max(float(getattr(self.config.federated, "pfedfda_beta", 0.5)), 0.0), 1.0)
        best_loss = float("inf")
        global_means = self.global_means.to(self.device)
        global_covariance = self.global_covariance.to(self.device)
        for candidate in torch.linspace(0.0, 1.0, steps=candidates, device=self.device):
            beta = float(candidate.item())
            means = beta * local_means + (1.0 - beta) * global_means
            covariance = beta * local_covariance + (1.0 - beta) * global_covariance
            logits = self._lda_logits(features, means, covariance, self.priors.to(self.device))
            loss = float(F.cross_entropy(logits, labels).detach().cpu().item())
            if loss < best_loss:
                best_loss = loss
                best_beta = beta
        return best_beta

    def _lda_logits(self, features: torch.Tensor, means: torch.Tensor, covariance: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        covariance = self._regularize_covariance(covariance)
        eye = torch.eye(self.feature_dim, device=covariance.device, dtype=covariance.dtype)
        covariance = (1.0 - self.eps) * covariance + self.eps * torch.trace(covariance) / self.feature_dim * eye
        try:
            coefs = torch.linalg.solve(covariance, means.t()).t()
        except RuntimeError:
            coefs = torch.matmul(torch.linalg.pinv(covariance), means.t()).t()
        intercepts = -0.5 * torch.diag(means.mm(coefs.t())) + torch.log(priors.to(self.device).clamp_min(self.eps))
        return features.mm(coefs.t()) + intercepts

    def _set_lda_classifier(self, model: nn.Module, means: torch.Tensor, covariance: torch.Tensor, priors: torch.Tensor) -> None:
        classifier = _classifier(model)
        with torch.no_grad():
            logits_weight, logits_bias = self._lda_classifier_parameters(means.to(self.device), covariance.to(self.device), priors.to(self.device))
            classifier.weight.data.copy_(logits_weight.detach())
            classifier.bias.data.copy_(logits_bias.detach())

    def _lda_classifier_parameters(self, means: torch.Tensor, covariance: torch.Tensor, priors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        covariance = self._regularize_covariance(covariance)
        eye = torch.eye(self.feature_dim, device=covariance.device, dtype=covariance.dtype)
        covariance = (1.0 - self.eps) * covariance + self.eps * torch.trace(covariance) / self.feature_dim * eye
        try:
            coefs = torch.linalg.solve(covariance, means.t()).t()
        except RuntimeError:
            coefs = torch.matmul(torch.linalg.pinv(covariance), means.t()).t()
        intercepts = -0.5 * torch.diag(means.mm(coefs.t())) + torch.log(priors.to(self.device).clamp_min(self.eps))
        return coefs, intercepts

    def _set_classifier_requires_grad(self, model: nn.Module, requires_grad: bool) -> None:
        for param in _classifier(model).parameters():
            param.requires_grad_(requires_grad)

    def _move_statistics(self, device: torch.device) -> None:
        self.means = self.means.to(device)
        self.covariance = self.covariance.to(device)
        self.global_means = self.global_means.to(device)
        self.global_covariance = self.global_covariance.to(device)
        self.adaptive_means = self.adaptive_means.to(device)
        self.adaptive_covariance = self.adaptive_covariance.to(device)
        self.priors = self.priors.to(device)


class PFedFDAServer(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.algorithm_name = "pfedfda"
        self.global_model = build_baseline_model(config).to(self.device)
        if not hasattr(self.global_model, "feature_dim"):
            raise ValueError("pFedFDA requires a baseline model with feature_dim and forward_with_features.")
        self.feature_dim = int(getattr(self.global_model, "feature_dim"))
        self.global_means = torch.randn(config.model.num_classes, self.feature_dim)
        self.global_covariance = torch.eye(self.feature_dim)
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
            client_id: PFedFDAClient(client_id, dataset, config.federated.device, config, data_module)
            for client_id, dataset in client_datasets.items()
        }
        for client in self.clients.values():
            client.receive_global_statistics(self.global_means, self.global_covariance)
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
            updates: List[PFedFDAClientUpdate] = []
            for client_id in sampled_ids:
                self.clients[client_id].receive_global_statistics(self.global_means, self.global_covariance)
                loader = self.data_module.make_loader(self.clients[client_id].dataset, shuffle=True)
                updates.append(self.clients[client_id].train(self.global_model, loader))

            self._aggregate_model(updates)
            self._aggregate_feature_distributions(updates)
            for client in self.clients.values():
                client.receive_global_statistics(self.global_means, self.global_covariance)
            if bool(getattr(self.config.federated, "pfedfda_refresh_eval_stats", True)):
                self._refresh_client_statistics(self.clients.keys())

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
            client_train_profile = self._build_client_training_profile(self.model_flops, sampled_ids, self.client_datasets)
            resource_metrics = self._resource_metric_values(self.resource_profiles, client_train_profile, compute_profile)
            round_upload_bytes = float(
                sum(
                    self._estimate_tensor_payload_bytes(update.delta)
                    + self._estimate_tensor_payload_bytes(update.adaptive_means)
                    + self._estimate_tensor_payload_bytes(update.adaptive_covariance)
                    for update in updates
                )
            )
            upload_bytes_total += round_upload_bytes
            self._device_synchronize()
            round_train_time_seconds = time.perf_counter() - round_start_time
            beta_mean = sum(update.means_beta for update in updates) / max(len(updates), 1)
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
                extra_metrics={"pfedfda_beta_mean": beta_mean},
                **resource_metrics,
            )
            LOGGER.info(
                "pfedfda round %d | loss=%.4f | personalized_acc=%.4f | weighted_acc=%.4f | hard_recall=%.4f | beta=%.4f | savings=%.4f%s",
                round_idx,
                avg_loss,
                macro["accuracy"],
                aggregate["accuracy"],
                aggregate["hard_recall"],
                beta_mean,
                compute_profile["savings_ratio"],
                self._format_resource_metrics_for_log(round_metrics),
            )
            self._log_round_metrics(self.algorithm_name, round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        if bool(getattr(self.config.federated, "pfedfda_refresh_eval_stats", True)):
            self._refresh_client_statistics(self.clients.keys())
        self._cached_eval_client_id = None
        personalized_eval = self._evaluate_predictor_on_client_tests(self._predict_with_personalized_model, prefix="pfedfda_final")
        global_eval = self._evaluate_predictor_on_client_tests(self._predict_with_global_model, prefix="pfedfda_global_final")
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
        general_reference_profile = self._build_compute_profile(self.model_flops, self.general_flops, 1.0, mode="general_only")
        final_history = self.last_history[-1] if self.last_history else RoundMetrics(0, 0.0, 0.0, 0.0, 0.0)
        final_client_train = {
            "avg_flops_per_client": final_history.client_train_flops,
            "total_flops": final_history.client_train_flops_total,
            "num_clients": self.config.federated.clients_per_round,
            "num_samples": 0,
        }
        final_resource_metrics = self._resource_metric_values(self.resource_profiles, final_client_train, compute_profile)
        average_round_train_time_seconds = (
            sum(item.round_train_time_seconds for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_train_time_seconds = sum(item.round_train_time_seconds for item in self.last_history) if self.last_history else 0.0
        average_upload_bytes_per_round = (
            sum(item.upload_bytes_per_round for item in self.last_history) / max(len(self.last_history), 1)
            if self.last_history
            else 0.0
        )
        total_upload_bytes = self.last_history[-1].upload_bytes_total if self.last_history else 0.0
        beta_mean = (
            sum(float(client.means_beta) for client in self.clients.values()) / max(len(self.clients), 1)
            if self.clients
            else 0.0
        )
        return {
            "algorithm": "pfedfda",
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
                "pfedfda_beta_mean": beta_mean,
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

    def _aggregate_model(self, updates: Sequence[PFedFDAClientUpdate]) -> None:
        aggregated_delta = average_weighted_deltas((update.num_samples, update.delta) for update in updates)
        updated_state = self.global_model.state_dict()
        for key, delta in aggregated_delta.items():
            if key in updated_state:
                updated_state[key] = updated_state[key] + delta.to(updated_state[key].device)
        self.global_model.load_state_dict(updated_state)

    def _aggregate_feature_distributions(self, updates: Sequence[PFedFDAClientUpdate]) -> None:
        total_samples = sum(update.num_samples for update in updates)
        if total_samples <= 0:
            return
        means = torch.zeros_like(self.global_means)
        covariance = torch.zeros_like(self.global_covariance)
        for update in updates:
            weight = update.num_samples / total_samples
            means += update.adaptive_means * weight
            covariance += update.adaptive_covariance * weight
        self.global_means = means.detach().cpu()
        self.global_covariance = covariance.detach().cpu()

    def _refresh_client_statistics(self, client_ids) -> None:
        for client_id in client_ids:
            self.clients[client_id].receive_global_statistics(self.global_means, self.global_covariance)
            self.clients[client_id].refresh_statistics(self.global_model)
        self._cached_eval_client_id = None

    def _predict_with_global_model(self, client_id, images, indices):
        logits = self.global_model(images)
        return logits.argmax(dim=1), 0

    def _predict_with_personalized_model(self, client_id, images, indices):
        model = self._load_personalized_eval_model(client_id)
        logits = model(images)
        return logits.argmax(dim=1), 0

    def _load_personalized_eval_model(self, client_id: str) -> nn.Module:
        if self._cached_eval_client_id != client_id:
            means, covariance, priors = self.clients[client_id].export_personal_statistics()
            self._personalized_eval_model.load_state_dict(
                {key: value.detach().cpu().clone() for key, value in self.global_model.state_dict().items()},
                strict=True,
            )
            self._personalized_eval_model.to(self.device)
            self.clients[client_id]._set_lda_classifier(self._personalized_eval_model, means, covariance, priors)
            self._personalized_eval_model.eval()
            self._cached_eval_client_id = client_id
        return self._personalized_eval_model
