"""
FedAsym: pure Gemini-dialogue implementation.

Design constraints:
  - no inheritance from other algorithm files,
  - client-side public distillation follows the FedEcho-style KL/CE mixing
    described in the Gemini dialogue,
  - server-side public distillation follows the DKDR bidirectional KL idea,
  - reliability comes only from an explicit client-side error predictor R_k,
  - routing uses only the predicted expert error probability.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from fedegs.federated.common import (
    BaseFederatedClient,
    BaseFederatedServer,
    LOGGER,
    RoundMetrics,
)
from fedegs.models import SmallCNN, build_teacher_model, estimate_model_flops


@dataclass
class FedAsymClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_logits: torch.Tensor
    predictor_state: Dict[str, torch.Tensor]
    error_threshold: float


class GeneralModel(nn.Module):
    """General model wrapper built on the teacher backbone."""

    def __init__(self, num_classes: int, pretrained_imagenet: bool = False) -> None:
        super().__init__()
        backbone = build_teacher_model(
            num_classes=num_classes,
            pretrained_imagenet=pretrained_imagenet,
        )
        self.num_classes = num_classes
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier.load_state_dict(backbone.fc.state_dict())
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))


def _stable_client_seed(base_seed: int, client_id: str) -> int:
    return int(base_seed) + sum((index + 1) * ord(char) for index, char in enumerate(client_id))


def _split_dataset_for_calibration(
    dataset: Dataset,
    ratio: float,
    min_samples: int,
    max_samples: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    total_samples = len(dataset)
    if total_samples <= 1 or ratio <= 0.0:
        return dataset, dataset

    calibration_samples = max(int(round(total_samples * ratio)), 1)
    if total_samples >= max(2 * max(min_samples, 1), 8):
        calibration_samples = max(calibration_samples, max(min_samples, 1))
    if max_samples > 0:
        calibration_samples = min(calibration_samples, max_samples)
    calibration_samples = min(max(calibration_samples, 1), total_samples - 1)
    if calibration_samples <= 0 or calibration_samples >= total_samples:
        return dataset, dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total_samples, generator=generator).tolist()
    calibration_indices = sorted(permutation[:calibration_samples])
    train_indices = sorted(permutation[calibration_samples:])
    if not calibration_indices or not train_indices:
        return dataset, dataset
    return Subset(dataset, train_indices), Subset(dataset, calibration_indices)


def _split_dataset_for_router_validation(
    dataset: Dataset,
    ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    total_samples = len(dataset)
    ratio = min(max(float(ratio), 0.0), 1.0)
    if total_samples <= 3 or ratio <= 0.0:
        return dataset, dataset

    validation_samples = int(round(total_samples * ratio))
    validation_samples = min(max(validation_samples, 1), total_samples - 1)
    if validation_samples <= 0 or validation_samples >= total_samples:
        return dataset, dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total_samples, generator=generator).tolist()
    validation_indices = sorted(permutation[:validation_samples])
    train_indices = sorted(permutation[validation_samples:])
    if not validation_indices or not train_indices:
        return dataset, dataset
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def _clone_tensor_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _predictor_feature_dim(num_classes: int) -> int:
    return 8 + int(num_classes)


def _predictor_features_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
    confidence = topk.values[:, 0]
    top2_prob = topk.values[:, 1] if topk.values.size(1) > 1 else torch.zeros_like(confidence)
    if topk.values.size(1) > 1:
        margin = topk.values[:, 0] - topk.values[:, 1]
    else:
        margin = torch.ones_like(confidence)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / max(math.log(float(probs.size(1))), 1e-8)
    max_logit = logits.max(dim=1).values
    top_logits = torch.topk(logits, k=min(2, logits.size(1)), dim=1).values
    if top_logits.size(1) > 1:
        logit_margin = top_logits[:, 0] - top_logits[:, 1]
    else:
        logit_margin = torch.ones_like(max_logit)
    energy = torch.logsumexp(logits, dim=1)
    prob_variance = probs.var(dim=1, unbiased=False)
    predicted_class = logits.argmax(dim=1)
    predicted_class_onehot = F.one_hot(predicted_class, num_classes=logits.size(1)).to(dtype=logits.dtype)
    dense_features = torch.stack(
        [
            confidence,
            entropy,
            margin,
            max_logit,
            top2_prob,
            logit_margin,
            energy,
            prob_variance,
        ],
        dim=1,
    )
    return torch.cat([dense_features, predicted_class_onehot], dim=1)


def _default_predictor_state(
    feature_dim: int,
    error_rate: float = 0.5,
    hidden_dim: int = 0,
) -> Dict[str, torch.Tensor]:
    p = min(max(float(error_rate), 1e-4), 1.0 - 1e-4)
    if hidden_dim > 0:
        return {
            "mean": torch.zeros(feature_dim, dtype=torch.float32),
            "std": torch.ones(feature_dim, dtype=torch.float32),
            "hidden_weight": torch.zeros(hidden_dim, feature_dim, dtype=torch.float32),
            "hidden_bias": torch.zeros(hidden_dim, dtype=torch.float32),
            "output_weight": torch.zeros(1, hidden_dim, dtype=torch.float32),
            "output_bias": torch.tensor([math.log(p / (1.0 - p))], dtype=torch.float32),
        }
    return {
        "mean": torch.zeros(feature_dim, dtype=torch.float32),
        "std": torch.ones(feature_dim, dtype=torch.float32),
        "weight": torch.zeros(1, feature_dim, dtype=torch.float32),
        "bias": torch.tensor([math.log(p / (1.0 - p))], dtype=torch.float32),
    }


def _align_features_to_predictor_state(
    predictor_state: Dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    expected_dim = int(predictor_state["mean"].numel())
    current_dim = int(features.size(1)) if features.ndim == 2 else 0
    if current_dim == expected_dim:
        return features
    if current_dim > expected_dim:
        return features[:, :expected_dim]
    padding = torch.zeros(
        features.size(0),
        expected_dim - current_dim,
        device=features.device,
        dtype=features.dtype,
    )
    return torch.cat([features, padding], dim=1)


def _predict_error_probabilities(
    predictor_state: Dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    features = _align_features_to_predictor_state(predictor_state, features)
    mean = predictor_state["mean"].to(features.device, dtype=features.dtype)
    std = predictor_state["std"].to(features.device, dtype=features.dtype).clamp_min(1e-6)
    normalized = (features - mean) / std
    if "hidden_weight" in predictor_state:
        hidden_weight = predictor_state["hidden_weight"].to(features.device, dtype=features.dtype)
        hidden_bias = predictor_state["hidden_bias"].to(features.device, dtype=features.dtype)
        output_weight = predictor_state["output_weight"].to(features.device, dtype=features.dtype)
        output_bias = predictor_state["output_bias"].to(features.device, dtype=features.dtype)
        hidden = F.relu(F.linear(normalized, hidden_weight, hidden_bias))
        logits = F.linear(hidden, output_weight, output_bias)
    else:
        weight = predictor_state["weight"].to(features.device, dtype=features.dtype)
        bias = predictor_state["bias"].to(features.device, dtype=features.dtype)
        logits = F.linear(normalized, weight, bias)
    return torch.sigmoid(logits).squeeze(1)


def _average_precision_score(labels: List[int], scores: List[float]) -> float:
    if not labels or not scores or len(labels) != len(scores):
        return 0.0
    positive_total = sum(1 for label in labels if int(label) == 1)
    if positive_total <= 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if int(label) == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_total


def _high_confidence_guard(config) -> float:
    return float(getattr(config.inference, "error_predictor_high_confidence_guard", 1.0))


def _build_error_fallback_mask(
    error_probs: torch.Tensor,
    features: torch.Tensor,
    threshold: float,
    config,
) -> torch.Tensor:
    fallback_mask = error_probs >= float(threshold)
    guard = _high_confidence_guard(config)
    if guard < 1.0:
        fallback_mask = fallback_mask & (features[:, 0] < guard)
    return fallback_mask


def _wilson_precision_lower_bound(true_positive: int, predicted_positive: int, z: float = 1.96) -> float:
    if predicted_positive <= 0:
        return 0.0
    n = float(predicted_positive)
    p_hat = float(true_positive) / n
    z2 = z * z
    denominator = 1.0 + (z2 / n)
    center = p_hat + (z2 / (2.0 * n))
    spread = z * math.sqrt(((p_hat * (1.0 - p_hat)) + (z2 / (4.0 * n))) / n)
    return max((center - spread) / denominator, 0.0)


def _select_precision_constrained_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    config,
) -> float:
    base_threshold = float(getattr(config.inference, "error_predictor_threshold", 0.5))
    mode = str(getattr(config.inference, "error_predictor_threshold_mode", "fixed")).lower()
    if mode not in {"precision", "precision_constrained"}:
        return base_threshold

    target_precision = min(max(float(getattr(config.inference, "error_predictor_target_precision", 0.8)), 0.0), 1.0)
    min_positive = max(int(getattr(config.inference, "error_predictor_min_predicted_positive", 3)), 1)
    disable_on_fail = bool(getattr(config.inference, "error_predictor_disable_on_precision_fail", True))
    min_threshold = float(getattr(config.inference, "routing_error_min_threshold", 0.0))
    max_threshold = float(getattr(config.inference, "routing_error_max_threshold", 1.0))
    use_wilson = mode in {"precision_wilson", "wilson"} or bool(
        getattr(config.inference, "error_predictor_use_wilson_lower_bound", False)
    )
    wilson_z = float(getattr(config.inference, "error_predictor_wilson_z", 1.96))

    scores = scores.detach().cpu().to(torch.float32).view(-1)
    labels = labels.detach().cpu().to(torch.bool).view(-1)
    features = features.detach().cpu().to(torch.float32)
    if scores.numel() == 0 or labels.numel() == 0 or scores.numel() != labels.numel():
        return 1.01 if disable_on_fail else base_threshold

    eligible = torch.ones_like(labels, dtype=torch.bool)
    guard = _high_confidence_guard(config)
    if guard < 1.0 and features.ndim == 2 and features.size(0) == scores.numel():
        eligible = features[:, 0] < guard

    candidate_scores = scores[eligible]
    if candidate_scores.numel() == 0:
        return 1.01 if disable_on_fail else base_threshold

    best_threshold: Optional[float] = None
    best_invoked = -1
    candidates = torch.unique(candidate_scores.clamp(min_threshold, max_threshold)).tolist()
    candidates.extend([base_threshold, min_threshold, max_threshold])
    for threshold in sorted({float(item) for item in candidates}):
        predicted = (scores >= threshold) & eligible
        invoked = int(predicted.sum().item())
        if invoked < min_positive:
            continue
        true_positive = int((predicted & labels).sum().item())
        precision = true_positive / max(invoked, 1)
        precision_score = _wilson_precision_lower_bound(true_positive, invoked, wilson_z) if use_wilson else precision
        if precision_score >= target_precision and invoked > best_invoked:
            best_invoked = invoked
            best_threshold = float(threshold)

    if best_threshold is not None:
        return best_threshold
    return 1.01 if disable_on_fail else base_threshold




def _fit_predictor_state(
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int = 0,
    dropout: float = 0.0,
) -> Dict[str, torch.Tensor]:
    feature_dim = int(features.size(1)) if features.ndim == 2 else 4
    hidden_dim = max(int(hidden_dim), 0)
    if features.numel() == 0 or labels.numel() == 0:
        return _default_predictor_state(feature_dim, hidden_dim=hidden_dim)

    labels = labels.float().view(-1)
    features = features.float()
    positive_rate = float(labels.mean().item())
    if positive_rate <= 1e-6 or positive_rate >= 1.0 - 1e-6:
        return _default_predictor_state(feature_dim, error_rate=positive_rate, hidden_dim=hidden_dim)

    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std

    if hidden_dim > 0:
        model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=min(max(float(dropout), 0.0), 0.9)),
            nn.Linear(hidden_dim, 1),
        ).to(device)
    else:
        model = nn.Linear(feature_dim, 1).to(device)
    pos_count = labels.sum().item()
    neg_count = labels.numel() - pos_count
    pos_weight = torch.tensor([max(neg_count / max(pos_count, 1.0), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(max(normalized.size(0), 1), 128)
    x_all = normalized.to(device)
    y_all = labels.to(device).unsqueeze(1)
    for _ in range(max(epochs, 1)):
        permutation = torch.randperm(x_all.size(0), device=device)
        for start in range(0, x_all.size(0), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_all[indices])
            loss = criterion(logits, y_all[indices])
            loss.backward()
            optimizer.step()

    state = {
        "mean": mean.detach().cpu().clone(),
        "std": std.detach().cpu().clone(),
    }
    if hidden_dim > 0:
        first_layer = model[0]
        output_layer = model[3]
        state.update(
            {
                "hidden_weight": first_layer.weight.detach().cpu().clone(),
                "hidden_bias": first_layer.bias.detach().cpu().clone(),
                "output_weight": output_layer.weight.detach().cpu().clone(),
                "output_bias": output_layer.bias.detach().cpu().clone(),
            }
        )
    else:
        state.update(
            {
                "weight": model.weight.detach().cpu().clone(),
                "bias": model.bias.detach().cpu().clone(),
            }
        )
    return state


class FedAsymClient(BaseFederatedClient):
    def __init__(
        self,
        client_id: str,
        train_dataset: Dataset,
        calibration_dataset: Dataset,
        threshold_dataset: Dataset,
        num_classes: int,
        device: str,
        config,
        data_module,
    ) -> None:
        super().__init__(client_id, train_dataset, device)
        self.config = config
        self.data_module = data_module
        self.calibration_dataset = calibration_dataset
        self.threshold_dataset = threshold_dataset
        self.num_classes = num_classes
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.predictor_state = _default_predictor_state(
            _predictor_feature_dim(num_classes),
            hidden_dim=int(getattr(config.federated, "risk_predictor_hidden_dim", 32)),
        )
        self.error_threshold = float(getattr(config.inference, "error_predictor_threshold", 0.5))

    def train_local(
        self,
        general_model: nn.Module,
        public_batches: Sequence,
    ) -> FedAsymClientUpdate:
        distill_loss = self._distill_expert_on_public(public_batches, general_model)
        private_loss = self._train_expert_on_private()
        self.predictor_state = self._fit_risk_predictor()
        public_logits = self._collect_public_logits(public_batches)
        loss_terms = [value for value in (distill_loss, private_loss) if value > 0.0]
        mean_loss = sum(loss_terms) / max(len(loss_terms), 1)
        return FedAsymClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=mean_loss,
            public_logits=public_logits,
            predictor_state=_clone_tensor_dict(self.predictor_state),
            error_threshold=float(self.error_threshold),
        )

    def _distill_expert_on_public(
        self,
        public_batches: Sequence,
        general_model: nn.Module,
    ) -> float:
        distill_epochs = max(int(getattr(self.config.federated, "public_distill_epochs", 1)), 0)
        if distill_epochs <= 0 or not public_batches:
            return 0.0

        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))
        alpha_min = min(max(float(getattr(self.config.federated, "uncertainty_alpha_min", 0.2)), 0.0), 1.0)
        alpha_max = min(max(float(getattr(self.config.federated, "uncertainty_alpha_max", 0.8)), alpha_min), 1.0)
        log_c = math.log(float(max(self.num_classes, 2)))

        self.expert_model.train()
        general_model.eval()

        total_loss = 0.0
        total_batches = 0
        for _ in range(distill_epochs):
            for batch in public_batches:
                images = batch[0].to(self.device)

                with torch.no_grad():
                    teacher_logits = general_model(images)
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                    teacher_hard = teacher_logits.argmax(dim=1)
                    teacher_entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=1)
                    normalized_entropy = float((teacher_entropy / max(log_c, 1e-8)).mean().item())
                    alpha = (normalized_entropy * alpha_max) + ((1.0 - normalized_entropy) * alpha_min)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.expert_model(images)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature ** 2)
                teacher_ce = F.cross_entropy(student_logits, teacher_hard)
                loss = (alpha * kd_loss) + ((1.0 - alpha) * teacher_ce)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _train_expert_on_private(self) -> float:
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        return self._optimize_model(
            model=self.expert_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )

    def _fit_risk_predictor(self) -> Dict[str, torch.Tensor]:
        if self.calibration_dataset is None or len(self.calibration_dataset) == 0:
            return _default_predictor_state(
                _predictor_feature_dim(self.num_classes),
                hidden_dim=int(getattr(self.config.federated, "risk_predictor_hidden_dim", 32)),
            )

        features, labels = self._collect_predictor_features(self.calibration_dataset)
        if features.numel() == 0 or labels.numel() == 0:
            return _default_predictor_state(
                _predictor_feature_dim(self.num_classes),
                hidden_dim=int(getattr(self.config.federated, "risk_predictor_hidden_dim", 32)),
            )

        predictor_state = _fit_predictor_state(
            features=features,
            labels=labels,
            device=self.device,
            epochs=int(getattr(self.config.federated, "risk_predictor_epochs", 40)),
            lr=float(getattr(self.config.federated, "risk_predictor_lr", 0.05)),
            weight_decay=float(getattr(self.config.federated, "risk_predictor_weight_decay", 0.0)),
            hidden_dim=int(getattr(self.config.federated, "risk_predictor_hidden_dim", 32)),
            dropout=float(getattr(self.config.federated, "risk_predictor_dropout", 0.1)),
        )
        threshold_features, threshold_labels = self._collect_predictor_features(self.threshold_dataset)
        if threshold_features.numel() == 0 or threshold_labels.numel() == 0:
            threshold_features, threshold_labels = features, labels
        error_probs = _predict_error_probabilities(predictor_state, threshold_features)
        self.error_threshold = _select_precision_constrained_threshold(
            scores=error_probs,
            labels=threshold_labels,
            features=threshold_features,
            config=self.config,
        )
        return predictor_state

    def _collect_predictor_features(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        if dataset is None or len(dataset) == 0:
            return (
                torch.empty((0, _predictor_feature_dim(self.num_classes)), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )

        loader = self.data_module.make_loader(dataset, shuffle=False)
        feature_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []

        self.expert_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                logits = self.expert_model(images)
                features = _predictor_features_from_logits(logits)
                predictions = logits.argmax(dim=1)
                label_batches.append(predictions.ne(targets_device).float().cpu())
                feature_batches.append(features.cpu())

        if not feature_batches:
            return (
                torch.empty((0, _predictor_feature_dim(self.num_classes)), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
        return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)

    def _collect_public_logits(self, public_batches: Sequence) -> torch.Tensor:
        logits_batches: List[torch.Tensor] = []
        self.expert_model.eval()
        with torch.no_grad():
            for batch in public_batches:
                images = batch[0].to(self.device)
                logits_batches.append(self.expert_model(images).detach().cpu())
        return torch.cat(logits_batches, dim=0) if logits_batches else torch.empty((0, self.num_classes), dtype=torch.float32)


class FedAsymServer(BaseFederatedServer):
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
        if public_dataset is None or len(public_dataset) == 0:
            raise ValueError("FedAsym requires a non-empty public_dataset.")

        BaseFederatedServer.__init__(
            self,
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )
        self.algorithm_name = "fedasym"
        self.general_model = GeneralModel(
            num_classes=config.model.num_classes,
            pretrained_imagenet=bool(getattr(config.federated, "general_pretrain_imagenet_init", False)),
        ).to(self.device)
        self.reference_expert = SmallCNN(
            num_classes=config.model.num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
        self.resource_profiles = self._build_dual_model_resource_profiles(
            self.reference_expert,
            self.general_model,
            self.expert_flops,
            self.general_flops,
        )

        calibration_ratio = max(float(getattr(config.federated, "calibration_ratio", 0.1)), 0.0)
        calibration_min = max(int(getattr(config.federated, "calibration_min_samples", 16)), 0)
        calibration_max = max(int(getattr(config.federated, "calibration_max_samples", 0)), 0)
        router_validation_ratio = min(max(float(getattr(config.federated, "router_validation_ratio", 0.5)), 0.0), 1.0)
        self.client_training_datasets: Dict[str, Dataset] = {}
        self.client_router_train_datasets: Dict[str, Dataset] = {}
        self.client_calibration_datasets: Dict[str, Dataset] = {}
        for client_id, dataset in client_datasets.items():
            train_dataset, calibration_dataset = _split_dataset_for_calibration(
                dataset=dataset,
                ratio=calibration_ratio,
                min_samples=calibration_min,
                max_samples=calibration_max,
                seed=_stable_client_seed(config.federated.seed, client_id),
            )
            router_train_dataset, router_validation_dataset = _split_dataset_for_router_validation(
                dataset=calibration_dataset,
                ratio=router_validation_ratio,
                seed=_stable_client_seed(config.federated.seed + 7919, client_id),
            )
            self.client_training_datasets[client_id] = train_dataset
            self.client_router_train_datasets[client_id] = router_train_dataset
            self.client_calibration_datasets[client_id] = router_validation_dataset

        self.clients: Dict[str, FedAsymClient] = {
            client_id: FedAsymClient(
                client_id=client_id,
                train_dataset=self.client_training_datasets[client_id],
                calibration_dataset=self.client_router_train_datasets[client_id],
                threshold_dataset=self.client_calibration_datasets[client_id],
                num_classes=config.model.num_classes,
                device=config.federated.device,
                config=config,
                data_module=data_module,
            )
            for client_id in client_datasets.keys()
        }

        self.public_loader = self.data_module.make_loader(public_dataset, shuffle=False)
        self.public_batches = list(self.public_loader)
        self.public_images_cpu = torch.cat([batch[0] for batch in self.public_batches], dim=0)

        self.client_predictor_states: Dict[str, Dict[str, torch.Tensor]] = {
            client_id: _clone_tensor_dict(client.predictor_state) for client_id, client in self.clients.items()
        }
        self.client_error_thresholds: Dict[str, float] = {
            client_id: float(getattr(config.inference, "error_predictor_threshold", 0.5))
            for client_id in client_datasets
        }
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.loaded_checkpoint_path: Optional[str] = None
        self.current_round = 0
        self.latest_distill_stats: Dict[str, float] = {}

    def _predict_public_error_probabilities(
        self,
        predictor_state: Dict[str, torch.Tensor],
        public_logits: torch.Tensor,
    ) -> torch.Tensor:
        features = _predictor_features_from_logits(public_logits)
        return _predict_error_probabilities(predictor_state, features)

    def _aggregate_public_knowledge(self, updates: List[FedAsymClientUpdate]) -> Dict[str, torch.Tensor | float]:
        if not updates:
            return {
                "fused_logits": torch.empty((0, self.config.model.num_classes), dtype=torch.float32),
                "sample_reliability": torch.empty((0,), dtype=torch.float32),
                "teacher_reliability_mean": 0.0,
                "teacher_error_mean": 0.0,
            }

        all_logits = []
        all_reliabilities = []
        for update in updates:
            error_probs = self._predict_public_error_probabilities(update.predictor_state, update.public_logits)
            reliability = (1.0 - error_probs).clamp(1e-4, 1.0)
            all_logits.append(update.public_logits)
            all_reliabilities.append(reliability.cpu())

        stacked_logits = torch.stack(all_logits, dim=0)
        stacked_reliability = torch.stack(all_reliabilities, dim=0)
        normalized_weights = stacked_reliability / stacked_reliability.sum(dim=0, keepdim=True).clamp_min(1e-8)
        fused_logits = (stacked_logits * normalized_weights.unsqueeze(-1)).sum(dim=0)
        sample_reliability = stacked_reliability.mean(dim=0).clamp(0.0, 1.0)
        return {
            "fused_logits": fused_logits,
            "sample_reliability": sample_reliability,
            "teacher_reliability_mean": float(stacked_reliability.mean().item()),
            "teacher_error_mean": float((1.0 - stacked_reliability).mean().item()),
        }

    def _distill_general_model(
        self,
        public_knowledge: Dict[str, torch.Tensor | float],
        round_idx: int,
    ) -> Dict[str, float]:
        fused_logits_all = public_knowledge["fused_logits"]
        if not torch.is_tensor(fused_logits_all) or fused_logits_all.numel() == 0:
            self.latest_distill_stats = {
                "total_loss": 0.0,
                "forward_kl": 0.0,
                "reverse_kl": 0.0,
                "gamma_forward": 0.5,
                "gamma_reverse": 0.5,
                "teacher_reliability_mean": float(public_knowledge.get("teacher_reliability_mean", 0.0)),
                "teacher_error_mean": float(public_knowledge.get("teacher_error_mean", 0.0)),
            }
            return dict(self.latest_distill_stats)

        temperature = float(self.config.federated.distill_temperature)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = self.config.dataset.batch_size
        dkdr_center = min(max(float(getattr(self.config.federated, "dkdr_reliability_center", 0.5)), 0.0), 1.0)
        dkdr_mu = max(float(getattr(self.config.federated, "dkdr_mu", 0.5)), 1e-6)
        sample_reliability_all = public_knowledge.get("sample_reliability")
        if not torch.is_tensor(sample_reliability_all) or sample_reliability_all.numel() != fused_logits_all.size(0):
            fallback_reliability = float(public_knowledge.get("teacher_reliability_mean", 0.5))
            sample_reliability_all = torch.full(
                (fused_logits_all.size(0),),
                min(max(fallback_reliability, 0.0), 1.0),
                dtype=torch.float32,
            )
        optimizer = torch.optim.Adam(self.general_model.parameters(), lr=float(self.config.federated.distill_lr))
        total_steps = distill_epochs * ((self.public_images_cpu.size(0) + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        self.general_model.train()
        total_loss = 0.0
        total_forward_kl = 0.0
        total_reverse_kl = 0.0
        total_gamma_forward = 0.0
        total_gamma_reverse = 0.0
        total_batches = 0

        for _ in range(distill_epochs):
            permutation = torch.randperm(self.public_images_cpu.size(0))
            for start in range(0, self.public_images_cpu.size(0), batch_size):
                indices = permutation[start:start + batch_size]
                images = self.public_images_cpu[indices].to(self.device)
                teacher_logits = fused_logits_all[indices].to(self.device)
                sample_reliability = sample_reliability_all[indices].to(self.device).clamp(0.0, 1.0)
                gamma_reverse = torch.sigmoid((sample_reliability - dkdr_center) / dkdr_mu)
                gamma_forward = 1.0 - gamma_reverse

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                student_probs = F.softmax(student_logits / temperature, dim=1)

                forward_kl_per_sample = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                reverse_kl_per_sample = F.kl_div(
                    teacher_log_probs,
                    student_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                loss = ((gamma_forward * forward_kl_per_sample) + (gamma_reverse * reverse_kl_per_sample)).mean()

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.detach().cpu().item())
                total_forward_kl += float(forward_kl_per_sample.detach().mean().cpu().item())
                total_reverse_kl += float(reverse_kl_per_sample.detach().mean().cpu().item())
                total_gamma_forward += float(gamma_forward.detach().mean().cpu().item())
                total_gamma_reverse += float(gamma_reverse.detach().mean().cpu().item())
                total_batches += 1

        divisor = max(total_batches, 1)
        self.latest_distill_stats = {
            "total_loss": total_loss / divisor,
            "forward_kl": total_forward_kl / divisor,
            "reverse_kl": total_reverse_kl / divisor,
            "gamma_forward": total_gamma_forward / divisor,
            "gamma_reverse": total_gamma_reverse / divisor,
            "teacher_reliability_mean": float(public_knowledge.get("teacher_reliability_mean", 0.0)),
            "teacher_error_mean": float(public_knowledge.get("teacher_error_mean", 0.0)),
        }
        return dict(self.latest_distill_stats)

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        return self.general_model(images).argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        self.clients[client_id].expert_model.eval()
        with torch.no_grad():
            logits = self.clients[client_id].expert_model(images)
        return logits.argmax(dim=1), 0

    def _predict_routed_impl(self, client_id, images, indices, full_metadata: bool = False):
        expert_model = self.clients[client_id].expert_model
        predictor_state = self.client_predictor_states[client_id]
        threshold = float(self.client_error_thresholds.get(client_id, getattr(self.config.inference, "error_predictor_threshold", 0.5)))

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            expert_logits = expert_model(images)
            expert_features = _predictor_features_from_logits(expert_logits)
            error_probs = _predict_error_probabilities(predictor_state, expert_features)
            expert_probs = torch.softmax(expert_logits, dim=1)
            expert_prediction = expert_logits.argmax(dim=1)
            fallback_mask = _build_error_fallback_mask(error_probs, expert_features, threshold, self.config)

            predictions = expert_prediction.clone()
            invoked_general = int(fallback_mask.sum().item())
            route_types = ["expert"] * images.size(0)
            general_prediction = torch.full_like(expert_prediction, -1)
            if full_metadata:
                general_logits = self.general_model(images)
                general_prediction = general_logits.argmax(dim=1)
                predictions[fallback_mask] = general_prediction[fallback_mask]
            elif fallback_mask.any():
                general_logits = self.general_model(images[fallback_mask])
                general_prediction[fallback_mask] = general_logits.argmax(dim=1)
                predictions[fallback_mask] = general_prediction[fallback_mask]
            if fallback_mask.any():
                for sample_index in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_index] = "general"

            metadata = {
                "route_type": route_types,
                "expert_confidence": expert_probs.max(dim=1).values.detach().cpu().tolist(),
                "expert_margin": expert_features[:, 2].detach().cpu().tolist(),
                "expert_entropy": expert_features[:, 1].detach().cpu().tolist(),
                "error_prob": error_probs.detach().cpu().tolist(),
                "error_threshold": [threshold] * images.size(0),
                "expert_pred": expert_prediction.detach().cpu().tolist(),
            }
            if full_metadata:
                metadata["general_pred"] = general_prediction.detach().cpu().tolist()
            return predictions, invoked_general, metadata

    def _predict_routed(self, client_id, images, indices):
        return self._predict_routed_impl(client_id, images, indices, full_metadata=False)

    def _predict_routed_full_metadata(self, client_id, images, indices):
        return self._predict_routed_impl(client_id, images, indices, full_metadata=True)

    def _evaluate_route_effectiveness_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._evaluate_route_effectiveness_metrics_from_predictors(
            expert_eval=expert_eval,
            general_eval=general_eval,
            routed_eval=routed_eval,
            predictor_expert=self._predict_expert_only,
            predictor_general=self._predict_general_only,
            predictor_routed=self._predict_routed,
        )

    def _evaluate_error_predictor_metrics(self) -> Dict[str, float]:
        error_labels: List[int] = []
        error_scores: List[float] = []
        predicted_positive = 0
        true_positive = 0
        actual_positive = 0
        false_positive = 0
        actual_negative = 0

        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                loader = self.data_module.make_loader(dataset, shuffle=False)
                predictor_state = self.client_predictor_states[client_id]
                threshold = float(
                    self.client_error_thresholds.get(
                        client_id,
                        getattr(self.config.inference, "error_predictor_threshold", 0.5),
                    )
                )
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    expert_logits = expert_model(images)
                    features = _predictor_features_from_logits(expert_logits)
                    error_probs = _predict_error_probabilities(predictor_state, features)
                    expert_predictions = expert_logits.argmax(dim=1)
                    batch_error_labels = expert_predictions.ne(targets_device)
                    batch_predicted_positive = _build_error_fallback_mask(error_probs, features, threshold, self.config)

                    predicted_positive += int(batch_predicted_positive.sum().item())
                    true_positive += int((batch_predicted_positive & batch_error_labels).sum().item())
                    actual_positive += int(batch_error_labels.sum().item())
                    false_positive += int((batch_predicted_positive & ~batch_error_labels).sum().item())
                    actual_negative += int((~batch_error_labels).sum().item())
                    error_labels.extend(batch_error_labels.detach().cpu().to(torch.int64).tolist())
                    error_scores.extend(error_probs.detach().cpu().to(torch.float32).tolist())

        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(actual_positive, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        auprc = _average_precision_score(error_labels, error_scores)
        false_positive_rate = false_positive / max(actual_negative, 1)
        predicted_positive_rate = predicted_positive / max(len(error_labels), 1)
        return {
            "error_predictor_precision": precision,
            "error_predictor_recall": recall,
            "error_predictor_f1": f1,
            "error_predictor_auprc": auprc,
            "error_predictor_false_positive_rate": false_positive_rate,
            "error_predictor_predicted_positive_rate": predicted_positive_rate,
        }

    def _build_round_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        metrics = self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)
        predictor_metrics = self._evaluate_error_predictor_metrics()
        threshold_mean = (
            sum(self.client_error_thresholds.values()) / max(len(self.client_error_thresholds), 1)
            if self.client_error_thresholds
            else 0.0
        )
        veto_rate = (
            sum(1 for threshold in self.client_error_thresholds.values() if float(threshold) > 1.0)
            / max(len(self.client_error_thresholds), 1)
            if self.client_error_thresholds
            else 0.0
        )
        metrics.update(
            {
                **predictor_metrics,
                "distill_loss": self.latest_distill_stats.get("total_loss", 0.0),
                "dkdr_forward_kl": self.latest_distill_stats.get("forward_kl", 0.0),
                "dkdr_reverse_kl": self.latest_distill_stats.get("reverse_kl", 0.0),
                "dkdr_gamma_forward": self.latest_distill_stats.get("gamma_forward", 0.5),
                "dkdr_gamma_reverse": self.latest_distill_stats.get("gamma_reverse", 0.5),
                "teacher_reliability_mean": self.latest_distill_stats.get("teacher_reliability_mean", 0.0),
                "teacher_error_mean": self.latest_distill_stats.get("teacher_error_mean", 0.0),
                "routing_error_threshold_mean": threshold_mean,
                "routing_client_veto_rate": veto_rate,
            }
        )
        return metrics

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        if not extra_metrics:
            return ""
        return (
            f" | g_gain={extra_metrics.get('general_gain_over_expert', 0.0):.4f}"
            f" | route_gain={extra_metrics.get('routed_gain_over_expert', 0.0):.4f}"
            f" | invoked_g={extra_metrics.get('invoked_general_gain', 0.0):.4f}"
            f" | ep_p={extra_metrics.get('error_predictor_precision', 0.0):.4f}"
            f" | ep_r={extra_metrics.get('error_predictor_recall', 0.0):.4f}"
            f" | ep_f1={extra_metrics.get('error_predictor_f1', 0.0):.4f}"
            f" | ep_auprc={extra_metrics.get('error_predictor_auprc', 0.0):.4f}"
            f" | ep_fpr={extra_metrics.get('error_predictor_false_positive_rate', 0.0):.4f}"
            f" | dkdr_f={extra_metrics.get('dkdr_forward_kl', 0.0):.4f}"
            f" | dkdr_r={extra_metrics.get('dkdr_reverse_kl', 0.0):.4f}"
            f" | gam_f={extra_metrics.get('dkdr_gamma_forward', 0.5):.4f}"
            f" | gam_r={extra_metrics.get('dkdr_gamma_reverse', 0.5):.4f}"
            f" | thr={extra_metrics.get('routing_error_threshold_mean', 0.0):.4f}"
            f" | veto={extra_metrics.get('routing_client_veto_rate', 0.0):.4f}"
        )

    def _maybe_update_best(self, round_idx: int, metrics: RoundMetrics, expert_accuracy: float, general_accuracy: float) -> None:
        better = self.best_snapshot is None or metrics.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not better:
            return
        self.best_snapshot = {
            "round_idx": round_idx,
            "routed_accuracy": metrics.routed_accuracy,
            "general_accuracy": general_accuracy,
            "expert_accuracy": expert_accuracy,
            "avg_client_loss": metrics.avg_client_loss,
            "general_model_state": _clone_tensor_dict(self.general_model.state_dict()),
            "client_expert_states": {
                client_id: _clone_tensor_dict(client.expert_model.state_dict())
                for client_id, client in self.clients.items()
            },
            "client_predictor_states": {
                client_id: _clone_tensor_dict(state) for client_id, state in self.client_predictor_states.items()
            },
            "client_error_thresholds": dict(self.client_error_thresholds),
            "latest_distill_stats": dict(self.latest_distill_stats),
        }
        LOGGER.info(
            "%s best | round=%d | routed=%.4f | general=%.4f | expert=%.4f",
            self.algorithm_name,
            round_idx,
            metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )
        if bool(getattr(self.config.federated, "save_best_checkpoint", True)):
            self._save_best_checkpoint()

    def _best_checkpoint_path(self) -> Path:
        configured_path = getattr(self.config.federated, "best_checkpoint_path", None)
        if configured_path:
            return Path(configured_path)
        run_name = self.config.run_name or "manual_run"
        return Path(self.config.output_dir) / "checkpoints" / run_name / f"{self.algorithm_name}_best.pt"

    def _save_best_checkpoint(self) -> Optional[Path]:
        if not self.best_snapshot:
            return None
        path = self._best_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "algorithm": self.algorithm_name,
            "run_name": self.config.run_name,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "best_snapshot": self.best_snapshot,
        }
        torch.save(payload, path)
        LOGGER.info("%s saved best checkpoint to %s", self.algorithm_name, path)
        return path

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"FedAsym checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        snapshot = payload.get("best_snapshot", payload) if isinstance(payload, dict) else payload
        if not isinstance(snapshot, dict):
            raise ValueError(f"FedAsym checkpoint has invalid format: {path}")
        self.best_snapshot = snapshot
        self._restore_best()
        self.loaded_checkpoint_path = str(path)
        LOGGER.info("%s loaded checkpoint from %s", self.algorithm_name, path)

    def _recalibrate_route_thresholds(self) -> None:
        updated_thresholds: Dict[str, float] = {}
        with torch.no_grad():
            for client_id, dataset in self.client_calibration_datasets.items():
                predictor_state = self.client_predictor_states[client_id]
                if str(getattr(self.config.inference, "error_predictor_threshold_mode", "fixed")).lower() == "fixed":
                    updated_thresholds[client_id] = float(getattr(self.config.inference, "error_predictor_threshold", 0.5))
                    continue

                loader = self.data_module.make_loader(dataset, shuffle=False)
                feature_batches: List[torch.Tensor] = []
                label_batches: List[torch.Tensor] = []
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()
                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    logits = expert_model(images)
                    features = _predictor_features_from_logits(logits)
                    predictions = logits.argmax(dim=1)
                    feature_batches.append(features.cpu())
                    label_batches.append(predictions.ne(targets_device).cpu())

                if not feature_batches:
                    updated_thresholds[client_id] = 1.01
                    continue

                features = torch.cat(feature_batches, dim=0)
                labels = torch.cat(label_batches, dim=0)
                scores = _predict_error_probabilities(predictor_state, features)
                updated_thresholds[client_id] = _select_precision_constrained_threshold(
                    scores=scores,
                    labels=labels,
                    features=features,
                    config=self.config,
                )

        self.client_error_thresholds = updated_thresholds
        LOGGER.info(
            "%s recalibrated route thresholds | mean=%.4f | veto=%.4f",
            self.algorithm_name,
            sum(updated_thresholds.values()) / max(len(updated_thresholds), 1),
            sum(1 for value in updated_thresholds.values() if float(value) > 1.0) / max(len(updated_thresholds), 1),
        )

    def _retrain_route_predictors(self) -> None:
        updated_thresholds: Dict[str, float] = {}
        for client_id, client in self.clients.items():
            predictor_state = client._fit_risk_predictor()
            client.predictor_state = _clone_tensor_dict(predictor_state)
            self.client_predictor_states[client_id] = _clone_tensor_dict(predictor_state)
            updated_thresholds[client_id] = float(client.error_threshold)
        self.client_error_thresholds = updated_thresholds
        LOGGER.info(
            "%s retrained route predictors | mean_threshold=%.4f | veto=%.4f",
            self.algorithm_name,
            sum(updated_thresholds.values()) / max(len(updated_thresholds), 1),
            sum(1 for value in updated_thresholds.values() if float(value) > 1.0) / max(len(updated_thresholds), 1),
        )

    def _restore_best(self) -> None:
        if not self.best_snapshot:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for client_id, state_dict in self.best_snapshot["client_expert_states"].items():
            self.clients[client_id].expert_model.load_state_dict(state_dict)
        self.client_predictor_states = {
            client_id: _clone_tensor_dict(state)
            for client_id, state in self.best_snapshot["client_predictor_states"].items()
        }
        self.client_error_thresholds = dict(self.best_snapshot["client_error_thresholds"])
        self.latest_distill_stats = dict(self.best_snapshot.get("latest_distill_stats", {}))
        self.current_round = int(self.best_snapshot["round_idx"])
        LOGGER.info("%s restored best from round %d", self.algorithm_name, self.best_snapshot["round_idx"])

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        checkpoint_path = getattr(self.config.federated, "load_checkpoint_path", None)
        if checkpoint_path and bool(getattr(self.config.federated, "eval_only_from_checkpoint", False)):
            self._load_checkpoint(checkpoint_path)
            if bool(getattr(self.config.federated, "risk_predictor_retrain_on_load", False)):
                self._retrain_route_predictors()
            elif bool(getattr(self.config.federated, "recalibrate_route_thresholds_on_load", True)):
                self._recalibrate_route_thresholds()
            self.last_history = []
            return []

        metrics: List[RoundMetrics] = []
        upload_bytes_total = 0.0

        for round_idx in range(1, self.config.federated.rounds + 1):
            self._device_synchronize()
            round_start_time = time.perf_counter()
            self.current_round = round_idx
            selected_client_ids = self._sample_client_ids()
            LOGGER.info("%s round %d | clients=%s", self.algorithm_name, round_idx, selected_client_ids)

            updates = [
                self.clients[client_id].train_local(
                    general_model=self.general_model,
                    public_batches=self.public_batches,
                )
                for client_id in selected_client_ids
            ]
            for update in updates:
                self.client_predictor_states[update.client_id] = _clone_tensor_dict(update.predictor_state)
                self.client_error_thresholds[update.client_id] = float(update.error_threshold)

            public_knowledge = self._aggregate_public_knowledge(updates)
            distill_stats = self._distill_general_model(public_knowledge, round_idx)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, f"{self.algorithm_name}-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, f"{self.algorithm_name}-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, f"{self.algorithm_name}-routed")
            extra_metrics = self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)

            average_loss = sum(update.loss for update in updates) / max(len(updates), 1)
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
                selected_client_ids,
                self.client_training_datasets,
            )
            resource_metrics = self._resource_metric_values(
                self.resource_profiles,
                client_train_profile,
                compute_profile,
            )
            round_upload_bytes = float(
                sum(
                    self._estimate_tensor_payload_bytes(update.public_logits)
                    + self._estimate_tensor_payload_bytes(update.predictor_state)
                    for update in updates
                )
            )
            upload_bytes_total += round_upload_bytes
            self._device_synchronize()
            round_train_time_seconds = time.perf_counter() - round_start_time

            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=average_loss,
                routed_accuracy=macro["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                weighted_accuracy=aggregate["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                inference_latency_ms=aggregate["latency_ms"],
                round_train_time_seconds=round_train_time_seconds,
                upload_bytes_per_round=round_upload_bytes,
                upload_bytes_total=upload_bytes_total,
                extra_metrics=extra_metrics,
                **resource_metrics,
            )
            extra_log = self._format_round_extra_metrics_for_log(extra_metrics)
            LOGGER.info(
                "%s round %d | loss=%.4f | distill=%.4f | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f%s%s",
                self.algorithm_name,
                round_idx,
                average_loss,
                distill_stats["total_loss"],
                macro["accuracy"],
                aggregate["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                extra_log,
                self._format_resource_metrics_for_log(round_metrics),
            )

            if self.writer is not None:
                self._log_compare_scalars(
                    self.algorithm_name,
                    round_idx,
                    {
                        "expert_loss": average_loss,
                        "distill_loss": distill_stats["total_loss"],
                        "dkdr_forward_kl": distill_stats.get("forward_kl", 0.0),
                        "dkdr_reverse_kl": distill_stats.get("reverse_kl", 0.0),
                        "dkdr_gamma_forward": distill_stats.get("gamma_forward", 0.5),
                        "dkdr_gamma_reverse": distill_stats.get("gamma_reverse", 0.5),
                    },
                )
                self._log_auxiliary_accuracy_metrics(
                    self.algorithm_name,
                    round_idx,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics(self.algorithm_name, round_metrics)
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
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, f"{self.algorithm_name}_final_expert")
        route_export_path = self._build_route_export_path(f"{self.algorithm_name}_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed_full_metadata,
            f"{self.algorithm_name}_final_routed",
            route_export_path=route_export_path,
        )
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, f"{self.algorithm_name}_final_general")

        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = 0
        if self.best_snapshot:
            final_loss = float(self.best_snapshot["avg_client_loss"])
            best_round = int(self.best_snapshot["round_idx"])

        routed_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            routed_eval["aggregate"]["invocation_rate"],
            "routed",
        )
        expert_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 0.0, "expert_only")
        general_compute = self._build_compute_profile(self.expert_flops, self.general_flops, 1.0, "general_only")
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
            routed_compute,
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
        extra_metrics = self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)
        return {
            "algorithm": self.algorithm_name,
            "metrics": {
                "accuracy": routed_eval["macro"]["accuracy"],
                "personalized_accuracy": routed_eval["macro"]["accuracy"],
                "weighted_accuracy": routed_eval["aggregate"]["accuracy"],
                "global_accuracy": routed_eval["macro"]["accuracy"],
                "local_accuracy": routed_eval["macro"]["accuracy"],
                "routed_accuracy": routed_eval["macro"]["accuracy"],
                "hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "hard_sample_recall": routed_eval["aggregate"]["hard_recall"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["macro"]["accuracy"],
                "general_only_accuracy": general_eval["macro"]["accuracy"],
                "final_training_loss": final_loss,
                "best_round": best_round,
                "average_inference_latency_ms": routed_eval["aggregate"]["latency_ms"],
                "average_round_train_time_seconds": average_round_train_time_seconds,
                "total_train_time_seconds": total_train_time_seconds,
                "average_upload_bytes_per_round": average_upload_bytes_per_round,
                "total_upload_bytes": total_upload_bytes,
                **final_resource_metrics,
                **extra_metrics,
            },
            "client_metrics": {
                "routed": routed_eval["clients"],
                "expert_only": expert_eval["clients"],
                "general_only": general_eval["clients"],
            },
            "group_metrics": {
                "routed": routed_eval["groups"],
                "expert_only": expert_eval["groups"],
                "general_only": general_eval["groups"],
            },
            "compute": {
                "routed": routed_compute,
                "expert_only": expert_compute,
                "general_only": general_compute,
                "client_train": final_client_train,
            },
            "memory_mb": self._resource_memory_table(self.resource_profiles),
            "artifacts": {
                "route_csv": str(route_export_path),
                "best_checkpoint": self.loaded_checkpoint_path or (str(self._best_checkpoint_path()) if self.best_snapshot else None),
            },
        }
