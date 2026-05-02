"""
FedAsym: pure Gemini-dialogue implementation.

Design constraints:
  - no inheritance from other algorithm files,
  - client-side public distillation follows the FedEcho-style KL/CE mixing
    described in the Gemini dialogue,
  - server-side public distillation follows the DKDR bidirectional KL idea,
  - reliability comes only from an explicit client-side error predictor R_k,
  - default routing uses only the predicted expert error probability;
    optional two-stage routing uses that probability as a candidate generator
    and a calibrated expert/general verifier for final adoption.
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
    public_tta_logits: Optional[torch.Tensor] = None
    candidate_predictor_state: Optional[Dict[str, torch.Tensor]] = None


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


def _parse_float_list(raw_values: object) -> List[float]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        values = [item.strip() for item in raw_values.split(",")]
        return [float(item) for item in values if item]
    if isinstance(raw_values, Sequence):
        return [float(item) for item in raw_values]
    return []


_PREDICTOR_TTA_FEATURE_DIM = 7


def _risk_predictor_tta_enabled(config) -> bool:
    return bool(getattr(config.federated, "risk_predictor_tta_enabled", False))


def _predictor_feature_dim(num_classes: int, tta_enabled: bool = False) -> int:
    return 8 + int(num_classes) + (_PREDICTOR_TTA_FEATURE_DIM if tta_enabled else 0)


def _configured_predictor_feature_dim(config, num_classes: int) -> int:
    return _predictor_feature_dim(num_classes, tta_enabled=_risk_predictor_tta_enabled(config))


def _route_verifier_feature_dim(num_classes: int) -> int:
    return 28 + (2 * int(num_classes))


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


def _probability_summary(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
    confidence = topk.values[:, 0]
    if topk.values.size(1) > 1:
        margin = topk.values[:, 0] - topk.values[:, 1]
    else:
        margin = torch.ones_like(confidence)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / max(math.log(float(probs.size(1))), 1e-8)
    return confidence, margin, entropy


def _predictor_tta_features_from_logits(logits: torch.Tensor, tta_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    tta_probs = torch.softmax(tta_logits, dim=1)
    confidence, margin, entropy = _probability_summary(probs)
    tta_confidence, tta_margin, tta_entropy = _probability_summary(tta_probs)
    log_probs = probs.clamp_min(1e-8).log()
    tta_log_probs = tta_probs.clamp_min(1e-8).log()
    forward_kl = (probs * (log_probs - tta_log_probs)).sum(dim=1)
    reverse_kl = (tta_probs * (tta_log_probs - log_probs)).sum(dim=1)
    mean_probs = 0.5 * (probs + tta_probs)
    mean_entropy = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    mean_entropy = mean_entropy / max(math.log(float(probs.size(1))), 1e-8)
    pred_agree = logits.argmax(dim=1).eq(tta_logits.argmax(dim=1)).to(dtype=logits.dtype)
    return torch.stack(
        [
            pred_agree,
            (confidence - tta_confidence).abs(),
            (margin - tta_margin).abs(),
            (entropy - tta_entropy).abs(),
            forward_kl,
            reverse_kl,
            mean_entropy,
        ],
        dim=1,
    )


def _predictor_features_from_logits_with_tta(
    logits: torch.Tensor,
    tta_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    features = _predictor_features_from_logits(logits)
    if tta_logits is None:
        return features
    return torch.cat([features, _predictor_tta_features_from_logits(logits, tta_logits)], dim=1)


def _predictor_features_from_model(
    config,
    model: nn.Module,
    images: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    if not _risk_predictor_tta_enabled(config):
        return _predictor_features_from_logits(logits)
    tta_images = torch.flip(images, dims=[-1])
    tta_logits = model(tta_images)
    return _predictor_features_from_logits_with_tta(logits, tta_logits)


def _route_verifier_features_from_logits(
    expert_logits: torch.Tensor,
    general_logits: torch.Tensor,
    error_probs: torch.Tensor,
) -> torch.Tensor:
    expert_features = _predictor_features_from_logits(expert_logits)
    general_features = _predictor_features_from_logits(general_logits)
    expert_probs = torch.softmax(expert_logits, dim=1)
    general_probs = torch.softmax(general_logits, dim=1)
    expert_log_probs = expert_probs.clamp_min(1e-8).log()
    general_log_probs = general_probs.clamp_min(1e-8).log()
    forward_kl = (expert_probs * (expert_log_probs - general_log_probs)).sum(dim=1)
    reverse_kl = (general_probs * (general_log_probs - expert_log_probs)).sum(dim=1)
    same_prediction = expert_logits.argmax(dim=1).eq(general_logits.argmax(dim=1)).to(dtype=expert_logits.dtype)
    dense_delta = general_features[:, :8] - expert_features[:, :8]
    return torch.cat(
        [
            error_probs.view(-1, 1).to(dtype=expert_logits.dtype),
            expert_features,
            general_features,
            dense_delta,
            same_prediction.view(-1, 1),
            forward_kl.view(-1, 1),
            reverse_kl.view(-1, 1),
        ],
        dim=1,
    )


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
    fpr_modes = {"fpr", "fpr_constrained", "fpr-constrained", "neyman_pearson", "neyman-pearson"}
    precision_modes = {"precision", "precision_constrained", "precision_wilson", "wilson"}
    if mode not in precision_modes | fpr_modes:
        return base_threshold

    target_precision = min(max(float(getattr(config.inference, "error_predictor_target_precision", 0.8)), 0.0), 1.0)
    target_fpr = min(max(float(getattr(config.inference, "error_predictor_target_fpr", 0.01)), 0.0), 1.0)
    max_false_positive = int(getattr(config.inference, "error_predictor_max_false_positive", 5))
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

    candidates = torch.unique(candidate_scores.clamp(min_threshold, max_threshold)).tolist()
    candidates.extend([base_threshold, min_threshold, max_threshold])

    if mode in fpr_modes:
        actual_positive = int((labels & eligible).sum().item())
        actual_negative = int(((~labels) & eligible).sum().item())
        if actual_positive <= 0 or actual_negative <= 0:
            return 1.01 if disable_on_fail else base_threshold

        best_threshold: Optional[float] = None
        best_key: Optional[Tuple[float, int, float, int]] = None
        for threshold in sorted({float(item) for item in candidates}):
            predicted = (scores >= threshold) & eligible
            invoked = int(predicted.sum().item())
            if invoked < min_positive:
                continue
            true_positive = int((predicted & labels).sum().item())
            false_positive = int((predicted & ~labels).sum().item())
            if max_false_positive >= 0 and false_positive > max_false_positive:
                continue
            fpr = false_positive / max(actual_negative, 1)
            if fpr > target_fpr:
                continue
            recall = true_positive / max(actual_positive, 1)
            precision = true_positive / max(invoked, 1)
            key = (recall, -false_positive, precision, invoked)
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)

        if best_threshold is not None:
            return best_threshold
        return 1.01 if disable_on_fail else base_threshold

    best_threshold: Optional[float] = None
    best_invoked = -1
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


def _select_fpr_constrained_group_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    eligible: torch.Tensor,
    base_threshold: float,
    target_fpr: float,
    max_false_positive: int,
    min_predicted_positive: int,
    min_support: int,
    min_errors: int,
    min_threshold: float,
    max_threshold: float,
) -> Tuple[Optional[float], Dict[str, float]]:
    scores = scores.detach().cpu().to(torch.float32).view(-1)
    labels = labels.detach().cpu().to(torch.bool).view(-1)
    eligible = eligible.detach().cpu().to(torch.bool).view(-1)
    stats: Dict[str, float] = {
        "selected_threshold": float(base_threshold),
        "threshold_lowered": 0.0,
        "selected_invoked": 0.0,
        "selected_true_positive": 0.0,
        "selected_false_positive": 0.0,
        "selected_precision": 0.0,
        "selected_recall": 0.0,
        "selected_fpr": 0.0,
    }
    if scores.numel() == 0 or scores.numel() != labels.numel() or scores.numel() != eligible.numel():
        return None, stats
    support = int(eligible.sum().item())
    actual_positive = int((eligible & labels).sum().item())
    actual_negative = int((eligible & ~labels).sum().item())
    if support < min_support or actual_positive < min_errors or actual_negative <= 0:
        return None, stats

    base_threshold = float(base_threshold)
    target_fpr = min(max(float(target_fpr), 0.0), 1.0)
    max_false_positive = int(max_false_positive)
    min_predicted_positive = max(int(min_predicted_positive), 1)
    min_threshold = float(min_threshold)
    max_threshold = float(max_threshold)
    candidates = torch.unique(scores[eligible].clamp(min_threshold, max_threshold)).tolist()
    candidates.extend([base_threshold, min_threshold, max_threshold])

    base_predicted = (scores >= base_threshold) & eligible
    base_true_positive = int((base_predicted & labels).sum().item())
    base_recall = base_true_positive / max(actual_positive, 1)

    best_threshold: Optional[float] = None
    best_key: Optional[Tuple[float, int, float, int]] = None
    best_stats: Dict[str, float] = dict(stats)
    for threshold in sorted({float(item) for item in candidates if float(item) <= base_threshold + 1e-8}):
        predicted = (scores >= threshold) & eligible
        invoked = int(predicted.sum().item())
        if invoked < min_predicted_positive:
            continue
        true_positive = int((predicted & labels).sum().item())
        false_positive = int((predicted & ~labels).sum().item())
        if max_false_positive >= 0 and false_positive > max_false_positive:
            continue
        fpr = false_positive / max(actual_negative, 1)
        if fpr > target_fpr:
            continue
        recall = true_positive / max(actual_positive, 1)
        precision = true_positive / max(invoked, 1)
        key = (recall, -false_positive, precision, invoked)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = {
                "selected_threshold": float(threshold),
                "selected_invoked": float(invoked),
                "selected_true_positive": float(true_positive),
                "selected_false_positive": float(false_positive),
                "selected_precision": float(precision),
                "selected_recall": float(recall),
                "selected_fpr": float(fpr),
            }

    if best_threshold is None or best_threshold >= base_threshold - 1e-8:
        return None, stats
    if float(best_stats.get("selected_recall", 0.0)) <= base_recall + 1e-8:
        return None, stats

    best_stats["threshold_lowered"] = 1.0
    return best_threshold, best_stats




def _fit_predictor_state(
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int = 0,
    dropout: float = 0.0,
    hard_negative_enabled: bool = False,
    hard_negative_quantile: float = 0.8,
    hard_negative_weight: float = 2.0,
    hard_negative_warmup_epochs: int = 20,
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(max(normalized.size(0), 1), 128)
    x_all = normalized.to(device)
    y_all = labels.to(device).unsqueeze(1)
    hard_negative_quantile = min(max(float(hard_negative_quantile), 0.0), 1.0)
    hard_negative_weight = max(float(hard_negative_weight), 1.0)
    hard_negative_warmup_epochs = max(int(hard_negative_warmup_epochs), 0)
    hard_negative_enabled = bool(hard_negative_enabled) and hard_negative_weight > 1.0
    sample_weights = torch.ones_like(y_all)
    hard_negative_weights_frozen = False

    for epoch_idx in range(max(epochs, 1)):
        if hard_negative_enabled and not hard_negative_weights_frozen and epoch_idx >= hard_negative_warmup_epochs:
            with torch.no_grad():
                scores = torch.sigmoid(model(x_all)).view(-1)
                negative_mask = y_all.view(-1) < 0.5
                negative_scores = scores[negative_mask]
                sample_weights = torch.ones_like(y_all)
                if negative_scores.numel() > 0:
                    sorted_scores, _ = torch.sort(negative_scores)
                    cutoff_index = int(math.floor(hard_negative_quantile * (sorted_scores.numel() - 1)))
                    cutoff = sorted_scores[min(max(cutoff_index, 0), sorted_scores.numel() - 1)]
                    hard_negative_mask = negative_mask & (scores >= cutoff)
                    sample_weights[hard_negative_mask.unsqueeze(1)] = hard_negative_weight
            hard_negative_weights_frozen = True

        permutation = torch.randperm(x_all.size(0), device=device)
        for start in range(0, x_all.size(0), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_all[indices])
            losses = criterion(logits, y_all[indices])
            loss = (losses * sample_weights[indices]).mean()
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


def _fit_weighted_binary_predictor_state(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int = 0,
    dropout: float = 0.0,
) -> Dict[str, torch.Tensor]:
    feature_dim = int(features.size(1)) if features.ndim == 2 else 1
    hidden_dim = max(int(hidden_dim), 0)
    if features.numel() == 0 or labels.numel() == 0 or sample_weights.numel() == 0:
        return _default_predictor_state(feature_dim, hidden_dim=hidden_dim)

    labels = labels.float().view(-1)
    features = features.float()
    sample_weights = sample_weights.float().view(-1).clamp_min(0.0)
    active_mask = sample_weights > 0.0
    if active_mask.sum().item() <= 0:
        return _default_predictor_state(feature_dim, hidden_dim=hidden_dim)

    features = features[active_mask]
    labels = labels[active_mask]
    sample_weights = sample_weights[active_mask]
    positive_weighted = float((sample_weights * labels).sum().item())
    negative_weighted = float((sample_weights * (1.0 - labels)).sum().item())
    total_weighted = max(positive_weighted + negative_weighted, 1e-8)
    positive_rate = positive_weighted / total_weighted
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
    pos_weight = torch.tensor([max(negative_weighted / max(positive_weighted, 1e-8), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(max(normalized.size(0), 1), 128)
    x_all = normalized.to(device)
    y_all = labels.to(device).unsqueeze(1)
    w_all = sample_weights.to(device).unsqueeze(1)

    for _ in range(max(int(epochs), 1)):
        permutation = torch.randperm(x_all.size(0), device=device)
        for start in range(0, x_all.size(0), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_all[indices])
            losses = criterion(logits, y_all[indices])
            loss = (losses * w_all[indices]).sum() / w_all[indices].sum().clamp_min(1e-8)
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


def _select_top_rate_threshold(scores: torch.Tensor, rate: float, min_score: float = 0.0) -> float:
    if scores.numel() == 0:
        return 1.01
    rate = min(max(float(rate), 0.0), 1.0)
    if rate <= 0.0:
        return 1.01
    scores = scores.detach().to(dtype=torch.float32).view(-1)
    candidate_count = min(max(int(math.ceil(scores.numel() * rate)), 1), int(scores.numel()))
    sorted_scores, _ = torch.sort(scores, descending=True)
    threshold = float(sorted_scores[candidate_count - 1].item())
    return max(threshold, float(min_score))


def _select_route_verifier_threshold(
    scores: torch.Tensor,
    rescue_mask: torch.Tensor,
    harm_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    config,
) -> Tuple[float, Dict[str, float]]:
    scores = scores.detach().to(dtype=torch.float32).view(-1)
    rescue_mask = rescue_mask.detach().to(dtype=torch.bool).view(-1)
    harm_mask = harm_mask.detach().to(dtype=torch.bool).view(-1)
    candidate_mask = candidate_mask.detach().to(dtype=torch.bool).view(-1)
    total = int(scores.numel())
    stats = {
        "samples": float(total),
        "candidate": float(candidate_mask.sum().item()),
        "adopted": 0.0,
        "rescue": 0.0,
        "harm": 0.0,
        "net_rescue": 0.0,
        "candidate_rate": float(candidate_mask.sum().item()) / max(total, 1),
        "adopted_rate": 0.0,
        "harm_rate": 0.0,
        "rescue_harm_ratio": 0.0,
        "selected_threshold": 1.01,
        "disabled": 1.0,
    }
    if total <= 0 or not candidate_mask.any():
        return 1.01, stats

    mode = str(getattr(config.inference, "route_verifier_threshold_mode", "harm_constrained")).lower()
    min_adopt_threshold = min(
        max(float(getattr(config.inference, "route_verifier_min_adopt_threshold", 0.0)), 0.0),
        1.0,
    )
    if mode in {"fixed", "manual"}:
        threshold = max(float(getattr(config.inference, "route_verifier_threshold", 0.5)), min_adopt_threshold)
        adopted = candidate_mask & scores.ge(threshold)
        rescue = int((adopted & rescue_mask).sum().item())
        harm = int((adopted & harm_mask).sum().item())
        adopted_count = int(adopted.sum().item())
        stats.update(
            {
                "adopted": float(adopted_count),
                "rescue": float(rescue),
                "harm": float(harm),
                "net_rescue": float(rescue - harm),
                "adopted_rate": adopted_count / max(total, 1),
                "harm_rate": harm / max(total, 1),
                "rescue_harm_ratio": rescue / max(harm, 1),
                "selected_threshold": threshold,
                "disabled": 0.0,
            }
        )
        return threshold, stats

    max_harm_rate = min(max(float(getattr(config.inference, "router_max_harm_rate", 0.01)), 0.0), 1.0)
    min_ratio = max(float(getattr(config.inference, "router_min_rescue_harm_ratio", 1.0)), 0.0)
    min_adopted = max(int(getattr(config.inference, "router_min_adopted", 3)), 1)
    min_rescue = max(int(getattr(config.inference, "router_min_rescue", 1)), 0)
    harm_lambda = max(float(getattr(config.inference, "route_verifier_harm_lambda", 1.0)), 0.0)

    candidate_scores = scores[candidate_mask]
    candidates = torch.unique(candidate_scores.clamp(0.0, 1.0)).tolist()
    candidates.extend([min_adopt_threshold, 0.5, 1.0])
    best_key: Optional[Tuple[float, int, int, int]] = None
    best_threshold: Optional[float] = None
    best_stats: Dict[str, float] = {}
    for threshold in sorted({float(item) for item in candidates}):
        if threshold < min_adopt_threshold:
            continue
        adopted = candidate_mask & scores.ge(threshold)
        adopted_count = int(adopted.sum().item())
        if adopted_count < min_adopted:
            continue
        rescue = int((adopted & rescue_mask).sum().item())
        harm = int((adopted & harm_mask).sum().item())
        if rescue < min_rescue:
            continue
        harm_rate = harm / max(total, 1)
        if harm_rate > max_harm_rate:
            continue
        rescue_harm_ratio = rescue / max(harm, 1)
        if rescue_harm_ratio < min_ratio:
            continue
        objective = rescue - (harm_lambda * harm)
        key = (float(objective), rescue, -harm, adopted_count)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = {
                "adopted": float(adopted_count),
                "rescue": float(rescue),
                "harm": float(harm),
                "net_rescue": float(rescue - harm),
                "adopted_rate": adopted_count / max(total, 1),
                "harm_rate": harm_rate,
                "rescue_harm_ratio": rescue_harm_ratio,
                "selected_threshold": float(threshold),
                "disabled": 0.0,
            }

    if best_threshold is None:
        return 1.01, stats
    stats.update(best_stats)
    return best_threshold, stats


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
            _predictor_feature_dim(num_classes, tta_enabled=False),
            hidden_dim=int(getattr(config.federated, "risk_predictor_hidden_dim", 32)),
        )
        self.candidate_predictor_state = _default_predictor_state(
            _configured_predictor_feature_dim(config, num_classes),
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
        self.predictor_state = self._fit_risk_predictor(use_tta=False, update_error_threshold=True)
        if _risk_predictor_tta_enabled(self.config):
            self.candidate_predictor_state = self._fit_risk_predictor(use_tta=True, update_error_threshold=False)
        else:
            self.candidate_predictor_state = _clone_tensor_dict(self.predictor_state)
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
            candidate_predictor_state=_clone_tensor_dict(self.candidate_predictor_state),
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

    def _fit_risk_predictor(
        self,
        use_tta: Optional[bool] = None,
        update_error_threshold: bool = True,
    ) -> Dict[str, torch.Tensor]:
        tta_enabled = _risk_predictor_tta_enabled(self.config) if use_tta is None else bool(use_tta)
        feature_dim = _predictor_feature_dim(self.num_classes, tta_enabled=tta_enabled)
        if self.calibration_dataset is None or len(self.calibration_dataset) == 0:
            return _default_predictor_state(
                feature_dim,
                hidden_dim=int(getattr(self.config.federated, "risk_predictor_hidden_dim", 32)),
            )

        features, labels = self._collect_predictor_features(self.calibration_dataset, use_tta=tta_enabled)
        if features.numel() == 0 or labels.numel() == 0:
            return _default_predictor_state(
                feature_dim,
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
            hard_negative_enabled=bool(
                getattr(self.config.federated, "risk_predictor_hard_negative_enabled", False)
            ),
            hard_negative_quantile=float(
                getattr(self.config.federated, "risk_predictor_hard_negative_quantile", 0.9)
            ),
            hard_negative_weight=float(
                getattr(self.config.federated, "risk_predictor_hard_negative_weight", 1.25)
            ),
            hard_negative_warmup_epochs=int(
                getattr(self.config.federated, "risk_predictor_hard_negative_warmup_epochs", 40)
            ),
        )
        if not update_error_threshold:
            return predictor_state
        threshold_features, threshold_labels = self._collect_predictor_features(
            self.threshold_dataset,
            use_tta=tta_enabled,
        )
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

    def _collect_predictor_features(
        self,
        dataset: Dataset,
        use_tta: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tta_enabled = _risk_predictor_tta_enabled(self.config) if use_tta is None else bool(use_tta)
        feature_dim = _predictor_feature_dim(self.num_classes, tta_enabled=tta_enabled)
        if dataset is None or len(dataset) == 0:
            return (
                torch.empty((0, feature_dim), dtype=torch.float32),
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
                if tta_enabled:
                    features = _predictor_features_from_model(self.config, self.expert_model, images, logits)
                else:
                    features = _predictor_features_from_logits(logits)
                predictions = logits.argmax(dim=1)
                label_batches.append(predictions.ne(targets_device).float().cpu())
                feature_batches.append(features.cpu())

        if not feature_batches:
            return (
                torch.empty((0, feature_dim), dtype=torch.float32),
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

    def _collect_public_tta_logits(self, public_batches: Sequence) -> torch.Tensor:
        logits_batches: List[torch.Tensor] = []
        self.expert_model.eval()
        with torch.no_grad():
            for batch in public_batches:
                images = torch.flip(batch[0].to(self.device), dims=[-1])
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
        self.client_candidate_predictor_states: Dict[str, Dict[str, torch.Tensor]] = {
            client_id: _clone_tensor_dict(client.candidate_predictor_state) for client_id, client in self.clients.items()
        }
        self.client_error_thresholds: Dict[str, float] = {
            client_id: float(getattr(config.inference, "error_predictor_threshold", 0.5))
            for client_id in client_datasets
        }
        self.client_raw_error_thresholds: Dict[str, float] = dict(self.client_error_thresholds)
        self.client_route_gain_stats: Dict[str, Dict[str, float]] = {}
        self.client_route_group_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.client_route_group_allowlists: Dict[str, List[int]] = {}
        self.client_route_group_blocklists: Dict[str, List[int]] = {}
        self.client_route_group_thresholds: Dict[str, Dict[int, float]] = {}
        self.client_route_group_threshold_boosts: Dict[str, Dict[int, float]] = {}
        self.client_route_candidate_thresholds: Dict[str, float] = {
            client_id: 1.01 for client_id in client_datasets
        }
        self.client_route_candidate_bin_thresholds: Dict[str, Dict[int, float]] = {
            client_id: {} for client_id in client_datasets
        }
        self.client_route_verifier_states: Dict[str, Dict[str, torch.Tensor]] = {
            client_id: _default_predictor_state(
                _route_verifier_feature_dim(config.model.num_classes),
                hidden_dim=int(getattr(config.inference, "route_verifier_hidden_dim", 32)),
            )
            for client_id in client_datasets
        }
        self.client_route_verifier_thresholds: Dict[str, float] = {
            client_id: 1.01 for client_id in client_datasets
        }
        self.client_route_verifier_bin_thresholds: Dict[str, Dict[int, float]] = {
            client_id: {} for client_id in client_datasets
        }
        self.client_route_verifier_stats: Dict[str, Dict[str, float]] = {}
        self.client_route_verifier_bin_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.loaded_checkpoint_path: Optional[str] = None
        self.current_round = 0
        self.latest_distill_stats: Dict[str, float] = {}

    def _predict_public_error_probabilities(
        self,
        predictor_state: Dict[str, torch.Tensor],
        public_logits: torch.Tensor,
        public_tta_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = _predictor_features_from_logits_with_tta(public_logits, public_tta_logits)
        return _predict_error_probabilities(predictor_state, features)

    def _candidate_predictor_state(self, client_id: str) -> Dict[str, torch.Tensor]:
        return self.client_candidate_predictor_states.get(
            client_id,
            self.client_predictor_states[client_id],
        )

    def _base_predictor_features(self, logits: torch.Tensor) -> torch.Tensor:
        return _predictor_features_from_logits(logits)

    def _candidate_predictor_features(
        self,
        model: nn.Module,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if not _risk_predictor_tta_enabled(self.config):
            return _predictor_features_from_logits(logits)
        return _predictor_features_from_model(self.config, model, images, logits)

    def _router_candidate_tta_weight(self) -> float:
        if not _risk_predictor_tta_enabled(self.config):
            return 0.0
        return min(max(float(getattr(self.config.inference, "router_candidate_tta_weight", 0.25)), 0.0), 1.0)

    def _candidate_error_probabilities(
        self,
        client_id: str,
        model: nn.Module,
        images: torch.Tensor,
        logits: torch.Tensor,
        base_features: torch.Tensor,
        base_error_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tta_weight = self._router_candidate_tta_weight()
        if tta_weight <= 0.0:
            return base_error_probs, base_features

        candidate_features = self._candidate_predictor_features(model, images, logits)
        tta_error_probs = _predict_error_probabilities(
            self._candidate_predictor_state(client_id),
            candidate_features,
        )
        if tta_weight >= 1.0:
            return tta_error_probs, candidate_features
        mixed_error_probs = ((1.0 - tta_weight) * base_error_probs) + (tta_weight * tta_error_probs)
        return mixed_error_probs.clamp(0.0, 1.0), candidate_features

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
            error_probs = self._predict_public_error_probabilities(
                update.predictor_state,
                update.public_logits,
            )
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

    def _two_stage_routing_enabled(self) -> bool:
        policy = str(getattr(self.config.inference, "routing_policy", "")).lower()
        return policy in {
            "two_stage_error_verifier",
            "two-stage-error-verifier",
            "two_stage",
            "candidate_verifier",
            "error_predictor_verifier",
        }

    def _confidence_bin_count(self) -> int:
        return len(self._router_diagnostic_confidence_edges()) + 1

    def _confidence_bin_label_by_index(self, bin_index: int) -> str:
        edges = self._router_diagnostic_confidence_edges()
        index = int(bin_index)
        if index <= 0:
            return f"<{edges[0]:.2f}" if edges else "all"
        if index >= len(edges):
            return f">={edges[-1]:.2f}" if edges else "all"
        return f"{edges[index - 1]:.2f}-{edges[index]:.2f}"

    def _confidence_bin_indices(self, confidence: torch.Tensor) -> torch.Tensor:
        bins = torch.zeros_like(confidence, dtype=torch.long)
        for edge in self._router_diagnostic_confidence_edges():
            bins = bins + confidence.ge(float(edge)).to(dtype=torch.long)
        return bins

    def _candidate_confidence_bins_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "router_candidate_confidence_bins_enabled", False))

    def _route_verifier_confidence_bins_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "route_verifier_confidence_bin_thresholds_enabled", False))

    def _route_fusion_confidence_alpha_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "route_fusion_confidence_alpha_enabled", False))

    def _candidate_confidence_bin_rates(self, default_rate: float) -> List[float]:
        raw_rates = _parse_float_list(getattr(self.config.inference, "router_candidate_confidence_bin_rates", []))
        bin_count = self._confidence_bin_count()
        rates: List[float] = []
        for bin_index in range(bin_count):
            rate = raw_rates[bin_index] if bin_index < len(raw_rates) else default_rate
            rates.append(min(max(float(rate), 0.0), 1.0))
        return rates

    def _fusion_confidence_alpha_max_values(self, default_max: float) -> List[float]:
        raw_values = _parse_float_list(getattr(self.config.inference, "route_fusion_confidence_alpha_max_values", []))
        bin_count = self._confidence_bin_count()
        values: List[float] = []
        for bin_index in range(bin_count):
            value = raw_values[bin_index] if bin_index < len(raw_values) else default_max
            values.append(min(max(float(value), 0.0), 1.0))
        return values

    def _select_candidate_thresholds_by_confidence(
        self,
        scores: torch.Tensor,
        confidence: torch.Tensor,
        default_rate: float,
        min_score: float,
    ) -> Dict[int, float]:
        if scores.numel() == 0 or confidence.numel() == 0:
            return {bin_index: 1.01 for bin_index in range(self._confidence_bin_count())}
        scores = scores.detach().to(dtype=torch.float32).view(-1)
        confidence = confidence.detach().to(dtype=torch.float32).view(-1)
        bins = self._confidence_bin_indices(confidence)
        rates = self._candidate_confidence_bin_rates(default_rate)
        thresholds: Dict[int, float] = {}
        for bin_index, rate in enumerate(rates):
            bin_mask = bins.eq(int(bin_index))
            if bin_mask.any():
                thresholds[int(bin_index)] = _select_top_rate_threshold(scores[bin_mask], rate, min_score=min_score)
            else:
                thresholds[int(bin_index)] = 1.01
        return thresholds

    def _candidate_thresholds_for_batch(
        self,
        client_id: str,
        expert_features: torch.Tensor,
    ) -> torch.Tensor:
        base_threshold = float(self.client_route_candidate_thresholds.get(client_id, 1.01))
        thresholds = torch.full(
            (expert_features.size(0),),
            base_threshold,
            device=expert_features.device,
            dtype=expert_features.dtype,
        )
        if not self._candidate_confidence_bins_enabled():
            return thresholds
        bin_thresholds = self.client_route_candidate_bin_thresholds.get(client_id, {})
        if not bin_thresholds:
            return thresholds
        bins = self._confidence_bin_indices(expert_features[:, 0])
        for bin_index, threshold in bin_thresholds.items():
            thresholds[bins.eq(int(bin_index))] = float(threshold)
        return thresholds

    def _candidate_mask_from_scores_and_confidence(
        self,
        client_id: str,
        scores: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        base_threshold = float(self.client_route_candidate_thresholds.get(client_id, 1.01))
        thresholds = torch.full_like(scores, base_threshold, dtype=torch.float32)
        if self._candidate_confidence_bins_enabled():
            bin_thresholds = self.client_route_candidate_bin_thresholds.get(client_id, {})
            bins = self._confidence_bin_indices(confidence.to(dtype=torch.float32))
            for bin_index, threshold in bin_thresholds.items():
                thresholds[bins.eq(int(bin_index))] = float(threshold)
        return scores.to(dtype=torch.float32).ge(thresholds)

    def _route_verifier_thresholds_for_batch(
        self,
        client_id: str,
        expert_features: torch.Tensor,
    ) -> torch.Tensor:
        base_threshold = float(self.client_route_verifier_thresholds.get(client_id, 1.01))
        thresholds = torch.full(
            (expert_features.size(0),),
            base_threshold,
            device=expert_features.device,
            dtype=expert_features.dtype,
        )
        if not self._route_verifier_confidence_bins_enabled():
            return thresholds
        bin_thresholds = self.client_route_verifier_bin_thresholds.get(client_id, {})
        if not bin_thresholds:
            return thresholds
        bins = self._confidence_bin_indices(expert_features[:, 0])
        for bin_index, threshold in bin_thresholds.items():
            thresholds[bins.eq(int(bin_index))] = float(threshold)
        return thresholds

    def _passes_route_verifier_validation_filter(
        self,
        threshold_stats: Dict[str, float],
        min_validation_adopted: int,
        min_validation_net: float,
        min_validation_ratio: float,
    ) -> Tuple[bool, Dict[str, float]]:
        validation_adopted = float(threshold_stats.get("adopted", 0.0))
        validation_net = float(threshold_stats.get("net_rescue", 0.0))
        validation_ratio = float(threshold_stats.get("rescue_harm_ratio", 0.0))
        adopted_fail = validation_adopted < float(min_validation_adopted)
        net_fail = validation_net < min_validation_net
        ratio_fail = validation_ratio < min_validation_ratio
        validation_fail = adopted_fail or net_fail or ratio_fail
        return not validation_fail, {
            "disabled_by_validation": float(validation_fail),
            "disabled_by_validation_adopted": float(adopted_fail),
            "disabled_by_validation_net": float(net_fail),
            "disabled_by_validation_ratio": float(ratio_fail),
        }

    def _disabled_route_verifier_stats(self, threshold_stats: Dict[str, float]) -> Dict[str, float]:
        disabled_stats = dict(threshold_stats)
        disabled_stats.update(
            {
                "candidate": 0.0,
                "adopted": 0.0,
                "rescue": 0.0,
                "harm": 0.0,
                "net_rescue": 0.0,
                "candidate_rate": 0.0,
                "adopted_rate": 0.0,
                "harm_rate": 0.0,
                "rescue_harm_ratio": 0.0,
                "selected_threshold": 1.01,
                "disabled": 1.0,
            }
        )
        return disabled_stats

    def _aggregate_route_verifier_bin_stats(
        self,
        bin_stats: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        stats = self._default_route_verifier_stats()
        if not bin_stats:
            return stats
        stats["samples"] = float(sum(item.get("samples", 0.0) for item in bin_stats.values()))
        stats["candidate"] = float(sum(item.get("candidate", 0.0) for item in bin_stats.values()))
        stats["adopted"] = float(sum(item.get("adopted", 0.0) for item in bin_stats.values()))
        stats["rescue"] = float(sum(item.get("rescue", 0.0) for item in bin_stats.values()))
        stats["harm"] = float(sum(item.get("harm", 0.0) for item in bin_stats.values()))
        stats["net_rescue"] = float(stats["rescue"] - stats["harm"])
        stats["candidate_rate"] = stats["candidate"] / max(stats["samples"], 1.0)
        stats["adopted_rate"] = stats["adopted"] / max(stats["samples"], 1.0)
        stats["harm_rate"] = stats["harm"] / max(stats["samples"], 1.0)
        stats["rescue_harm_ratio"] = stats["rescue"] / max(stats["harm"], 1.0)
        enabled_thresholds = [
            float(item.get("selected_threshold", 1.01))
            for item in bin_stats.values()
            if float(item.get("disabled", 1.0)) <= 0.0
        ]
        stats["selected_threshold"] = sum(enabled_thresholds) / max(len(enabled_thresholds), 1) if enabled_thresholds else 1.01
        stats["disabled"] = float(not enabled_thresholds)
        stats["disabled_by_validation"] = float(not enabled_thresholds)
        stats["disabled_by_validation_adopted"] = float(
            sum(item.get("disabled_by_validation_adopted", 0.0) for item in bin_stats.values()) > 0.0
        )
        stats["disabled_by_validation_net"] = float(
            sum(item.get("disabled_by_validation_net", 0.0) for item in bin_stats.values()) > 0.0
        )
        stats["disabled_by_validation_ratio"] = float(
            sum(item.get("disabled_by_validation_ratio", 0.0) for item in bin_stats.values()) > 0.0
        )
        return stats

    def _build_two_stage_candidate_mask(
        self,
        client_id: str,
        error_probs: torch.Tensor,
        expert_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_thresholds = self._candidate_thresholds_for_batch(client_id, expert_features)
        candidate_mask = error_probs >= candidate_thresholds
        disable_guard = bool(getattr(self.config.inference, "router_candidate_disable_high_confidence_guard", True))
        if not disable_guard:
            guard = _high_confidence_guard(self.config)
            if guard < 1.0:
                candidate_mask = candidate_mask & expert_features[:, 0].lt(guard)
        return candidate_mask

    def _score_route_verifier(
        self,
        client_id: str,
        expert_logits: torch.Tensor,
        general_logits: torch.Tensor,
        error_probs: torch.Tensor,
    ) -> torch.Tensor:
        state = self.client_route_verifier_states.get(client_id)
        if state is None:
            return torch.zeros(expert_logits.size(0), device=expert_logits.device, dtype=expert_logits.dtype)
        features = _route_verifier_features_from_logits(expert_logits, general_logits, error_probs)
        return _predict_error_probabilities(state, features)

    def _route_verifier_uses_fusion(self) -> bool:
        mode = str(getattr(self.config.inference, "route_verifier_adoption_mode", "hard")).lower()
        return mode in {"fusion", "soft_fusion", "soft-fusion", "prob_mix", "prob-mix"}

    def _route_fusion_alpha(
        self,
        error_probs: torch.Tensor,
        verifier_scores: torch.Tensor,
        adopt_threshold: float | torch.Tensor,
        expert_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha_min = min(max(float(getattr(self.config.inference, "route_fusion_alpha_min", 0.35)), 0.0), 1.0)
        alpha_max = min(max(float(getattr(self.config.inference, "route_fusion_alpha_max", 0.85)), alpha_min), 1.0)
        alpha_fixed = min(max(float(getattr(self.config.inference, "route_fusion_alpha_fixed", 0.50)), 0.0), 1.0)
        source = str(getattr(self.config.inference, "route_fusion_alpha_source", "verifier")).lower()
        if source in {"fixed", "constant"}:
            return torch.full_like(error_probs, alpha_fixed)
        if source in {"error", "error_prob", "error-prob", "risk"}:
            base = error_probs.clamp(0.0, 1.0)
        elif source in {"mean", "average", "avg"}:
            verifier_base = verifier_scores.clamp(0.0, 1.0)
            base = 0.5 * (error_probs.clamp(0.0, 1.0) + verifier_base)
        else:
            if isinstance(adopt_threshold, torch.Tensor):
                threshold_tensor = adopt_threshold.to(device=verifier_scores.device, dtype=verifier_scores.dtype)
            else:
                threshold_tensor = torch.full_like(verifier_scores, float(adopt_threshold))
            denominator = (1.0 - threshold_tensor).clamp_min(1e-6)
            base = ((verifier_scores - threshold_tensor) / denominator).clamp(0.0, 1.0)
        alpha = alpha_min + ((alpha_max - alpha_min) * base)
        if self._route_fusion_confidence_alpha_enabled() and expert_confidence is not None:
            max_values = self._fusion_confidence_alpha_max_values(alpha_max)
            bins = self._confidence_bin_indices(expert_confidence.to(device=alpha.device, dtype=alpha.dtype))
            alpha_max_by_bin = torch.full_like(alpha, alpha_max)
            for bin_index, alpha_cap in enumerate(max_values):
                alpha_max_by_bin[bins.eq(int(bin_index))] = float(alpha_cap)
            alpha = torch.minimum(alpha, alpha_max_by_bin)
        return alpha

    def _predict_routed_two_stage_impl(self, client_id, images, indices, full_metadata: bool = False):
        expert_model = self.clients[client_id].expert_model
        predictor_state = self.client_predictor_states[client_id]

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            expert_logits = expert_model(images)
            expert_features = self._base_predictor_features(expert_logits)
            verifier_error_probs = _predict_error_probabilities(predictor_state, expert_features)
            error_probs, candidate_features = self._candidate_error_probabilities(
                client_id,
                expert_model,
                images,
                expert_logits,
                expert_features,
                verifier_error_probs,
            )
            expert_probs = torch.softmax(expert_logits, dim=1)
            expert_prediction = expert_logits.argmax(dim=1)
            candidate_mask = self._build_two_stage_candidate_mask(client_id, error_probs, candidate_features)

            predictions = expert_prediction.clone()
            route_types = ["expert"] * images.size(0)
            general_prediction = torch.full_like(expert_prediction, -1)
            verifier_scores = torch.zeros_like(error_probs)
            fusion_alpha = torch.zeros_like(error_probs)
            adopted_mask = torch.zeros_like(candidate_mask)
            candidate_count = int(candidate_mask.sum().item())
            candidate_thresholds = self._candidate_thresholds_for_batch(client_id, candidate_features)
            adopt_thresholds = self._route_verifier_thresholds_for_batch(client_id, expert_features)
            use_fusion = self._route_verifier_uses_fusion()

            if full_metadata:
                general_logits = self.general_model(images)
                general_prediction = general_logits.argmax(dim=1)
                if candidate_mask.any():
                    verifier_scores = self._score_route_verifier(client_id, expert_logits, general_logits, verifier_error_probs)
                    adopted_mask = candidate_mask & verifier_scores.ge(adopt_thresholds)
                    if adopted_mask.any():
                        if use_fusion:
                            alpha = self._route_fusion_alpha(
                                verifier_error_probs[adopted_mask],
                                verifier_scores[adopted_mask],
                                adopt_thresholds[adopted_mask],
                                expert_confidence=expert_features[adopted_mask, 0],
                            )
                            fusion_alpha[adopted_mask] = alpha
                            general_probs = torch.softmax(general_logits[adopted_mask], dim=1)
                            expert_probs_adopted = expert_probs[adopted_mask]
                            fused_probs = ((1.0 - alpha).unsqueeze(1) * expert_probs_adopted) + (
                                alpha.unsqueeze(1) * general_probs
                            )
                            predictions[adopted_mask] = fused_probs.argmax(dim=1)
                        else:
                            predictions[adopted_mask] = general_prediction[adopted_mask]
            elif candidate_mask.any():
                candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
                general_logits_candidate = self.general_model(images[candidate_mask])
                general_prediction[candidate_mask] = general_logits_candidate.argmax(dim=1)
                verifier_scores_candidate = self._score_route_verifier(
                    client_id,
                    expert_logits[candidate_mask],
                    general_logits_candidate,
                    verifier_error_probs[candidate_mask],
                )
                verifier_scores[candidate_mask] = verifier_scores_candidate
                candidate_adopt_thresholds = adopt_thresholds[candidate_indices]
                adopted_candidates = verifier_scores_candidate.ge(candidate_adopt_thresholds)
                if adopted_candidates.any():
                    adopted_indices = candidate_indices[adopted_candidates]
                    adopted_mask[adopted_indices] = True
                    if use_fusion:
                        alpha = self._route_fusion_alpha(
                            verifier_error_probs[adopted_indices],
                            verifier_scores_candidate[adopted_candidates],
                            candidate_adopt_thresholds[adopted_candidates],
                            expert_confidence=expert_features[adopted_indices, 0],
                        )
                        fusion_alpha[adopted_indices] = alpha
                        general_probs = torch.softmax(general_logits_candidate[adopted_candidates], dim=1)
                        expert_probs_adopted = expert_probs[adopted_indices]
                        fused_probs = ((1.0 - alpha).unsqueeze(1) * expert_probs_adopted) + (
                            alpha.unsqueeze(1) * general_probs
                        )
                        predictions[adopted_indices] = fused_probs.argmax(dim=1)
                    else:
                        predictions[adopted_indices] = general_prediction[adopted_indices]

            if candidate_mask.any():
                for sample_index in candidate_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_index] = "candidate"
            if adopted_mask.any():
                for sample_index in adopted_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_index] = "fusion" if use_fusion else "general"

            metadata = {
                "route_type": route_types,
                "expert_confidence": expert_probs.max(dim=1).values.detach().cpu().tolist(),
                "expert_margin": expert_features[:, 2].detach().cpu().tolist(),
                "expert_entropy": expert_features[:, 1].detach().cpu().tolist(),
                "error_prob": error_probs.detach().cpu().tolist(),
                "verifier_error_prob": verifier_error_probs.detach().cpu().tolist(),
                "error_threshold": candidate_thresholds.detach().cpu().tolist(),
                "expert_pred": expert_prediction.detach().cpu().tolist(),
                "route_candidate": candidate_mask.detach().cpu().to(torch.int64).tolist(),
                "route_adopted": adopted_mask.detach().cpu().to(torch.int64).tolist(),
                "route_verifier_score": verifier_scores.detach().cpu().tolist(),
                "route_verifier_threshold": adopt_thresholds.detach().cpu().tolist(),
                "route_fusion_alpha": fusion_alpha.detach().cpu().tolist(),
                "route_confidence_bin": self._confidence_bin_indices(expert_features[:, 0]).detach().cpu().tolist(),
                "route_adoption_mode": ["fusion" if use_fusion else "hard"] * images.size(0),
            }
            if full_metadata:
                metadata["general_pred"] = general_prediction.detach().cpu().tolist()
            return predictions, candidate_count, metadata

    def _predict_routed_impl(self, client_id, images, indices, full_metadata: bool = False):
        if self._two_stage_routing_enabled():
            return self._predict_routed_two_stage_impl(client_id, images, indices, full_metadata=full_metadata)

        expert_model = self.clients[client_id].expert_model
        predictor_state = self.client_predictor_states[client_id]
        threshold = float(self.client_error_thresholds.get(client_id, getattr(self.config.inference, "error_predictor_threshold", 0.5)))

        expert_model.eval()
        self.general_model.eval()
        with torch.no_grad():
            expert_logits = expert_model(images)
            expert_features = self._base_predictor_features(expert_logits)
            error_probs = _predict_error_probabilities(predictor_state, expert_features)
            expert_probs = torch.softmax(expert_logits, dim=1)
            expert_prediction = expert_logits.argmax(dim=1)
            fallback_mask = _build_error_fallback_mask(error_probs, expert_features, threshold, self.config)
            fallback_mask = self._apply_route_group_filter(
                client_id,
                fallback_mask,
                expert_prediction,
                error_probs,
                threshold,
                expert_features,
            )

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

    def _route_gain_filter_enabled(self) -> bool:
        return bool(getattr(self.config.federated, "route_disable_when_no_gain", False))

    def _route_group_filter_enabled(self) -> bool:
        mode = str(getattr(self.config.federated, "router_group_mode", "none")).lower()
        return mode in {"predicted_class", "class", "predicted-class"}

    def _route_group_filter_strategy(self) -> str:
        strategy = str(getattr(self.config.federated, "router_group_filter_strategy", "blocklist")).lower()
        if strategy in {"none", "off"}:
            return "none"
        if strategy in {"threshold", "threshold_only", "threshold-only", "fpr", "fpr_constrained"}:
            return "threshold"
        if strategy in {"negative_net", "negative-net", "net_negative", "net-negative", "net_only", "net-only"}:
            return "negative_net"
        if strategy in {"allowlist", "positive", "positive_allowlist", "positive-allowlist"}:
            return "allowlist"
        return "blocklist"

    def _apply_route_group_filter(
        self,
        client_id: str,
        fallback_mask: torch.Tensor,
        expert_prediction: torch.Tensor,
        error_probs: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._route_group_filter_enabled():
            return fallback_mask

        filtered = fallback_mask.clone()
        group_thresholds = self.client_route_group_thresholds.get(client_id, {})
        if group_thresholds and error_probs is not None and threshold is not None and features is not None:
            base_threshold = float(threshold)
            for class_id, group_threshold in group_thresholds.items():
                group_threshold = float(group_threshold)
                if group_threshold >= base_threshold:
                    continue
                class_mask = expert_prediction.eq(int(class_id))
                group_mask = _build_error_fallback_mask(error_probs, features, group_threshold, self.config)
                filtered = filtered | (class_mask & group_mask)

        if not filtered.any():
            return filtered

        fallback_to_client = bool(getattr(self.config.federated, "router_group_fallback_to_client", True))
        strategy = self._route_group_filter_strategy()
        if strategy == "allowlist":
            allowlist = self.client_route_group_allowlists.get(client_id)
            if allowlist is None or not allowlist:
                return filtered if fallback_to_client else torch.zeros_like(filtered)
            allowed = torch.zeros_like(fallback_mask)
            for class_id in allowlist:
                allowed = allowed | expert_prediction.eq(int(class_id))
            filtered = filtered & allowed
        elif strategy in {"blocklist", "negative_net"}:
            blocklist = self.client_route_group_blocklists.get(client_id)
            if blocklist is None or not blocklist:
                filtered = filtered if fallback_to_client else torch.zeros_like(filtered)
            else:
                blocked = torch.zeros_like(fallback_mask)
                for class_id in blocklist:
                    blocked = blocked | expert_prediction.eq(int(class_id))
                filtered = filtered & (~blocked)

        boosts = self.client_route_group_threshold_boosts.get(client_id, {})
        if not boosts or error_probs is None or threshold is None or not filtered.any():
            return filtered

        boosted_allowed = torch.ones_like(filtered)
        base_threshold = float(threshold)
        for class_id, boost in boosts.items():
            boost_value = float(boost)
            if boost_value <= 0.0:
                continue
            class_mask = expert_prediction.eq(int(class_id))
            boosted_allowed = boosted_allowed & (~class_mask | error_probs.ge(min(base_threshold + boost_value, 1.01)))
        return filtered & boosted_allowed

    @staticmethod
    def _default_route_group_stats(class_id: int) -> Dict[str, float]:
        return {
            "class_id": float(class_id),
            "enabled": 0.0,
            "blocked": 0.0,
            "samples": 0.0,
            "invoked": 0.0,
            "invocation_rate": 0.0,
            "expert_acc": 0.0,
            "routed_acc": 0.0,
            "route_gain": 0.0,
            "expert_errors": 0.0,
            "actual_positive": 0.0,
            "actual_negative": 0.0,
            "true_positive": 0.0,
            "false_positive": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0,
            "rescue": 0.0,
            "harm": 0.0,
            "net_rescue": 0.0,
            "harm_rate": 0.0,
            "selected_threshold": 0.0,
            "threshold_lowered": 0.0,
            "selected_threshold_delta": 0.0,
            "selected_invoked": 0.0,
            "selected_true_positive": 0.0,
            "selected_false_positive": 0.0,
            "selected_precision": 0.0,
            "selected_recall": 0.0,
            "selected_fpr": 0.0,
            "threshold_boost": 0.0,
            "disabled_by_min_support": 0.0,
            "disabled_by_min_invoked": 0.0,
            "disabled_by_boost_min_support": 0.0,
            "disabled_by_boost_min_invoked": 0.0,
            "disabled_by_nonpositive_net": 0.0,
            "disabled_by_negative_net": 0.0,
        }

    def _calibrate_client_route_groups(
        self, client_id: str, threshold: float
    ) -> Tuple[List[int], List[int], Dict[int, Dict[str, float]]]:
        if not self._route_group_filter_enabled() or threshold > 1.0:
            return [], [], {}

        dataset = self.client_calibration_datasets.get(client_id)
        if dataset is None or len(dataset) == 0:
            return [], [], {}

        predictor_state = self.client_predictor_states[client_id]
        expert_model = self.clients[client_id].expert_model
        expert_model.eval()

        group_stats: Dict[int, Dict[str, float]] = {}
        group_scores: Dict[int, List[float]] = {}
        group_labels: Dict[int, List[int]] = {}
        group_eligible: Dict[int, List[int]] = {}
        loader = self.data_module.make_loader(dataset, shuffle=False)
        guard = _high_confidence_guard(self.config)
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                expert_logits = expert_model(images)
                expert_features = _predictor_features_from_logits(expert_logits)
                error_probs = _predict_error_probabilities(predictor_state, expert_features)
                expert_prediction = expert_logits.argmax(dim=1)
                fallback_mask = _build_error_fallback_mask(error_probs, expert_features, threshold, self.config)
                expert_error = expert_prediction.ne(targets_device)
                eligible_mask = torch.ones_like(expert_error, dtype=torch.bool)
                if guard < 1.0:
                    eligible_mask = expert_features[:, 0] < guard

                for sample_index in range(images.size(0)):
                    class_id = int(expert_prediction[sample_index].item())
                    stats = group_stats.setdefault(class_id, self._default_route_group_stats(class_id))
                    invoked = bool(fallback_mask[sample_index].item())
                    error = bool(expert_error[sample_index].item())
                    eligible = bool(eligible_mask[sample_index].item())
                    group_scores.setdefault(class_id, []).append(float(error_probs[sample_index].item()))
                    group_labels.setdefault(class_id, []).append(int(error))
                    group_eligible.setdefault(class_id, []).append(int(eligible))
                    stats["samples"] += 1.0
                    stats["invoked"] += float(invoked)
                    stats["expert_errors"] += float(error)
                    stats["actual_positive"] += float(error and eligible)
                    stats["actual_negative"] += float((not error) and eligible)
                    if invoked and error:
                        stats["true_positive"] += 1.0
                        stats["rescue"] += 1.0
                    if invoked and not error:
                        stats["false_positive"] += 1.0
                        stats["harm"] += 1.0

        min_support = max(int(getattr(self.config.federated, "router_group_min_support", 20)), 1)
        min_invoked = max(int(getattr(self.config.federated, "router_group_min_invoked", 5)), 1)
        require_positive_net = bool(getattr(self.config.federated, "router_group_require_positive_net", True))
        threshold_mode = str(getattr(self.config.federated, "router_group_threshold_mode", "none")).lower()
        threshold_modes = {"fpr", "fpr_constrained", "fpr-constrained", "neyman_pearson", "neyman-pearson"}
        boost_modes = {"boost", "threshold_boost", "threshold-boost", "harm_boost", "harm-boost"}
        group_threshold_min_support = max(
            int(getattr(self.config.federated, "router_group_threshold_min_support", 20)),
            1,
        )
        group_threshold_min_errors = max(
            int(getattr(self.config.federated, "router_group_threshold_min_errors", 3)),
            1,
        )
        group_threshold_min_positive = max(
            int(getattr(self.config.federated, "router_group_threshold_min_predicted_positive", 3)),
            1,
        )
        group_threshold_target_fpr = min(
            max(float(getattr(self.config.federated, "router_group_threshold_target_fpr", 0.10)), 0.0),
            1.0,
        )
        group_threshold_max_fp = int(getattr(self.config.federated, "router_group_threshold_max_false_positive", 2))
        min_threshold = float(getattr(self.config.inference, "routing_error_min_threshold", 0.0))
        max_threshold = float(getattr(self.config.inference, "routing_error_max_threshold", 1.0))
        threshold_boost = max(float(getattr(self.config.federated, "router_group_threshold_boost", 0.0)), 0.0)
        boost_min_support = max(int(getattr(self.config.federated, "router_group_boost_min_support", 20)), 1)
        boost_min_invoked = max(int(getattr(self.config.federated, "router_group_boost_min_invoked", 5)), 1)
        boost_max_net_rescue = float(getattr(self.config.federated, "router_group_boost_max_net_rescue", 1.0))
        boost_min_harm = max(int(getattr(self.config.federated, "router_group_boost_min_harm", 1)), 1)
        strategy = self._route_group_filter_strategy()
        allowlist: List[int] = []
        blocklist: List[int] = []
        for class_id, stats in group_stats.items():
            samples = max(float(stats.get("samples", 0.0)), 1.0)
            invoked = float(stats.get("invoked", 0.0))
            actual_positive = float(stats.get("actual_positive", 0.0))
            actual_negative = float(stats.get("actual_negative", 0.0))
            true_positive = float(stats.get("true_positive", 0.0))
            false_positive = float(stats.get("false_positive", 0.0))
            net_rescue = true_positive - false_positive
            error_rate = float(stats.get("expert_errors", 0.0)) / samples
            expert_acc = 1.0 - error_rate
            routed_acc = expert_acc
            precision = true_positive / max(invoked, 1.0)
            recall = true_positive / max(actual_positive, 1.0)
            fpr = false_positive / max(actual_negative, 1.0)
            harm_rate = false_positive / max(invoked, 1.0)
            min_support_fail = samples < min_support
            min_invoked_fail = invoked < min_invoked
            boost_min_support_fail = samples < boost_min_support
            boost_min_invoked_fail = invoked < boost_min_invoked
            negative_net_fail = net_rescue < 0.0
            selected_threshold = float(threshold)
            threshold_lowered = 0.0
            selected_stats: Dict[str, float] = {}
            if threshold_mode in threshold_modes:
                selected_threshold_optional, selected_stats = _select_fpr_constrained_group_threshold(
                    scores=torch.tensor(group_scores.get(class_id, []), dtype=torch.float32),
                    labels=torch.tensor(group_labels.get(class_id, []), dtype=torch.bool),
                    eligible=torch.tensor(group_eligible.get(class_id, []), dtype=torch.bool),
                    base_threshold=float(threshold),
                    target_fpr=group_threshold_target_fpr,
                    max_false_positive=group_threshold_max_fp,
                    min_predicted_positive=group_threshold_min_positive,
                    min_support=group_threshold_min_support,
                    min_errors=group_threshold_min_errors,
                    min_threshold=min_threshold,
                    max_threshold=max_threshold,
                )
                if selected_threshold_optional is not None:
                    selected_threshold = float(selected_threshold_optional)
                    threshold_lowered = 1.0
            group_threshold_boost = (
                threshold_boost
                if threshold_boost > 0.0
                and (threshold_mode in boost_modes or threshold_mode not in threshold_modes)
                and not boost_min_support_fail
                and not boost_min_invoked_fail
                and false_positive >= boost_min_harm
                and net_rescue <= boost_max_net_rescue
                else 0.0
            )
            if strategy == "allowlist":
                nonpositive_net_fail = require_positive_net and net_rescue <= 0.0
                enabled = not (min_support_fail or min_invoked_fail or nonpositive_net_fail)
                blocked = not enabled
            elif strategy in {"none", "threshold"}:
                nonpositive_net_fail = False
                blocked = False
                enabled = True
            elif strategy == "negative_net":
                nonpositive_net_fail = False
                blocked = (not min_support_fail) and (not min_invoked_fail) and negative_net_fail
                enabled = not blocked
            else:
                nonpositive_net_fail = False
                fpr_fail = (false_positive > group_threshold_max_fp >= 0) or (fpr > group_threshold_target_fpr)
                blocked = (not min_support_fail) and (not min_invoked_fail) and fpr_fail
                enabled = not blocked
            stats.update(
                {
                    "enabled": float(enabled),
                    "blocked": float(blocked),
                    "invocation_rate": float(invoked / samples),
                    "expert_acc": float(expert_acc),
                    "routed_acc": float(routed_acc),
                    "route_gain": 0.0,
                    "precision": float(precision),
                    "recall": float(recall),
                    "fpr": float(fpr),
                    "net_rescue": float(net_rescue),
                    "harm_rate": float(harm_rate),
                    "selected_threshold": float(selected_threshold),
                    "threshold_lowered": float(threshold_lowered),
                    "selected_threshold_delta": float(float(threshold) - selected_threshold),
                    "selected_invoked": float(selected_stats.get("selected_invoked", 0.0)),
                    "selected_true_positive": float(selected_stats.get("selected_true_positive", 0.0)),
                    "selected_false_positive": float(selected_stats.get("selected_false_positive", 0.0)),
                    "selected_precision": float(selected_stats.get("selected_precision", 0.0)),
                    "selected_recall": float(selected_stats.get("selected_recall", 0.0)),
                    "selected_fpr": float(selected_stats.get("selected_fpr", 0.0)),
                    "threshold_boost": float(group_threshold_boost),
                    "disabled_by_min_support": float(min_support_fail),
                    "disabled_by_min_invoked": float(min_invoked_fail),
                    "disabled_by_boost_min_support": float(boost_min_support_fail),
                    "disabled_by_boost_min_invoked": float(boost_min_invoked_fail),
                    "disabled_by_nonpositive_net": float(nonpositive_net_fail),
                    "disabled_by_negative_net": float(negative_net_fail),
                }
            )
            if enabled:
                allowlist.append(int(class_id))
            if blocked:
                blocklist.append(int(class_id))

        return sorted(allowlist), sorted(blocklist), group_stats

    def _refresh_route_group_filters(self, client_ids: Optional[Sequence[str]] = None, context: str = "refresh") -> None:
        if not self._route_group_filter_enabled():
            self.client_route_group_stats = {}
            self.client_route_group_allowlists = {}
            self.client_route_group_blocklists = {}
            self.client_route_group_thresholds = {}
            self.client_route_group_threshold_boosts = {}
            return

        target_client_ids = list(client_ids) if client_ids is not None else list(self.clients.keys())
        enabled_classes = 0
        disabled_classes = 0
        lowered_classes = 0
        for client_id in target_client_ids:
            threshold = float(
                self.client_error_thresholds.get(
                    client_id,
                    getattr(self.config.inference, "error_predictor_threshold", 0.5),
                )
            )
            allowlist, blocklist, group_stats = self._calibrate_client_route_groups(client_id, threshold)
            self.client_route_group_allowlists[client_id] = allowlist
            self.client_route_group_blocklists[client_id] = blocklist
            self.client_route_group_stats[client_id] = group_stats
            self.client_route_group_thresholds[client_id] = {
                int(class_id): float(stats.get("selected_threshold", threshold))
                for class_id, stats in group_stats.items()
                if float(stats.get("threshold_lowered", 0.0)) > 0.0
            }
            self.client_route_group_threshold_boosts[client_id] = {
                int(class_id): float(stats.get("threshold_boost", 0.0))
                for class_id, stats in group_stats.items()
                if float(stats.get("threshold_boost", 0.0)) > 0.0
            }
            enabled_classes += sum(1 for stats in group_stats.values() if stats.get("enabled", 0.0) > 0.0)
            disabled_classes += sum(1 for stats in group_stats.values() if stats.get("blocked", 0.0) > 0.0)
            lowered_classes += sum(1 for stats in group_stats.values() if stats.get("threshold_lowered", 0.0) > 0.0)

        LOGGER.info(
            "%s route group filter %s | strategy=%s | clients=%d"
            " | enabled_classes=%d | blocked_classes=%d | lowered_classes=%d",
            self.algorithm_name,
            context,
            self._route_group_filter_strategy(),
            len(target_client_ids),
            enabled_classes,
            disabled_classes,
            lowered_classes,
        )
        self._log_route_group_filter_details(context, target_client_ids)

    def _log_route_group_filter_details(self, context: str, client_ids: Optional[Sequence[str]] = None) -> None:
        if not self._route_group_filter_enabled() or not self.client_route_group_stats:
            return
        target_client_ids = list(client_ids) if client_ids is not None else sorted(self.client_route_group_stats.keys())
        for client_id in target_client_ids:
            for class_id, stats in sorted(self.client_route_group_stats.get(client_id, {}).items()):
                if stats.get("invoked", 0.0) <= 0.0 and stats.get("samples", 0.0) < float(
                    getattr(self.config.federated, "router_group_min_support", 20)
                ):
                    continue
                LOGGER.info(
                    "%s route group filter %s client=%s class=%d"
                    " | enabled=%d | blocked=%d | n=%d | invoke=%d | invoke_rate=%.4f"
                    " | expert_acc=%.4f | routed_acc=%.4f | gain=%.4f"
                    " | ep_p=%.4f | ep_r=%.4f | ep_fpr=%.4f"
                    " | rescue=%d | harm=%d | net=%d | harm_rate=%.4f | threshold_boost=%.4f"
                    " | threshold_lowered=%d | selected_thr=%.4f | selected_delta=%.4f"
                    " | selected_p=%.4f | selected_r=%.4f | selected_fpr=%.4f"
                    " | min_support_fail=%d | min_invoked_fail=%d"
                    " | boost_min_support_fail=%d | boost_min_invoked_fail=%d"
                    " | nonpositive_net_fail=%d | negative_net=%d",
                    self.algorithm_name,
                    context,
                    client_id,
                    int(class_id),
                    int(stats.get("enabled", 0.0) > 0.0),
                    int(stats.get("blocked", 0.0) > 0.0),
                    int(stats.get("samples", 0.0)),
                    int(stats.get("invoked", 0.0)),
                    float(stats.get("invocation_rate", 0.0)),
                    float(stats.get("expert_acc", 0.0)),
                    float(stats.get("routed_acc", 0.0)),
                    float(stats.get("route_gain", 0.0)),
                    float(stats.get("precision", 0.0)),
                    float(stats.get("recall", 0.0)),
                    float(stats.get("fpr", 0.0)),
                    int(stats.get("rescue", 0.0)),
                    int(stats.get("harm", 0.0)),
                    int(stats.get("net_rescue", 0.0)),
                    float(stats.get("harm_rate", 0.0)),
                    float(stats.get("threshold_boost", 0.0)),
                    int(stats.get("threshold_lowered", 0.0) > 0.0),
                    float(stats.get("selected_threshold", 0.0)),
                    float(stats.get("selected_threshold_delta", 0.0)),
                    float(stats.get("selected_precision", 0.0)),
                    float(stats.get("selected_recall", 0.0)),
                    float(stats.get("selected_fpr", 0.0)),
                    int(stats.get("disabled_by_min_support", 0.0) > 0.0),
                    int(stats.get("disabled_by_min_invoked", 0.0) > 0.0),
                    int(stats.get("disabled_by_boost_min_support", 0.0) > 0.0),
                    int(stats.get("disabled_by_boost_min_invoked", 0.0) > 0.0),
                    int(stats.get("disabled_by_nonpositive_net", 0.0) > 0.0),
                    int(stats.get("disabled_by_negative_net", 0.0) > 0.0),
                )

    def _calibrate_client_route_gain(self, client_id: str, threshold: float) -> Tuple[float, Dict[str, float]]:
        threshold = float(threshold)
        stats: Dict[str, float] = {
            "raw_threshold": threshold,
            "filtered_threshold": threshold,
            "disabled_by_gain": 0.0,
            "disabled_by_min_invoked": 0.0,
            "disabled_by_min_gain": 0.0,
            "disabled_by_nonpositive_net": 0.0,
            "samples": 0.0,
            "invoked": 0.0,
            "invocation_rate": 0.0,
            "expert_acc": 0.0,
            "routed_acc": 0.0,
            "route_gain": 0.0,
            "rescue": 0.0,
            "harm": 0.0,
            "net_rescue": 0.0,
        }
        if not self._route_gain_filter_enabled() or threshold > 1.0:
            return threshold, stats

        dataset = self.client_calibration_datasets.get(client_id)
        if dataset is None or len(dataset) == 0:
            return threshold, stats

        predictor_state = self.client_predictor_states[client_id]
        expert_model = self.clients[client_id].expert_model
        expert_model.eval()
        self.general_model.eval()

        total = 0
        invoked = 0
        expert_correct_total = 0
        routed_correct_total = 0
        rescue = 0
        harm = 0
        loader = self.data_module.make_loader(dataset, shuffle=False)
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)

                expert_logits = expert_model(images)
                expert_features = _predictor_features_from_logits(expert_logits)
                error_probs = _predict_error_probabilities(predictor_state, expert_features)
                fallback_mask = _build_error_fallback_mask(error_probs, expert_features, threshold, self.config)
                expert_correct = expert_logits.argmax(dim=1).eq(targets_device)
                routed_correct = expert_correct.clone()

                if fallback_mask.any():
                    general_logits = self.general_model(images[fallback_mask])
                    general_correct = general_logits.argmax(dim=1).eq(targets_device[fallback_mask])
                    routed_correct[fallback_mask] = general_correct
                    expert_fallback_correct = expert_correct[fallback_mask]
                    rescue += int((~expert_fallback_correct & general_correct).sum().item())
                    harm += int((expert_fallback_correct & ~general_correct).sum().item())

                total += int(images.size(0))
                invoked += int(fallback_mask.sum().item())
                expert_correct_total += int(expert_correct.sum().item())
                routed_correct_total += int(routed_correct.sum().item())

        expert_acc = expert_correct_total / max(total, 1)
        routed_acc = routed_correct_total / max(total, 1)
        route_gain = routed_acc - expert_acc
        stats.update(
            {
                "samples": float(total),
                "invoked": float(invoked),
                "invocation_rate": float(invoked / max(total, 1)),
                "expert_acc": float(expert_acc),
                "routed_acc": float(routed_acc),
                "route_gain": float(route_gain),
                "rescue": float(rescue),
                "harm": float(harm),
                "net_rescue": float(rescue - harm),
            }
        )

        min_gain = float(getattr(self.config.federated, "route_min_gain", 0.0))
        default_min_invoked = int(getattr(self.config.inference, "error_predictor_min_predicted_positive", 3))
        min_invoked = max(int(getattr(self.config.federated, "route_gain_filter_min_invoked", default_min_invoked)), 1)
        require_positive_net = bool(getattr(self.config.federated, "route_gain_filter_require_positive_net", True))
        net_rescue = rescue - harm
        min_invoked_fail = invoked < min_invoked
        min_gain_fail = route_gain <= min_gain
        nonpositive_net_fail = require_positive_net and net_rescue <= 0
        if min_invoked_fail or min_gain_fail or nonpositive_net_fail:
            stats["disabled_by_gain"] = 1.0
            stats["disabled_by_min_invoked"] = float(min_invoked_fail)
            stats["disabled_by_min_gain"] = float(min_gain_fail)
            stats["disabled_by_nonpositive_net"] = float(nonpositive_net_fail)
            stats["filtered_threshold"] = 1.01
            return 1.01, stats

        return threshold, stats

    def _log_route_gain_filter_details(self, context: str, client_ids: Optional[Sequence[str]] = None) -> None:
        if not self._route_gain_filter_enabled() or not self.client_route_gain_stats:
            return
        target_client_ids = list(client_ids) if client_ids is not None else sorted(self.client_route_gain_stats.keys())
        for client_id in target_client_ids:
            stats = self.client_route_gain_stats.get(client_id)
            if not stats:
                continue
            LOGGER.info(
                "%s route gain filter %s client=%s"
                " | raw_thr=%.4f | final_thr=%.4f | disabled=%d"
                " | min_invoked_fail=%d | min_gain_fail=%d | nonpositive_net_fail=%d"
                " | n=%d | invoke=%d | invoke_rate=%.4f"
                " | expert_acc=%.4f | routed_acc=%.4f | gain=%.4f"
                " | rescue=%d | harm=%d | net=%d",
                self.algorithm_name,
                context,
                client_id,
                float(stats.get("raw_threshold", 0.0)),
                float(stats.get("filtered_threshold", 0.0)),
                int(stats.get("disabled_by_gain", 0.0) > 0.0),
                int(stats.get("disabled_by_min_invoked", 0.0) > 0.0),
                int(stats.get("disabled_by_min_gain", 0.0) > 0.0),
                int(stats.get("disabled_by_nonpositive_net", 0.0) > 0.0),
                int(stats.get("samples", 0.0)),
                int(stats.get("invoked", 0.0)),
                float(stats.get("invocation_rate", 0.0)),
                float(stats.get("expert_acc", 0.0)),
                float(stats.get("routed_acc", 0.0)),
                float(stats.get("route_gain", 0.0)),
                int(stats.get("rescue", 0.0)),
                int(stats.get("harm", 0.0)),
                int(stats.get("net_rescue", 0.0)),
            )

    def _apply_route_gain_filter_to_thresholds(self, client_ids: Optional[Sequence[str]] = None) -> None:
        if not self._route_gain_filter_enabled():
            return
        target_client_ids = list(client_ids) if client_ids is not None else list(self.clients.keys())
        disabled = 0
        gains: List[float] = []
        for client_id in target_client_ids:
            raw_threshold = float(
                self.client_raw_error_thresholds.get(
                    client_id,
                    self.client_error_thresholds.get(
                        client_id,
                        getattr(self.config.inference, "error_predictor_threshold", 0.5),
                    ),
                )
            )
            filtered_threshold, stats = self._calibrate_client_route_gain(client_id, raw_threshold)
            self.client_error_thresholds[client_id] = filtered_threshold
            self.client_route_gain_stats[client_id] = stats
            disabled += int(stats.get("disabled_by_gain", 0.0) > 0.0)
            if stats.get("samples", 0.0) > 0.0:
                gains.append(float(stats.get("route_gain", 0.0)))

        if target_client_ids:
            LOGGER.info(
                "%s route gain filter | clients=%d | disabled=%d | mean_gain=%.4f",
                self.algorithm_name,
                len(target_client_ids),
                disabled,
                sum(gains) / max(len(gains), 1),
            )
        self._refresh_route_group_filters(target_client_ids, context="route_gain")

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
            general_route_types=("general", "fusion"),
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
                    expert_features = self._base_predictor_features(expert_logits)
                    base_error_probs = _predict_error_probabilities(predictor_state, expert_features)
                    if self._two_stage_routing_enabled():
                        error_probs, features = self._candidate_error_probabilities(
                            client_id,
                            expert_model,
                            images,
                            expert_logits,
                            expert_features,
                            base_error_probs,
                        )
                    else:
                        features = expert_features
                        error_probs = base_error_probs
                    expert_predictions = expert_logits.argmax(dim=1)
                    batch_error_labels = expert_predictions.ne(targets_device)
                    if self._two_stage_routing_enabled():
                        batch_predicted_positive = self._build_two_stage_candidate_mask(client_id, error_probs, features)
                    else:
                        batch_predicted_positive = _build_error_fallback_mask(error_probs, features, threshold, self.config)
                        batch_predicted_positive = self._apply_route_group_filter(
                            client_id,
                            batch_predicted_positive,
                            expert_predictions,
                            error_probs,
                            threshold,
                            features,
                        )

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

    def _router_diagnostics_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "router_diagnostics_enabled", False))

    def _router_diagnostic_confidence_edges(self) -> List[float]:
        raw_edges = getattr(self.config.inference, "router_diagnostics_confidence_bins", [0.5, 0.7, 0.85, 0.95])
        try:
            edges = [float(item) for item in raw_edges]
        except TypeError:
            edges = [0.5, 0.7, 0.85, 0.95]
        return sorted({edge for edge in edges if 0.0 < edge < 1.0})

    @staticmethod
    def _router_diagnostic_bucket() -> Dict[str, float]:
        return {
            "samples": 0.0,
            "expert_errors": 0.0,
            "expert_correct": 0.0,
            "predicted_positive": 0.0,
            "true_positive": 0.0,
            "false_positive": 0.0,
            "false_negative": 0.0,
            "true_negative": 0.0,
            "rescue": 0.0,
            "harm": 0.0,
            "guard_blocked": 0.0,
            "guard_blocked_errors": 0.0,
        }

    @staticmethod
    def _router_confidence_bin(confidence: float, edges: Sequence[float]) -> str:
        lower = 0.0
        for edge in edges:
            if confidence < edge:
                return f"[{lower:.2f},{edge:.2f})"
            lower = edge
        return f"[{lower:.2f},1.00]"

    @staticmethod
    def _update_router_diagnostic_bucket(
        bucket: Dict[str, float],
        *,
        expert_error: bool,
        predicted_positive: bool,
        general_correct: bool,
        guard_blocked: bool,
    ) -> None:
        bucket["samples"] += 1.0
        bucket["expert_errors"] += float(expert_error)
        bucket["expert_correct"] += float(not expert_error)
        bucket["predicted_positive"] += float(predicted_positive)
        bucket["guard_blocked"] += float(guard_blocked)
        bucket["guard_blocked_errors"] += float(guard_blocked and expert_error)
        if predicted_positive and expert_error:
            bucket["true_positive"] += 1.0
            bucket["rescue"] += float(general_correct)
        elif predicted_positive and not expert_error:
            bucket["false_positive"] += 1.0
            bucket["harm"] += float(not general_correct)
        elif not predicted_positive and expert_error:
            bucket["false_negative"] += 1.0
        else:
            bucket["true_negative"] += 1.0

    @staticmethod
    def _router_bucket_rates(bucket: Dict[str, float]) -> Dict[str, float]:
        predicted = float(bucket.get("predicted_positive", 0.0))
        errors = float(bucket.get("expert_errors", 0.0))
        correct = float(bucket.get("expert_correct", 0.0))
        samples = float(bucket.get("samples", 0.0))
        true_positive = float(bucket.get("true_positive", 0.0))
        false_positive = float(bucket.get("false_positive", 0.0))
        return {
            "error_rate": errors / max(samples, 1.0),
            "predicted_positive_rate": predicted / max(samples, 1.0),
            "precision": true_positive / max(predicted, 1.0),
            "recall": true_positive / max(errors, 1.0),
            "fpr": false_positive / max(correct, 1.0),
            "net_rescue": float(bucket.get("rescue", 0.0)) - float(bucket.get("harm", 0.0)),
        }

    def _log_router_diagnostic_bucket(self, label: str, key: str, bucket: Dict[str, float]) -> None:
        rates = self._router_bucket_rates(bucket)
        LOGGER.info(
            "%s router diag %s=%s"
            " | n=%d | errors=%d | err_rate=%.4f"
            " | pred=%d | pred_rate=%.4f"
            " | tp=%d | fp=%d | fn=%d | tn=%d"
            " | p=%.4f | r=%.4f | fpr=%.4f"
            " | rescue=%d | harm=%d | net=%d"
            " | guard_blocked=%d | guard_blocked_errors=%d",
            self.algorithm_name,
            label,
            key,
            int(bucket.get("samples", 0.0)),
            int(bucket.get("expert_errors", 0.0)),
            rates["error_rate"],
            int(bucket.get("predicted_positive", 0.0)),
            rates["predicted_positive_rate"],
            int(bucket.get("true_positive", 0.0)),
            int(bucket.get("false_positive", 0.0)),
            int(bucket.get("false_negative", 0.0)),
            int(bucket.get("true_negative", 0.0)),
            rates["precision"],
            rates["recall"],
            rates["fpr"],
            int(bucket.get("rescue", 0.0)),
            int(bucket.get("harm", 0.0)),
            int(rates["net_rescue"]),
            int(bucket.get("guard_blocked", 0.0)),
            int(bucket.get("guard_blocked_errors", 0.0)),
        )

    def _evaluate_router_diagnostics(self) -> Dict[str, float]:
        if not self._router_diagnostics_enabled():
            return {}

        min_samples = max(int(getattr(self.config.inference, "router_diagnostics_min_samples", 20)), 1)
        include_classes = bool(getattr(self.config.inference, "router_diagnostics_include_classes", True))
        confidence_edges = self._router_diagnostic_confidence_edges()
        guard = _high_confidence_guard(self.config)

        summary = self._router_diagnostic_bucket()
        confidence_buckets: Dict[Tuple[str, str], Dict[str, float]] = {}
        class_buckets: Dict[Tuple[str, int], Dict[str, float]] = {}
        high_confidence_samples = 0
        high_confidence_errors = 0
        high_confidence_true_positive = 0

        self.general_model.eval()
        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                predictor_state = self.client_predictor_states.get(client_id)
                if predictor_state is None:
                    continue
                threshold = float(
                    self.client_error_thresholds.get(
                        client_id,
                        getattr(self.config.inference, "error_predictor_threshold", 0.5),
                    )
                )
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()
                loader = self.data_module.make_loader(dataset, shuffle=False)

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)
                    expert_logits = expert_model(images)
                    features = self._base_predictor_features(expert_logits)
                    error_probs = _predict_error_probabilities(predictor_state, features)
                    raw_positive = error_probs >= threshold
                    expert_predictions = expert_logits.argmax(dim=1)
                    base_fallback_mask = _build_error_fallback_mask(error_probs, features, threshold, self.config)
                    guard_blocked = raw_positive & ~base_fallback_mask
                    fallback_mask = self._apply_route_group_filter(
                        client_id,
                        base_fallback_mask,
                        expert_predictions,
                        error_probs,
                        threshold,
                        features,
                    )
                    expert_error = expert_predictions.ne(targets_device)
                    confidence = features[:, 0]

                    general_correct = torch.zeros_like(expert_error, dtype=torch.bool)
                    if fallback_mask.any():
                        general_logits = self.general_model(images[fallback_mask])
                        general_correct[fallback_mask] = general_logits.argmax(dim=1).eq(targets_device[fallback_mask])

                    for sample_index in range(images.size(0)):
                        sample_confidence = float(confidence[sample_index].item())
                        sample_error = bool(expert_error[sample_index].item())
                        sample_predicted = bool(fallback_mask[sample_index].item())
                        sample_general_correct = bool(general_correct[sample_index].item())
                        sample_guard_blocked = bool(guard_blocked[sample_index].item())
                        predicted_class = int(expert_predictions[sample_index].item())

                        if guard < 1.0 and sample_confidence >= guard:
                            high_confidence_samples += 1
                            high_confidence_errors += int(sample_error)
                            high_confidence_true_positive += int(sample_predicted and sample_error)

                        update_kwargs = {
                            "expert_error": sample_error,
                            "predicted_positive": sample_predicted,
                            "general_correct": sample_general_correct,
                            "guard_blocked": sample_guard_blocked,
                        }
                        self._update_router_diagnostic_bucket(summary, **update_kwargs)
                        confidence_key = (client_id, self._router_confidence_bin(sample_confidence, confidence_edges))
                        confidence_bucket = confidence_buckets.setdefault(confidence_key, self._router_diagnostic_bucket())
                        self._update_router_diagnostic_bucket(confidence_bucket, **update_kwargs)
                        if include_classes:
                            class_key = (client_id, predicted_class)
                            class_bucket = class_buckets.setdefault(class_key, self._router_diagnostic_bucket())
                            self._update_router_diagnostic_bucket(class_bucket, **update_kwargs)

        self._log_router_diagnostic_bucket("summary", "all", summary)
        high_confidence_recall = high_confidence_true_positive / max(high_confidence_errors, 1)
        LOGGER.info(
            "%s router diag high_confidence"
            " | guard=%.4f | n=%d | errors=%d | recall=%.4f",
            self.algorithm_name,
            guard,
            high_confidence_samples,
            high_confidence_errors,
            high_confidence_recall,
        )

        for (client_id, confidence_bin), bucket in sorted(confidence_buckets.items()):
            if int(bucket.get("samples", 0.0)) < min_samples:
                continue
            if bucket.get("expert_errors", 0.0) <= 0.0 and bucket.get("predicted_positive", 0.0) <= 0.0:
                continue
            self._log_router_diagnostic_bucket("confidence", f"{client_id}:{confidence_bin}", bucket)

        if include_classes:
            for (client_id, predicted_class), bucket in sorted(class_buckets.items()):
                if int(bucket.get("samples", 0.0)) < min_samples:
                    continue
                if bucket.get("expert_errors", 0.0) <= 0.0 and bucket.get("predicted_positive", 0.0) <= 0.0:
                    continue
                self._log_router_diagnostic_bucket("class", f"{client_id}:{predicted_class}", bucket)

        rates = self._router_bucket_rates(summary)
        return {
            "router_diag_precision": rates["precision"],
            "router_diag_recall": rates["recall"],
            "router_diag_fpr": rates["fpr"],
            "router_diag_predicted_positive_rate": rates["predicted_positive_rate"],
            "router_diag_rescue": float(summary.get("rescue", 0.0)),
            "router_diag_harm": float(summary.get("harm", 0.0)),
            "router_diag_net_rescue": rates["net_rescue"],
            "router_diag_guard_blocked": float(summary.get("guard_blocked", 0.0)),
            "router_diag_guard_blocked_errors": float(summary.get("guard_blocked_errors", 0.0)),
            "router_diag_high_confidence_samples": float(high_confidence_samples),
            "router_diag_high_confidence_errors": float(high_confidence_errors),
            "router_diag_high_confidence_recall": float(high_confidence_recall),
        }

    def _router_regret_diagnostics_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "router_regret_diagnostics_enabled", False))

    def _router_candidate_diagnostics_enabled(self) -> bool:
        return bool(getattr(self.config.inference, "router_candidate_diagnostics_enabled", False))

    def _router_candidate_rates(self) -> List[float]:
        raw_rates = getattr(self.config.inference, "router_candidate_rates", [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
        if isinstance(raw_rates, str):
            parts = [part.strip() for part in raw_rates.split(",")]
        else:
            try:
                parts = list(raw_rates)
            except TypeError:
                parts = [raw_rates]
        rates: List[float] = []
        for part in parts:
            try:
                rate = float(part)
            except (TypeError, ValueError):
                continue
            if 0.0 < rate <= 1.0:
                rates.append(rate)
        return sorted(set(rates))

    @staticmethod
    def _router_candidate_rate_label(rate: float) -> str:
        percent = rate * 100.0
        if abs(percent - round(percent)) < 1e-6:
            return f"top{int(round(percent))}"
        return "top" + f"{percent:.1f}".replace(".", "p")

    @staticmethod
    def _router_regret_bucket() -> Dict[str, float]:
        return {
            "samples": 0.0,
            "expert_errors": 0.0,
            "oracle_positive": 0.0,
            "routed": 0.0,
            "rescue": 0.0,
            "missed_rescue": 0.0,
            "harm": 0.0,
            "neutral_call": 0.0,
            "both_correct_call": 0.0,
            "both_wrong_call": 0.0,
        }

    def _log_router_regret_bucket(self, label: str, key: str, bucket: Dict[str, float]) -> None:
        samples = float(bucket.get("samples", 0.0))
        oracle_positive = float(bucket.get("oracle_positive", 0.0))
        routed = float(bucket.get("routed", 0.0))
        rescue = float(bucket.get("rescue", 0.0))
        missed_rescue = float(bucket.get("missed_rescue", 0.0))
        harm = float(bucket.get("harm", 0.0))
        LOGGER.info(
            "%s router regret %s=%s"
            " | n=%d | errors=%d | oracle_pos=%d | oracle_pos_rate=%.4f"
            " | routed=%d | routed_rate=%.4f"
            " | rescue=%d | missed=%d | harm=%d | neutral=%d"
            " | both_correct_call=%d | both_wrong_call=%d"
            " | oracle_capture=%.4f | missed_rate=%.4f | harm_rate=%.4f | regret_rate=%.4f",
            self.algorithm_name,
            label,
            key,
            int(samples),
            int(bucket.get("expert_errors", 0.0)),
            int(oracle_positive),
            oracle_positive / max(samples, 1.0),
            int(routed),
            routed / max(samples, 1.0),
            int(rescue),
            int(missed_rescue),
            int(harm),
            int(bucket.get("neutral_call", 0.0)),
            int(bucket.get("both_correct_call", 0.0)),
            int(bucket.get("both_wrong_call", 0.0)),
            rescue / max(oracle_positive, 1.0),
            missed_rescue / max(samples, 1.0),
            harm / max(samples, 1.0),
            (missed_rescue + harm) / max(samples, 1.0),
        )

    def _evaluate_router_oracle_diagnostics(self) -> Dict[str, float]:
        regret_enabled = self._router_regret_diagnostics_enabled()
        candidate_enabled = self._router_candidate_diagnostics_enabled()
        if not regret_enabled and not candidate_enabled:
            return {}

        confidence_edges = self._router_diagnostic_confidence_edges()
        summary_bucket = self._router_regret_bucket()
        confidence_buckets: Dict[str, Dict[str, float]] = {}

        score_batches: List[torch.Tensor] = []
        oracle_positive_batches: List[torch.Tensor] = []
        harm_candidate_batches: List[torch.Tensor] = []
        both_correct_batches: List[torch.Tensor] = []
        both_wrong_batches: List[torch.Tensor] = []
        expert_correct_batches: List[torch.Tensor] = []
        general_correct_batches: List[torch.Tensor] = []

        total_samples = 0
        expert_correct_total = 0
        general_correct_total = 0
        routed_correct_total = 0
        oracle_correct_total = 0
        disagreement_total = 0
        two_stage_candidate_total = 0
        two_stage_enabled = self._two_stage_routing_enabled()

        self.general_model.eval()
        with torch.no_grad():
            for client_id, dataset in self.client_test_datasets.items():
                predictor_state = self.client_predictor_states.get(client_id)
                if predictor_state is None:
                    continue
                threshold = float(
                    self.client_error_thresholds.get(
                        client_id,
                        getattr(self.config.inference, "error_predictor_threshold", 0.5),
                    )
                )
                expert_model = self.clients[client_id].expert_model
                expert_model.eval()
                loader = self.data_module.make_loader(dataset, shuffle=False)

                for images, targets, _ in loader:
                    images = images.to(self.device)
                    targets_device = targets.to(self.device)

                    expert_logits = expert_model(images)
                    expert_features = self._base_predictor_features(expert_logits)
                    verifier_error_probs = _predict_error_probabilities(predictor_state, expert_features)
                    expert_predictions = expert_logits.argmax(dim=1)

                    general_logits = self.general_model(images)
                    general_predictions = general_logits.argmax(dim=1)
                    if two_stage_enabled:
                        error_probs, candidate_features = self._candidate_error_probabilities(
                            client_id,
                            expert_model,
                            images,
                            expert_logits,
                            expert_features,
                            verifier_error_probs,
                        )
                        candidate_mask = self._build_two_stage_candidate_mask(client_id, error_probs, candidate_features)
                        verifier_scores = self._score_route_verifier(
                            client_id,
                            expert_logits,
                            general_logits,
                            verifier_error_probs,
                        )
                        adopt_thresholds = self._route_verifier_thresholds_for_batch(client_id, expert_features)
                        routed_mask = candidate_mask & verifier_scores.ge(adopt_thresholds)
                        two_stage_candidate_total += int(candidate_mask.sum().item())
                    else:
                        base_fallback_mask = _build_error_fallback_mask(
                            verifier_error_probs,
                            expert_features,
                            threshold,
                            self.config,
                        )
                        routed_mask = self._apply_route_group_filter(
                            client_id,
                            base_fallback_mask,
                            expert_predictions,
                            verifier_error_probs,
                            threshold,
                            expert_features,
                        )
                    expert_correct = expert_predictions.eq(targets_device)
                    general_correct = general_predictions.eq(targets_device)
                    if two_stage_enabled and self._route_verifier_uses_fusion() and routed_mask.any():
                        alpha = self._route_fusion_alpha(
                            verifier_error_probs[routed_mask],
                            verifier_scores[routed_mask],
                            adopt_thresholds[routed_mask],
                            expert_confidence=expert_features[routed_mask, 0],
                        )
                        fused_probs = ((1.0 - alpha).unsqueeze(1) * torch.softmax(expert_logits[routed_mask], dim=1)) + (
                            alpha.unsqueeze(1) * torch.softmax(general_logits[routed_mask], dim=1)
                        )
                        route_correct = torch.zeros_like(expert_correct, dtype=torch.bool)
                        route_correct[routed_mask] = fused_probs.argmax(dim=1).eq(targets_device[routed_mask])
                    else:
                        route_correct = general_correct
                    expert_error = ~expert_correct
                    oracle_positive = expert_error & general_correct
                    harm_candidate = expert_correct & (~general_correct)
                    rescue = routed_mask & expert_error & route_correct
                    missed_rescue = oracle_positive & (~rescue)
                    harm = routed_mask & expert_correct & (~route_correct)
                    both_correct = expert_correct & general_correct
                    both_wrong = (~expert_correct) & (~general_correct)
                    neutral_call = routed_mask & (~rescue) & (~harm)
                    routed_correct = torch.where(routed_mask, route_correct, expert_correct)

                    batch_size = int(targets_device.numel())
                    total_samples += batch_size
                    expert_correct_total += int(expert_correct.sum().item())
                    general_correct_total += int(general_correct.sum().item())
                    routed_correct_total += int(routed_correct.sum().item())
                    oracle_correct_total += int((expert_correct | general_correct).sum().item())
                    disagreement_total += int(expert_predictions.ne(general_predictions).sum().item())

                    summary_bucket["samples"] += float(batch_size)
                    summary_bucket["expert_errors"] += float(expert_error.sum().item())
                    summary_bucket["oracle_positive"] += float(oracle_positive.sum().item())
                    summary_bucket["routed"] += float(routed_mask.sum().item())
                    summary_bucket["rescue"] += float(rescue.sum().item())
                    summary_bucket["missed_rescue"] += float(missed_rescue.sum().item())
                    summary_bucket["harm"] += float(harm.sum().item())
                    summary_bucket["neutral_call"] += float(neutral_call.sum().item())
                    summary_bucket["both_correct_call"] += float((routed_mask & both_correct).sum().item())
                    summary_bucket["both_wrong_call"] += float((routed_mask & both_wrong).sum().item())

                    if regret_enabled:
                        confidence = expert_features[:, 0]
                        for sample_index in range(batch_size):
                            confidence_bin = self._router_confidence_bin(
                                float(confidence[sample_index].item()),
                                confidence_edges,
                            )
                            bucket = confidence_buckets.setdefault(confidence_bin, self._router_regret_bucket())
                            bucket["samples"] += 1.0
                            bucket["expert_errors"] += float(expert_error[sample_index].item())
                            bucket["oracle_positive"] += float(oracle_positive[sample_index].item())
                            bucket["routed"] += float(routed_mask[sample_index].item())
                            bucket["rescue"] += float(rescue[sample_index].item())
                            bucket["missed_rescue"] += float(missed_rescue[sample_index].item())
                            bucket["harm"] += float(harm[sample_index].item())
                            bucket["neutral_call"] += float(neutral_call[sample_index].item())
                            bucket["both_correct_call"] += float((routed_mask & both_correct)[sample_index].item())
                            bucket["both_wrong_call"] += float((routed_mask & both_wrong)[sample_index].item())

                    if candidate_enabled:
                        score_batches.append(error_probs.detach().cpu().to(torch.float32))
                        oracle_positive_batches.append(oracle_positive.detach().cpu())
                        harm_candidate_batches.append(harm_candidate.detach().cpu())
                        both_correct_batches.append(both_correct.detach().cpu())
                        both_wrong_batches.append(both_wrong.detach().cpu())
                        expert_correct_batches.append(expert_correct.detach().cpu())
                        general_correct_batches.append(general_correct.detach().cpu())

        metrics: Dict[str, float] = {}
        samples = float(max(total_samples, 1))
        oracle_positive_total = float(summary_bucket.get("oracle_positive", 0.0))
        rescue_total = float(summary_bucket.get("rescue", 0.0))
        missed_rescue_total = float(summary_bucket.get("missed_rescue", 0.0))
        harm_total = float(summary_bucket.get("harm", 0.0))
        routed_total = float(summary_bucket.get("routed", 0.0))
        metrics.update(
            {
                "router_regret_rescue": rescue_total,
                "router_regret_harm": harm_total,
                "router_regret_missed_rescue": missed_rescue_total,
                "router_regret_neutral_call": float(summary_bucket.get("neutral_call", 0.0)),
                "router_regret_rescue_rate": rescue_total / samples,
                "router_regret_harm_rate": harm_total / samples,
                "router_regret_missed_rescue_rate": missed_rescue_total / samples,
                "router_regret_rate_from_masks": (missed_rescue_total + harm_total) / samples,
                "router_oracle_positive_rate": oracle_positive_total / samples,
                "router_current_route_rate_from_masks": routed_total / samples,
                "router_oracle_capture_rate": rescue_total / max(oracle_positive_total, 1.0),
                "router_oracle_accuracy_from_masks": oracle_correct_total / samples,
                "router_routed_accuracy_from_masks": routed_correct_total / samples,
                "router_expert_accuracy_from_masks": expert_correct_total / samples,
                "router_general_accuracy_from_masks": general_correct_total / samples,
                "router_expert_general_disagreement_from_masks": disagreement_total / samples,
                "router_two_stage_candidate_rate_from_masks": two_stage_candidate_total / samples if two_stage_enabled else 0.0,
                "router_two_stage_adopted_rate_from_masks": routed_total / samples if two_stage_enabled else 0.0,
            }
        )

        if regret_enabled:
            self._log_router_regret_bucket("summary", "all", summary_bucket)
            for confidence_bin, bucket in sorted(confidence_buckets.items()):
                self._log_router_regret_bucket("confidence", confidence_bin, bucket)

        if candidate_enabled and score_batches:
            scores = torch.cat(score_batches, dim=0)
            oracle_positive = torch.cat(oracle_positive_batches, dim=0).to(dtype=torch.bool)
            harm_candidate = torch.cat(harm_candidate_batches, dim=0).to(dtype=torch.bool)
            both_correct = torch.cat(both_correct_batches, dim=0).to(dtype=torch.bool)
            both_wrong = torch.cat(both_wrong_batches, dim=0).to(dtype=torch.bool)
            expert_correct = torch.cat(expert_correct_batches, dim=0).to(dtype=torch.bool)
            general_correct = torch.cat(general_correct_batches, dim=0).to(dtype=torch.bool)
            order = torch.argsort(scores, descending=True)
            total = int(scores.numel())
            total_oracle_positive = int(oracle_positive.sum().item())
            for rate in self._router_candidate_rates():
                candidate_size = min(max(int(math.ceil(total * rate)), 1), total)
                selected = order[:candidate_size]
                candidate_oracle_positive = int(oracle_positive[selected].sum().item())
                candidate_harm = int(harm_candidate[selected].sum().item())
                candidate_both_correct = int(both_correct[selected].sum().item())
                candidate_both_wrong = int(both_wrong[selected].sum().item())
                candidate_expert_correct = int(expert_correct[selected].sum().item())
                candidate_general_correct = int(general_correct[selected].sum().item())
                candidate_hard_correct = expert_correct.clone()
                candidate_hard_correct[selected] = general_correct[selected]
                candidate_hard_accuracy = float(candidate_hard_correct.to(torch.float32).mean().item())
                capture = candidate_oracle_positive / max(total_oracle_positive, 1)
                net = candidate_oracle_positive - candidate_harm
                label = self._router_candidate_rate_label(rate)
                metrics.update(
                    {
                        f"router_candidate_{label}_rate": candidate_size / max(total, 1),
                        f"router_candidate_{label}_capture": capture,
                        f"router_candidate_{label}_oracle_positive": float(candidate_oracle_positive),
                        f"router_candidate_{label}_harm": float(candidate_harm),
                        f"router_candidate_{label}_net": float(net),
                        f"router_candidate_{label}_both_correct": float(candidate_both_correct),
                        f"router_candidate_{label}_both_wrong": float(candidate_both_wrong),
                        f"router_candidate_{label}_expert_acc": candidate_expert_correct / max(candidate_size, 1),
                        f"router_candidate_{label}_general_acc": candidate_general_correct / max(candidate_size, 1),
                        f"router_candidate_{label}_hard_general_acc": candidate_hard_accuracy,
                    }
                )
                LOGGER.info(
                    "%s router candidate dryrun rate=%.4f"
                    " | k=%d | capture=%.4f | oracle_pos=%d/%d"
                    " | harm=%d | net=%d | both_correct=%d | both_wrong=%d"
                    " | candidate_expert_acc=%.4f | candidate_general_acc=%.4f"
                    " | hard_general_acc=%.4f",
                    self.algorithm_name,
                    rate,
                    candidate_size,
                    capture,
                    candidate_oracle_positive,
                    total_oracle_positive,
                    candidate_harm,
                    net,
                    candidate_both_correct,
                    candidate_both_wrong,
                    candidate_expert_correct / max(candidate_size, 1),
                    candidate_general_correct / max(candidate_size, 1),
                    candidate_hard_accuracy,
                )

        return metrics

    def _build_round_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
        include_oracle_diagnostics: bool = False,
    ) -> Dict[str, float]:
        metrics = self._evaluate_route_effectiveness_metrics(expert_eval, general_eval, routed_eval)
        predictor_metrics = self._evaluate_error_predictor_metrics()
        router_diagnostics = self._evaluate_router_diagnostics()
        oracle_diagnostics = self._evaluate_router_oracle_diagnostics() if include_oracle_diagnostics else {}
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
        gain_filter_stats = list(self.client_route_gain_stats.values())
        gain_filter_disabled_rate = (
            sum(1 for item in gain_filter_stats if item.get("disabled_by_gain", 0.0) > 0.0)
            / max(len(gain_filter_stats), 1)
            if gain_filter_stats
            else 0.0
        )
        gain_filter_mean_gain = (
            sum(float(item.get("route_gain", 0.0)) for item in gain_filter_stats)
            / max(len(gain_filter_stats), 1)
            if gain_filter_stats
            else 0.0
        )
        gain_filter_mean_invocation = (
            sum(float(item.get("invoked", 0.0)) / max(float(item.get("samples", 0.0)), 1.0) for item in gain_filter_stats)
            / max(len(gain_filter_stats), 1)
            if gain_filter_stats
            else 0.0
        )
        route_group_stats = [
            stats
            for client_stats in self.client_route_group_stats.values()
            for stats in client_stats.values()
        ]
        route_group_enabled = sum(1 for item in route_group_stats if item.get("enabled", 0.0) > 0.0)
        route_group_disabled = sum(1 for item in route_group_stats if item.get("blocked", 0.0) > 0.0)
        route_group_lowered = sum(1 for item in route_group_stats if item.get("threshold_lowered", 0.0) > 0.0)
        route_group_boosted = sum(1 for item in route_group_stats if item.get("threshold_boost", 0.0) > 0.0)
        route_group_disabled_rate = (
            route_group_disabled / max(len(route_group_stats), 1)
            if route_group_stats
            else 0.0
        )
        route_group_net_rescue = sum(float(item.get("net_rescue", 0.0)) for item in route_group_stats if item.get("enabled", 0.0) > 0.0)
        route_group_blocked_net_rescue = sum(
            float(item.get("net_rescue", 0.0)) for item in route_group_stats if item.get("blocked", 0.0) > 0.0
        )
        verifier_stats = list(self.client_route_verifier_stats.values())
        verifier_enabled_rate = (
            sum(1 for item in verifier_stats if item.get("disabled", 1.0) <= 0.0) / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_candidate_rate = (
            sum(float(item.get("candidate_rate", 0.0)) for item in verifier_stats) / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_adopted_rate = (
            sum(float(item.get("adopted_rate", 0.0)) for item in verifier_stats) / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_rescue = sum(float(item.get("rescue", 0.0)) for item in verifier_stats)
        verifier_harm = sum(float(item.get("harm", 0.0)) for item in verifier_stats)
        verifier_candidate_threshold = (
            sum(float(item.get("candidate_threshold", 1.01)) for item in verifier_stats) / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_adopt_threshold = (
            sum(float(item.get("selected_threshold", 1.01)) for item in verifier_stats) / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_validation_disabled_rate = (
            sum(1 for item in verifier_stats if item.get("disabled_by_validation", 0.0) > 0.0)
            / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        verifier_train_harm_guard_disabled_rate = (
            sum(1 for item in verifier_stats if item.get("disabled_by_train_harm_guard", 0.0) > 0.0)
            / max(len(verifier_stats), 1)
            if verifier_stats
            else 0.0
        )
        metrics.update(
            {
                **predictor_metrics,
                **router_diagnostics,
                **oracle_diagnostics,
                "distill_loss": self.latest_distill_stats.get("total_loss", 0.0),
                "dkdr_forward_kl": self.latest_distill_stats.get("forward_kl", 0.0),
                "dkdr_reverse_kl": self.latest_distill_stats.get("reverse_kl", 0.0),
                "dkdr_gamma_forward": self.latest_distill_stats.get("gamma_forward", 0.5),
                "dkdr_gamma_reverse": self.latest_distill_stats.get("gamma_reverse", 0.5),
                "teacher_reliability_mean": self.latest_distill_stats.get("teacher_reliability_mean", 0.0),
                "teacher_error_mean": self.latest_distill_stats.get("teacher_error_mean", 0.0),
                "routing_error_threshold_mean": threshold_mean,
                "routing_client_veto_rate": veto_rate,
                "route_gain_filter_disabled_rate": gain_filter_disabled_rate,
                "route_gain_filter_mean_gain": gain_filter_mean_gain,
                "route_gain_filter_mean_invocation_rate": gain_filter_mean_invocation,
                "route_group_filter_enabled_classes": float(route_group_enabled),
                "route_group_filter_disabled_classes": float(route_group_disabled),
                "route_group_filter_disabled_rate": float(route_group_disabled_rate),
                "route_group_filter_enabled_net_rescue": float(route_group_net_rescue),
                "route_group_filter_blocked_net_rescue": float(route_group_blocked_net_rescue),
                "route_group_filter_lowered_classes": float(route_group_lowered),
                "route_group_filter_boosted_classes": float(route_group_boosted),
                "route_verifier_enabled_rate": float(verifier_enabled_rate),
                "route_verifier_candidate_rate": float(verifier_candidate_rate),
                "route_verifier_adopted_rate": float(verifier_adopted_rate),
                "route_verifier_rescue": float(verifier_rescue),
                "route_verifier_harm": float(verifier_harm),
                "route_verifier_net_rescue": float(verifier_rescue - verifier_harm),
                "route_verifier_candidate_threshold_mean": float(verifier_candidate_threshold),
                "route_verifier_adopt_threshold_mean": float(verifier_adopt_threshold),
                "route_verifier_validation_disabled_rate": float(verifier_validation_disabled_rate),
                "route_verifier_train_harm_guard_disabled_rate": float(verifier_train_harm_guard_disabled_rate),
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
            f" | ep_pos={extra_metrics.get('error_predictor_predicted_positive_rate', 0.0):.4f}"
            f" | rgf_disable={extra_metrics.get('route_gain_filter_disabled_rate', 0.0):.4f}"
            f" | rgf_gain={extra_metrics.get('route_gain_filter_mean_gain', 0.0):.4f}"
            f" | rgg_on={extra_metrics.get('route_group_filter_enabled_classes', 0.0):.0f}"
            f" | rgg_disable={extra_metrics.get('route_group_filter_disabled_rate', 0.0):.4f}"
            f" | rv_cand={extra_metrics.get('route_verifier_candidate_rate', 0.0):.4f}"
            f" | rv_adopt={extra_metrics.get('route_verifier_adopted_rate', 0.0):.4f}"
            f" | rv_net={extra_metrics.get('route_verifier_net_rescue', 0.0):.0f}"
            f" | rv_val_disable={extra_metrics.get('route_verifier_validation_disabled_rate', 0.0):.4f}"
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
            "client_candidate_predictor_states": {
                client_id: _clone_tensor_dict(state)
                for client_id, state in self.client_candidate_predictor_states.items()
            },
            "client_error_thresholds": dict(self.client_error_thresholds),
            "client_raw_error_thresholds": dict(self.client_raw_error_thresholds),
            "client_route_group_allowlists": {
                client_id: list(allowlist) for client_id, allowlist in self.client_route_group_allowlists.items()
            },
            "client_route_group_blocklists": {
                client_id: list(blocklist) for client_id, blocklist in self.client_route_group_blocklists.items()
            },
            "client_route_group_thresholds": copy.deepcopy(self.client_route_group_thresholds),
            "client_route_group_threshold_boosts": copy.deepcopy(self.client_route_group_threshold_boosts),
            "client_route_group_stats": copy.deepcopy(self.client_route_group_stats),
            "client_route_candidate_thresholds": dict(self.client_route_candidate_thresholds),
            "client_route_candidate_bin_thresholds": copy.deepcopy(self.client_route_candidate_bin_thresholds),
            "client_route_verifier_states": {
                client_id: _clone_tensor_dict(state) for client_id, state in self.client_route_verifier_states.items()
            },
            "client_route_verifier_thresholds": dict(self.client_route_verifier_thresholds),
            "client_route_verifier_bin_thresholds": copy.deepcopy(self.client_route_verifier_bin_thresholds),
            "client_route_verifier_stats": copy.deepcopy(self.client_route_verifier_stats),
            "client_route_verifier_bin_stats": copy.deepcopy(self.client_route_verifier_bin_stats),
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
                    features = self._base_predictor_features(logits)
                    predictions = logits.argmax(dim=1)
                    feature_batches.append(features.cpu())
                    label_batches.append(predictions.ne(targets_device).cpu())

                if not feature_batches:
                    updated_thresholds[client_id] = 1.01
                    continue

                features = torch.cat(feature_batches, dim=0)
                labels = torch.cat(label_batches, dim=0)
                scores = _predict_error_probabilities(predictor_state, features)
                raw_threshold = _select_precision_constrained_threshold(
                    scores=scores,
                    labels=labels,
                    features=features,
                    config=self.config,
                )
                self.client_raw_error_thresholds[client_id] = raw_threshold
                updated_thresholds[client_id], stats = self._calibrate_client_route_gain(client_id, raw_threshold)
                self.client_route_gain_stats[client_id] = stats

        self.client_error_thresholds = updated_thresholds
        LOGGER.info(
            "%s recalibrated route thresholds | mean=%.4f | veto=%.4f",
            self.algorithm_name,
            sum(updated_thresholds.values()) / max(len(updated_thresholds), 1),
            sum(1 for value in updated_thresholds.values() if float(value) > 1.0) / max(len(updated_thresholds), 1),
        )
        self._log_route_gain_filter_details("recalibrate")
        self._refresh_route_group_filters(context="recalibrate")

    def _retrain_route_predictors(self) -> None:
        updated_thresholds: Dict[str, float] = {}
        for client_id, client in self.clients.items():
            predictor_state = client._fit_risk_predictor(use_tta=False, update_error_threshold=True)
            client.predictor_state = _clone_tensor_dict(predictor_state)
            self.client_predictor_states[client_id] = _clone_tensor_dict(predictor_state)
            if _risk_predictor_tta_enabled(self.config):
                candidate_predictor_state = client._fit_risk_predictor(use_tta=True, update_error_threshold=False)
            else:
                candidate_predictor_state = _clone_tensor_dict(predictor_state)
            client.candidate_predictor_state = _clone_tensor_dict(candidate_predictor_state)
            self.client_candidate_predictor_states[client_id] = _clone_tensor_dict(candidate_predictor_state)
            raw_threshold = float(client.error_threshold)
            self.client_raw_error_thresholds[client_id] = raw_threshold
            updated_thresholds[client_id], stats = self._calibrate_client_route_gain(client_id, raw_threshold)
            self.client_route_gain_stats[client_id] = stats
        self.client_error_thresholds = updated_thresholds
        LOGGER.info(
            "%s retrained route predictors | mean_threshold=%.4f | veto=%.4f",
            self.algorithm_name,
            sum(updated_thresholds.values()) / max(len(updated_thresholds), 1),
            sum(1 for value in updated_thresholds.values() if float(value) > 1.0) / max(len(updated_thresholds), 1),
        )
        self._log_route_gain_filter_details("retrain")
        self._refresh_route_group_filters(context="retrain")

    def _empty_route_verifier_data(self) -> Dict[str, torch.Tensor]:
        feature_dim = _route_verifier_feature_dim(self.config.model.num_classes)
        return {
            "features": torch.empty((0, feature_dim), dtype=torch.float32),
            "error_probs": torch.empty((0,), dtype=torch.float32),
            "confidence": torch.empty((0,), dtype=torch.float32),
            "rescue": torch.empty((0,), dtype=torch.bool),
            "harm": torch.empty((0,), dtype=torch.bool),
            "neutral": torch.empty((0,), dtype=torch.bool),
        }

    def _collect_route_verifier_data(self, client_id: str, dataset: Dataset) -> Dict[str, torch.Tensor]:
        if dataset is None or len(dataset) == 0:
            return self._empty_route_verifier_data()

        predictor_state = self.client_predictor_states.get(client_id)
        if predictor_state is None:
            return self._empty_route_verifier_data()

        expert_model = self.clients[client_id].expert_model
        expert_model.eval()
        self.general_model.eval()
        feature_batches: List[torch.Tensor] = []
        score_batches: List[torch.Tensor] = []
        confidence_batches: List[torch.Tensor] = []
        rescue_batches: List[torch.Tensor] = []
        harm_batches: List[torch.Tensor] = []
        neutral_batches: List[torch.Tensor] = []
        loader = self.data_module.make_loader(dataset, shuffle=False)
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets_device = targets.to(self.device)
                expert_logits = expert_model(images)
                expert_features = self._base_predictor_features(expert_logits)
                verifier_error_probs = _predict_error_probabilities(predictor_state, expert_features)
                error_probs, _ = self._candidate_error_probabilities(
                    client_id,
                    expert_model,
                    images,
                    expert_logits,
                    expert_features,
                    verifier_error_probs,
                )
                general_logits = self.general_model(images)
                route_features = _route_verifier_features_from_logits(
                    expert_logits,
                    general_logits,
                    verifier_error_probs,
                )

                expert_correct = expert_logits.argmax(dim=1).eq(targets_device)
                general_correct = general_logits.argmax(dim=1).eq(targets_device)
                rescue = (~expert_correct) & general_correct
                harm = expert_correct & (~general_correct)
                neutral = ~(rescue | harm)

                feature_batches.append(route_features.detach().cpu())
                score_batches.append(error_probs.detach().cpu())
                confidence_batches.append(expert_features[:, 0].detach().cpu())
                rescue_batches.append(rescue.detach().cpu())
                harm_batches.append(harm.detach().cpu())
                neutral_batches.append(neutral.detach().cpu())

        if not feature_batches:
            return self._empty_route_verifier_data()
        return {
            "features": torch.cat(feature_batches, dim=0).to(dtype=torch.float32),
            "error_probs": torch.cat(score_batches, dim=0).to(dtype=torch.float32),
            "confidence": torch.cat(confidence_batches, dim=0).to(dtype=torch.float32),
            "rescue": torch.cat(rescue_batches, dim=0).to(dtype=torch.bool),
            "harm": torch.cat(harm_batches, dim=0).to(dtype=torch.bool),
            "neutral": torch.cat(neutral_batches, dim=0).to(dtype=torch.bool),
        }

    def _default_route_verifier_stats(self) -> Dict[str, float]:
        return {
            "samples": 0.0,
            "train_samples": 0.0,
            "candidate": 0.0,
            "adopted": 0.0,
            "rescue": 0.0,
            "harm": 0.0,
            "net_rescue": 0.0,
            "candidate_rate": 0.0,
            "adopted_rate": 0.0,
            "harm_rate": 0.0,
            "rescue_harm_ratio": 0.0,
            "candidate_threshold": 1.01,
            "selected_threshold": 1.01,
            "train_candidate": 0.0,
            "train_rescue": 0.0,
            "train_harm": 0.0,
            "train_neutral": 0.0,
            "disabled": 1.0,
            "disabled_by_validation": 0.0,
            "disabled_by_validation_adopted": 0.0,
            "disabled_by_validation_net": 0.0,
            "disabled_by_validation_ratio": 0.0,
            "disabled_by_train_harm_guard": 0.0,
        }

    def _refresh_route_verifiers(self, client_ids: Optional[Sequence[str]] = None, context: str = "refresh") -> None:
        if not self._two_stage_routing_enabled():
            self.client_route_verifier_stats = {}
            self.client_route_verifier_bin_stats = {}
            return

        target_client_ids = list(client_ids) if client_ids is not None else list(self.clients.keys())
        candidate_rate = min(max(float(getattr(self.config.inference, "router_candidate_rate", 0.10)), 0.0), 1.0)
        candidate_min_score = min(max(float(getattr(self.config.inference, "router_candidate_min_score", 0.0)), 0.0), 1.0)
        hidden_dim = max(int(getattr(self.config.inference, "route_verifier_hidden_dim", 32)), 0)
        dropout = min(max(float(getattr(self.config.inference, "route_verifier_dropout", 0.10)), 0.0), 0.9)
        epochs = max(int(getattr(self.config.inference, "route_verifier_epochs", 60)), 1)
        lr = float(getattr(self.config.inference, "route_verifier_lr", 0.001))
        weight_decay = float(getattr(self.config.inference, "route_verifier_weight_decay", 0.0))
        negative_weight = max(float(getattr(self.config.inference, "route_verifier_negative_weight", 2.0)), 0.0)
        neutral_weight = max(float(getattr(self.config.inference, "route_verifier_neutral_weight", 0.0)), 0.0)
        min_validation_adopted = max(
            int(getattr(self.config.inference, "route_verifier_min_validation_adopted", 0)),
            0,
        )
        bin_min_validation_adopted = max(
            int(getattr(self.config.inference, "route_verifier_bin_min_validation_adopted", min_validation_adopted)),
            0,
        )
        min_validation_net = float(getattr(self.config.inference, "route_verifier_min_validation_net", 0.0))
        min_validation_ratio = max(
            float(getattr(self.config.inference, "route_verifier_min_validation_rescue_harm_ratio", 1.0)),
            0.0,
        )
        min_adopt_threshold = min(
            max(float(getattr(self.config.inference, "route_verifier_min_adopt_threshold", 0.0)), 0.0),
            1.0,
        )
        low_threshold_train_harm_ratio = max(
            float(getattr(self.config.inference, "route_verifier_low_threshold_train_harm_ratio", 0.0)),
            0.0,
        )
        low_threshold_train_harm_min_candidates = max(
            int(getattr(self.config.inference, "route_verifier_low_threshold_train_harm_min_candidates", 10)),
            1,
        )
        disable_on_validation_fail = bool(
            getattr(self.config.inference, "route_verifier_disable_on_validation_fail", True)
        )

        enabled = 0
        total_candidate_rate = 0.0
        total_adopted_rate = 0.0
        for client_id in target_client_ids:
            stats = self._default_route_verifier_stats()
            train_data = self._collect_route_verifier_data(client_id, self.client_router_train_datasets.get(client_id))
            validation_data = self._collect_route_verifier_data(client_id, self.client_calibration_datasets.get(client_id))
            score_source = validation_data["error_probs"] if validation_data["error_probs"].numel() > 0 else train_data["error_probs"]
            confidence_source = (
                validation_data["confidence"] if validation_data["confidence"].numel() > 0 else train_data["confidence"]
            )
            candidate_threshold = _select_top_rate_threshold(score_source, candidate_rate, min_score=candidate_min_score)
            self.client_route_candidate_thresholds[client_id] = candidate_threshold
            if self._candidate_confidence_bins_enabled():
                self.client_route_candidate_bin_thresholds[client_id] = self._select_candidate_thresholds_by_confidence(
                    score_source,
                    confidence_source,
                    candidate_rate,
                    min_score=candidate_min_score,
                )
            else:
                self.client_route_candidate_bin_thresholds[client_id] = {}
            stats["candidate_threshold"] = float(candidate_threshold)
            stats["train_samples"] = float(train_data["error_probs"].numel())

            train_candidate = self._candidate_mask_from_scores_and_confidence(
                client_id,
                train_data["error_probs"],
                train_data["confidence"],
            )
            train_rescue = train_data["rescue"] & train_candidate
            train_harm = train_data["harm"] & train_candidate
            train_neutral = train_data["neutral"] & train_candidate
            stats["train_candidate"] = float(train_candidate.sum().item())
            stats["train_rescue"] = float(train_rescue.sum().item())
            stats["train_harm"] = float(train_harm.sum().item())
            stats["train_neutral"] = float(train_neutral.sum().item())

            labels = train_rescue.to(dtype=torch.float32)
            weights = torch.zeros_like(labels, dtype=torch.float32)
            weights[train_rescue] = 1.0
            weights[train_harm] = negative_weight
            if neutral_weight > 0.0:
                weights[train_neutral] = neutral_weight

            if train_data["features"].numel() > 0 and weights.sum().item() > 0.0:
                verifier_state = _fit_weighted_binary_predictor_state(
                    features=train_data["features"],
                    labels=labels,
                    sample_weights=weights,
                    device=self.device,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            else:
                verifier_state = _default_predictor_state(
                    _route_verifier_feature_dim(self.config.model.num_classes),
                    hidden_dim=hidden_dim,
                )
            self.client_route_verifier_states[client_id] = _clone_tensor_dict(verifier_state)

            validation_scores = validation_data["error_probs"]
            validation_candidate = self._candidate_mask_from_scores_and_confidence(
                client_id,
                validation_scores,
                validation_data["confidence"],
            )
            self.client_route_verifier_bin_thresholds[client_id] = {}
            self.client_route_verifier_bin_stats[client_id] = {}
            if validation_data["features"].numel() > 0:
                verifier_scores = _predict_error_probabilities(verifier_state, validation_data["features"])
                if self._route_verifier_confidence_bins_enabled():
                    bins = self._confidence_bin_indices(validation_data["confidence"])
                    bin_thresholds: Dict[int, float] = {}
                    bin_stats_map: Dict[int, Dict[str, float]] = {}
                    candidate_bin_thresholds = dict(self.client_route_candidate_bin_thresholds.get(client_id, {}))
                    for bin_index in range(self._confidence_bin_count()):
                        bin_mask = bins.eq(int(bin_index))
                        if bin_mask.any():
                            bin_candidate = validation_candidate & bin_mask
                            bin_threshold, bin_stats = _select_route_verifier_threshold(
                                scores=verifier_scores,
                                rescue_mask=validation_data["rescue"],
                                harm_mask=validation_data["harm"],
                                candidate_mask=bin_candidate,
                                config=self.config,
                            )
                            bin_stats["samples"] = float(bin_mask.sum().item())
                        else:
                            bin_threshold, bin_stats = 1.01, self._default_route_verifier_stats()
                        bin_stats["confidence_bin"] = float(bin_index)
                        bin_stats["candidate_threshold"] = float(
                            candidate_bin_thresholds.get(bin_index, candidate_threshold)
                        )
                        bin_stats["confidence_bin_label"] = self._confidence_bin_label_by_index(bin_index)
                        passes_filter, filter_stats = self._passes_route_verifier_validation_filter(
                            bin_stats,
                            bin_min_validation_adopted,
                            min_validation_net,
                            min_validation_ratio,
                        )
                        bin_stats.update(filter_stats)
                        if disable_on_validation_fail and not passes_filter:
                            bin_threshold = 1.01
                            bin_stats = self._disabled_route_verifier_stats(bin_stats)
                            bin_stats.update(filter_stats)
                            bin_stats["confidence_bin"] = float(bin_index)
                            bin_stats["confidence_bin_label"] = self._confidence_bin_label_by_index(bin_index)
                            bin_stats["candidate_threshold"] = float(
                                candidate_bin_thresholds.get(bin_index, candidate_threshold)
                            )
                        bin_thresholds[bin_index] = float(bin_threshold)
                        bin_stats["selected_threshold"] = float(bin_threshold)
                        bin_stats_map[bin_index] = bin_stats
                    self.client_route_candidate_bin_thresholds[client_id] = candidate_bin_thresholds
                    self.client_route_verifier_bin_thresholds[client_id] = bin_thresholds
                    self.client_route_verifier_bin_stats[client_id] = bin_stats_map
                    adopt_threshold = 1.01
                    threshold_stats = self._aggregate_route_verifier_bin_stats(bin_stats_map)
                else:
                    adopt_threshold, threshold_stats = _select_route_verifier_threshold(
                        scores=verifier_scores,
                        rescue_mask=validation_data["rescue"],
                        harm_mask=validation_data["harm"],
                        candidate_mask=validation_candidate,
                        config=self.config,
                    )
            else:
                adopt_threshold, threshold_stats = 1.01, self._default_route_verifier_stats()
            self.client_route_verifier_thresholds[client_id] = float(adopt_threshold)
            stats.update(threshold_stats)
            stats["candidate_threshold"] = float(candidate_threshold)
            train_harm_guard_failed = False
            if (
                low_threshold_train_harm_ratio > 0.0
                and float(stats.get("selected_threshold", 1.01)) <= min_adopt_threshold + 1e-8
                and float(stats.get("train_candidate", 0.0)) >= float(low_threshold_train_harm_min_candidates)
            ):
                train_rescue_count = float(stats.get("train_rescue", 0.0))
                train_harm_count = float(stats.get("train_harm", 0.0))
                train_harm_guard_failed = train_harm_count > max(train_rescue_count * low_threshold_train_harm_ratio, 0.0)
            stats["disabled_by_train_harm_guard"] = float(train_harm_guard_failed)
            if not self._route_verifier_confidence_bins_enabled():
                passes_filter, filter_stats = self._passes_route_verifier_validation_filter(
                    stats,
                    min_validation_adopted,
                    min_validation_net,
                    min_validation_ratio,
                )
                stats.update(filter_stats)
                if disable_on_validation_fail and (not passes_filter or train_harm_guard_failed):
                    self.client_route_verifier_thresholds[client_id] = 1.01
                    stats["selected_threshold"] = 1.01
                    stats["disabled"] = 1.0
            self.client_route_verifier_stats[client_id] = stats
            enabled += int(float(stats.get("disabled", 1.0)) <= 0.0)
            total_candidate_rate += float(stats.get("candidate_rate", 0.0))
            total_adopted_rate += float(stats.get("adopted_rate", 0.0))

        divisor = max(len(target_client_ids), 1)
        LOGGER.info(
            "%s route verifier %s | clients=%d | enabled=%d | candidate_rate=%.4f | adopted_rate=%.4f",
            self.algorithm_name,
            context,
            len(target_client_ids),
            enabled,
            total_candidate_rate / divisor,
            total_adopted_rate / divisor,
        )
        self._log_route_verifier_details(context, target_client_ids)

    def _log_route_verifier_details(self, context: str, client_ids: Optional[Sequence[str]] = None) -> None:
        if not self._two_stage_routing_enabled() or not self.client_route_verifier_stats:
            return
        target_client_ids = list(client_ids) if client_ids is not None else sorted(self.client_route_verifier_stats.keys())
        for client_id in target_client_ids:
            stats = self.client_route_verifier_stats.get(client_id)
            if not stats:
                continue
            LOGGER.info(
                "%s route verifier %s client=%s"
                " | disabled=%d | cand_thr=%.4f | adopt_thr=%.4f"
                " | n=%d | cand=%d | adopt=%d | cand_rate=%.4f | adopt_rate=%.4f"
                " | rescue=%d | harm=%d | net=%d | harm_rate=%.4f | rescue_harm=%.4f"
                " | val_fail=%d | val_adopt_fail=%d | val_net_fail=%d | val_ratio_fail=%d | train_harm_guard=%d"
                " | train_n=%d | train_cand=%d | train_rescue=%d | train_harm=%d | train_neutral=%d",
                self.algorithm_name,
                context,
                client_id,
                int(stats.get("disabled", 1.0) > 0.0),
                float(stats.get("candidate_threshold", 1.01)),
                float(stats.get("selected_threshold", 1.01)),
                int(stats.get("samples", 0.0)),
                int(stats.get("candidate", 0.0)),
                int(stats.get("adopted", 0.0)),
                float(stats.get("candidate_rate", 0.0)),
                float(stats.get("adopted_rate", 0.0)),
                int(stats.get("rescue", 0.0)),
                int(stats.get("harm", 0.0)),
                int(stats.get("net_rescue", 0.0)),
                float(stats.get("harm_rate", 0.0)),
                float(stats.get("rescue_harm_ratio", 0.0)),
                int(stats.get("disabled_by_validation", 0.0) > 0.0),
                int(stats.get("disabled_by_validation_adopted", 0.0) > 0.0),
                int(stats.get("disabled_by_validation_net", 0.0) > 0.0),
                int(stats.get("disabled_by_validation_ratio", 0.0) > 0.0),
                int(stats.get("disabled_by_train_harm_guard", 0.0) > 0.0),
                int(stats.get("train_samples", 0.0)),
                int(stats.get("train_candidate", 0.0)),
                int(stats.get("train_rescue", 0.0)),
                int(stats.get("train_harm", 0.0)),
                int(stats.get("train_neutral", 0.0)),
            )
            for bin_index, bin_stats in sorted(self.client_route_verifier_bin_stats.get(client_id, {}).items()):
                LOGGER.info(
                    "%s route verifier %s client=%s bin=%s"
                    " | disabled=%d | cand_thr=%.4f | adopt_thr=%.4f"
                    " | n=%d | cand=%d | adopt=%d | cand_rate=%.4f | adopt_rate=%.4f"
                    " | rescue=%d | harm=%d | net=%d | harm_rate=%.4f | rescue_harm=%.4f"
                    " | val_fail=%d | val_adopt_fail=%d | val_net_fail=%d | val_ratio_fail=%d",
                    self.algorithm_name,
                    context,
                    client_id,
                    str(bin_stats.get("confidence_bin_label", self._confidence_bin_label_by_index(bin_index))),
                    int(bin_stats.get("disabled", 1.0) > 0.0),
                    float(bin_stats.get("candidate_threshold", 1.01)),
                    float(bin_stats.get("selected_threshold", 1.01)),
                    int(bin_stats.get("samples", 0.0)),
                    int(bin_stats.get("candidate", 0.0)),
                    int(bin_stats.get("adopted", 0.0)),
                    float(bin_stats.get("candidate_rate", 0.0)),
                    float(bin_stats.get("adopted_rate", 0.0)),
                    int(bin_stats.get("rescue", 0.0)),
                    int(bin_stats.get("harm", 0.0)),
                    int(bin_stats.get("net_rescue", 0.0)),
                    float(bin_stats.get("harm_rate", 0.0)),
                    float(bin_stats.get("rescue_harm_ratio", 0.0)),
                    int(bin_stats.get("disabled_by_validation", 0.0) > 0.0),
                    int(bin_stats.get("disabled_by_validation_adopted", 0.0) > 0.0),
                    int(bin_stats.get("disabled_by_validation_net", 0.0) > 0.0),
                    int(bin_stats.get("disabled_by_validation_ratio", 0.0) > 0.0),
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
        self.client_candidate_predictor_states = {
            client_id: _clone_tensor_dict(state)
            for client_id, state in self.best_snapshot.get("client_candidate_predictor_states", {}).items()
        }
        candidate_hidden_dim = int(getattr(self.config.federated, "risk_predictor_hidden_dim", 32))
        for client_id in self.clients.keys():
            self.client_candidate_predictor_states.setdefault(
                client_id,
                _clone_tensor_dict(self.client_predictor_states[client_id]),
            )
            if _risk_predictor_tta_enabled(self.config) and int(
                self.client_candidate_predictor_states[client_id]["mean"].numel()
            ) == int(self.client_predictor_states[client_id]["mean"].numel()):
                self.client_candidate_predictor_states[client_id] = _default_predictor_state(
                    _configured_predictor_feature_dim(self.config, self.config.model.num_classes),
                    hidden_dim=candidate_hidden_dim,
                )
        self.client_error_thresholds = dict(self.best_snapshot["client_error_thresholds"])
        self.client_raw_error_thresholds = dict(
            self.best_snapshot.get("client_raw_error_thresholds", self.client_error_thresholds)
        )
        self.client_route_gain_stats = {}
        self.client_route_group_stats = copy.deepcopy(self.best_snapshot.get("client_route_group_stats", {}))
        self.client_route_group_allowlists = {
            client_id: [int(class_id) for class_id in allowlist]
            for client_id, allowlist in self.best_snapshot.get("client_route_group_allowlists", {}).items()
        }
        self.client_route_group_blocklists = {
            client_id: [int(class_id) for class_id in blocklist]
            for client_id, blocklist in self.best_snapshot.get("client_route_group_blocklists", {}).items()
        }
        self.client_route_group_thresholds = {
            client_id: {int(class_id): float(threshold) for class_id, threshold in thresholds.items()}
            for client_id, thresholds in self.best_snapshot.get("client_route_group_thresholds", {}).items()
        }
        self.client_route_group_threshold_boosts = {
            client_id: {int(class_id): float(boost) for class_id, boost in boosts.items()}
            for client_id, boosts in self.best_snapshot.get("client_route_group_threshold_boosts", {}).items()
        }
        self.client_route_candidate_thresholds = {
            client_id: float(threshold)
            for client_id, threshold in self.best_snapshot.get("client_route_candidate_thresholds", {}).items()
        }
        for client_id in self.clients.keys():
            self.client_route_candidate_thresholds.setdefault(client_id, 1.01)
        self.client_route_candidate_bin_thresholds = {
            client_id: {int(bin_index): float(threshold) for bin_index, threshold in thresholds.items()}
            for client_id, thresholds in self.best_snapshot.get("client_route_candidate_bin_thresholds", {}).items()
        }
        for client_id in self.clients.keys():
            self.client_route_candidate_bin_thresholds.setdefault(client_id, {})
        self.client_route_verifier_states = {
            client_id: _clone_tensor_dict(state)
            for client_id, state in self.best_snapshot.get("client_route_verifier_states", {}).items()
        }
        hidden_dim = int(getattr(self.config.inference, "route_verifier_hidden_dim", 32))
        for client_id in self.clients.keys():
            self.client_route_verifier_states.setdefault(
                client_id,
                _default_predictor_state(
                    _route_verifier_feature_dim(self.config.model.num_classes),
                    hidden_dim=hidden_dim,
                ),
            )
        self.client_route_verifier_thresholds = {
            client_id: float(threshold)
            for client_id, threshold in self.best_snapshot.get("client_route_verifier_thresholds", {}).items()
        }
        for client_id in self.clients.keys():
            self.client_route_verifier_thresholds.setdefault(client_id, 1.01)
        self.client_route_verifier_bin_thresholds = {
            client_id: {int(bin_index): float(threshold) for bin_index, threshold in thresholds.items()}
            for client_id, thresholds in self.best_snapshot.get("client_route_verifier_bin_thresholds", {}).items()
        }
        for client_id in self.clients.keys():
            self.client_route_verifier_bin_thresholds.setdefault(client_id, {})
        self.client_route_verifier_stats = copy.deepcopy(self.best_snapshot.get("client_route_verifier_stats", {}))
        self.client_route_verifier_bin_stats = copy.deepcopy(
            self.best_snapshot.get("client_route_verifier_bin_stats", {})
        )
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
            self._refresh_route_verifiers(context="load")
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
                if update.candidate_predictor_state is not None:
                    self.client_candidate_predictor_states[update.client_id] = _clone_tensor_dict(
                        update.candidate_predictor_state
                    )
                else:
                    self.client_candidate_predictor_states[update.client_id] = _clone_tensor_dict(update.predictor_state)
                self.client_raw_error_thresholds[update.client_id] = float(update.error_threshold)
                self.client_error_thresholds[update.client_id] = float(update.error_threshold)

            public_knowledge = self._aggregate_public_knowledge(updates)
            distill_stats = self._distill_general_model(public_knowledge, round_idx)
            self._apply_route_gain_filter_to_thresholds()
            self._refresh_route_verifiers(context=f"round_{round_idx}")

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
                    + self._estimate_tensor_payload_bytes(update.candidate_predictor_state)
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
        extra_metrics = self._build_round_extra_metrics(
            expert_eval,
            general_eval,
            routed_eval,
            include_oracle_diagnostics=True,
        )
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
