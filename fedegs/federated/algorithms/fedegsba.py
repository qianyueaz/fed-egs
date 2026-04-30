"""
FedEGS-BA: Federated Expert-General System with Bidirectional Anchored Distillation.

Design goals:
  - The server maintains a larger width-scalable general model.
  - Each client slices a lightweight expert subnet from the general model.
  - Client-side knowledge flow: general -> expert on private data via DREL.
  - Server-side knowledge flow: expert -> general on a labeled public anchor set
    via staged logit aggregation and CE+KD distillation.
  - Inference is expert-first with confidence-based general fallback.

This implementation intentionally keeps the pipeline compact and close to the
paper plan: it does not upload expert parameters, but instead uploads only
public-anchor predictions from the expert.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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
from fedegs.models import (
    WidthScalableResNet,
    estimate_model_flops,
    get_expert_state_dict,
    get_num_expert_blocks,
    load_expert_state_dict,
    model_memory_mb,
)


def drel_loss_per_sample(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Decoupled relative entropy loss used for general -> expert transfer."""
    temperature = max(float(temperature), 1e-6)

    teacher_target = logits_teacher.gather(1, targets.unsqueeze(1)) / temperature
    student_target = logits_student.gather(1, targets.unsqueeze(1)) / temperature
    target_class_loss = ((student_target - teacher_target.detach()) ** 2).squeeze(1) * (temperature ** 2)

    mask = torch.zeros_like(logits_teacher)
    mask.scatter_(1, targets.unsqueeze(1), -1e9)
    teacher_non_target = F.softmax((logits_teacher + mask) / temperature, dim=1)
    student_non_target = F.log_softmax((logits_student + mask) / temperature, dim=1)
    non_target_loss = F.kl_div(
        student_non_target,
        teacher_non_target.detach(),
        reduction="none",
    ).sum(dim=1) * (temperature ** 2)

    return alpha * target_class_loss + beta * non_target_loss


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _stable_client_seed(base_seed: int, client_id: str) -> int:
    return int(base_seed) + sum((index + 1) * ord(char) for index, char in enumerate(client_id))


def _client_holdout_seed(base_seed: int, client_id: str, seed_offset: int) -> int:
    return _stable_client_seed(int(base_seed) + int(seed_offset), client_id)


def _split_dataset_for_holdout(
    dataset: Dataset,
    holdout_ratio: float,
    min_holdout_samples: int,
    max_holdout_samples: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    total_samples = len(dataset)
    if total_samples <= 1 or holdout_ratio <= 0.0:
        return dataset, dataset

    requested_samples = max(int(round(total_samples * holdout_ratio)), 1)
    if total_samples >= max(2 * max(min_holdout_samples, 1), 8):
        requested_samples = max(requested_samples, max(min_holdout_samples, 1))
    if max_holdout_samples > 0:
        requested_samples = min(requested_samples, max_holdout_samples)
    holdout_samples = min(max(requested_samples, 1), total_samples - 1)
    if holdout_samples <= 0 or holdout_samples >= total_samples:
        return dataset, dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total_samples, generator=generator).tolist()
    holdout_indices = sorted(permutation[:holdout_samples])
    train_indices = sorted(permutation[holdout_samples:])
    if not train_indices or not holdout_indices:
        return dataset, dataset
    return Subset(dataset, train_indices), Subset(dataset, holdout_indices)


@dataclass
class FedEGSBAClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_logits: torch.Tensor
    public_confidence: torch.Tensor
    public_predictions: torch.Tensor
    expert_temperature: float = 1.0


class FedEGSBAClient(BaseFederatedClient):
    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        expert_width: float,
        num_classes: int,
        device: str,
        config,
        data_module,
        block_index: int,
    ) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.block_index = block_index
        self.num_classes = num_classes
        self.expert_model = WidthScalableResNet(width_factor=expert_width, num_classes=num_classes).to(self.device)
        self.personalized_residual: Dict[str, torch.Tensor] = {}
        self.expert_temperature = 1.0
        self.last_drel_gate_ratio = 0.0

    def refresh_from_general(self, general_model: WidthScalableResNet) -> None:
        expert_state = get_expert_state_dict(general_model, self.expert_model, block_index=self.block_index)
        if self.personalized_residual:
            for key, residual in self.personalized_residual.items():
                if key not in expert_state:
                    continue
                if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked")):
                    continue
                expert_state[key] = expert_state[key] + residual.to(
                    device=expert_state[key].device,
                    dtype=expert_state[key].dtype,
                )
        self.expert_model.load_state_dict(expert_state)

    def train_round(
        self,
        round_idx: int,
        general_model: WidthScalableResNet,
        public_batches: Sequence,
        use_kd: bool,
    ) -> FedEGSBAClientUpdate:
        self.refresh_from_general(general_model)
        if use_kd:
            loss = self._train_with_general_drel(general_model)
        else:
            loader = self.data_module.make_loader(self.dataset, shuffle=True)
            loss = self._optimize_model(
                model=self.expert_model,
                loader=loader,
                epochs=self.config.federated.local_epochs,
                lr=self.config.federated.local_lr,
                momentum=self.config.federated.local_momentum,
                weight_decay=self.config.federated.local_weight_decay,
            )

        self._update_personalized_residual(general_model)
        self.expert_temperature = self._estimate_temperature(round_idx)
        public_logits, public_confidence, public_predictions = self._collect_public_predictions(public_batches)

        return FedEGSBAClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            public_logits=public_logits,
            public_confidence=public_confidence,
            public_predictions=public_predictions,
            expert_temperature=self.expert_temperature,
        )

    def _build_teacher_gate(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        gate_temperature: float,
    ) -> torch.Tensor:
        safe_temperature = max(float(gate_temperature), 1e-4)
        teacher_probs = F.softmax(teacher_logits.detach() / safe_temperature, dim=1)
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_prediction = teacher_topk.indices[:, 0]
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        confidence_threshold = float(getattr(self.config.federated, "expert_kd_confidence_threshold", 0.60))
        margin_threshold = float(getattr(self.config.federated, "expert_kd_margin_threshold", 0.08))
        hard_boost = max(float(getattr(self.config.federated, "expert_kd_hard_boost", 0.0)), 0.0)
        gate_power = max(float(getattr(self.config.federated, "expert_kd_gate_power", 1.0)), 1e-3)

        conf_gate = ((teacher_confidence - confidence_threshold) / max(1.0 - confidence_threshold, 1e-6)).clamp(0.0, 1.0)
        margin_gate = ((teacher_margin - margin_threshold) / max(1.0 - margin_threshold, 1e-6)).clamp(0.0, 1.0)
        teacher_correct = teacher_prediction.eq(targets).to(dtype=teacher_confidence.dtype)

        student_probs = F.softmax(student_logits.detach() / safe_temperature, dim=1)
        student_topk = torch.topk(student_probs, k=min(2, student_probs.size(1)), dim=1)
        student_prediction = student_topk.indices[:, 0]
        student_confidence = student_topk.values[:, 0]
        if student_topk.values.size(1) > 1:
            student_margin = student_topk.values[:, 0] - student_topk.values[:, 1]
        else:
            student_margin = torch.ones_like(student_confidence)

        student_hard = student_prediction.ne(targets).to(dtype=teacher_confidence.dtype)
        teacher_advantage = torch.maximum(
            (teacher_confidence - student_confidence).clamp(0.0, 1.0),
            (teacher_margin - student_margin).clamp(0.0, 1.0),
        )

        gate = teacher_correct * conf_gate * margin_gate * torch.maximum(student_hard, teacher_advantage)
        if hard_boost > 0.0:
            gate = gate * (1.0 + (hard_boost * student_hard))
        if not math.isclose(gate_power, 1.0):
            gate = gate.pow(gate_power)
        return gate

    def _ensure_min_gate_coverage(
        self,
        gate: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        gate_temperature: float,
    ) -> torch.Tensor:
        min_gate_ratio = min(max(float(getattr(self.config.federated, "expert_kd_min_gate_ratio", 0.0)), 0.0), 1.0)
        if gate.numel() == 0 or min_gate_ratio <= 0.0:
            return gate

        positive_mask = gate > 0
        current_count = int(positive_mask.sum().item())
        desired_count = max(int(math.ceil(gate.numel() * min_gate_ratio)), 1)
        if current_count >= desired_count:
            return gate

        safe_temperature = max(float(gate_temperature), 1e-4)
        teacher_probs = F.softmax(teacher_logits.detach() / safe_temperature, dim=1)
        teacher_topk = torch.topk(teacher_probs, k=min(2, teacher_probs.size(1)), dim=1)
        teacher_prediction = teacher_topk.indices[:, 0]
        teacher_confidence = teacher_topk.values[:, 0]
        if teacher_topk.values.size(1) > 1:
            teacher_margin = teacher_topk.values[:, 0] - teacher_topk.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_confidence)

        student_probs = F.softmax(student_logits.detach() / safe_temperature, dim=1)
        student_topk = torch.topk(student_probs, k=min(2, student_probs.size(1)), dim=1)
        student_prediction = student_topk.indices[:, 0]
        student_confidence = student_topk.values[:, 0]
        if student_topk.values.size(1) > 1:
            student_margin = student_topk.values[:, 0] - student_topk.values[:, 1]
        else:
            student_margin = torch.ones_like(student_confidence)

        student_hard = student_prediction.ne(targets).to(dtype=gate.dtype)
        teacher_advantage = torch.maximum(
            (teacher_confidence - student_confidence).clamp(0.0, 1.0),
            (teacher_margin - student_margin).clamp(0.0, 1.0),
        )
        candidate_scores = teacher_prediction.eq(targets).to(dtype=gate.dtype)
        candidate_scores = candidate_scores * ((0.7 * teacher_confidence) + (0.3 * teacher_margin))
        candidate_scores = candidate_scores * (0.5 + (0.5 * torch.maximum(student_hard, teacher_advantage)))
        candidate_scores = candidate_scores * positive_mask.logical_not().to(dtype=gate.dtype)

        available_count = int(candidate_scores.gt(0).sum().item())
        if available_count <= 0:
            return gate

        missing_count = min(max(desired_count - current_count, 0), available_count)
        if missing_count <= 0:
            return gate

        gate_floor = max(float(getattr(self.config.federated, "expert_kd_gate_floor", 0.0)), 1e-4)
        selected_indices = torch.topk(candidate_scores, k=missing_count).indices
        adjusted_gate = gate.clone()
        adjusted_gate[selected_indices] = torch.maximum(
            adjusted_gate[selected_indices],
            candidate_scores[selected_indices].clamp_min(gate_floor),
        )
        return adjusted_gate

    def _train_with_general_drel(self, general_model: WidthScalableResNet) -> float:
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )

        lambda_ge = float(getattr(self.config.federated, "lambda_ge", 1.0))
        drel_alpha = float(getattr(self.config.federated, "drel_alpha", 1.0))
        drel_beta = float(getattr(self.config.federated, "drel_beta", 8.0))
        drel_temperature = float(getattr(self.config.federated, "expert_kd_temperature", 3.0))

        self.expert_model.train()
        general_model.eval()
        total_loss = 0.0
        total_batches = 0
        total_gate_ratio = 0.0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    general_logits = general_model(images)

                expert_logits = self.expert_model(images)
                ce_loss = criterion(expert_logits, targets)
                gate = self._build_teacher_gate(
                    teacher_logits=general_logits,
                    student_logits=expert_logits,
                    targets=targets,
                    gate_temperature=drel_temperature,
                )
                gate = self._ensure_min_gate_coverage(
                    gate=gate,
                    teacher_logits=general_logits,
                    student_logits=expert_logits,
                    targets=targets,
                    gate_temperature=drel_temperature,
                )
                gate_ratio = float(gate.gt(0).to(dtype=torch.float32).mean().item())

                per_sample_kd = drel_loss_per_sample(
                    logits_teacher=general_logits,
                    logits_student=expert_logits,
                    targets=targets,
                    alpha=drel_alpha,
                    beta=drel_beta,
                    temperature=drel_temperature,
                )
                if float(gate.sum().detach().cpu().item()) > 0.0:
                    kd_loss = (per_sample_kd * gate).sum() / gate.sum().clamp_min(1e-8)
                else:
                    kd_loss = torch.zeros((), device=self.device)
                loss = ce_loss + (lambda_ge * kd_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
                total_gate_ratio += gate_ratio

        self.last_drel_gate_ratio = total_gate_ratio / max(total_batches, 1)
        return total_loss / max(total_batches, 1)

    def _update_personalized_residual(self, general_model: WidthScalableResNet) -> None:
        base_state = get_expert_state_dict(general_model, self.expert_model, block_index=self.block_index)
        current_state = self.expert_model.state_dict()
        residual: Dict[str, torch.Tensor] = {}
        for key, tensor in current_state.items():
            if key not in base_state:
                continue
            if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked")):
                continue
            if not tensor.dtype.is_floating_point:
                continue
            residual[key] = tensor.detach().cpu() - base_state[key].detach().cpu()
        self.personalized_residual = residual

    def _estimate_temperature(self, round_idx: int) -> float:
        if not bool(getattr(self.config.federated, "temperature_calibration_enabled", False)):
            return 1.0

        candidate_count = max(int(getattr(self.config.federated, "temperature_calibration_candidates", 11)), 2)
        min_temperature = float(getattr(self.config.federated, "temperature_calibration_min", 0.8))
        max_temperature = float(getattr(self.config.federated, "temperature_calibration_max", 2.5))
        subset_size = min(max(int(len(self.dataset) * 0.1), 32), 256, len(self.dataset))
        if subset_size <= 0:
            return 1.0

        rng = random.Random(_stable_client_seed(self.config.federated.seed + round_idx, self.client_id))
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        calibration_dataset = Subset(self.dataset, indices[:subset_size])
        calibration_loader = self.data_module.make_loader(calibration_dataset, shuffle=False)

        candidate_temperatures = torch.linspace(min_temperature, max_temperature, steps=candidate_count).tolist()
        best_temperature = 1.0
        best_nll = float("inf")

        self.expert_model.eval()
        with torch.no_grad():
            for temperature in candidate_temperatures:
                total_nll = 0.0
                total_samples = 0
                for images, targets, _ in calibration_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    logits = self.expert_model(images) / max(float(temperature), 1e-6)
                    total_nll += float(F.cross_entropy(logits, targets, reduction="sum").detach().cpu().item())
                    total_samples += int(targets.numel())
                mean_nll = total_nll / max(total_samples, 1)
                if mean_nll < best_nll:
                    best_nll = mean_nll
                    best_temperature = float(temperature)

        return best_temperature

    def _collect_public_predictions(
        self,
        public_batches: Sequence,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        predictions: List[torch.Tensor] = []

        self.expert_model.eval()
        with torch.no_grad():
            for images, _, _ in public_batches:
                images = images.to(self.device)
                batch_logits = self.expert_model(images) / max(self.expert_temperature, 1e-6)
                batch_probs = torch.softmax(batch_logits, dim=1)
                logits.append(batch_logits.detach().cpu())
                confidences.append(batch_probs.max(dim=1).values.detach().cpu())
                predictions.append(batch_probs.argmax(dim=1).detach().cpu())

        return (
            torch.cat(logits, dim=0),
            torch.cat(confidences, dim=0),
            torch.cat(predictions, dim=0),
        )

    @torch.no_grad()
    def predict_expert(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.expert_model.eval()
        logits = self.expert_model(images) / max(self.expert_temperature, 1e-6)
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1), probs.max(dim=1).values


class FedEGSBAServer(BaseFederatedServer):
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
            raise ValueError("fedegsba requires a non-empty public_dataset for anchor distillation.")
        super().__init__(
            config,
            client_datasets,
            client_test_datasets,
            data_module,
            test_hard_indices,
            writer,
            public_dataset=public_dataset,
        )

        holdout_ratio = max(float(getattr(config.inference, "routing_holdout_ratio", 0.0)), 0.0)
        holdout_min_samples = max(int(getattr(config.inference, "routing_holdout_min_samples", 0)), 0)
        holdout_max_samples = max(int(getattr(config.inference, "routing_holdout_max_samples", 0)), 0)
        holdout_seed_offset = int(getattr(config.inference, "routing_holdout_seed_offset", 17))
        self.client_routing_datasets: Dict[str, Dataset] = {}
        routed_client_datasets: Dict[str, Dataset] = {}
        for client_id, dataset in client_datasets.items():
            train_dataset, routing_dataset = _split_dataset_for_holdout(
                dataset,
                holdout_ratio=holdout_ratio,
                min_holdout_samples=holdout_min_samples,
                max_holdout_samples=holdout_max_samples,
                seed=_client_holdout_seed(config.federated.seed, client_id, holdout_seed_offset),
            )
            routed_client_datasets[client_id] = train_dataset
            self.client_routing_datasets[client_id] = routing_dataset

        self.client_datasets = routed_client_datasets

        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.deploy_general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.deploy_general_model.load_state_dict(_clone_state_dict(self.general_model.state_dict()))
        self.reference_expert = WidthScalableResNet(
            width_factor=config.model.expert_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.deploy_general_model)
        self.num_expert_blocks = get_num_expert_blocks(self.general_model, self.reference_expert)

        sorted_ids = sorted(self.client_datasets.keys())
        self.client_block_map = {client_id: index % self.num_expert_blocks for index, client_id in enumerate(sorted_ids)}
        self.clients: Dict[str, FedEGSBAClient] = {
            client_id: FedEGSBAClient(
                client_id=client_id,
                dataset=dataset,
                expert_width=config.model.expert_width,
                num_classes=config.model.num_classes,
                device=config.federated.device,
                config=config,
                data_module=data_module,
                block_index=self.client_block_map[client_id],
            )
            for client_id, dataset in self.client_datasets.items()
        }
        for client_id in sorted_ids:
            LOGGER.info(
                "FedEGS-BA client %s -> expert block %d/%d",
                client_id,
                self.client_block_map[client_id],
                self.num_expert_blocks,
            )
            self.clients[client_id].refresh_from_general(self.deploy_general_model)

        self.public_loader = self.data_module.make_loader(public_dataset, shuffle=False)
        self.public_batches = list(self.public_loader)
        self.public_images_cpu, self.public_targets_cpu = self._cache_public_tensors(self.public_batches)
        self.public_size = int(self.public_targets_cpu.size(0))

        base_threshold = float(config.inference.confidence_threshold)
        self.client_confidence_thresholds = {client_id: base_threshold for client_id in self.client_datasets}
        self.client_force_general = {client_id: False for client_id in self.client_datasets}
        self.client_routing_metrics: Dict[str, Dict[str, float]] = {}
        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict[str, object]] = None
        self.current_round = 0

        LOGGER.info(
            "fedegsba routing holdout | ratio=%.3f min=%d max=%d avg_holdout=%.1f avg_train=%.1f",
            holdout_ratio,
            holdout_min_samples,
            holdout_max_samples,
            (
                sum(len(dataset) for dataset in self.client_routing_datasets.values())
                / max(len(self.client_routing_datasets), 1)
            ),
            (
                sum(len(dataset) for dataset in self.client_datasets.values())
                / max(len(self.client_datasets), 1)
            ),
        )

        if bool(getattr(self.config.federated, "general_pretrain_on_public", False)):
            self._pretrain_general_on_public()
            self._sync_deploy_general()

    def _cache_public_tensors(self, public_batches: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        images: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        for batch in public_batches:
            batch_images, batch_targets, _ = batch
            images.append(batch_images.detach().cpu())
            targets.append(batch_targets.detach().cpu())
        return torch.cat(images, dim=0), torch.cat(targets, dim=0)

    def _pretrain_general_on_public(self) -> None:
        epochs = max(int(getattr(self.config.federated, "general_pretrain_epochs", 0)), 0)
        if epochs <= 0:
            return
        optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=float(getattr(self.config.federated, "general_pretrain_lr", 0.01)),
            momentum=0.9,
            weight_decay=float(getattr(self.config.federated, "local_weight_decay", 5e-4)),
        )
        criterion = nn.CrossEntropyLoss()
        self.general_model.train()
        for _ in range(epochs):
            for images, targets, _ in self.public_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.general_model(images)
                loss = criterion(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        LOGGER.info("fedegsba pre-trained general on public anchors for %d epochs", epochs)

    def _sync_deploy_general(self) -> None:
        self.deploy_general_model.load_state_dict(_clone_state_dict(self.general_model.state_dict()))

    def _public_accuracy_for_model(self, model: nn.Module) -> float:
        if self.public_size == 0:
            return 0.0
        model.eval()
        predictions: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, self.public_size, self.config.dataset.batch_size):
                images = self.public_images_cpu[start:start + self.config.dataset.batch_size].to(self.device)
                predictions.append(model(images).argmax(dim=1).detach().cpu())
        predicted = torch.cat(predictions, dim=0)
        return float((predicted == self.public_targets_cpu).float().mean().item())

    def _aggregate_public_teachers(
        self,
        updates: List[FedEGSBAClientUpdate],
        round_idx: int,
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("fedegsba requires at least one client update to aggregate public teachers.")

        temperature = max(float(self.config.federated.distill_temperature), 1e-6)
        warmup_rounds = max(int(getattr(self.config.federated, "general_warmup_rounds", 0)), 0)
        confidence_threshold = float(getattr(self.config.federated, "public_teacher_confidence_threshold", 0.60))
        weight_power = max(float(getattr(self.config.federated, "public_teacher_weight_power", 2.0)), 0.0)
        topk = max(int(getattr(self.config.federated, "public_teacher_topk", 0)), 0)
        min_reliability = max(float(getattr(self.config.federated, "min_client_reliability", 0.05)), 0.0)

        stacked_logits = torch.stack([update.public_logits for update in updates], dim=0)
        stacked_confidence = torch.stack([update.public_confidence for update in updates], dim=0)
        stacked_predictions = torch.stack([update.public_predictions for update in updates], dim=0)
        targets = self.public_targets_cpu
        num_clients = stacked_logits.size(0)
        num_samples = stacked_logits.size(1)

        class_accuracy = torch.full(
            (num_clients, self.config.model.num_classes),
            min_reliability,
            dtype=torch.float32,
        )
        for client_idx in range(num_clients):
            predictions = stacked_predictions[client_idx]
            for class_idx in range(self.config.model.num_classes):
                mask = targets == class_idx
                if bool(mask.any().item()):
                    class_accuracy[client_idx, class_idx] = (
                        (predictions[mask] == targets[mask]).float().mean().item()
                    )

        aggregated_logits = torch.zeros_like(stacked_logits[0])
        sample_weights = torch.zeros(num_samples, dtype=torch.float32)
        teacher_strength = torch.zeros(num_samples, dtype=torch.float32)

        for sample_idx in range(num_samples):
            if round_idx <= warmup_rounds:
                weights = torch.ones(num_clients, dtype=torch.float32)
            else:
                target_class = int(targets[sample_idx].item())
                weights = (stacked_confidence[:, sample_idx].float().clamp_min(0.0) ** weight_power)
                weights = weights * (class_accuracy[:, target_class].float() + min_reliability)
                weights = torch.where(
                    stacked_confidence[:, sample_idx] >= confidence_threshold,
                    weights,
                    torch.zeros_like(weights),
                )
                if topk > 0 and topk < num_clients:
                    top_indices = torch.topk(weights, k=topk).indices
                    keep_mask = torch.zeros_like(weights, dtype=torch.bool)
                    keep_mask[top_indices] = True
                    weights = torch.where(keep_mask, weights, torch.zeros_like(weights))
                if float(weights.sum().item()) <= 0.0:
                    weights = torch.ones(num_clients, dtype=torch.float32)

            normalized = weights / weights.sum().clamp_min(1e-8)
            aggregated_logits[sample_idx] = torch.sum(
                stacked_logits[:, sample_idx, :] * normalized.unsqueeze(1),
                dim=0,
            )
            sample_weights[sample_idx] = torch.sum(normalized * stacked_confidence[:, sample_idx].float())
            teacher_strength[sample_idx] = float(weights.max().item())

        soft_labels = F.softmax(aggregated_logits / temperature, dim=1)
        return {
            "aggregated_logits": aggregated_logits,
            "soft_labels": soft_labels,
            "sample_weights": sample_weights.clamp_min(0.05),
            "teacher_strength": teacher_strength,
            "mean_confidence": stacked_confidence.mean(),
        }

    def _distill_general_model(
        self,
        teacher_bundle: Dict[str, torch.Tensor],
        round_idx: int,
    ) -> Dict[str, float]:
        temperature = max(float(self.config.federated.distill_temperature), 1e-6)
        distill_epochs = max(int(self.config.federated.distill_epochs), 1)
        batch_size = max(int(self.config.dataset.batch_size), 1)
        public_ce_weight = max(float(getattr(self.config.federated, "public_ce_weight", 1.0)), 0.0)
        public_kd_weight = max(float(getattr(self.config.federated, "public_logit_align_weight", 1.0)), 0.0)
        warmup_rounds = max(int(getattr(self.config.federated, "general_warmup_rounds", 0)), 0)
        ramp_rounds = max(int(getattr(self.config.federated, "general_distill_ramp_rounds", 1)), 1)

        if round_idx <= warmup_rounds:
            public_kd_weight = 0.0
        else:
            ramp_progress = min(float(round_idx - warmup_rounds) / float(ramp_rounds), 1.0)
            public_kd_weight = public_kd_weight * ramp_progress

        optimizer = torch.optim.Adam(self.general_model.parameters(), lr=float(self.config.federated.distill_lr))
        sample_weights = teacher_bundle["sample_weights"]
        soft_labels = teacher_bundle["soft_labels"]
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_batches = 0

        self.general_model.train()
        num_samples = int(self.public_size)
        for _ in range(distill_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                indices = permutation[start:start + batch_size]
                images = self.public_images_cpu[indices].to(self.device)
                targets = self.public_targets_cpu[indices].to(self.device)
                teacher_probs = soft_labels[indices].to(self.device)
                batch_weights = sample_weights[indices].to(self.device)

                logits = self.general_model(images)
                ce_loss = F.cross_entropy(logits, targets)
                log_probs = F.log_softmax(logits / temperature, dim=1)
                per_sample_kd = F.kl_div(
                    log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=1) * (temperature ** 2)
                kd_loss = (per_sample_kd * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
                loss = (public_ce_weight * ce_loss) + (public_kd_weight * kd_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_ce += float(ce_loss.detach().cpu().item())
                total_kd += float(kd_loss.detach().cpu().item())
                total_batches += 1

        denominator = max(total_batches, 1)
        return {
            "total_loss": total_loss / denominator,
            "public_ce_loss": total_ce / denominator,
            "public_kd_loss": total_kd / denominator,
            "kd_weight": public_kd_weight,
        }

    def _routing_threshold_for_client(self, client_id: str) -> float:
        threshold = float(
            self.client_confidence_thresholds.get(
                client_id,
                float(self.config.inference.confidence_threshold),
            )
        )
        if self.current_round <= int(getattr(self.config.inference, "route_warmup_rounds", 0)):
            threshold = max(
                threshold,
                float(getattr(self.config.inference, "route_warmup_confidence_threshold", threshold)),
            )
        return threshold

    def _threshold_candidates(self, current_threshold: float, confidences: torch.Tensor) -> List[float]:
        min_threshold = float(getattr(self.config.inference, "min_confidence_threshold", 0.0))
        max_threshold = float(getattr(self.config.inference, "max_confidence_threshold", 1.0))
        step = max(float(getattr(self.config.inference, "personalized_threshold_step", 0.0)), 0.0)
        search_radius = max(int(getattr(self.config.inference, "routing_search_radius", 2)), 0)

        candidates = {min(max(current_threshold, min_threshold), max_threshold)}
        if step > 0.0:
            for offset in range(-search_radius, search_radius + 1):
                candidates.add(min(max(current_threshold + (offset * step), min_threshold), max_threshold))
        if confidences.numel() > 0:
            for quantile in (0.20, 0.35, 0.50, 0.65, 0.80):
                value = float(torch.quantile(confidences, quantile).item())
                candidates.add(min(max(value, min_threshold), max_threshold))
        candidates.add(min_threshold)
        candidates.add(max_threshold)
        return sorted(candidates)

    def _collect_client_routing_statistics(self, client_id: str) -> Optional[Dict[str, torch.Tensor]]:
        dataset = self.client_routing_datasets.get(client_id)
        if dataset is None or len(dataset) == 0:
            return None

        loader = self.data_module.make_loader(dataset, shuffle=False)
        expert_predictions: List[torch.Tensor] = []
        expert_confidences: List[torch.Tensor] = []
        general_predictions: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []

        client = self.clients[client_id]
        client.expert_model.eval()
        self.deploy_general_model.eval()

        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                expert_logits = client.expert_model(images) / max(client.expert_temperature, 1e-6)
                expert_probs = torch.softmax(expert_logits, dim=1)
                expert_predictions.append(expert_probs.argmax(dim=1).detach().cpu())
                expert_confidences.append(expert_probs.max(dim=1).values.detach().cpu())
                general_predictions.append(self.deploy_general_model(images).argmax(dim=1).detach().cpu())
                targets_all.append(targets.detach().cpu())

        if not targets_all:
            return None

        return {
            "expert_predictions": torch.cat(expert_predictions, dim=0),
            "confidences": torch.cat(expert_confidences, dim=0),
            "general_predictions": torch.cat(general_predictions, dim=0),
            "targets": torch.cat(targets_all, dim=0),
        }

    def _update_routing_policy(self, client_ids: Sequence[str]) -> Dict[str, float]:
        warmup_rounds = int(getattr(self.config.inference, "route_warmup_rounds", 0))
        force_general_gap = float(getattr(self.config.inference, "client_force_general_gap", 0.12))

        forced_clients = 0
        holdout_routed_acc = 0.0
        average_threshold = 0.0
        updated_clients = 0

        for client_id in client_ids:
            routing_stats = self._collect_client_routing_statistics(client_id)
            if routing_stats is None:
                continue

            expert_predictions = routing_stats["expert_predictions"]
            confidences = routing_stats["confidences"]
            general_predictions = routing_stats["general_predictions"]
            targets = routing_stats["targets"]

            expert_accuracy = float(expert_predictions.eq(targets).to(dtype=torch.float32).mean().item())
            general_accuracy = float(general_predictions.eq(targets).to(dtype=torch.float32).mean().item())

            force_general = (
                self.current_round > warmup_rounds
                and (general_accuracy > (expert_accuracy + force_general_gap))
            )
            self.client_force_general[client_id] = force_general
            if force_general:
                forced_clients += 1
                self.client_routing_metrics[client_id] = {
                    "expert_accuracy": expert_accuracy,
                    "general_accuracy": general_accuracy,
                    "routed_accuracy": general_accuracy,
                    "invocation_rate": 1.0,
                    "threshold": 1.0,
                    "forced_general": 1.0,
                }
                holdout_routed_acc += general_accuracy
                average_threshold += 1.0
                updated_clients += 1
                continue

            threshold = self.client_confidence_thresholds.get(
                client_id,
                float(self.config.inference.confidence_threshold),
            )
            if self.current_round > warmup_rounds:
                best_candidate = None
                for candidate_threshold in self._threshold_candidates(float(threshold), confidences):
                    fallback_mask = confidences < candidate_threshold
                    routed_predictions = torch.where(
                        fallback_mask,
                        general_predictions,
                        expert_predictions,
                    )
                    accuracy = float(routed_predictions.eq(targets).to(dtype=torch.float32).mean().item())
                    invocation = float(fallback_mask.to(dtype=torch.float32).mean().item())
                    candidate = {
                        "threshold": candidate_threshold,
                        "accuracy": accuracy,
                        "invocation": invocation,
                    }
                    if best_candidate is None:
                        best_candidate = candidate
                        continue
                    if accuracy > best_candidate["accuracy"] + 1e-8:
                        best_candidate = candidate
                    elif abs(accuracy - best_candidate["accuracy"]) <= 1e-8:
                        if invocation < best_candidate["invocation"] - 1e-8:
                            best_candidate = candidate
                        elif abs(invocation - best_candidate["invocation"]) <= 1e-8 and candidate_threshold < best_candidate["threshold"]:
                            best_candidate = candidate

                if best_candidate is not None:
                    threshold = float(best_candidate["threshold"])

            self.client_confidence_thresholds[client_id] = float(threshold)
            effective_threshold = self._routing_threshold_for_client(client_id)
            fallback_mask = confidences < effective_threshold
            routed_predictions = torch.where(fallback_mask, general_predictions, expert_predictions)
            routed_accuracy = float(routed_predictions.eq(targets).to(dtype=torch.float32).mean().item())
            invocation_rate = float(fallback_mask.to(dtype=torch.float32).mean().item())

            self.client_routing_metrics[client_id] = {
                "expert_accuracy": expert_accuracy,
                "general_accuracy": general_accuracy,
                "routed_accuracy": routed_accuracy,
                "invocation_rate": invocation_rate,
                "threshold": effective_threshold,
                "forced_general": 0.0,
            }
            holdout_routed_acc += routed_accuracy
            average_threshold += effective_threshold
            updated_clients += 1

        if updated_clients == 0:
            return {
                "holdout_routed_accuracy": 0.0,
                "average_threshold": float(self.config.inference.confidence_threshold),
                "forced_clients": 0.0,
            }

        return {
            "holdout_routed_accuracy": holdout_routed_acc / updated_clients,
            "average_threshold": average_threshold / updated_clients,
            "forced_clients": float(forced_clients),
        }

    def _predict_general_only(self, client_id, images, indices):
        self.deploy_general_model.eval()
        with torch.no_grad():
            predictions = self.deploy_general_model(images).argmax(dim=1)
        return predictions, 0

    def _predict_expert_only(self, client_id, images, indices):
        predictions, _ = self.clients[client_id].predict_expert(images)
        return predictions, 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        self.deploy_general_model.eval()

        if (
            self.current_round > int(getattr(self.config.inference, "route_warmup_rounds", 0))
            and self.client_force_general.get(client_id, False)
        ):
            with torch.no_grad():
                general_predictions = self.deploy_general_model(images).argmax(dim=1)
            return general_predictions, int(images.size(0)), {
                "route_type": ["general"] * images.size(0),
                "expert_confidence": [0.0] * images.size(0),
            }

        threshold = self._routing_threshold_for_client(client_id)
        with torch.no_grad():
            expert_predictions, expert_confidence = client.predict_expert(images)
            fallback_mask = expert_confidence < threshold
            predictions = expert_predictions.clone()
            invocation_count = int(fallback_mask.sum().item())
            route_types = ["expert"] * images.size(0)
            if invocation_count > 0:
                general_predictions = self.deploy_general_model(images[fallback_mask]).argmax(dim=1)
                predictions[fallback_mask] = general_predictions
                for sample_idx in fallback_mask.nonzero(as_tuple=False).flatten().tolist():
                    route_types[sample_idx] = "general"

        return predictions, invocation_count, {
            "route_type": route_types,
            "expert_confidence": expert_confidence.detach().cpu().tolist(),
        }

    def _maybe_update_best(
        self,
        round_idx: int,
        round_metrics: RoundMetrics,
        expert_accuracy: float,
        general_accuracy: float,
    ) -> bool:
        if self.best_snapshot is not None and round_metrics.routed_accuracy <= float(self.best_snapshot["routed_accuracy"]):
            return False

        self.best_snapshot = {
            "round_idx": round_idx,
            "routed_accuracy": round_metrics.routed_accuracy,
            "general_accuracy": general_accuracy,
            "expert_accuracy": expert_accuracy,
            "avg_client_loss": round_metrics.avg_client_loss,
            "general_model_state": _clone_state_dict(self.general_model.state_dict()),
            "deploy_general_model_state": _clone_state_dict(self.deploy_general_model.state_dict()),
            "client_expert_states": {
                client_id: _clone_state_dict(client.expert_model.state_dict())
                for client_id, client in self.clients.items()
            },
            "client_temperatures": {
                client_id: float(client.expert_temperature)
                for client_id, client in self.clients.items()
            },
            "client_confidence_thresholds": {
                client_id: float(threshold)
                for client_id, threshold in self.client_confidence_thresholds.items()
            },
            "client_force_general": {
                client_id: bool(flag)
                for client_id, flag in self.client_force_general.items()
            },
        }
        LOGGER.info(
            "fedegsba best | round=%d | routed=%.4f | general=%.4f | expert=%.4f",
            round_idx,
            round_metrics.routed_accuracy,
            general_accuracy,
            expert_accuracy,
        )
        return True

    def _restore_best(self) -> None:
        if not self.best_snapshot:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        self.deploy_general_model.load_state_dict(self.best_snapshot["deploy_general_model_state"])
        for client_id, expert_state in self.best_snapshot["client_expert_states"].items():
            self.clients[client_id].expert_model.load_state_dict(expert_state)
        for client_id, temperature in self.best_snapshot["client_temperatures"].items():
            self.clients[client_id].expert_temperature = float(temperature)
        if "client_confidence_thresholds" in self.best_snapshot:
            self.client_confidence_thresholds = {
                client_id: float(threshold)
                for client_id, threshold in self.best_snapshot["client_confidence_thresholds"].items()
            }
        if "client_force_general" in self.best_snapshot:
            self.client_force_general = {
                client_id: bool(flag)
                for client_id, flag in self.best_snapshot["client_force_general"].items()
            }
        self.current_round = int(self.best_snapshot["round_idx"])
        LOGGER.info("fedegsba restored best snapshot from round %d", int(self.best_snapshot["round_idx"]))

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        kd_warmup_rounds = max(int(getattr(self.config.federated, "expert_kd_warmup_rounds", 0)), 0)
        min_general_accuracy = max(float(getattr(self.config.federated, "expert_kd_min_general_accuracy", 0.0)), 0.0)

        for round_idx in range(1, self.config.federated.rounds + 1):
            self.current_round = round_idx
            selected_client_ids = self._sample_client_ids()
            deploy_public_accuracy = self._public_accuracy_for_model(self.deploy_general_model)
            use_kd = (round_idx > kd_warmup_rounds) and (deploy_public_accuracy >= min_general_accuracy)

            LOGGER.info(
                "fedegsba round %d | clients=%s | kd=%s | deploy_public_acc=%.4f",
                round_idx,
                selected_client_ids,
                use_kd,
                deploy_public_accuracy,
            )

            updates = [
                self.clients[client_id].train_round(
                    round_idx=round_idx,
                    general_model=self.deploy_general_model,
                    public_batches=self.public_batches,
                    use_kd=use_kd,
                )
                for client_id in selected_client_ids
            ]
            average_client_loss = sum(update.loss for update in updates) / max(len(updates), 1)

            teacher_bundle = self._aggregate_public_teachers(updates, round_idx)
            distill_stats = self._distill_general_model(teacher_bundle, round_idx)
            self._sync_deploy_general()
            routing_stats = self._update_routing_policy(sorted(self.client_datasets.keys()))

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsba-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsba-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, "fedegsba-routed")
            extra_metrics = self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)

            aggregate_metrics = routed_eval["aggregate"]
            macro_metrics = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate_metrics["invocation_rate"],
                mode="routed",
            )
            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=average_client_loss,
                routed_accuracy=macro_metrics["accuracy"],
                hard_accuracy=aggregate_metrics["hard_recall"],
                invocation_rate=aggregate_metrics["invocation_rate"],
                local_accuracy=macro_metrics["accuracy"],
                weighted_accuracy=aggregate_metrics["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
                extra_metrics=extra_metrics,
            )

            LOGGER.info(
                "fedegsba round %d | loss=%.4f | distill=%.4f | public_ce=%.4f | public_kd=%.4f | kd_weight=%.4f | teacher_conf=%.4f | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f | holdout_routed=%.4f | threshold=%.4f | forced=%d",
                round_idx,
                average_client_loss,
                distill_stats["total_loss"],
                distill_stats["public_ce_loss"],
                distill_stats["public_kd_loss"],
                distill_stats["kd_weight"],
                float(teacher_bundle["mean_confidence"].item()),
                macro_metrics["accuracy"],
                aggregate_metrics["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate_metrics["hard_recall"],
                aggregate_metrics["invocation_rate"],
                routing_stats["holdout_routed_accuracy"],
                routing_stats["average_threshold"],
                int(routing_stats["forced_clients"]),
            )

            if self.writer is not None:
                self.writer.add_scalar("expert_loss/fedegsba", average_client_loss, round_idx)
                self.writer.add_scalar("distill_loss/fedegsba", distill_stats["total_loss"], round_idx)
                self.writer.add_scalar("distill_ce/fedegsba", distill_stats["public_ce_loss"], round_idx)
                self.writer.add_scalar("distill_kd/fedegsba", distill_stats["public_kd_loss"], round_idx)
                self.writer.add_scalar("teacher_confidence/fedegsba", float(teacher_bundle["mean_confidence"].item()), round_idx)
                self.writer.add_scalar("routing_holdout_accuracy/fedegsba", routing_stats["holdout_routed_accuracy"], round_idx)
                self.writer.add_scalar("routing_threshold/fedegsba", routing_stats["average_threshold"], round_idx)
                self.writer.add_scalar("routing_forced_clients/fedegsba", routing_stats["forced_clients"], round_idx)
                self._log_auxiliary_accuracy_metrics(
                    "fedegsba",
                    round_idx,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics("fedegsba", round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best(
                round_idx,
                round_metrics,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

        self.last_history = metrics
        if bool(getattr(self.config.federated, "restore_best_checkpoint", True)):
            self._restore_best()
        return metrics

    def evaluate_baselines(self, test_dataset):
        route_path = self._build_route_export_path("fedegsba_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            "fedegsba_final_routed",
            route_export_path=route_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, "fedegsba_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, "fedegsba_final_general")
        extra_metrics = self._build_final_extra_metrics(expert_eval, general_eval, routed_eval)
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0
        best_round = 0
        if self.best_snapshot is not None:
            final_loss = float(self.best_snapshot["avg_client_loss"])
            best_round = int(self.best_snapshot["round_idx"])

        routed_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            routed_eval["aggregate"]["invocation_rate"],
            mode="routed",
        )
        expert_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            0.0,
            mode="expert_only",
        )
        general_compute = self._build_compute_profile(
            self.expert_flops,
            self.general_flops,
            1.0,
            mode="general_only",
        )

        return {
            "algorithm": "fedegsba",
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
                "public_dataset_size": self.public_size,
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
            },
            "memory_mb": {
                "expert": model_memory_mb(self.reference_expert),
                "general": model_memory_mb(self.deploy_general_model),
            },
            "artifacts": {"route_csv": str(route_path)},
        }

    def _build_round_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._evaluate_route_effectiveness_metrics_from_predictors(
            expert_eval,
            general_eval,
            routed_eval,
            self._predict_expert_only,
            self._predict_general_only,
            self._predict_routed,
            general_route_types=("general",),
        )

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return self._build_round_extra_metrics(expert_eval, general_eval, routed_eval)
