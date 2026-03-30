from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.models import SmallCNN, WidthScalableResNet, model_memory_mb


@dataclass
class FedEGS2ClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    public_logits: torch.Tensor


class FedEGS2Client(BaseFederatedClient):
    def __init__(self, client_id: str, dataset: Dataset, num_classes: int, device: str, config, data_module) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.data_module = data_module
        self.expert_model = SmallCNN(num_classes=num_classes).to(self.device)

    def train(self, public_loader: DataLoader) -> FedEGS2ClientUpdate:
        local_loader = self.data_module.make_loader(self.dataset, shuffle=True)
        loss = self._optimize_model(
            model=self.expert_model,
            loader=local_loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        public_logits = self._predict_public_logits(public_loader)
        return FedEGS2ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            public_logits=public_logits,
        )

    def _predict_public_logits(self, public_loader: DataLoader) -> torch.Tensor:
        self.expert_model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for images, _, _ in public_loader:
                images = images.to(self.device)
                outputs.append(self.expert_model(images).detach().cpu())
        return torch.cat(outputs, dim=0)


class FedEGS2Server(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None, public_dataset: Dataset | None = None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
        if self.public_dataset is None or len(self.public_dataset) == 0:
            raise ValueError("FedEGS-2 requires a non-empty public distillation dataset.")

        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.clients = {
            client_id: FedEGS2Client(client_id, dataset, config.model.num_classes, config.federated.device, config, data_module)
            for client_id, dataset in client_datasets.items()
        }
        self.public_eval_loader = self.data_module.make_loader(self.public_dataset, shuffle=False)
        self.last_history: List[RoundMetrics] = []

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            LOGGER.info("fedegs2 round %d sampled clients=%s", round_idx, sampled_ids)
            updates: List[FedEGS2ClientUpdate] = []

            for client_id in sampled_ids:
                updates.append(self.clients[client_id].train(self.public_eval_loader))

            teacher_logits = self._aggregate_public_logits(updates)
            distill_loss = self._distill_general_model(teacher_logits)
            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs2-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs2-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs2-routed")
            aggregate = routed_eval["aggregate"]
            round_metrics = RoundMetrics(round_idx, avg_loss, aggregate["accuracy"], aggregate["hard_accuracy"], aggregate["invocation_rate"])

            LOGGER.info(
                "fedegs2 round %d | local_loss=%.4f | distill_loss=%.4f | routed_acc=%.4f | hard_acc=%.4f | invocation=%.4f",
                round_idx,
                avg_loss,
                distill_loss,
                aggregate["accuracy"],
                aggregate["hard_accuracy"],
                aggregate["invocation_rate"],
            )
            LOGGER.info(
                "fedegs2 auxiliary round %d | expert_acc=%.4f | general_acc=%.4f",
                round_idx,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )

            if self.writer is not None:
                self.writer.add_scalar("distill_loss/fedegs2", distill_loss, round_idx)
                self.writer.add_scalar("accuracy/fedegs2_expert", expert_eval["aggregate"]["accuracy"], round_idx)
                self.writer.add_scalar("accuracy/fedegs2_general", general_eval["aggregate"]["accuracy"], round_idx)

            self._log_round_metrics("fedegs2", round_metrics)
            metrics.append(round_metrics)

        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset):
        routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs2_final_routed")
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs2_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs2_final_general")
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0

        return {
            "algorithm": "fedegs2",
            "metrics": {
                "routed_accuracy": routed_eval["aggregate"]["accuracy"],
                "routed_hard_accuracy": routed_eval["aggregate"]["hard_accuracy"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["aggregate"]["accuracy"],
                "expert_only_recall_macro": expert_eval["aggregate"]["recall_macro"],
                "general_only_accuracy": general_eval["aggregate"]["accuracy"],
                "general_only_recall_macro": general_eval["aggregate"]["recall_macro"],
                "public_dataset_size": len(self.public_dataset),
                "final_training_loss": final_loss,
            },
            "client_metrics": {
                "routed": routed_eval["clients"],
                "expert_only": expert_eval["clients"],
                "general_only": general_eval["clients"],
            },
            "memory_mb": {
                "expert": model_memory_mb(next(iter(self.clients.values())).expert_model),
                "general": model_memory_mb(self.general_model),
            },
        }

    def _aggregate_public_logits(self, updates: List[FedEGS2ClientUpdate]) -> torch.Tensor:
        total_weight = sum(update.num_samples for update in updates)
        aggregate = torch.zeros_like(updates[0].public_logits)
        for update in updates:
            aggregate += update.public_logits * float(update.num_samples)
        aggregate /= max(float(total_weight), 1.0)
        return aggregate

    def _distill_general_model(self, teacher_logits: torch.Tensor) -> float:
        temperature = self.config.federated.distill_temperature
        optimizer = torch.optim.SGD(
            self.general_model.parameters(),
            lr=self.config.federated.distill_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        losses: List[float] = []
        teacher_cursor = 0

        for _ in range(self.config.federated.distill_epochs):
            self.general_model.train()
            teacher_cursor = 0
            for images, _, _ in self.public_eval_loader:
                batch_size = images.size(0)
                batch_teacher = teacher_logits[teacher_cursor : teacher_cursor + batch_size].to(self.device)
                teacher_cursor += batch_size
                images = images.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.general_model(images)
                loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(batch_teacher / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        return sum(losses) / max(len(losses), 1)

    def _predict_general_only(self, client_id, images, indices):
        self.general_model.eval()
        logits = self.general_model(images)
        return logits.argmax(dim=1), 0

    def _predict_expert_only(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        logits = client.expert_model(images)
        return logits.argmax(dim=1), 0

    def _predict_routed(self, client_id, images, indices):
        client = self.clients[client_id]
        client.expert_model.eval()
        self.general_model.eval()
        expert_logits = client.expert_model(images)
        expert_probs = torch.softmax(expert_logits, dim=1)
        expert_confidence, expert_prediction = torch.max(expert_probs, dim=1)
        use_expert_mask = expert_confidence > self.config.inference.high_threshold
        predictions = expert_prediction.clone()
        invoked_general = 0
        if (~use_expert_mask).any():
            general_logits = self.general_model(images[~use_expert_mask])
            predictions[~use_expert_mask] = torch.argmax(general_logits, dim=1)
            invoked_general = int((~use_expert_mask).sum().item())
        return predictions, invoked_general
