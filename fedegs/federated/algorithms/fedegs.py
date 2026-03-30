from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedClient, BaseFederatedServer, ClientUpdate, RoundMetrics, LOGGER
from fedegs.models import (
    WidthScalableResNet,
    apply_expert_delta_to_general,
    average_weighted_deltas,
    get_num_expert_blocks,
    load_expert_state_dict,
    model_memory_mb,
)
from fedegs.models.width_scalable_resnet import state_dict_delta


class FedEGSClient(BaseFederatedClient):
    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        expert_width: float,
        num_classes: int,
        device: str,
        config,
        block_index: int,
    ) -> None:
        super().__init__(client_id, dataset, device)
        self.config = config
        self.block_index = block_index
        self.expert_model = WidthScalableResNet(width_factor=expert_width, num_classes=num_classes).to(self.device)

    def train(self, general_model: WidthScalableResNet, loader: DataLoader) -> ClientUpdate:
        load_expert_state_dict(general_model, self.expert_model, block_index=self.block_index)
        before_state = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        loss = self._optimize_model(
            model=self.expert_model,
            loader=loader,
            epochs=self.config.federated.local_epochs,
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        after_state = self.expert_model.state_dict()
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            delta=state_dict_delta(after_state, before_state),
        )


class FedEGSServer(BaseFederatedServer):
    def __init__(self, config, client_datasets: Dict[str, Dataset], client_test_datasets: Dict[str, Dataset], data_module, test_hard_indices, writer=None) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.reference_expert = WidthScalableResNet(
            width_factor=config.model.expert_width,
            num_classes=config.model.num_classes,
        )
        self.num_expert_blocks = get_num_expert_blocks(self.general_model, self.reference_expert)

        sorted_client_ids = sorted(client_datasets.keys())
        self.client_block_map = {
            client_id: index % self.num_expert_blocks for index, client_id in enumerate(sorted_client_ids)
        }
        self.clients = {
            client_id: FedEGSClient(
                client_id=client_id,
                dataset=dataset,
                expert_width=config.model.expert_width,
                num_classes=config.model.num_classes,
                device=config.federated.device,
                config=config,
                block_index=self.client_block_map[client_id],
            )
            for client_id, dataset in client_datasets.items()
        }

        for client_id in sorted_client_ids:
            LOGGER.info("FedEGS client %s assigned expert block %d/%d", client_id, self.client_block_map[client_id], self.num_expert_blocks)

        self.last_history: List[RoundMetrics] = []

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            updates: List[ClientUpdate] = []
            sampled_blocks = [self.client_block_map[client_id] for client_id in sampled_ids]
            LOGGER.info("fedegs round %d sampled clients=%s blocks=%s", round_idx, sampled_ids, sampled_blocks)

            for client_id in sampled_ids:
                loader = self.data_module.make_loader(self.clients[client_id].dataset, shuffle=True)
                updates.append(self.clients[client_id].train(self.general_model, loader))

            block_groups: Dict[int, List[ClientUpdate]] = {}
            for update in updates:
                block_index = self.client_block_map[update.client_id]
                block_groups.setdefault(block_index, []).append(update)

            for block_index, group_updates in block_groups.items():
                aggregated_delta = average_weighted_deltas((update.num_samples, update.delta) for update in group_updates)
                apply_expert_delta_to_general(
                    self.general_model,
                    aggregated_delta,
                    self.reference_expert,
                    block_index=block_index,
                )

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs-routed")
            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            metrics = [
                RoundMetrics(
                    round_idx=round_idx,
                    avg_client_loss=avg_loss,
                    routed_accuracy=aggregate["accuracy"],
                    hard_accuracy=aggregate["hard_accuracy"],
                    invocation_rate=aggregate["invocation_rate"],
                )
                for aggregate in [expert_eval["aggregate"], general_eval["aggregate"], routed_eval["aggregate"]]
            ]

            for prefix, aggregate in zip(["expert ", "general", "routed "],[expert_eval["aggregate"], general_eval["aggregate"], routed_eval["aggregate"]]):
                LOGGER.info(
                    f"fedegs-{prefix} round {round_idx} | loss={avg_loss:.4f} | acc={aggregate['accuracy']:.4f} | f1={aggregate['f1_macro']:.4f} | invocation={aggregate['invocation_rate']:.4f}")
            
            round_metrics = metrics[-1]  # routed metrics are primary
            self._log_round_metrics("fedegs", round_metrics)
            metrics.append(round_metrics)
        self.last_history = metrics
        return metrics

    def evaluate_baselines(self, test_dataset: Dataset) -> Dict[str, Dict[str, float]]:
        routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs_final_routed")
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs_final_general")
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0

        if self.writer is not None:
            self.writer.add_scalar("fedegs_summary/routed_accuracy", routed_eval["aggregate"]["accuracy"], 0)
            self.writer.add_scalar("fedegs_summary/expert_only_accuracy", expert_eval["aggregate"]["accuracy"], 0)
            self.writer.add_scalar("fedegs_summary/general_only_accuracy", general_eval["aggregate"]["accuracy"], 0)
            self.writer.add_scalar("memory/expert_mb", model_memory_mb(self.reference_expert), 0)
            self.writer.add_scalar("memory/general_mb", model_memory_mb(self.general_model), 0)

        return {
            "algorithm": "fedegs",
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
                "final_training_loss": final_loss,
            },
            "client_metrics": {
                "routed": routed_eval["clients"],
                "expert_only": expert_eval["clients"],
                "general_only": general_eval["clients"],
            },
            "memory_mb": {
                "expert": model_memory_mb(self.reference_expert),
                "general": model_memory_mb(self.general_model),
            },
        }

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
