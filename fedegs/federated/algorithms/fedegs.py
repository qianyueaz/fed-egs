from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import (
    BaseFederatedClient,
    BaseFederatedServer,
    ClientUpdate,
    HyperKnowledge,
    LOGGER,
    RoundMetrics,
)
from fedegs.models import (
    WidthScalableResNet,
    apply_expert_delta_to_general,
    average_weighted_deltas,
    estimate_model_flops,
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

    def train(
        self,
        general_model: WidthScalableResNet,
        global_knowledge: HyperKnowledge,
        train_loader: DataLoader,
        knowledge_loader: DataLoader,
    ) -> ClientUpdate:
        load_expert_state_dict(general_model, self.expert_model, block_index=self.block_index)
        before_state = {key: value.detach().cpu().clone() for key, value in self.expert_model.state_dict().items()}
        loss = self._train_local_expert(train_loader, global_knowledge)
        hyper_knowledge = self._extract_hyper_knowledge(knowledge_loader)
        after_state = self.expert_model.state_dict()
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=len(self.dataset),
            loss=loss,
            delta=state_dict_delta(after_state, before_state),
            hyper_knowledge=hyper_knowledge,
        )

    def _train_local_expert(self, loader: DataLoader, global_knowledge: HyperKnowledge) -> float:
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()
        mse = torch.nn.MSELoss()
        temperature = self.config.federated.distill_temperature
        feature_weight = self.config.federated.feature_align_weight
        logit_weight = self.config.federated.logit_align_weight

        knowledge_features = {
            label: tensor.to(self.device)
            for label, tensor in (global_knowledge.features if global_knowledge else {}).items()
        }
        knowledge_soft_predictions = {
            label: tensor.to(self.device)
            for label, tensor in (global_knowledge.soft_predictions if global_knowledge else {}).items()
        }

        self.expert_model.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                features, logits = self.expert_model.forward_with_features(images)
                ce_loss = criterion(logits, targets)

                alignment_mask = torch.tensor(
                    [int(label.item()) in knowledge_features for label in targets],
                    device=self.device,
                    dtype=torch.bool,
                )

                if alignment_mask.any():
                    masked_targets = targets[alignment_mask].detach().cpu().tolist()
                    target_features = torch.stack([knowledge_features[label] for label in masked_targets], dim=0)
                    target_probs = torch.stack([knowledge_soft_predictions[label] for label in masked_targets], dim=0)
                    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    feature_loss = mse(features[alignment_mask], target_features)
                    logit_loss = F.kl_div(
                        F.log_softmax(logits[alignment_mask] / temperature, dim=1),
                        target_probs,
                        reduction="batchmean",
                    ) * (temperature ** 2)
                else:
                    feature_loss = ce_loss.new_zeros(())
                    logit_loss = ce_loss.new_zeros(())

                loss = ce_loss + feature_weight * feature_loss + logit_weight * logit_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _extract_hyper_knowledge(self, loader: DataLoader) -> HyperKnowledge:
        temperature = self.config.federated.distill_temperature
        noise_std = self.config.federated.feature_noise_std
        feature_sums: Dict[int, torch.Tensor] = {}
        prob_sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        self.expert_model.eval()
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(self.device)
                features, logits = self.expert_model.forward_with_features(images)
                probs = torch.softmax(logits / temperature, dim=1)

                for sample_idx, target in enumerate(targets.tolist()):
                    label = int(target)
                    feature = features[sample_idx].detach().cpu()
                    prob = probs[sample_idx].detach().cpu()
                    counts[label] = counts.get(label, 0) + 1
                    if label not in feature_sums:
                        feature_sums[label] = feature.clone()
                        prob_sums[label] = prob.clone()
                    else:
                        feature_sums[label] += feature
                        prob_sums[label] += prob

        features = {}
        soft_predictions = {}
        for label, count in counts.items():
            feature_mean = feature_sums[label] / float(count)
            if noise_std > 0:
                feature_mean = feature_mean + torch.randn_like(feature_mean) * noise_std
            soft_mean = prob_sums[label] / float(count)
            soft_predictions[label] = soft_mean / soft_mean.sum().clamp_min(1e-8)
            features[label] = feature_mean

        return HyperKnowledge(features=features, soft_predictions=soft_predictions, counts=counts)


class FedEGSServer(BaseFederatedServer):
    def __init__(
        self,
        config,
        client_datasets: Dict[str, Dataset],
        client_test_datasets: Dict[str, Dataset],
        data_module,
        test_hard_indices,
        writer=None,
    ) -> None:
        super().__init__(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
        self.general_model = WidthScalableResNet(
            width_factor=config.model.general_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.reference_expert = WidthScalableResNet(
            width_factor=config.model.expert_width,
            num_classes=config.model.num_classes,
        ).to(self.device)
        self.expert_flops = estimate_model_flops(self.reference_expert)
        self.general_flops = estimate_model_flops(self.general_model)
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
        self.global_hyper_knowledge = HyperKnowledge()
        self.last_history: List[RoundMetrics] = []

        for client_id in sorted_client_ids:
            LOGGER.info(
                "FedEGS client %s assigned expert block %d/%d",
                client_id,
                self.client_block_map[client_id],
                self.num_expert_blocks,
            )

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        history: List[RoundMetrics] = []
        for round_idx in range(1, self.config.federated.rounds + 1):
            sampled_ids = self._sample_client_ids()
            sampled_blocks = [self.client_block_map[client_id] for client_id in sampled_ids]
            LOGGER.info("fedegs round %d sampled clients=%s blocks=%s", round_idx, sampled_ids, sampled_blocks)

            updates: List[ClientUpdate] = []
            for client_id in sampled_ids:
                train_loader = self.data_module.make_loader(self.clients[client_id].dataset, shuffle=True)
                knowledge_loader = self.data_module.make_loader(self.clients[client_id].dataset, shuffle=False)
                update = self.clients[client_id].train(
                    self.general_model,
                    self.global_hyper_knowledge,
                    train_loader,
                    knowledge_loader,
                )
                updates.append(update)

            block_groups: Dict[int, List[ClientUpdate]] = {}
            for update in updates:
                block_index = self.client_block_map[update.client_id]
                block_groups.setdefault(block_index, []).append(update)

            for block_index, group_updates in block_groups.items():
                aggregated_delta = average_weighted_deltas(
                    (update.num_samples, update.delta) for update in group_updates
                )
                apply_expert_delta_to_general(
                    self.general_model,
                    aggregated_delta,
                    self.reference_expert,
                    block_index=block_index,
                )

            self.global_hyper_knowledge = self._aggregate_hyper_knowledge(updates)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs-routed")

            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
            aggregate = routed_eval["aggregate"]
            macro = routed_eval["macro"]
            compute_profile = self._build_compute_profile(
                self.expert_flops,
                self.general_flops,
                aggregate["invocation_rate"],
                mode="routed",
            )
            round_metrics = RoundMetrics(
                round_idx=round_idx,
                avg_client_loss=avg_loss,
                routed_accuracy=aggregate["accuracy"],
                hard_accuracy=aggregate["hard_recall"],
                invocation_rate=aggregate["invocation_rate"],
                local_accuracy=macro["accuracy"],
                compute_savings=compute_profile["savings_ratio"],
            )

            LOGGER.info(
                "fedegs round %d | loss=%.4f | global_acc=%.4f | local_acc=%.4f | hard_recall=%.4f | invocation=%.4f | savings=%.4f | knowledge_classes=%d",
                round_idx,
                avg_loss,
                aggregate["accuracy"],
                macro["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                compute_profile["savings_ratio"],
                len(self.global_hyper_knowledge.counts),
            )
            LOGGER.info(
                "fedegs auxiliary round %d | expert_acc=%.4f | general_acc=%.4f",
                round_idx,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )

            self._log_auxiliary_accuracy_metrics(
                "fedegs",
                round_idx,
                expert_eval["aggregate"]["accuracy"],
                general_eval["aggregate"]["accuracy"],
            )
            self._log_round_metrics("fedegs", round_metrics)
            history.append(round_metrics)

        self.last_history = history
        return history

    def evaluate_baselines(self, test_dataset: Dataset) -> Dict[str, Dict[str, float]]:
        routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, prefix="fedegs_final_routed")
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, prefix="fedegs_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, prefix="fedegs_final_general")
        final_loss = self.last_history[-1].avg_client_loss if self.last_history else 0.0

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

        if self.writer is not None:
            self.writer.add_scalar("fedegs_summary/global_accuracy", routed_eval["aggregate"]["accuracy"], 0)
            self.writer.add_scalar("fedegs_summary/local_accuracy", routed_eval["macro"]["accuracy"], 0)
            self.writer.add_scalar("fedegs_summary/hard_recall", routed_eval["aggregate"]["hard_recall"], 0)
            self.writer.add_scalar("fedegs_summary/compute_savings", routed_compute["savings_ratio"], 0)
            self.writer.add_scalar("memory/expert_mb", model_memory_mb(self.reference_expert), 0)
            self.writer.add_scalar("memory/general_mb", model_memory_mb(self.general_model), 0)

        return {
            "algorithm": "fedegs",
            "metrics": {
                "accuracy": routed_eval["aggregate"]["accuracy"],
                "global_accuracy": routed_eval["aggregate"]["accuracy"],
                "local_accuracy": routed_eval["macro"]["accuracy"],
                "routed_accuracy": routed_eval["aggregate"]["accuracy"],
                "hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "hard_sample_recall": routed_eval["aggregate"]["hard_recall"],
                "routed_hard_accuracy": routed_eval["aggregate"]["hard_recall"],
                "general_invocation_rate": routed_eval["aggregate"]["invocation_rate"],
                "compute_savings": routed_compute["savings_ratio"],
                "precision_macro": routed_eval["aggregate"]["precision_macro"],
                "recall_macro": routed_eval["aggregate"]["recall_macro"],
                "f1_macro": routed_eval["aggregate"]["f1_macro"],
                "expert_only_accuracy": expert_eval["aggregate"]["accuracy"],
                "general_only_accuracy": general_eval["aggregate"]["accuracy"],
                "expert_only_recall_macro": expert_eval["aggregate"]["recall_macro"],
                "general_only_recall_macro": general_eval["aggregate"]["recall_macro"],
                "global_hyper_knowledge_classes": len(self.global_hyper_knowledge.counts),
                "final_training_loss": final_loss,
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
            "hyper_knowledge": {
                "class_counts": {str(label): count for label, count in self.global_hyper_knowledge.counts.items()},
            },
            "memory_mb": {
                "expert": model_memory_mb(self.reference_expert),
                "general": model_memory_mb(self.general_model),
            },
        }

    def _aggregate_hyper_knowledge(self, updates: List[ClientUpdate]) -> HyperKnowledge:
        feature_accumulator: Dict[int, torch.Tensor] = {}
        soft_accumulator: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        for update in updates:
            knowledge = update.hyper_knowledge
            if knowledge is None or knowledge.is_empty():
                continue
            for label, count in knowledge.counts.items():
                counts[label] = counts.get(label, 0) + count
                weighted_feature = knowledge.features[label] * float(count)
                weighted_soft = knowledge.soft_predictions[label] * float(count)
                if label not in feature_accumulator:
                    feature_accumulator[label] = weighted_feature.clone()
                    soft_accumulator[label] = weighted_soft.clone()
                else:
                    feature_accumulator[label] += weighted_feature
                    soft_accumulator[label] += weighted_soft

        features = {}
        soft_predictions = {}
        for label, count in counts.items():
            features[label] = feature_accumulator[label] / float(count)
            soft_mean = soft_accumulator[label] / float(count)
            soft_predictions[label] = soft_mean / soft_mean.sum().clamp_min(1e-8)
        return HyperKnowledge(features=features, soft_predictions=soft_predictions, counts=counts)

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
        return self._confidence_route(client.expert_model, self.general_model, images)
