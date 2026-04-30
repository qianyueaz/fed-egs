"""
FedEGS-S: single-direction client-to-server distillation.

This variant keeps the current FedEGSD server distillation and routing design,
but removes the client-side reverse KD path entirely:
  - Clients train only the local SmallCNN expert with supervised CE.
  - Clients upload expert weights.
  - Server extracts expert ensemble logits on the external distillation set.
  - Server distills the general model from the expert ensemble.
  - Inference still uses the calibrated routed expert/general policy.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import BaseFederatedServer, LOGGER, RoundMetrics
from fedegs.federated.algorithms.fedegsd import (
    FedEGSDServer,
    GeneralModel,
    _client_holdout_seed,
    _split_dataset_for_holdout,
    load_distillation_dataset,
)
from fedegs.models import SmallCNN, estimate_model_flops


@dataclass
class FedEGSSClientUpdate:
    client_id: str
    num_samples: int
    loss: float
    expert_state_dict: Dict[str, torch.Tensor]


class FedEGSSClient:
    def __init__(self, client_id, dataset, num_classes, device, config, data_module):
        self.client_id = client_id
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.config = config
        self.data_module = data_module
        self.device = device
        self.expert_model = SmallCNN(
            num_classes=num_classes,
            base_channels=config.model.expert_base_channels,
        ).to(self.device)

    def train_local(self) -> FedEGSSClientUpdate:
        loader = self.data_module.make_loader(self.dataset, shuffle=True)
        loss = self._optimize(loader)
        state = {k: v.detach().cpu().clone() for k, v in self.expert_model.state_dict().items()}
        return FedEGSSClientUpdate(
            client_id=self.client_id,
            num_samples=self.num_samples,
            loss=loss,
            expert_state_dict=state,
        )

    def _optimize(self, loader) -> float:
        optimizer = torch.optim.SGD(
            self.expert_model.parameters(),
            lr=self.config.federated.local_lr,
            momentum=self.config.federated.local_momentum,
            weight_decay=self.config.federated.local_weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()
        self.expert_model.train()

        total_loss = 0.0
        total_batches = 0
        for _ in range(self.config.federated.local_epochs):
            for images, targets, _ in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.expert_model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
        return total_loss / max(total_batches, 1)


class FedEGSSServer(FedEGSDServer):
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
        if not hasattr(self, "algorithm_name"):
            self.algorithm_name = "fedegss"
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

        holdout_ratio = max(float(getattr(config.inference, "routing_holdout_ratio", 0.0)), 0.0)
        holdout_min_samples = max(int(getattr(config.inference, "routing_holdout_min_samples", 0)), 0)
        holdout_max_samples = max(int(getattr(config.inference, "routing_holdout_max_samples", 0)), 0)
        holdout_seed_offset = int(getattr(config.inference, "routing_holdout_seed_offset", 17))
        self.client_routing_datasets: Dict[str, Dataset] = {}
        self.client_training_datasets: Dict[str, Dataset] = {}
        for client_id, dataset in client_datasets.items():
            train_dataset, routing_dataset = _split_dataset_for_holdout(
                dataset=dataset,
                holdout_ratio=holdout_ratio,
                min_holdout_samples=holdout_min_samples,
                max_holdout_samples=holdout_max_samples,
                seed=_client_holdout_seed(config.federated.seed, client_id, holdout_seed_offset),
            )
            self.client_training_datasets[client_id] = train_dataset
            self.client_routing_datasets[client_id] = routing_dataset

        self.clients: Dict[str, FedEGSSClient] = {
            cid: FedEGSSClient(
                cid,
                self.client_training_datasets[cid],
                config.model.num_classes,
                self.device,
                config,
                data_module,
            )
            for cid in client_datasets.keys()
        }

        distill_name = str(getattr(config.federated, "distill_dataset", "cifar100"))
        distill_root = str(getattr(config.federated, "distill_dataset_root", config.dataset.root))
        distill_max = int(getattr(config.federated, "distill_max_samples", 0))
        self.distill_dataset = load_distillation_dataset(distill_name, distill_root, distill_max)
        self.distill_loader = DataLoader(
            self.distill_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
        )

        self.last_history: List[RoundMetrics] = []
        self.best_snapshot: Optional[Dict] = None
        self.current_round = 0

        initial_route_threshold = self._initial_route_score_threshold()
        self.client_route_score_thresholds = {c: initial_route_threshold for c in client_datasets}
        self.client_expert_temperatures = {c: 1.0 for c in client_datasets}
        self.client_prototypes: Dict[str, object] = {}
        self.client_prototype_scales: Dict[str, object] = {}
        self.client_prototype_masks: Dict[str, object] = {}
        self.client_routing_metrics: Dict[str, Dict[str, float]] = {}
        if holdout_ratio > 0.0:
            algorithm_name = getattr(self, "algorithm_name", "fedegss")
            LOGGER.info(
                "%s routing holdout | ratio=%.3f min=%d max=%d avg_holdout=%.1f avg_train=%.1f",
                algorithm_name,
                holdout_ratio,
                holdout_min_samples,
                holdout_max_samples,
                sum(len(dataset) for dataset in self.client_routing_datasets.values())
                / max(len(self.client_routing_datasets), 1),
                sum(len(dataset) for dataset in self.client_training_datasets.values())
                / max(len(self.client_training_datasets), 1),
            )

    def _update_routing_thresholds(self, updates):
        algorithm_name = getattr(self, "algorithm_name", "fedegss")
        updated_metrics: List[Dict[str, float]] = []
        for update in updates:
            cid = update.client_id
            client = self.clients[cid]
            temperature = self._estimate_expert_temperature(cid, client.expert_model, self.current_round)
            prototypes, prototype_scales, prototype_mask = self._compute_client_prototypes(cid, client.expert_model)
            self.client_expert_temperatures[cid] = temperature
            self.client_prototypes[cid] = prototypes.detach().cpu()
            self.client_prototype_scales[cid] = prototype_scales.detach().cpu()
            self.client_prototype_masks[cid] = prototype_mask.detach().cpu()

            routing_statistics = self._collect_client_routing_statistics(
                client_id=cid,
                expert_model=client.expert_model,
                temperature=temperature,
                prototypes=prototypes,
                prototype_scales=prototype_scales,
                prototype_mask=prototype_mask,
            )
            previous_threshold = float(
                self.client_route_score_thresholds.get(cid, self._initial_route_score_threshold())
            )
            if routing_statistics is None:
                self.client_route_score_thresholds[cid] = previous_threshold
                continue

            threshold, metrics = self._select_route_threshold(
                client_id=cid,
                statistics=routing_statistics,
                previous_threshold=previous_threshold,
            )
            self.client_route_score_thresholds[cid] = threshold
            metrics["temperature"] = temperature
            self.client_routing_metrics[cid] = metrics
            updated_metrics.append(metrics)

        if updated_metrics:
            mean_threshold = sum(item["threshold"] for item in updated_metrics) / len(updated_metrics)
            mean_temperature = sum(item["temperature"] for item in updated_metrics) / len(updated_metrics)
            mean_holdout_accuracy = sum(item["holdout_routed_accuracy"] for item in updated_metrics) / len(updated_metrics)
            mean_invocation = sum(item["invocation_rate"] for item in updated_metrics) / len(updated_metrics)
            LOGGER.info(
                "%s routing update | round=%d | clients=%d | holdout_acc=%.4f | invoke=%.4f | threshold=%.4f | temp=%.4f",
                algorithm_name,
                self.current_round,
                len(updated_metrics),
                mean_holdout_accuracy,
                mean_invocation,
                mean_threshold,
                mean_temperature,
            )
            if self.writer is not None:
                self._log_compare_scalars(
                    algorithm_name,
                    self.current_round,
                    {
                        "routing_holdout_accuracy": mean_holdout_accuracy,
                        "routing_score_threshold": mean_threshold,
                        "routing_temperature": mean_temperature,
                    },
                )

    def _build_round_extra_metrics(
        self,
        round_idx: int,
        distill_stats: Dict[str, float],
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return super()._build_round_extra_metrics(
            round_idx=round_idx,
            expert_eval=expert_eval,
            general_eval=general_eval,
            routed_eval=routed_eval,
        )

    def _build_final_extra_metrics(
        self,
        expert_eval: Dict[str, object],
        general_eval: Dict[str, object],
        routed_eval: Dict[str, object],
    ) -> Dict[str, float]:
        return super()._build_final_extra_metrics(
            expert_eval=expert_eval,
            general_eval=general_eval,
            routed_eval=routed_eval,
        )

    def _format_round_extra_metrics_for_log(self, extra_metrics: Dict[str, float]) -> str:
        return super()._format_round_extra_metrics_for_log(extra_metrics)

    def _log_algorithm_process_metrics(
        self,
        algorithm_name: str,
        round_idx: int,
        avg_loss: float,
        distill_stats: Dict[str, float],
    ) -> None:
        if self.writer is None:
            return
        self._log_compare_scalars(
            algorithm_name,
            round_idx,
            {
                "expert_loss": avg_loss,
                "distill_loss": distill_stats.get("total_loss"),
                "distill_kd_loss": distill_stats.get("kd_loss"),
                "distill_ce_loss": distill_stats.get("ce_loss"),
                "distill_alpha_mean": distill_stats.get("alpha_mean"),
                "teacher_bank_size": distill_stats.get("teacher_bank_size"),
                "teacher_bank_avg_staleness": distill_stats.get("teacher_bank_avg_staleness"),
                "teacher_bank_max_staleness": distill_stats.get("teacher_bank_max_staleness"),
                "teacher_bank_memory_mb": distill_stats.get("teacher_bank_memory_mb"),
                "teacher_bank_effective_size": distill_stats.get("teacher_bank_effective_size"),
                "teacher_weight_max_share": distill_stats.get("teacher_weight_max_share"),
                "teacher_confidence_mean": distill_stats.get("teacher_confidence_mean"),
                "teacher_entropy_mean": distill_stats.get("teacher_entropy_mean"),
                "selected_teacher_count_mean": distill_stats.get("selected_teacher_count_mean"),
                "selected_teacher_coverage": distill_stats.get("selected_teacher_coverage"),
                "ema_kd_loss": distill_stats.get("ema_kd_loss"),
            },
        )

    def _maybe_update_best(self, ri, rm, ea, ga):
        algorithm_name = getattr(self, "algorithm_name", "fedegss")
        better = self.best_snapshot is None or rm.routed_accuracy > float(self.best_snapshot["routed_accuracy"]) + 1e-8
        if not better:
            return False
        self.best_snapshot = {
            "round_idx": ri,
            "routed_accuracy": rm.routed_accuracy,
            "general_accuracy": ga,
            "expert_accuracy": ea,
            "avg_client_loss": rm.avg_client_loss,
            "general_model_state": {k: v.cpu().clone() for k, v in self.general_model.state_dict().items()},
            "client_expert_states": {
                c: {k: v.cpu().clone() for k, v in cl.expert_model.state_dict().items()}
                for c, cl in self.clients.items()
            },
            "client_route_score_thresholds": dict(self.client_route_score_thresholds),
            "client_expert_temperatures": dict(self.client_expert_temperatures),
            "client_prototypes": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototypes.items()
            },
            "client_prototype_scales": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototype_scales.items()
            },
            "client_prototype_masks": {
                c: tensor.detach().cpu().clone() for c, tensor in self.client_prototype_masks.items()
            },
        }
        LOGGER.info(
            "%s best | round=%d | routed=%.4f | general=%.4f | expert=%.4f",
            algorithm_name,
            ri,
            rm.routed_accuracy,
            ga,
            ea,
        )
        return True

    def _restore_best(self):
        algorithm_name = getattr(self, "algorithm_name", "fedegss")
        if not self.best_snapshot:
            return
        self.general_model.load_state_dict(self.best_snapshot["general_model_state"])
        for c, st in self.best_snapshot["client_expert_states"].items():
            self.clients[c].expert_model.load_state_dict(st)
        self.client_route_score_thresholds = dict(self.best_snapshot["client_route_score_thresholds"])
        self.client_expert_temperatures = dict(self.best_snapshot["client_expert_temperatures"])
        self.client_prototypes = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototypes"].items()
        }
        self.client_prototype_scales = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototype_scales"].items()
        }
        self.client_prototype_masks = {
            c: tensor.detach().cpu().clone() for c, tensor in self.best_snapshot["client_prototype_masks"].items()
        }
        self.current_round = int(self.best_snapshot["round_idx"])
        LOGGER.info("%s restored best from round %d", algorithm_name, self.best_snapshot["round_idx"])

    def train(self, test_dataset: Dataset) -> List[RoundMetrics]:
        algorithm_name = getattr(self, "algorithm_name", "fedegss")
        metrics: List[RoundMetrics] = []
        upload_bytes_total = 0.0

        for ri in range(1, self.config.federated.rounds + 1):
            self._device_synchronize()
            round_start_time = time.perf_counter()
            self.current_round = ri
            sampled_client_ids = self._sample_client_ids()
            LOGGER.info("%s round %d | clients=%s", algorithm_name, ri, sampled_client_ids)

            updates = [self.clients[cid].train_local() for cid in sampled_client_ids]
            ensemble = self._extract_ensemble_logits(updates)
            distill_stats = self._distill_general_model(ensemble, ri)
            self._update_routing_thresholds(updates)

            expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, f"{algorithm_name}-expert")
            general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, f"{algorithm_name}-general")
            routed_eval = self._evaluate_predictor_on_client_tests(self._predict_routed, f"{algorithm_name}-routed")
            extra_metrics = self._build_round_extra_metrics(
                round_idx=ri,
                distill_stats=distill_stats,
                expert_eval=expert_eval,
                general_eval=general_eval,
                routed_eval=routed_eval,
            )

            avg_loss = sum(update.loss for update in updates) / max(len(updates), 1)
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
                sampled_client_ids,
                self.client_training_datasets,
            )
            resource_metrics = self._resource_metric_values(
                self.resource_profiles,
                client_train_profile,
                compute_profile,
            )
            round_upload_bytes = float(
                sum(self._estimate_tensor_payload_bytes(update.expert_state_dict) for update in updates)
            )
            upload_bytes_total += round_upload_bytes
            self._device_synchronize()
            round_train_time_seconds = time.perf_counter() - round_start_time

            round_metrics = RoundMetrics(
                round_idx=ri,
                avg_client_loss=avg_loss,
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
            distill_extras = ""
            if "kd_loss" in distill_stats:
                distill_extras += f" | kd={distill_stats['kd_loss']:.4f}"
            if "ce_loss" in distill_stats:
                distill_extras += f" | ce={distill_stats['ce_loss']:.4f}"
            if "alpha_mean" in distill_stats:
                distill_extras += f" | alpha={distill_stats['alpha_mean']:.4f}"
            if "teacher_bank_size" in distill_stats:
                distill_extras += f" | bank={distill_stats['teacher_bank_size']:.0f}"
            if "teacher_bank_avg_staleness" in distill_stats:
                distill_extras += f" | stale={distill_stats['teacher_bank_avg_staleness']:.2f}"
            extra_log = self._format_round_extra_metrics_for_log(extra_metrics)

            LOGGER.info(
                "%s round %d | loss=%.4f | distill=%.4f | personalized=%.4f | weighted=%.4f | expert=%.4f | general=%.4f | hard=%.4f | invoke=%.4f%s%s%s",
                algorithm_name,
                ri,
                avg_loss,
                distill_stats["total_loss"],
                macro["accuracy"],
                aggregate["accuracy"],
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
                aggregate["hard_recall"],
                aggregate["invocation_rate"],
                distill_extras,
                extra_log,
                self._format_resource_metrics_for_log(round_metrics),
            )

            if self.writer:
                self._log_algorithm_process_metrics(algorithm_name, ri, avg_loss, distill_stats)
                self._log_auxiliary_accuracy_metrics(
                    algorithm_name,
                    ri,
                    expert_eval["macro"]["accuracy"],
                    general_eval["macro"]["accuracy"],
                )

            self._log_round_metrics(algorithm_name, round_metrics)
            metrics.append(round_metrics)
            self._maybe_update_best(
                ri,
                round_metrics,
                expert_eval["macro"]["accuracy"],
                general_eval["macro"]["accuracy"],
            )

        self.last_history = metrics
        if self.config.federated.restore_best_checkpoint:
            self._restore_best()
        return metrics

    def evaluate_baselines(self, test_dataset):
        algorithm_name = getattr(self, "algorithm_name", "fedegss")
        route_export_path = self._build_route_export_path(f"{algorithm_name}_final_routed")
        routed_eval = self._evaluate_predictor_on_client_tests(
            self._predict_routed,
            f"{algorithm_name}_final_routed",
            route_export_path=route_export_path,
        )
        expert_eval = self._evaluate_predictor_on_client_tests(self._predict_expert_only, f"{algorithm_name}_final_expert")
        general_eval = self._evaluate_predictor_on_client_tests(self._predict_general_only, f"{algorithm_name}_final_general")
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
        extra_metrics = self._build_final_extra_metrics(expert_eval, general_eval, routed_eval)
        return {
            "algorithm": algorithm_name,
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
            "artifacts": {"route_csv": str(route_export_path)},
        }
