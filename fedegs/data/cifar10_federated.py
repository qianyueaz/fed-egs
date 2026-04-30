import json
import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from fedegs.config import DatasetConfig
from fedegs.models import build_teacher_model, load_teacher_checkpoint


LOGGER = logging.getLogger(__name__)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, target, index

    def __len__(self) -> int:
        return len(self.dataset)


PARTITION_CACHE_VERSION = 5
PUBLIC_CACHE_VERSION = 2
DIRICHLET_CACHE_VERSION = 2


class CIFAR10FederatedDataModule:
    def __init__(self, config: DatasetConfig, device: str, seed: int) -> None:
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.scoring_transform = self.eval_transform

    def prepare(self) -> None:
        datasets.CIFAR10(root=self.config.root, train=True, download=True)
        datasets.CIFAR10(root=self.config.root, train=False, download=True)

    def build(self) -> Dict[str, object]:
        self.prepare()
        train_raw = datasets.CIFAR10(root=self.config.root, train=True, transform=self.train_transform, download=False)
        test_raw = datasets.CIFAR10(root=self.config.root, train=False, transform=self.eval_transform, download=False)
        public_raw = datasets.CIFAR10(root=self.config.root, train=True, transform=self.eval_transform, download=False)

        public_indices = self._load_or_build_public_indices(total_samples=len(train_raw), labels=public_raw.targets)
        public_index_set = set(public_indices)

        strategy = self.config.partition_strategy.lower()

        if strategy in ("dirichlet", "dir"):
            client_train_indices, client_test_indices = self._load_or_build_aligned_dirichlet_partitions(
                train_labels=train_raw.targets,
                test_labels=test_raw.targets,
                train_exclude=public_index_set,
            )
            train_hard_indices: List[int] = []
            train_easy_indices: List[int] = list(range(len(train_raw)))
            test_hard_indices: List[int] = []
            test_easy_indices: List[int] = list(range(len(test_raw)))
        elif strategy in ("dirichlet_quantity", "dirichlet_quantity_skew", "dirq"):
            client_train_indices = self._build_dirichlet_quantity_skew_partition(
                labels=train_raw.targets, exclude=public_index_set, split_name="train",
            )
            client_test_indices = self._build_dirichlet_quantity_skew_partition(
                labels=test_raw.targets, exclude=set(), split_name="test",
            )
            train_hard_indices = []
            train_easy_indices = list(range(len(train_raw)))
            test_hard_indices = []
            test_easy_indices = list(range(len(test_raw)))
        elif strategy in ("longtail", "long_tail"):
            client_train_indices = self._build_longtail_partition(
                labels=train_raw.targets, exclude=public_index_set, split_name="train",
            )
            client_test_indices = self._build_longtail_partition(
                labels=test_raw.targets, exclude=set(), split_name="test",
            )
            train_hard_indices = []
            train_easy_indices = list(range(len(train_raw)))
            test_hard_indices = []
            test_easy_indices = list(range(len(test_raw)))
        else:
            # Original difficulty_skewed partition — needs teacher checkpoint
            train_scoring = datasets.CIFAR10(root=self.config.root, train=True, transform=self.scoring_transform, download=False)
            test_scoring = datasets.CIFAR10(root=self.config.root, train=False, transform=self.scoring_transform, download=False)
            train_hard_indices, train_easy_indices = self._load_or_build_difficulty_split(train_scoring, split_name="train")
            test_hard_indices, test_easy_indices = self._load_or_build_difficulty_split(test_scoring, split_name="test")
            if public_index_set:
                train_hard_indices = [idx for idx in train_hard_indices if idx not in public_index_set]
                train_easy_indices = [idx for idx in train_easy_indices if idx not in public_index_set]
            client_train_indices = self._load_or_build_client_partitions(train_hard_indices, train_easy_indices, split_name="train")
            client_test_indices = self._load_or_build_client_partitions(test_hard_indices, test_easy_indices, split_name="test")

        self._log_partition_summary(client_train_indices, train_hard_indices, split_name="train")
        self._log_partition_summary(client_test_indices, test_hard_indices, split_name="test")
        self._log_class_distribution(client_train_indices, train_raw.targets, split_name="train")

        indexed_train = IndexedDataset(train_raw)
        indexed_test = IndexedDataset(test_raw)
        indexed_public = IndexedDataset(public_raw)

        return {
            "client_datasets": {client_id: Subset(indexed_train, indices) for client_id, indices in client_train_indices.items()},
            "client_test_datasets": {client_id: Subset(indexed_test, indices) for client_id, indices in client_test_indices.items()},
            "client_indices": client_train_indices,
            "client_test_indices": client_test_indices,
            "train_dataset": indexed_train,
            "test_dataset": indexed_test,
            "train_hard_indices": set(train_hard_indices),
            "train_easy_indices": set(train_easy_indices),
            "test_hard_indices": set(test_hard_indices),
            "test_easy_indices": set(test_easy_indices),
            "public_dataset": Subset(indexed_public, public_indices) if public_indices else None,
            "public_indices": public_indices,
        }

    def _difficulty_cache_path(self, split_name: str) -> Path:
        return Path(self.config.cache_dir) / f"difficulty_split_{split_name}.json"

    def _partition_cache_path(self, split_name: str) -> Path:
        return Path(self.config.cache_dir) / f"client_partitions_{split_name}.json"

    def _public_cache_path(self) -> Path:
        return Path(self.config.cache_dir) / "public_data_indices.json"

    def _load_or_build_public_indices(self, total_samples: int, labels: Sequence[int]) -> List[int]:
        strategy = self.config.public_split_strategy
        configured_public_size = max(int(self.config.public_dataset_size), 0)
        per_class_ratio = max(float(self.config.public_per_class_ratio), 0.0)
        if strategy == "random" and configured_public_size == 0:
            return []
        if strategy == "per_class_ratio" and per_class_ratio == 0.0:
            return []

        cache_path = self._public_cache_path()
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            expected = {
                "seed": self.seed,
                "configured_public_dataset_size": configured_public_size,
                "public_split_strategy": strategy,
                "public_per_class_ratio": per_class_ratio,
                "public_cache_version": PUBLIC_CACHE_VERSION,
            }
            if all(metadata.get(key) == value for key, value in expected.items()):
                LOGGER.info("Loaded cached public train split from %s", cache_path)
                return list(payload.get("indices", []))

        rng = random.Random(self.seed)
        if strategy == "random":
            if configured_public_size >= total_samples:
                raise ValueError(
                    f"public_dataset_size={configured_public_size} must be smaller than the train split size {total_samples}."
                )
            indices = list(range(total_samples))
            rng.shuffle(indices)
            public_indices = sorted(indices[:configured_public_size])
        elif strategy == "per_class_ratio":
            if not 0.0 < per_class_ratio < 1.0:
                raise ValueError(
                    f"public_per_class_ratio must be in (0, 1) for per_class_ratio strategy, got {per_class_ratio}."
                )
            class_to_indices: Dict[int, List[int]] = defaultdict(list)
            for sample_index, label in enumerate(labels):
                class_to_indices[int(label)].append(sample_index)

            public_indices = []
            for label, class_indices in sorted(class_to_indices.items()):
                class_sample_count = int(round(len(class_indices) * per_class_ratio))
                if class_sample_count <= 0:
                    raise ValueError(
                        f"public_per_class_ratio={per_class_ratio} produced zero public samples for class {label}."
                    )
                shuffled = list(class_indices)
                rng.shuffle(shuffled)
                public_indices.extend(shuffled[:class_sample_count])
            public_indices.sort()
        else:
            raise ValueError(f"Unsupported public split strategy: {strategy}")
        realized_public_size = len(public_indices)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "seed": self.seed,
                        "configured_public_dataset_size": configured_public_size,
                        "realized_public_dataset_size": realized_public_size,
                        "public_split_strategy": strategy,
                        "public_per_class_ratio": per_class_ratio,
                        "public_cache_version": PUBLIC_CACHE_VERSION,
                    },
                    "indices": public_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Saved public train split cache to %s", cache_path)
        return public_indices

    def _load_or_build_difficulty_split(self, scoring_dataset: Dataset, split_name: str) -> Tuple[List[int], List[int]]:
        cache_path = self._difficulty_cache_path(split_name)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            LOGGER.info("Loaded cached %s difficulty split from %s", split_name, cache_path)
            return payload["hard_indices"], payload["easy_indices"]

        LOGGER.info("Scoring CIFAR-10 %s split with a ResNet18 teacher for difficulty partitioning.", split_name)
        losses = self._score_samples(scoring_dataset)
        sorted_indices = [idx for idx, _ in sorted(losses, key=lambda item: item[1], reverse=True)]
        hard_cutoff = int(len(sorted_indices) * self.config.hard_ratio)
        hard_indices = sorted_indices[:hard_cutoff]
        easy_indices = sorted_indices[hard_cutoff:]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"hard_indices": hard_indices, "easy_indices": easy_indices}, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved %s difficulty split cache to %s", split_name, cache_path)
        return hard_indices, easy_indices

    def _load_or_build_client_partitions(
        self,
        hard_indices: Sequence[int],
        easy_indices: Sequence[int],
        split_name: str,
    ) -> Dict[str, List[int]]:
        cache_path = self._partition_cache_path(split_name)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            if self._partition_cache_matches(metadata, split_name):
                LOGGER.info("Loaded cached %s client partitions from %s", split_name, cache_path)
                return {client_id: list(indices) for client_id, indices in payload["partitions"].items()}
            LOGGER.warning("%s client partition cache config mismatch detected. Rebuilding partitions.", split_name)

        partitions = self._build_client_partitions(hard_indices, easy_indices)
        cache_payload = {
            "metadata": {
                "seed": self.seed,
                "num_clients": self.config.num_clients,
                "simple_clients": self.config.simple_clients,
                "complex_clients": self.config.complex_clients,
                "simple_easy_ratio": self.config.simple_easy_ratio,
                "complex_easy_ratio": self.config.complex_easy_ratio,
                "hard_ratio": self.config.hard_ratio,
                "public_dataset_size": self.config.public_dataset_size,
                "public_split_strategy": self.config.public_split_strategy,
                "public_per_class_ratio": self.config.public_per_class_ratio,
                "partition_cache_version": PARTITION_CACHE_VERSION,
            },
            "partitions": partitions,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved %s client partition cache to %s", split_name, cache_path)
        return partitions

    def _partition_cache_matches(self, metadata: Dict[str, object], split_name: str) -> bool:
        expected = {
            "seed": self.seed,
            "num_clients": self.config.num_clients,
            "simple_clients": self.config.simple_clients,
            "complex_clients": self.config.complex_clients,
            "simple_easy_ratio": self.config.simple_easy_ratio,
            "complex_easy_ratio": self.config.complex_easy_ratio,
            "hard_ratio": self.config.hard_ratio,
            "public_dataset_size": self.config.public_dataset_size,
            "public_split_strategy": self.config.public_split_strategy,
            "public_per_class_ratio": self.config.public_per_class_ratio,
            "partition_cache_version": PARTITION_CACHE_VERSION,
        }
        return all(metadata.get(key) == value for key, value in expected.items())

    def _score_samples(self, scoring_dataset: Dataset) -> List[Tuple[int, float]]:
        loader = DataLoader(
            IndexedDataset(scoring_dataset),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        model = self._build_difficulty_model().to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        losses: List[Tuple[int, float]] = []

        with torch.no_grad():
            for images, targets, indices in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = model(images)
                batch_losses = criterion(logits, targets)
                for sample_index, loss_value in zip(indices.tolist(), batch_losses.detach().cpu().tolist()):
                    losses.append((sample_index, float(loss_value)))
        return losses

    def _build_difficulty_model(self) -> torch.nn.Module:
        if not self.config.difficulty_checkpoint:
            raise ValueError(
                "A CIFAR-10 teacher checkpoint is required for difficulty-aware partitioning. "
                "Run pretrain.py first and set dataset.difficulty_checkpoint."
            )

        checkpoint_path = Path(self.config.difficulty_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Difficulty checkpoint not found: {checkpoint_path}")

        model = build_teacher_model(num_classes=10)
        state = torch.load(checkpoint_path, map_location="cpu")
        load_teacher_checkpoint(model, state)
        return model

    def _build_client_partitions(self, hard_indices: Sequence[int], easy_indices: Sequence[int]) -> Dict[str, List[int]]:
        rng = random.Random(self.seed)
        hard_pool = list(hard_indices)
        easy_pool = list(easy_indices)
        rng.shuffle(hard_pool)
        rng.shuffle(easy_pool)

        total_samples = len(hard_pool) + len(easy_pool)
        total_hard = len(hard_pool)
        simple_clients = self.config.simple_clients
        complex_clients = self.config.complex_clients
        simple_hard_ratio = 1.0 - self.config.simple_easy_ratio
        complex_hard_ratio = 1.0 - self.config.complex_easy_ratio

        simple_total, complex_total = self._solve_group_sample_totals(
            total_samples=total_samples,
            total_hard=total_hard,
            simple_clients=simple_clients,
            complex_clients=complex_clients,
            simple_hard_ratio=simple_hard_ratio,
            complex_hard_ratio=complex_hard_ratio,
        )

        simple_hard_total = int(round(simple_total * simple_hard_ratio))
        simple_hard_total = min(simple_hard_total, total_hard)
        complex_hard_total = total_hard - simple_hard_total
        simple_easy_total = simple_total - simple_hard_total
        complex_easy_total = len(easy_pool) - simple_easy_total

        simple_total_counts = self._split_integer_total(simple_total, simple_clients)
        complex_total_counts = self._split_integer_total(complex_total, complex_clients)
        simple_hard_counts = self._split_integer_total(simple_hard_total, simple_clients)
        complex_hard_counts = self._split_integer_total(complex_hard_total, complex_clients)
        simple_easy_counts = [total - hard for total, hard in zip(simple_total_counts, simple_hard_counts)]
        complex_easy_counts = [total - hard for total, hard in zip(complex_total_counts, complex_hard_counts)]

        partitions: Dict[str, List[int]] = {}
        hard_cursor = 0
        easy_cursor = 0

        for local_idx in range(simple_clients):
            client_id = f"simple_{local_idx:02d}"
            selected_easy = easy_pool[easy_cursor : easy_cursor + simple_easy_counts[local_idx]]
            selected_hard = hard_pool[hard_cursor : hard_cursor + simple_hard_counts[local_idx]]
            easy_cursor += simple_easy_counts[local_idx]
            hard_cursor += simple_hard_counts[local_idx]
            client_samples = selected_easy + selected_hard
            rng.shuffle(client_samples)
            partitions[client_id] = client_samples

        for offset in range(complex_clients):
            client_index = simple_clients + offset
            client_id = f"complex_{client_index:02d}"
            selected_easy = easy_pool[easy_cursor : easy_cursor + complex_easy_counts[offset]]
            selected_hard = hard_pool[hard_cursor : hard_cursor + complex_hard_counts[offset]]
            easy_cursor += complex_easy_counts[offset]
            hard_cursor += complex_hard_counts[offset]
            client_samples = selected_easy + selected_hard
            rng.shuffle(client_samples)
            partitions[client_id] = client_samples

        if hard_cursor != len(hard_pool) or easy_cursor != len(easy_pool):
            raise RuntimeError(
                "Client partitioning did not consume all samples. "
                f"hard_used={hard_cursor}/{len(hard_pool)} easy_used={easy_cursor}/{len(easy_pool)}"
            )

        return partitions

    def _solve_group_sample_totals(
        self,
        total_samples: int,
        total_hard: int,
        simple_clients: int,
        complex_clients: int,
        simple_hard_ratio: float,
        complex_hard_ratio: float,
    ) -> Tuple[int, int]:
        if simple_clients == 0:
            return 0, total_samples
        if complex_clients == 0:
            return total_samples, 0
        if abs(simple_hard_ratio - complex_hard_ratio) < 1e-8:
            return total_samples * simple_clients // (simple_clients + complex_clients), total_samples - (total_samples * simple_clients // (simple_clients + complex_clients))

        simple_total_float = (total_hard - complex_hard_ratio * total_samples) / (simple_hard_ratio - complex_hard_ratio)
        simple_total = int(round(simple_total_float))
        simple_total = max(0, min(simple_total, total_samples))
        complex_total = total_samples - simple_total

        weighted_hard_ratio = total_hard / max(total_samples, 1)
        lower = min(simple_hard_ratio, complex_hard_ratio)
        upper = max(simple_hard_ratio, complex_hard_ratio)
        if not (lower - 1e-8 <= weighted_hard_ratio <= upper + 1e-8):
            raise ValueError(
                "Requested client hardness ratios are incompatible with the available hard/easy split. "
                f"global_hard_ratio={weighted_hard_ratio:.4f}, simple_hard_ratio={simple_hard_ratio:.4f}, complex_hard_ratio={complex_hard_ratio:.4f}"
            )
        return simple_total, complex_total

    def _split_integer_total(self, total: int, parts: int) -> List[int]:
        if parts <= 0:
            return []
        base = total // parts
        remainder = total % parts
        return [base + (1 if idx < remainder else 0) for idx in range(parts)]

    def _log_partition_summary(self, partitions: Dict[str, List[int]], hard_indices: Sequence[int], split_name: str) -> None:
        hard_index_set = set(hard_indices)
        for client_id, indices in partitions.items():
            hard_count = sum(1 for idx in indices if idx in hard_index_set)
            easy_count = len(indices) - hard_count
            LOGGER.info(
                "%s client %s | samples=%d | easy=%d | hard=%d",
                split_name,
                client_id,
                len(indices),
                easy_count,
                hard_count,
            )

    def _log_class_distribution(self, partitions: Dict[str, List[int]], labels: Sequence[int], split_name: str) -> None:
        """Log per-client class distribution for debugging partition quality."""
        for client_id, indices in sorted(partitions.items()):
            class_counts: Dict[int, int] = defaultdict(int)
            for idx in indices:
                class_counts[labels[idx]] += 1
            classes_present = sorted(class_counts.keys())
            dist_str = " ".join(f"{c}:{class_counts[c]}" for c in classes_present)
            LOGGER.info("%s client %s | classes=%d/%d | %s",
                        split_name, client_id, len(classes_present), 10, dist_str)

    def _dirichlet_client_names(self) -> List[str]:
        return [f"client_{i:02d}" for i in range(int(self.config.num_clients))]

    def _index_signature(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha1()
        for index in sorted(int(idx) for idx in indices):
            digest.update(f"{index},".encode("utf-8"))
        return digest.hexdigest()[:16]

    def _resolve_dirichlet_min_size(self, total_available: int) -> int:
        requested = max(int(getattr(self.config, "dirichlet_min_client_size", 0)), 0)
        num_clients = int(self.config.num_clients)
        if num_clients <= 0:
            raise ValueError("num_clients must be positive for Dirichlet partitioning.")
        if total_available <= 0:
            return 0
        return min(requested, total_available // num_clients)

    def _dirichlet_pair_cache_path(
        self,
        alpha: float,
        train_min_size: int,
        test_min_size: int,
        public_signature: str,
    ) -> Path:
        return (
            Path(self.config.cache_dir)
            / (
                f"dirichlet_pair_a{alpha}_tr{train_min_size}_te{test_min_size}"
                f"_c{self.config.num_clients}_s{self.seed}_p{public_signature}.json"
            )
        )

    def _dirichlet_pair_cache_metadata(
        self,
        alpha: float,
        train_min_size: int,
        test_min_size: int,
        public_signature: str,
        train_exclude_count: int,
        class_proportions: List[List[float]],
    ) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "num_clients": int(self.config.num_clients),
            "dirichlet_alpha": alpha,
            "dirichlet_min_client_size": max(int(getattr(self.config, "dirichlet_min_client_size", 0)), 0),
            "effective_train_min_client_size": train_min_size,
            "effective_test_min_client_size": test_min_size,
            "public_dataset_size": int(self.config.public_dataset_size),
            "public_split_strategy": self.config.public_split_strategy,
            "public_per_class_ratio": float(self.config.public_per_class_ratio),
            "public_indices_signature": public_signature,
            "public_indices_count": train_exclude_count,
            "partition_mode": "shared_client_distribution_across_train_and_test",
            "dirichlet_cache_version": DIRICHLET_CACHE_VERSION,
            "class_proportions_signature": self._index_signature(
                [int(round(value * 1_000_000)) for row in class_proportions for value in row]
            ),
        }

    def _dirichlet_pair_cache_matches(
        self,
        metadata: Dict[str, object],
        alpha: float,
        train_min_size: int,
        test_min_size: int,
        public_signature: str,
        train_exclude_count: int,
    ) -> bool:
        expected = {
            "seed": self.seed,
            "num_clients": int(self.config.num_clients),
            "dirichlet_alpha": alpha,
            "dirichlet_min_client_size": max(int(getattr(self.config, "dirichlet_min_client_size", 0)), 0),
            "effective_train_min_client_size": train_min_size,
            "effective_test_min_client_size": test_min_size,
            "public_dataset_size": int(self.config.public_dataset_size),
            "public_split_strategy": self.config.public_split_strategy,
            "public_per_class_ratio": float(self.config.public_per_class_ratio),
            "public_indices_signature": public_signature,
            "public_indices_count": train_exclude_count,
            "partition_mode": "shared_client_distribution_across_train_and_test",
            "dirichlet_cache_version": DIRICHLET_CACHE_VERSION,
        }
        return all(metadata.get(key) == value for key, value in expected.items())

    def _load_or_build_aligned_dirichlet_partitions(
        self,
        train_labels: Sequence[int],
        test_labels: Sequence[int],
        train_exclude: set,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Sample one shared client-class Dirichlet distribution on the official train
        split and use the same client distribution to allocate both train and test.
        This keeps each client's evaluation distribution aligned with its training
        distribution while still preserving the canonical CIFAR-10 train/test split.
        """
        import numpy as np

        alpha = float(self.config.dirichlet_alpha)
        num_clients = int(self.config.num_clients)
        if num_clients <= 0:
            raise ValueError("num_clients must be positive for Dirichlet partitioning.")
        if alpha <= 0.0:
            raise ValueError(f"dirichlet_alpha must be positive, got {alpha}.")

        train_available = sum(1 for idx in range(len(train_labels)) if idx not in train_exclude)
        test_available = len(test_labels)
        train_min_size = self._resolve_dirichlet_min_size(train_available)
        test_min_size = self._resolve_dirichlet_min_size(test_available)
        public_signature = self._index_signature(sorted(train_exclude))
        cache_path = self._dirichlet_pair_cache_path(alpha, train_min_size, test_min_size, public_signature)

        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if self._dirichlet_pair_cache_matches(
                metadata=payload.get("metadata", {}),
                alpha=alpha,
                train_min_size=train_min_size,
                test_min_size=test_min_size,
                public_signature=public_signature,
                train_exclude_count=len(train_exclude),
            ):
                LOGGER.info("Loaded cached aligned dirichlet partitions from %s", cache_path)
                return (
                    {client_id: list(indices) for client_id, indices in payload["train_partitions"].items()},
                    {client_id: list(indices) for client_id, indices in payload["test_partitions"].items()},
                )
            LOGGER.warning("Aligned dirichlet cache config mismatch detected. Rebuilding partitions.")

        num_classes = len(set(int(label) for label in list(train_labels) + list(test_labels)))
        max_attempts = 512
        rng = np.random.RandomState(self.seed)
        best_snapshot: Optional[Tuple[int, int]] = None

        for attempt in range(1, max_attempts + 1):
            class_proportions = self._sample_dirichlet_class_proportions(
                num_classes=num_classes,
                num_clients=num_clients,
                alpha=alpha,
                rng=rng,
            )
            train_partitions = self._allocate_dirichlet_partition_from_proportions(
                labels=train_labels,
                exclude=train_exclude,
                class_proportions=class_proportions,
                rng=rng,
            )
            test_partitions = self._allocate_dirichlet_partition_from_proportions(
                labels=test_labels,
                exclude=set(),
                class_proportions=class_proportions,
                rng=rng,
            )
            observed_train_min = min(len(indices) for indices in train_partitions.values())
            observed_test_min = min(len(indices) for indices in test_partitions.values())
            if best_snapshot is None or (observed_train_min, observed_test_min) > best_snapshot:
                best_snapshot = (observed_train_min, observed_test_min)
            if observed_train_min >= train_min_size and observed_test_min >= test_min_size:
                cache_payload = {
                    "metadata": self._dirichlet_pair_cache_metadata(
                        alpha=alpha,
                        train_min_size=train_min_size,
                        test_min_size=test_min_size,
                        public_signature=public_signature,
                        train_exclude_count=len(train_exclude),
                        class_proportions=class_proportions,
                    ),
                    "train_partitions": train_partitions,
                    "test_partitions": test_partitions,
                    "class_proportions": class_proportions,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
                LOGGER.info(
                    "Saved aligned dirichlet partitions to %s | alpha=%.2f train_min=%d test_min=%d attempts=%d",
                    cache_path,
                    alpha,
                    observed_train_min,
                    observed_test_min,
                    attempt,
                )
                return train_partitions, test_partitions

        raise RuntimeError(
            "Failed to sample aligned Dirichlet train/test partitions that satisfy the minimum client size "
            f"after {max_attempts} attempts. "
            f"requested_train_min={train_min_size} requested_test_min={test_min_size} "
            f"best_observed={best_snapshot}. "
            "Consider lowering dataset.dirichlet_min_client_size or increasing dirichlet_alpha."
        )

    def _sample_dirichlet_class_proportions(
        self,
        num_classes: int,
        num_clients: int,
        alpha: float,
        rng,
    ) -> List[List[float]]:
        return rng.dirichlet([alpha] * num_clients, size=num_classes).tolist()

    def _allocate_dirichlet_partition_from_proportions(
        self,
        labels: Sequence[int],
        exclude: set,
        class_proportions: List[List[float]],
        rng,
    ) -> Dict[str, List[int]]:
        client_names = self._dirichlet_client_names()
        if not client_names:
            raise ValueError("num_clients must be positive for Dirichlet partitioning.")

        class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if idx not in exclude:
                class_indices[int(label)].append(idx)

        client_indices: Dict[str, List[int]] = {name: [] for name in client_names}
        for cls, class_distribution in enumerate(class_proportions):
            indices = list(class_indices.get(cls, []))
            if not indices:
                continue
            rng.shuffle(indices)
            counts = rng.multinomial(len(indices), class_distribution)
            cursor = 0
            for client_idx, name in enumerate(client_names):
                take_count = int(counts[client_idx])
                if take_count <= 0:
                    continue
                client_indices[name].extend(indices[cursor : cursor + take_count])
                cursor += take_count

        for name in client_names:
            rng.shuffle(client_indices[name])
        return client_indices

    def _build_longtail_partition(
        self,
        labels: Sequence[int],
        exclude: set,
        split_name: str,
    ) -> Dict[str, List[int]]:
        """
        Long-tail partition: each client has a few "major" classes (with lots of
        samples) and many "minor" classes (with very few samples, or none).

        This creates extreme class imbalance per client, ideal for testing
        prototype-based OOD routing: the expert is strong on major classes
        but weak/absent on minor classes.

        Config:
          longtail_major_classes: how many classes each client dominates (default: 3)
          longtail_major_ratio: fraction of client's data from major classes (default: 0.9)
        """
        num_clients = self.config.num_clients
        num_classes = 10
        major_k = min(int(self.config.longtail_major_classes), num_classes)
        major_ratio = float(self.config.longtail_major_ratio)

        cache_path = Path(self.config.cache_dir) / f"longtail_{split_name}_k{major_k}_r{major_ratio}_c{num_clients}_s{self.seed}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            LOGGER.info("Loaded cached longtail %s partition from %s", split_name, cache_path)
            return {k: list(v) for k, v in payload["partitions"].items()}

        rng = random.Random(self.seed)

        # Group indices by class
        class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if idx not in exclude:
                class_indices[label].append(idx)
        for cls in class_indices:
            rng.shuffle(class_indices[cls])

        # Assign major classes to each client (round-robin with offset)
        all_classes = list(range(num_classes))
        client_major_classes: Dict[str, List[int]] = {}
        for i in range(num_clients):
            start = (i * major_k) % num_classes
            majors = [(start + j) % num_classes for j in range(major_k)]
            client_major_classes[f"client_{i:02d}"] = majors

        # Total samples per client (roughly equal)
        total_available = sum(len(v) for v in class_indices.values())
        per_client = total_available // num_clients

        # Per-class cursors
        class_cursors: Dict[int, int] = {c: 0 for c in range(num_classes)}

        def take_from_class(cls: int, count: int) -> List[int]:
            cursor = class_cursors[cls]
            available = class_indices[cls]
            actual = min(count, len(available) - cursor)
            result = available[cursor:cursor + actual]
            class_cursors[cls] = cursor + actual
            return result

        client_indices: Dict[str, List[int]] = {}

        for client_name in sorted(client_major_classes.keys()):
            majors = client_major_classes[client_name]
            minors = [c for c in all_classes if c not in majors]

            major_count = int(per_client * major_ratio)
            minor_count = per_client - major_count

            samples = []
            # Major classes: split evenly
            per_major = major_count // max(len(majors), 1)
            for cls in majors:
                samples.extend(take_from_class(cls, per_major))

            # Minor classes: small amount each (or 0)
            if minors:
                per_minor = minor_count // len(minors)
                for cls in minors:
                    samples.extend(take_from_class(cls, per_minor))

            rng.shuffle(samples)
            client_indices[client_name] = samples

        client_indices = {k: v for k, v in client_indices.items() if len(v) > 0}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"partitions": client_indices}, indent=2), encoding="utf-8")
        LOGGER.info("Saved longtail %s partition to %s | major_k=%d major_ratio=%.2f",
                     split_name, cache_path, major_k, major_ratio)
        return client_indices

    def _build_dirichlet_quantity_skew_partition(
        self,
        labels: Sequence[int],
        exclude: set,
        split_name: str,
    ) -> Dict[str, List[int]]:
        """
        Dirichlet + quantity skew partition.

        Each client first receives a target dataset size sampled from a log-normal
        distribution, then samples labels according to its own Dirichlet class
        preference. This creates both label imbalance and client-size imbalance.
        """
        import numpy as np

        alpha = float(self.config.dirichlet_alpha)
        sigma = max(float(getattr(self.config, "quantity_skew_sigma", 1.0)), 0.0)
        num_clients = int(self.config.num_clients)
        num_classes = len(set(int(label) for label in labels))
        total_available = sum(1 for idx in range(len(labels)) if idx not in exclude)
        effective_min_size = max(int(getattr(self.config, "quantity_min_size", 32)), 0)
        if num_clients > 0:
            effective_min_size = min(effective_min_size, total_available // num_clients)

        cache_path = (
            Path(self.config.cache_dir)
            / f"dirq_{split_name}_a{alpha}_q{sigma}_m{effective_min_size}_c{num_clients}_s{self.seed}.json"
        )
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            LOGGER.info("Loaded cached dirichlet+quantity %s partition from %s", split_name, cache_path)
            return {k: list(v) for k, v in payload["partitions"].items()}

        rng = np.random.RandomState(self.seed if split_name == "train" else self.seed + 1000)
        py_rng = random.Random(self.seed if split_name == "train" else self.seed + 1000)

        class_pools: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if idx not in exclude:
                class_pools[int(label)].append(idx)
        for cls in class_pools:
            py_rng.shuffle(class_pools[cls])

        raw_size_weights = rng.lognormal(mean=0.0, sigma=sigma, size=num_clients) if sigma > 0 else np.ones(num_clients)
        raw_size_weights = raw_size_weights / raw_size_weights.sum()
        free_budget = max(total_available - (effective_min_size * num_clients), 0)
        client_target_sizes = np.floor(raw_size_weights * free_budget).astype(int) + effective_min_size
        remainder = total_available - int(client_target_sizes.sum())
        if remainder > 0:
            ordering = np.argsort(-raw_size_weights)
            for idx in ordering[:remainder]:
                client_target_sizes[idx] += 1
        elif remainder < 0:
            ordering = np.argsort(raw_size_weights)
            for idx in ordering:
                if remainder == 0:
                    break
                removable = max(int(client_target_sizes[idx]) - effective_min_size, 0)
                if removable <= 0:
                    continue
                delta = min(removable, -remainder)
                client_target_sizes[idx] -= delta
                remainder += delta

        client_class_preferences = rng.dirichlet([alpha] * num_classes, size=num_clients)
        client_indices: Dict[str, List[int]] = {f"client_{i:02d}": [] for i in range(num_clients)}

        for client_idx in range(num_clients):
            client_name = f"client_{client_idx:02d}"
            target_size = int(client_target_sizes[client_idx])
            if target_size <= 0:
                continue
            for _ in range(target_size):
                available_classes = [cls for cls, pool in class_pools.items() if pool]
                if not available_classes:
                    break
                class_probs = np.array(
                    [client_class_preferences[client_idx, cls] for cls in available_classes],
                    dtype=np.float64,
                )
                if float(class_probs.sum()) <= 0.0:
                    class_probs = np.ones(len(available_classes), dtype=np.float64)
                class_probs = class_probs / class_probs.sum()
                chosen_class = int(rng.choice(available_classes, p=class_probs))
                client_indices[client_name].append(class_pools[chosen_class].pop())
            py_rng.shuffle(client_indices[client_name])

        leftovers: List[int] = []
        for remaining in class_pools.values():
            leftovers.extend(remaining)
        py_rng.shuffle(leftovers)
        if leftovers:
            client_names = sorted(client_indices.keys(), key=lambda name: len(client_indices[name]))
            for offset, sample_index in enumerate(leftovers):
                client_indices[client_names[offset % len(client_names)]].append(sample_index)

        client_indices = {name: indices for name, indices in client_indices.items() if len(indices) > 0}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"partitions": client_indices}, indent=2), encoding="utf-8")
        LOGGER.info(
            "Saved dirichlet+quantity %s partition to %s | alpha=%.2f sigma=%.2f min_size=%d",
            split_name,
            cache_path,
            alpha,
            sigma,
            effective_min_size,
        )
        return client_indices

    def make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        num_workers = max(int(self.config.num_workers), 0)
        # In sequential federated simulation we create many short-lived loaders,
        # especially during per-client evaluation. Multiprocess workers there can
        # exhaust file descriptors on Linux, so keep eval loaders single-process
        # and avoid persistent workers for these transient loaders.
        if not shuffle:
            num_workers = 0
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=self.device.type == "cuda",
        )
