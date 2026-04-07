import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
        train_scoring = datasets.CIFAR10(root=self.config.root, train=True, transform=self.scoring_transform, download=False)
        test_raw = datasets.CIFAR10(root=self.config.root, train=False, transform=self.eval_transform, download=False)
        test_scoring = datasets.CIFAR10(root=self.config.root, train=False, transform=self.scoring_transform, download=False)
        public_raw = datasets.CIFAR10(root=self.config.root, train=True, transform=self.eval_transform, download=False)

        public_indices = self._load_or_build_public_indices(total_samples=len(train_raw), labels=public_raw.targets)
        public_index_set = set(public_indices)

        train_hard_indices, train_easy_indices = self._load_or_build_difficulty_split(train_scoring, split_name="train")
        test_hard_indices, test_easy_indices = self._load_or_build_difficulty_split(test_scoring, split_name="test")
        if public_index_set:
            train_hard_indices = [idx for idx in train_hard_indices if idx not in public_index_set]
            train_easy_indices = [idx for idx in train_easy_indices if idx not in public_index_set]

        client_train_indices = self._load_or_build_client_partitions(train_hard_indices, train_easy_indices, split_name="train")
        client_test_indices = self._load_or_build_client_partitions(test_hard_indices, test_easy_indices, split_name="test")
        self._log_partition_summary(client_train_indices, train_hard_indices, split_name="train")
        self._log_partition_summary(client_test_indices, test_hard_indices, split_name="test")

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

    def make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )
