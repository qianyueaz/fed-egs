import copy
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedegs.federated.common import LOGGER
from fedegs.models import build_baseline_model, initialize_model_from_teacher


def prepare_server_model(
    model: nn.Module,
    config,
    device,
    data_module,
    public_dataset: Optional[Dataset] = None,
    algorithm_name: str = "baseline",
    writer: Optional[object] = None,
    enable_external_weight_init: bool = False,
    enable_public_pretrain: bool = False,
) -> nn.Module:
    model = model.to(device)
    checkpoint_state = None
    if enable_external_weight_init:
        checkpoint_state = _load_teacher_state_if_needed(config)
        initialized = initialize_model_from_teacher(
            model,
            num_classes=config.model.num_classes,
            checkpoint_or_state=checkpoint_state,
            pretrained_imagenet=bool(config.federated.general_pretrain_imagenet_init) and checkpoint_state is None,
        )
        if initialized:
            source = "teacher checkpoint" if checkpoint_state is not None else "ImageNet initialization"
            LOGGER.info("%s model initialized from %s.", algorithm_name, source)

    if enable_public_pretrain:
        _pretrain_model_on_public(
            model=model,
            config=config,
            data_module=data_module,
            device=device,
            public_dataset=public_dataset,
            algorithm_name=algorithm_name,
            writer=writer,
        )
    return model


def build_server_baseline_model(
    config,
    device,
    data_module,
    public_dataset: Optional[Dataset] = None,
    algorithm_name: str = "baseline",
    writer: Optional[object] = None,
) -> nn.Module:
    return prepare_server_model(
        model=build_baseline_model(config),
        config=config,
        device=device,
        data_module=data_module,
        public_dataset=public_dataset,
        algorithm_name=algorithm_name,
        writer=writer,
        enable_external_weight_init=bool(config.federated.baseline_match_general_resources),
        enable_public_pretrain=(
            bool(config.federated.baseline_match_general_resources)
            and bool(config.federated.general_pretrain_on_public)
        ),
    )


def _load_teacher_state_if_needed(config):
    if not bool(config.federated.general_init_from_teacher):
        return None
    checkpoint_value = config.dataset.difficulty_checkpoint
    if not checkpoint_value:
        LOGGER.warning("baseline_match_general_resources is enabled, but dataset.difficulty_checkpoint is not set.")
        return None

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        LOGGER.warning("Baseline teacher checkpoint not found at %s. Falling back to other initialization.", checkpoint_path)
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def _pretrain_model_on_public(
    model: nn.Module,
    config,
    data_module,
    device,
    public_dataset: Optional[Dataset],
    algorithm_name: str,
    writer: Optional[object] = None,
) -> None:
    pretrain_epochs = max(int(config.federated.general_pretrain_epochs), 0)
    if pretrain_epochs == 0:
        return
    if public_dataset is None or len(public_dataset) == 0:
        LOGGER.warning("%s baseline public pretraining was requested, but no public dataset is available.", algorithm_name)
        return

    pretrain_lr = float(config.federated.general_pretrain_lr)
    LOGGER.info(
        "%s baseline pretrain on public dataset | epochs=%d | lr=%.4f | samples=%d",
        algorithm_name,
        pretrain_epochs,
        pretrain_lr,
        len(public_dataset),
    )

    train_loader = data_module.make_loader(public_dataset, shuffle=True)
    eval_loader = data_module.make_loader(public_dataset, shuffle=False)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=pretrain_lr,
        momentum=config.federated.local_momentum,
        weight_decay=config.federated.local_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    for epoch in range(1, pretrain_epochs + 1):
        avg_loss = _run_public_train_epoch(model, train_loader, optimizer, criterion, device)
        public_acc = _evaluate_classifier(model, eval_loader, device)
        if public_acc >= best_acc:
            best_acc = public_acc
            best_state = copy.deepcopy(model.state_dict())

        if writer is not None:
            writer.add_scalar(f"pretrain/{algorithm_name}_baseline_public_loss", avg_loss, epoch)
            writer.add_scalar(f"pretrain/{algorithm_name}_baseline_public_acc", public_acc, epoch)
        LOGGER.info(
            "%s baseline pretrain epoch %d/%d | loss=%.4f | acc=%.4f",
            algorithm_name,
            epoch,
            pretrain_epochs,
            avg_loss,
            public_acc,
        )
        scheduler.step()

    model.load_state_dict(best_state)
    LOGGER.info("%s baseline public pretrain complete | best_acc=%.4f", algorithm_name, best_acc)


def _run_public_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def _evaluate_classifier(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    return correct / max(total, 1)
