import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fedegs.models import build_teacher_model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the CIFAR-10 difficulty teacher")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="artifacts/checkpoints/cifar10_resnet18_teacher.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train_aug = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_teacher_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Teacher pretraining did not produce a checkpoint.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"saved best teacher to: {output_path}")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
