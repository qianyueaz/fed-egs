from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

seed = 42
epochs = 30
batch_size = 128
lr = 0.01
data_root = "./data"
save_path = Path("artifacts/checkpoints/cifar10_resnet18_teacher.pt")

torch.manual_seed(seed)

device = torch.device("cuda:0")
print(f"device: {device}")

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

full_train_aug = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
full_train_eval = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_transform)
test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)

g = torch.Generator().manual_seed(seed)
indices = torch.randperm(len(full_train_aug), generator=g).tolist()
train_indices = indices[:45000]
val_indices = indices[45000:]

train_set = Subset(full_train_aug, train_indices)
val_set = Subset(full_train_eval, val_indices)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def evaluate(loader):
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
    return loss_sum / total, correct / total

best_val_acc = -1.0
best_state = None

for epoch in range(1, epochs + 1):
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
    train_loss = loss_sum / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(best_state, save_path)
print(f"saved best teacher to: {save_path}")

model.load_state_dict(best_state)
test_loss, test_acc = evaluate(test_loader)
print(f"best_val_acc={best_val_acc:.4f}")
print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
