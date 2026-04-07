import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Compact CNN used as the personalized expert in FedEGS-2."""

    def __init__(self, num_classes: int = 10, base_channels: int = 32, knowledge_dim: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)
        self.feature_dim = base_channels * 4
        self.knowledge_dim = knowledge_dim or self.feature_dim
        if self.knowledge_dim == self.feature_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(self.feature_dim, self.knowledge_dim, bias=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.projector(features)
        return F.normalize(embedding, dim=1)

    def forward_with_features(self, x: torch.Tensor):
        features = self.forward_features(x)
        return features, self.classify_features(features)

    def forward_with_embedding(self, x: torch.Tensor):
        features = self.forward_features(x)
        logits = self.classify_features(features)
        embedding = self.project_features(features)
        return features, embedding, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))
