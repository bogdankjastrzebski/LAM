import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features=256, num_classes=14, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z):
        return self.classifier(z)
