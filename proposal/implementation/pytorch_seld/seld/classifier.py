"""
Classification head: maps pooled features to semantic class probabilities.

13 DCASE target classes + 1 inactive/background class = 14 classes.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Lightweight MLP: pooled features → class logits.

    Args:
        in_features: C_v (YOLO feature channels)
        num_classes: typically 14 (13 DCASE + inactive)
        hidden_dim: hidden layer size
    """
    def __init__(self, in_features: int = 256, num_classes: int = 14,
                 hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z):
        """z: (B, K, C) pooled feature vectors
        Returns: (B, K, num_classes) class logits per source"""
        return self.classifier(z)
