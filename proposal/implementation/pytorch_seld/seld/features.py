"""
Feature extraction and implicit pooling.

Visual stream:
  YOLO backbone → dense feature field H_v ∈ R^{C_v × H × W}

Implicit pooling:
  z_{v,k} = ∫ H_v(x) · N(x | μ_k, Σ_k) dx

Instead of rendering a 484-pixel map, we integrate the YOLO feature
field over each Gaussian's spatial footprint. This is a differentiable
attention mechanism: Gaussians act as soft spatial probes.
"""

import torch
import torch.nn as nn
import numpy as np


class YOLOFeatureExtractor(nn.Module):
    """Extract intermediate feature maps from YOLO backbone.

    Placeholder: in practice, load YOLOv8-nano and extract from neck.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        # Minimal CNN placeholder — replace with actual YOLO backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        """x: (B, 3, H_in, W_in) 360° frames
        Returns: (B, C_v, H, W) feature field"""
        return self.backbone(x)


class ImplicitFeaturePooling(nn.Module):
    """z_k = ∫ H(x) · N(x | μ_k, σ²I) dx

    Computed as a weighted sum of feature vectors at spatial positions,
    weighted by the Gaussian density at each position.

    Args:
        sigma: (K,) or scalar Gaussian spread (shared or per-source)
    """
    def __init__(self, azimuth_range=(0, 2 * np.pi), elevation_range=(-0.5 * np.pi, 0.5 * np.pi)):
        super().__init__()
        self.az_min, self.az_max = azimuth_range
        self.el_min, self.el_max = elevation_range

    def _build_coordinate_grid(self, H, W, device):
        """Build (H, W) grids of azimuth/elevation coordinates."""
        az = torch.linspace(self.az_min, self.az_max, W, device=device)
        el = torch.linspace(self.el_max, self.el_min, H, device=device)
        az_grid, el_grid = torch.meshgrid(az, el, indexing='xy')
        return az_grid, el_grid  # (H, W), (H, W)

    def forward(self, feature_field, mu_az, mu_el, sigma=0.15):
        """
        Args:
            feature_field: (B, C, H, W) YOLO features
            mu_az, mu_el: (B, K) Gaussian centroids
            sigma: float or (B, K) spread
        Returns: (B, K, C) pooled feature vectors
        """
        B, C, H, W = feature_field.shape
        K = mu_az.shape[1]
        device = feature_field.device

        az_grid, el_grid = self._build_coordinate_grid(H, W, device)

        # Angular distance with azimuth wraparound
        daz = az_grid.unsqueeze(0).unsqueeze(0) - mu_az.view(B, K, 1, 1)
        daz = torch.min(torch.abs(daz), 2 * np.pi - torch.abs(daz))
        del_ = el_grid.unsqueeze(0).unsqueeze(0) - mu_el.view(B, K, 1, 1)

        dist2 = daz ** 2 + del_ ** 2  # (B, K, H, W)

        # Convert sigma to tensor if scalar
        if isinstance(sigma, (int, float)):
            sigma = torch.full((B, K), sigma, device=device)

        weights = torch.exp(-dist2 / (2 * sigma.view(B, K, 1, 1) ** 2))
        weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Pool: z_{b,k,c} = Σ_{h,w} H_{b,c,h,w} · w_{b,k,h,w}
        pooled = torch.einsum('bchw,bkhw->bkc', feature_field, weights)
        return pooled
