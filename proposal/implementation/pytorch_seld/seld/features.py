import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YOLOFeatureExtractor(nn.Module):
    """
    Feature extractor utilizing YOLO backbone (YOLOv8 from ultralytics)
    with automatic fallback to custom convolutional backbone if YOLO weights are unavailable.
    """
    def __init__(self, out_channels=256, weights="yolov8n.pt", use_real_yolo=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_real_yolo = False
        self.yolo_model = None

        if use_real_yolo:
            try:
                from ultralytics import YOLO
                # Check for weights locally or load
                yolo = YOLO(weights)
                self.yolo_model = yolo.model
                # Freeze YOLO backbone weights
                for p in self.yolo_model.parameters():
                    p.requires_grad = False
                self.yolo_model.eval()

                # Layer 15 has 64 channels in YOLOv8n, project to out_channels
                self.proj = nn.Conv2d(64, out_channels, kernel_size=1)
                self.use_real_yolo = True
            except Exception as e:
                print(f"[YOLOFeatureExtractor] Falling back to CNN backbone ({e})")
                self.use_real_yolo = False

        if not self.use_real_yolo:
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
        """
        x: (B, 3, H, W)
        Returns: (B, out_channels, H_feat, W_feat)
        """
        if self.use_real_yolo and self.yolo_model is not None:
            # Extract features from layer 15 using forward hook
            feat_holder = []
            def hook_fn(mod, inp, out):
                feat_holder.append(out)

            hook = self.yolo_model.model[15].register_forward_hook(hook_fn)
            with torch.no_grad():
                self.yolo_model(x)
            hook.remove()

            if len(feat_holder) > 0:
                raw_feat = feat_holder[0]
                return self.proj(raw_feat)

        return self.backbone(x)


class ImplicitFeaturePooling(nn.Module):
    """
    Implicit differentiable Gaussian feature pooling:
    z_k = ∫ H_v(x) N(x | μ_k, Σ_k) dx

    Uses smooth 2π-periodic azimuth distance: 2(1 - cos(az - mu_az)) + (el - mu_el)^2
    """
    def __init__(self, azimuth_range=(-np.pi, np.pi), elevation_range=(-0.5 * np.pi, 0.5 * np.pi)):
        super().__init__()
        self.az_min, self.az_max = azimuth_range
        self.el_min, self.el_max = elevation_range

    def _build_coordinate_grid(self, H, W, device):
        az = torch.linspace(self.az_min, self.az_max, W, device=device)
        el = torch.linspace(self.el_max, self.el_min, H, device=device)
        az_grid, el_grid = torch.meshgrid(az, el, indexing='xy')
        return az_grid, el_grid

    def forward(self, feature_field, mu_az, mu_el, sigma=0.15):
        B, C, H, W = feature_field.shape
        K = mu_az.shape[1]
        device = feature_field.device
        az_grid, el_grid = self._build_coordinate_grid(H, W, device)

        # Smooth periodic angular distance
        daz = az_grid.unsqueeze(0).unsqueeze(0) - mu_az.view(B, K, 1, 1)
        dist2_az = 2.0 * (1.0 - torch.cos(daz))
        del_ = el_grid.unsqueeze(0).unsqueeze(0) - mu_el.view(B, K, 1, 1)
        dist2 = dist2_az + del_ ** 2

        if isinstance(sigma, (int, float)):
            sigma = torch.full((B, K), sigma, device=device)
        weights = torch.exp(-dist2 / (2 * sigma.view(B, K, 1, 1) ** 2))
        weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)

        pooled = torch.einsum('bchw,bkhw->bkc', feature_field, weights)
        return pooled
