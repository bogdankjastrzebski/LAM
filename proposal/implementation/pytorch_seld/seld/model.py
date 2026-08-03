"""
Top-level model: YOLO features + Gaussian fitting + classification.

Envelope theorem training:
  Spatial fitting produces (μ_k, π_k) which are detached before
  classification. Gradients flow: L_cls → classifier → pooled features
  → YOLO backbone. No gradient through the spatial optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import YOLOFeatureExtractor, ImplicitFeaturePooling
from .gaussians import adam_fit, GaussianParameters
from .classifier import ClassificationHead


class SELDModel(nn.Module):
    """Full parametric audio-visual SELD model.

    Args:
        K: number of Gaussian sources (default 32)
        num_classes: 14 (13 DCASE + inactive)
        yolo_channels: YOLO feature channels
    """
    def __init__(self, K: int = 32, num_classes: int = 14,
                 yolo_channels: int = 256):
        super().__init__()
        self.K = K

        self.visual_backbone = YOLOFeatureExtractor(out_channels=yolo_channels)
        self.feature_pooling = ImplicitFeaturePooling()
        self.classifier = ClassificationHead(
            in_features=yolo_channels, num_classes=num_classes
        )

    def forward(self, image, C_obs, mu_az_init=None, mu_el_init=None,
                log_pi_init=None, fit_iters: int = 300, fit_lr: float = 1e-2):
        """
        Args:
            image: (B, 3, H, W) 360° camera frame
            C_obs: (B, 2, 4, 4) microphone covariance [real, imag]
            mu_az_init, mu_el_init, log_pi_init: (B, K) optional warm-start
            fit_iters: gradient steps for spatial fitting
            fit_lr: learning rate for spatial fitting

        Returns:
            class_logits: (B, K, num_classes) per-source class predictions
            mu_az, mu_el, pi: (B, K) fitted Gaussian parameters (detached)
            fit_loss: (B,) spatial fitting loss
        """
        B = image.shape[0]
        device = image.device

        # Initialize Gaussian parameters if not provided (cold start)
        if mu_az_init is None:
            mu_az = torch.empty(B, self.K, device=device).uniform_(0, 2 * torch.pi)
            mu_el = torch.empty(B, self.K, device=device).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
            mu_az.requires_grad_(True)
            mu_el.requires_grad_(True)
            log_pi = torch.full((B, self.K), -4.6, device=device)
            log_pi.requires_grad_(True)
        else:
            mu_az = mu_az_init.clone().detach().requires_grad_(True)
            mu_el = mu_el_init.clone().detach().requires_grad_(True)
            log_pi = log_pi_init.clone().detach().requires_grad_(True)

        # --- Step 1: Spatial fitting (envelope theorem barrier) ---
        mu_az, mu_el, pi = adam_fit(mu_az, mu_el, log_pi, C_obs,
                                     max_iters=fit_iters, lr=fit_lr)

        # Compute fit loss for logging (detached)
        from .steering import covariance_from_gaussians, frobenius_loss
        C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
        fit_loss = frobenius_loss(C_pred, C_obs)

        # --- Step 2: Visual feature extraction ---
        H_v = self.visual_backbone(image)  # (B, C_v, H, W)

        # --- Step 3: Feature pooling ---
        z = self.feature_pooling(H_v, mu_az, mu_el, sigma=0.15)  # (B, K, C_v)

        # --- Step 4: Classification ---
        class_logits = self.classifier(z)  # (B, K, num_classes)

        return class_logits, mu_az, mu_el, pi, fit_loss
