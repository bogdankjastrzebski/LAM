import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import YOLOFeatureExtractor, ImplicitFeaturePooling
from .classifier import ClassificationHead
from .steering import covariance_from_gaussians, frobenius_loss, STARSS_MIC_POSITIONS
from .visual_driven_fitting import VisualDrivenGaussianFitting


class SELDOutput(dict):
    """
    Structured dictionary return supporting both key indexing (out['mu_az'])
    and legacy tuple unpacking (logits, mu_az, mu_el, pi, loss = model(...)).
    """
    def __iter__(self):
        return iter((
            self["class_logits"],
            self["mu_az"],
            self["mu_el"],
            self["pi"],
            self["spatial_loss"]
        ))


class SELDModel(nn.Module):
    """
    Parametric Audio-Visual SELD model with YOLO feature extraction,
    joint gradient-based spatial fitting, and Envelope Theorem decoupling.
    
    Given:
      a) video image / frames: computes dense YOLO feature field H_v
      b) microphone covariance matrix C_obs
    Runs a gradient method (Adam) to estimate point of sound (mu_az, mu_el)
    driven by both microphone covariance and visual features.
    
    When a sound source becomes silent, appearance tracking (prev_features)
    and class guidance allow the model to continue following the source
    smoothly using video features alone.
    """
    def __init__(self, K=8, num_classes=14, yolo_channels=256,
                 lambda_cls=1.0, lambda_track=1.0, use_real_yolo=True):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.yolo_channels = yolo_channels

        # 1. Visual backbone (YOLOv8 feature extractor, frozen)
        self.visual_backbone = YOLOFeatureExtractor(
            out_channels=yolo_channels, weights="yolov8n.pt", use_real_yolo=use_real_yolo
        )

        # 2. Visual-driven spatial fitter (runs gradient method over mu_az, mu_el, log_pi)
        self.fitter = VisualDrivenGaussianFitting(
            K=K, num_classes=num_classes, yolo_channels=yolo_channels,
            lambda_cls=lambda_cls, lambda_track=lambda_track
        )

        # 3. Soft Gaussian feature pooling
        self.feature_pooling = ImplicitFeaturePooling()

        # 4. Classification head (for end-to-end training via envelope theorem)
        self.classifier = ClassificationHead(in_features=yolo_channels, num_classes=num_classes)

    def extract_visual_features(self, image):
        """Extract dense YOLO feature field: (B, 3, H, W) -> (B, C, H_feat, W_feat)"""
        with torch.no_grad():
            return self.visual_backbone(image)

    def forward(self, image, C_obs, target_classes=None, prev_features=None,
                mu_az_init=None, mu_el_init=None, log_pi_init=None,
                fit_iters=100, fit_lr=1e-2, lambda_cls=None, lambda_track=None,
                mic_positions=None, wavelength=None):
        """
        Forward pass:
          1. Compute video features from YOLO backbone: H_v
          2. Run gradient method to estimate source directions (mu_az, mu_el)
             using microphone covariance C_obs and video features H_v
          3. Apply Envelope Theorem: detach spatial positions and pool visual features
          4. Compute classification logits
        
        Args:
            image: (B, 3, H, W) video frame tensor
            C_obs: (B, 2, 4, 4) microphone covariance tensor
            target_classes: optional (B, K) or (B,) class guidance
            prev_features: optional (B, K, C) visual features from previous frame (for silent tracking)
            mu_az_init, mu_el_init, log_pi_init: optional warm-start positions
            fit_iters: number of Adam gradient steps (e.g. 100 cold, 5 warm)
            fit_lr: learning rate for spatial gradient fitting
            lambda_cls: weight for classification loss
            lambda_track: weight for appearance tracking loss
        """
        device = image.device
        if mic_positions is None:
            mic_positions = STARSS_MIC_POSITIONS.to(device)

        # a) Compute video features from YOLO
        H_v = self.extract_visual_features(image)

        # b) Run gradient method to estimate point of sound
        # Temporarily configure fitter iterations and learning rate
        self.fitter.fit_iters = fit_iters
        self.fitter.lr = fit_lr
        if lambda_cls is not None:
            self.fitter.lambda_cls = lambda_cls
        if lambda_track is not None:
            self.fitter.lambda_track = lambda_track

        mu_az, mu_el, pi, inner_logits, z_fitted, total_loss = self.fitter(
            feature_field=H_v,
            C_obs=C_obs,
            mu_az_init=mu_az_init,
            mu_el_init=mu_el_init,
            log_pi_init=log_pi_init,
            target_classes=target_classes,
            prev_features=prev_features,
            mic_positions=mic_positions,
            wavelength=wavelength
        )

        # c) Envelope Theorem: detach spatial parameters
        mu_az_final = mu_az.detach()
        mu_el_final = mu_el.detach()
        pi_final = pi.detach()

        # Compute audio covariance reconstruction loss
        C_pred = covariance_from_gaussians(mu_az_final, mu_el_final, pi_final,
                                           mic_positions=mic_positions, wavelength=wavelength)
        spatial_loss = frobenius_loss(C_pred, C_obs).mean().detach()

        # Pool features at final converged positions and classify
        z_final = self.feature_pooling(H_v, mu_az_final, mu_el_final, sigma=0.15)
        class_logits = self.classifier(z_final)

        return SELDOutput({
            "class_logits": class_logits,
            "mu_az": mu_az_final,
            "mu_el": mu_el_final,
            "pi": pi_final,
            "features": z_final.detach(),
            "feature_field": H_v,
            "spatial_loss": spatial_loss,
            "total_fit_loss": total_loss,
        })
