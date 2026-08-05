"""
YOLO-based Audio-Visual SELD with Envelope Theorem Gradient Propagation
======================================================================

The pipeline:
  1. Visual encoder (YOLO backbone) → feature field H_v
  2. Gaussian fitting: θ* = argmin L(θ; H_v, C_obs)  [analytical, via Julia]
  3. Feature pooling: z_k = ∫ H_v(x) N(x | μ_k, Σ_k) dx
  4. Classification head: ŷ_k = Softmax(W · z_k + b)

Gradient flow via the envelope theorem:
  At the optimum θ*, ∂L/∂θ = 0 by definition. Therefore the total derivative
  of the classification loss L_cls with respect to encoder parameters φ is:

    dL_cls/dφ = ∂L_cls/∂φ + ∂L_cls/∂θ* · dθ*/dφ

  The term ∂L_cls/∂θ* · dθ*/dφ requires differentiating through the
  optimization. By the implicit function theorem applied to the optimality
  condition ∇_θ L(θ*; φ) = 0:

    dθ*/dφ = -(∇_θ² L)^(-1) · ∇_φ∇_θ L

  In our case, φ are the encoder weights (YOLO backbone), θ are the
  Gaussian parameters (μ_k, π_k), and L is the Frobenius covariance loss.

  Crucially, the spatial fitting L depends on φ only through the steering
  vectors a(μ) — the encoder features enter through the classification
  head, not the spatial loss. This decouples the optimization:

    ∂L/∂φ = 0  (spatial loss does not depend on encoder weights)
    ⇒ dθ*/dφ = 0  (optimizer treats each frame independently)
    ⇒ dL_cls/dφ = ∂L_cls/∂φ  (standard backprop through classification)

  The envelope theorem simplifies the training: we run Gaussian fitting
  as a non-differentiable black box (or differentiable via implicit
  differentiation), then backprop only through the classification and
  feature pooling stages.

Status: proof-of-concept scaffolding. Gaussian fitting currently runs
        in Julia (see SoundSources.jl); this module defines the PyTorch
        wrapper for the neural pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import detection
import numpy as np


class SteeringVector(nn.Module):
    """
    Complex steering vector a(μ) ∈ C⁴ for direction μ = (θ, φ).
    Mic positions: 4-element planar Eigenmike subset at r=42mm.
    Frequency: center of max STFT bin (~4833 Hz).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mic_positions", torch.tensor([
            [ 0.022257,  0.0,  0.035618],
            [ 0.022257,  0.0, -0.035618],
            [-0.022257,  0.0, -0.035618],
            [-0.022257,  0.0,  0.035618],
        ], dtype=torch.float32))
        self.wavelength = 343.0 / 4833.0

    def forward(self, az: torch.Tensor, el: torch.Tensor):
        """
        az, el: (K,) tensors in radians
        Returns: (K, 4) complex steering vectors as (K, 4, 2) real tensors
        """
        K = az.shape[0]
        sx = torch.cos(el) * torch.cos(az)
        sy = torch.cos(el) * torch.sin(az)
        sz = torch.sin(el)
        s = torch.stack([sx, sy, sz], dim=-1)

        k = 2 * np.pi / self.wavelength
        phases = k * (self.mic_positions @ s.T).T

        a_real = torch.cos(phases)
        a_imag = -torch.sin(phases)
        return torch.stack([a_real, a_imag], dim=-1)


class CovarianceFromGaussians(nn.Module):
    """
    C_pred = Σ_k π_k · a(μ_k) · a(μ_k)^H
    Computes 4×4 complex covariance from K Gaussian parameters.
    """
    def __init__(self, K: int = 32):
        super().__init__()
        self.K = K
        self.steering = SteeringVector()

    def forward(self, mu_az: torch.Tensor, mu_el: torch.Tensor, pi: torch.Tensor):
        """
        mu_az, mu_el: (K,) azimuth/elevation in radians
        pi: (K,) mixing coefficients (non-negative)
        Returns: (2, 4, 4) covariance [real, imag]
        """
        a = self.steering(mu_az, mu_el)
        K = a.shape[0]

        a_cplx = a[..., 0] + 1j * a[..., 1]
        C = torch.einsum('ki,kj->kij', a_cplx, a_cplx.conj())
        C_weighted = C * pi.view(K, 1, 1)
        C_total = C_weighted.sum(dim=0)

        return torch.stack([C_total.real, C_total.imag], dim=0)


class GaussianFittingLoss(nn.Module):
    """
    Frobenius loss: ||C_obs - C_pred||^2_F

    This module encapsules the spatial fitting objective.
    For training with envelope theorem: after fitting θ* = argmin L,
    we treat θ* as detached and only backprop through the classification
    and feature pooling stages.
    """
    def forward(self, mu_az, mu_el, pi, C_obs):
        C_pred = CovarianceFromGaussians(K=mu_az.shape[0])(
            mu_az, mu_el, F.softplus(pi)
        )
        diff = C_obs - C_pred
        return (diff ** 2).sum()


class FeaturePooling(nn.Module):
    """
    Implicit differentiable feature pooling:
    z_{v,k} = ∫ H_v(x) N(x | μ_k, Σ_k) dx

    Args:
        feature_grid: (C, H, W) feature map from YOLO backbone
        az_coords: (H, W) azimuth coordinates for each grid cell
        el_coords: (H, W) elevation coordinates
        mu_az, mu_el: (K,) Gaussian centroids
        sigma: (K,) Gaussian spreads

    Returns: (K, C) pooled feature vectors
    """
    def forward(self, feature_grid, az_coords, el_coords,
                mu_az, mu_el, sigma):
        C, H, W = feature_grid.shape
        K = mu_az.shape[0]

        daz = (az_coords.unsqueeze(0) - mu_az.view(K, 1, 1))
        daz = torch.min(torch.abs(daz), 2 * np.pi - torch.abs(daz))
        del_ = el_coords.unsqueeze(0) - mu_el.view(K, 1, 1)
        dist2 = daz ** 2 + del_ ** 2

        weights = torch.exp(-dist2 / (2 * sigma.view(K, 1, 1) ** 2))
        weights = weights / (weights.sum(dim=(1,2), keepdim=True) + 1e-8)

        pooled = torch.einsum('chw,khw->kc', feature_grid, weights)
        return pooled


class YOLOToFeatureField(nn.Module):
    """
    Extracts dense feature field from YOLO backbone.
    Uses intermediate feature maps (not final detection heads).
    """
    def __init__(self, backbone: str = "yolov8n"):
        super().__init__()
        self.backbone_name = backbone
        self.feature_channels = 256

    def forward(self, x):
        """
        x: (B, 3, H_in, W_in) input 360-degree frames
        Returns: (B, C_v, H, W) feature field
        """
        raise NotImplementedError("YOLO backbone not yet integrated")


if __name__ == "__main__":
    K = 8
    mu_az = torch.randn(K, requires_grad=True)
    mu_el = torch.randn(K, requires_grad=True)
    pi = torch.randn(K, requires_grad=True)
    C_obs = torch.randn(2, 4, 4)

    loss_fn = GaussianFittingLoss()
    loss = loss_fn(mu_az, mu_el, pi, C_obs)
    loss.backward()

    print("gradient flow test")
    print(f"  loss={loss.item():.3f}")
    print(f"  dL/d(az)[0]={mu_az.grad[0].item():.4f}  dL/d(pi)[0]={pi.grad[0].item():.4f}")
    print("ok")
