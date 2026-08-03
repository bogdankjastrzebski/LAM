"""
Gaussian fitting via Adam on the Frobenius covariance loss.

Parameters (per Gaussian k, total 4K):
  μ_k^az, μ_k^el  — spatial centroid (azimuth, elevation in radians)
  log σ_k         — spread (not used in direct variant)
  log π_k         — mixing coefficient (intensity)

The optimization minimizes:
  L(μ, π) = ||C_obs - Σ_k π_k · a(μ_k) · a(μ_k)^H||²_F

at 14 μs/iter (Julia, analytical gradients) or ~40 μs/iter (PyTorch autograd).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .steering import covariance_from_gaussians, frobenius_loss


class GaussianParameters(nn.Module):
    """K Gaussian point sources: (μ_az, μ_el, log_π) per source.

    π is parameterized in log-space for positivity. σ not used in the
    direct covariance matching variant (steering vector beamwidth is implicit).
    """
    def __init__(self, K: int = 32):
        super().__init__()
        self.K = K

        # Initialize: random azimuth, random elevation, small intensity
        mu_az_init = 2 * torch.pi * torch.rand(K)
        mu_el_init = (torch.rand(K) - 0.5) * torch.pi
        log_pi_init = torch.full((K,), -4.6)  # π ≈ 0.01

        self.mu_az = nn.Parameter(mu_az_init)
        self.mu_el = nn.Parameter(mu_el_init)
        self.log_pi = nn.Parameter(log_pi_init)

    def get_pi(self):
        return F.softplus(self.log_pi)  # softplus ensures π > 0

    def forward(self):
        return self.mu_az, self.mu_el, self.get_pi()


class GaussianFittingLoss(nn.Module):
    """Frobenius norm between predicted and observed covariance."""
    def forward(self, mu_az, mu_el, pi, C_obs):
        C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
        return frobenius_loss(C_pred, C_obs).mean()


def adam_fit(mu_az, mu_el, log_pi, C_obs,
             max_iters: int = 300, lr: float = 1e-2):
    """Run K Adam iterations to fit Gaussians to observed covariance.

    Args:
        mu_az, mu_el: (B, K) initial centroids (tensors with grad)
        log_pi: (B, K) initial log-intensities (tensor with grad)
        C_obs: (B, 2, 4, 4) observed covariance
        max_iters: gradient steps per frame
        lr: learning rate

    Returns: (mu_az, mu_el, pi) detached — per envelope theorem, gradients
             do NOT flow through the optimization.
    """
    params = [mu_az, mu_el, log_pi]
    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        pi = F.softplus(log_pi)
        C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
        loss = frobenius_loss(C_pred, C_obs).mean()
        loss.backward()
        optimizer.step()

    return mu_az.detach(), mu_el.detach(), F.softplus(log_pi).detach()
