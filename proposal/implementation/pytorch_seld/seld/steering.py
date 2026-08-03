"""
Steering vectors and microphone geometry for the 4-channel Eigenmike subset.

Mic positions (capsules 5, 9, 25, 21 of EM32):
  All lie in the xz-plane at radius 42mm, forming a linear-ish array.
  This 4-mic planar array has front-back ambiguity (θ vs θ+π).

Wavelength: center frequency of max STFT bin ≈ 4833 Hz, c ≈ 343 m/s.
"""

import torch
import torch.nn as nn
import numpy as np

# Eigenmike 4-mic subset positions (meters)
MIC_POSITIONS = torch.tensor([
    [0.022257, 0.0, 0.035618],
    [0.022257, 0.0, -0.035618],
    [-0.022257, 0.0, -0.035618],
    [-0.022257, 0.0, 0.035618],
], dtype=torch.float32)

WAVELENGTH = 343.0 / 4833.0  # speed of sound / frequency (Hz)
WAVENUMBER = 2 * np.pi / WAVELENGTH


def source_direction(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Unit vector toward direction (azimuth, elevation) in radians.
    Args: az, el: (...,) shaped tensors
    Returns: (..., 3) unit vectors
    """
    sx = torch.cos(el) * torch.cos(az)
    sy = torch.cos(el) * torch.sin(az)
    sz = torch.sin(el)
    return torch.stack([sx, sy, sz], dim=-1)


def dsource_daz(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Derivative of source direction w.r.t. azimuth."""
    return torch.stack([
        -torch.cos(el) * torch.sin(az),
        torch.cos(el) * torch.cos(az),
        torch.zeros_like(az),
    ], dim=-1)


def dsource_del(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Derivative of source direction w.r.t. elevation."""
    return torch.stack([
        -torch.sin(el) * torch.cos(az),
        -torch.sin(el) * torch.sin(az),
        torch.cos(el),
    ], dim=-1)


def steering_vector(az: torch.Tensor, el: torch.Tensor,
                    mic_positions: torch.Tensor = MIC_POSITIONS) -> torch.Tensor:
    """Complex steering vector a(μ) ∈ C⁴.
    a_m(μ) = exp(-i · k · r_m · s(μ))

    Args: az, el: (B, K) source directions in radians
    Returns: (B, K, 4) complex as (real, imag) in last dim
    """
    s = source_direction(az, el)  # (B, K, 3)
    # dot(r_m, s) for all mics: (B, K, 3) · (4, 3) → (B, K, 4)
    dot_products = torch.einsum('...d,md->...m', s, mic_positions)
    phases = WAVENUMBER * dot_products  # (B, K, 4)
    a_real = torch.cos(phases)
    a_imag = -torch.sin(phases)
    return torch.stack([a_real, a_imag], dim=-1)  # (B, K, 4, 2)


def steering_derivative_az(az: torch.Tensor, el: torch.Tensor,
                           mic_positions: torch.Tensor = MIC_POSITIONS) -> torch.Tensor:
    """Derivative ∂a/∂az ∈ C⁴.
    ∂a_m/∂az = -i · k · (r_m · ∂s/∂az) · a_m
    """
    ds = dsource_daz(az, el)  # (B, K, 3)
    a = steering_vector(az, el, mic_positions)  # (B, K, 4, 2)
    a_cplx = a[..., 0] + 1j * a[..., 1]  # (B, K, 4)
    dot_ds = WAVENUMBER * torch.einsum('...d,md->...m', ds, mic_positions)
    da_cplx = -1j * dot_ds * a_cplx
    return torch.stack([da_cplx.real, da_cplx.imag], dim=-1)


def steering_derivative_el(az: torch.Tensor, el: torch.Tensor,
                           mic_positions: torch.Tensor = MIC_POSITIONS) -> torch.Tensor:
    """Derivative ∂a/∂el ∈ C⁴."""
    ds = dsource_del(az, el)
    a = steering_vector(az, el, mic_positions)
    a_cplx = a[..., 0] + 1j * a[..., 1]
    dot_ds = WAVENUMBER * torch.einsum('...d,md->...m', ds, mic_positions)
    da_cplx = -1j * dot_ds * a_cplx
    return torch.stack([da_cplx.real, da_cplx.imag], dim=-1)


def covariance_from_gaussians(mu_az: torch.Tensor, mu_el: torch.Tensor,
                               pi: torch.Tensor) -> torch.Tensor:
    """C_pred = Σ_k π_k · a(μ_k) · a(μ_k)^H  ∈ C⁴ˣ⁴

    Args:
        mu_az, mu_el: (B, K) centroid directions in radians
        pi: (B, K) mixing coefficients (non-negative)
    Returns: (B, 2, 4, 4) covariance [real, imag]
    """
    B, K = mu_az.shape
    a = steering_vector(mu_az, mu_el)  # (B, K, 4, 2)
    a_cplx = a[..., 0] + 1j * a[..., 1]  # (B, K, 4)

    # Outer products: C_k = a_k · a_k^H
    C_all = torch.einsum('bki,bkj->bkij', a_cplx, a_cplx.conj())  # (B, K, 4, 4)
    C_weighted = C_all * pi.view(B, K, 1, 1)
    C_total = C_weighted.sum(dim=1)  # (B, 4, 4)

    return torch.stack([C_total.real, C_total.imag], dim=1)  # (B, 2, 4, 4)


def frobenius_loss(C_pred, C_obs):
    """L = ||C_obs - C_pred||²_F. Both (B, 2, 4, 4)."""
    return ((C_obs - C_pred) ** 2).sum(dim=(1, 2, 3))
