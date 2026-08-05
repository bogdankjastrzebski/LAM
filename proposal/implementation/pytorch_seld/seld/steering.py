import torch
import torch.nn as nn
import numpy as np

MIC_POSITIONS = torch.tensor([
    [0.022257, 0.0, 0.035618],
    [0.022257, 0.0, -0.035618],
    [-0.022257, 0.0, -0.035618],
    [-0.022257, 0.0, 0.035618],
], dtype=torch.float32)

WAVELENGTH = 343.0 / 4833.0
WAVENUMBER = 2 * np.pi / WAVELENGTH


def source_direction(az, el):
    sx = torch.cos(el) * torch.cos(az)
    sy = torch.cos(el) * torch.sin(az)
    sz = torch.sin(el)
    return torch.stack([sx, sy, sz], dim=-1)


def dsource_daz(az, el):
    return torch.stack([
        -torch.cos(el) * torch.sin(az),
        torch.cos(el) * torch.cos(az),
        torch.zeros_like(az),
    ], dim=-1)


def dsource_del(az, el):
    return torch.stack([
        -torch.sin(el) * torch.cos(az),
        -torch.sin(el) * torch.sin(az),
        torch.cos(el),
    ], dim=-1)


def steering_vector(az, el, mic_positions=MIC_POSITIONS):
    s = source_direction(az, el)
    dot_products = torch.einsum('...d,md->...m', s, mic_positions)
    phases = WAVENUMBER * dot_products
    a_real = torch.cos(phases)
    a_imag = -torch.sin(phases)
    return torch.stack([a_real, a_imag], dim=-1)


def steering_derivative_az(az, el, mic_positions=MIC_POSITIONS):
    ds = dsource_daz(az, el)
    a = steering_vector(az, el, mic_positions)
    a_cplx = a[..., 0] + 1j * a[..., 1]
    dot_ds = WAVENUMBER * torch.einsum('...d,md->...m', ds, mic_positions)
    da_cplx = -1j * dot_ds * a_cplx
    return torch.stack([da_cplx.real, da_cplx.imag], dim=-1)


def steering_derivative_el(az, el, mic_positions=MIC_POSITIONS):
    ds = dsource_del(az, el)
    a = steering_vector(az, el, mic_positions)
    a_cplx = a[..., 0] + 1j * a[..., 1]
    dot_ds = WAVENUMBER * torch.einsum('...d,md->...m', ds, mic_positions)
    da_cplx = -1j * dot_ds * a_cplx
    return torch.stack([da_cplx.real, da_cplx.imag], dim=-1)


def covariance_from_gaussians(mu_az, mu_el, pi):
    B, K = mu_az.shape
    a = steering_vector(mu_az, mu_el)
    a_cplx = a[..., 0] + 1j * a[..., 1]
    C_all = torch.einsum('bki,bkj->bkij', a_cplx, a_cplx.conj())
    C_weighted = C_all * pi.view(B, K, 1, 1)
    C_total = C_weighted.sum(dim=1)
    return torch.stack([C_total.real, C_total.imag], dim=1)


def frobenius_loss(C_pred, C_obs):
    return ((C_obs - C_pred) ** 2).sum(dim=(1, 2, 3))
