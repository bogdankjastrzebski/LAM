import torch
import torch.nn as nn
import torch.nn.functional as F
from .steering import covariance_from_gaussians, frobenius_loss


class GaussianParameters(nn.Module):
    def __init__(self, K=32):
        super().__init__()
        self.K = K
        mu_az_init = 2 * torch.pi * torch.rand(K)
        mu_el_init = (torch.rand(K) - 0.5) * torch.pi
        log_pi_init = torch.full((K,), -4.6)
        self.mu_az = nn.Parameter(mu_az_init)
        self.mu_el = nn.Parameter(mu_el_init)
        self.log_pi = nn.Parameter(log_pi_init)

    def get_pi(self):
        return F.softplus(self.log_pi)

    def forward(self):
        return self.mu_az, self.mu_el, self.get_pi()


class GaussianFittingLoss(nn.Module):
    def forward(self, mu_az, mu_el, pi, C_obs):
        C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
        return frobenius_loss(C_pred, C_obs).mean()


def adam_fit(mu_az, mu_el, log_pi, C_obs, max_iters=300, lr=1e-2):
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
