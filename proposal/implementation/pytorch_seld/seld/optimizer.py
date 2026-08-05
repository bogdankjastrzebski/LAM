import torch
import torch.nn.functional as F
from .gaussians import adam_fit


class TemporalTracker:
    def __init__(self, K=32, cold_iters=300, warm_iters=5, lr_cold=1e-2, lr_warm=1e-3):
        self.K = K
        self.cold_iters = cold_iters
        self.warm_iters = warm_iters
        self.lr_cold = lr_cold
        self.lr_warm = lr_warm
        self.is_first_frame = True
        self.mu_az = None
        self.mu_el = None
        self.log_pi = None

    def fit_frame(self, C_obs):
        if self.is_first_frame:
            self.mu_az = 2 * torch.pi * torch.rand(1, self.K, requires_grad=True)
            self.mu_el = (torch.rand(1, self.K) - 0.5) * torch.pi
            self.mu_el = self.mu_el.requires_grad_(True)
            self.log_pi = torch.full((1, self.K), -4.6, requires_grad=True)
            iters, lr = self.cold_iters, self.lr_cold
            self.is_first_frame = False
        else:
            self.mu_az = self.mu_az.clone().detach().requires_grad_(True)
            self.mu_el = self.mu_el.clone().detach().requires_grad_(True)
            self.log_pi = self.log_pi.clone().detach().requires_grad_(True)
            iters, lr = self.warm_iters, self.lr_warm

        self.mu_az, self.mu_el, pi = adam_fit(
            self.mu_az, self.mu_el, self.log_pi, C_obs, max_iters=iters, lr=lr
        )
        self.log_pi = (pi + 1e-8).log().detach()
        return self.mu_az, self.mu_el, pi
