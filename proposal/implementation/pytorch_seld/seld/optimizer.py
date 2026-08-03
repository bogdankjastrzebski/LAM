"""
Warm-start optimization: reuse previous-frame parameters.

Cold start: 300 iters (4.2 ms in Julia, ~10 ms in PyTorch)
Warm start: 3-5 iters (<100 μs in Julia, ~200 μs in PyTorch)

The temporal prior from the Bayesian filter is implemented simply:
parameters from frame t-1 are used as initialization for frame t.
"""

import torch
import torch.nn.functional as F
from .gaussians import adam_fit


class TemporalTracker:
    """Manages Gaussian parameters across frames with warm-start."""

    def __init__(self, K: int = 32, cold_iters: int = 300, warm_iters: int = 5,
                 lr_cold: float = 1e-2, lr_warm: float = 1e-3):
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
        """Fit Gaussians to one frame of microphone data.

        Args: C_obs: (1, 2, 4, 4) observed covariance
        Returns: (mu_az, mu_el, pi) detached
        """
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
        # Store log_pi for next frame
        self.log_pi = (pi + 1e-8).log().detach()
        return self.mu_az, self.mu_el, pi
