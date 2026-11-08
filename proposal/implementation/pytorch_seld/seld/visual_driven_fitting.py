import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .steering import covariance_from_gaussians, frobenius_loss, STARSS_MIC_POSITIONS


def init_from_feature_field(feature_field, K=8):
    """Initialize Gaussian centroids at the top-K energy peaks of the YOLO feature field."""
    B, C, H, W = feature_field.shape
    device = feature_field.device
    energy = (feature_field ** 2).sum(dim=1)  # (B, H, W)
    flat_energy = energy.view(B, -1)
    _, topk_indices = torch.topk(flat_energy, min(K, H * W), dim=-1)
    py = topk_indices // W
    px = topk_indices % W
    mu_az = ((px.float() / max(W - 1, 1)) * 2.0 * np.pi) - np.pi
    mu_el = 0.5 * np.pi - ((py.float() / max(H - 1, 1)) * np.pi)
    if K > H * W:
        pad = K - H * W
        mu_az = torch.cat([mu_az, (2.0 * torch.rand(B, pad, device=device) - 1.0) * np.pi], dim=-1)
        mu_el = torch.cat([mu_el, (torch.rand(B, pad, device=device) - 0.5) * np.pi], dim=-1)
    return mu_az, mu_el


class VisualDrivenGaussianFitting(nn.Module):
    """
    Position Gaussians using a joint gradient method driven by:
      1) Audio: Microphone covariance matching (L_spatial)
      2) Video: YOLO visual features via classification and appearance tracking (L_cls, L_track)
    
    When a sound source is silent, audio covariance gradients diminish,
    and visual features (appearance matching / class prediction) seamlessly
    continue tracking the source (e.g. moving person) across video frames.
    """
    def __init__(self, K=32, num_classes=14, yolo_channels=256,
                 lambda_cls=1.0, lambda_track=2.0, fit_iters=150, lr=2e-2,
                 sigma_start=0.6, sigma_end=0.15):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_track = lambda_track
        self.fit_iters = fit_iters
        self.lr = lr
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

        # Frozen classifier (weights not trained during spatial fitting)
        self.classifier = nn.Sequential(
            nn.Linear(yolo_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, feature_field, C_obs, mu_az_init=None, mu_el_init=None,
                log_pi_init=None, target_classes=None, prev_features=None,
                mic_positions=None, wavelength=None, init_from_visual=True):
        """
        Args:
            feature_field: (B, C, H, W) YOLO features (detached)
            C_obs: (B, 2, 4, 4) microphone covariance
            mu_az_init, mu_el_init, log_pi_init: optional warm-start positions
            target_classes: (B, K) or (B,) ground truth / active class indices
            prev_features: (B, K, C) visual features from previous frame for silent tracking
            mic_positions: optional custom microphone array coordinates
            wavelength: acoustic wavelength
            init_from_visual: whether to initialize cold centroids on visual feature peaks
        Returns:
            mu_az, mu_el, pi: fitted parameters (detached)
            class_logits: (B, K, num_classes)
            z: (B, K, C) pooled visual features at fitted positions
            total_loss: scalar tracking total fitting objective
        """
        B = feature_field.shape[0]
        feature_field = feature_field.detach()
        device = feature_field.device
        if mic_positions is None:
            mic_positions = STARSS_MIC_POSITIONS.to(device)

        # Initialize Gaussian parameters
        if mu_az_init is None:
            if init_from_visual:
                mu_az_val, mu_el_val = init_from_feature_field(feature_field, K=self.K)
                mu_az = mu_az_val.clone().requires_grad_(True)
                mu_el = mu_el_val.clone().requires_grad_(True)
            else:
                mu_az = ((2.0 * torch.rand(B, self.K, device=device) - 1.0) * np.pi).requires_grad_(True)
                mu_el = ((torch.rand(B, self.K, device=device) - 0.5) * np.pi).requires_grad_(True)
            log_pi = torch.full((B, self.K), -4.6, device=device).requires_grad_(True)
        else:
            mu_az = mu_az_init.clone().detach().requires_grad_(True)
            mu_el = mu_el_init.clone().detach().requires_grad_(True)
            if log_pi_init is not None:
                log_pi = log_pi_init.clone().detach().requires_grad_(True)
            else:
                log_pi = torch.full((B, self.K), -4.6, device=device).requires_grad_(True)

        # Build equirectangular coordinate grid: Azimuth [-pi, pi], Elevation [pi/2, -pi/2]
        H, W = feature_field.shape[2], feature_field.shape[3]
        az_grid = torch.linspace(-np.pi, np.pi, W, device=device)
        el_grid = torch.linspace(0.5 * np.pi, -0.5 * np.pi, H, device=device)
        az_grid, el_grid = torch.meshgrid(az_grid, el_grid, indexing='xy')

        optimizer = torch.optim.Adam([mu_az, mu_el, log_pi], lr=self.lr)

        total_iters = max(self.fit_iters, 1)
        for step in range(total_iters):
            optimizer.zero_grad()
            pi = F.softplus(log_pi)

            # Coarse-to-fine Gaussian bandwidth annealing
            cur_sigma = self.sigma_start - (self.sigma_start - self.sigma_end) * (step / float(total_iters))

            # 1. --- Spatial loss (audio covariance matching) ---
            C_pred = covariance_from_gaussians(mu_az, mu_el, pi, mic_positions=mic_positions, wavelength=wavelength)
            L_spatial = frobenius_loss(C_pred, C_obs).mean()

            # 2. --- Smooth periodic feature pooling ---
            daz = az_grid.unsqueeze(0).unsqueeze(0) - mu_az.view(B, self.K, 1, 1)
            dist2_az = 2.0 * (1.0 - torch.cos(daz))
            dist2_el = (el_grid.unsqueeze(0).unsqueeze(0) - mu_el.view(B, self.K, 1, 1)) ** 2
            dist2 = dist2_az + dist2_el

            weights = torch.exp(-dist2 / (2.0 * (cur_sigma ** 2)))
            weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)
            z = torch.einsum('bchw,bkhw->bkc', feature_field, weights)

            # 3. --- Visual objective ---
            total_visual_loss = 0.0

            # 3a. Classification guidance
            class_logits = self.classifier(z)
            if target_classes is not None:
                if target_classes.dim() == 1:
                    target_expanded = target_classes.unsqueeze(1).expand(-1, self.K)
                else:
                    target_expanded = target_classes
                L_cls = F.cross_entropy(
                    class_logits.view(-1, self.num_classes),
                    target_expanded.reshape(-1)
                )
                total_visual_loss = total_visual_loss + self.lambda_cls * L_cls
            elif prev_features is None:
                probs = F.softmax(class_logits, dim=-1)
                L_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                total_visual_loss = total_visual_loss + self.lambda_cls * L_entropy

            # 3b. Appearance feature tracking (enables following silent persons)
            if prev_features is not None:
                cos_sim = F.cosine_similarity(z, prev_features.detach(), dim=-1).mean()
                L_track = 1.0 - cos_sim
                total_visual_loss = total_visual_loss + self.lambda_track * L_track

            total_loss = L_spatial + total_visual_loss
            total_loss.backward()
            optimizer.step()

            # Elevation clamping and azimuth wrapping
            with torch.no_grad():
                mu_el.data.clamp_(-0.5 * np.pi, 0.5 * np.pi)
                mu_az.data = ((mu_az.data + np.pi) % (2.0 * np.pi)) - np.pi

        return (mu_az.detach(), mu_el.detach(), F.softplus(log_pi).detach(),
                class_logits.detach(), z.detach(), total_loss.detach())
