import torch
import torch.nn as nn
import torch.nn.functional as F
from .steering import covariance_from_gaussians, frobenius_loss


class VisualDrivenGaussianFitting(nn.Module):
    """
    Position Gaussians to maximize classification probability.
    
    YOLO features and classifier weights are frozen.
    Only Gaussian parameters (mu_az, mu_el, log_pi) are optimized.
    
    Loss = L_spatial(audio) + lambda * L_cls(visual)
    """
    def __init__(self, K=32, num_classes=14, yolo_channels=256,
                 lambda_cls=1.0, fit_iters=300, lr=1e-2):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.fit_iters = fit_iters
        self.lr = lr
        
        # Frozen classifier (weights not trained)
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
                log_pi_init=None, target_classes=None):
        """
        Args:
            feature_field: (B, C, H, W) YOLO features (detached)
            C_obs: (B, 2, 4, 4) microphone covariance
            mu_az_init, mu_el_init, log_pi_init: optional warm-start
            target_classes: (B, K) ground truth class indices (for L_cls)
        Returns:
            mu_az, mu_el, pi: fitted parameters
            class_logits: (B, K, num_classes)
            total_loss: L_spatial + lambda * L_cls
        """
        B = feature_field.shape[0]
        device = feature_field.device

        # Initialize Gaussian parameters
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

        # Build angular coordinate grid for pooling
        H, W = feature_field.shape[2], feature_field.shape[3]
        az_grid = torch.linspace(0, 2 * torch.pi, W, device=device)
        el_grid = torch.linspace(0.5 * torch.pi, -0.5 * torch.pi, H, device=device)
        az_grid, el_grid = torch.meshgrid(az_grid, el_grid, indexing='xy')

        optimizer = torch.optim.Adam([mu_az, mu_el, log_pi], lr=self.lr)

        for _ in range(self.fit_iters):
            optimizer.zero_grad()
            pi = F.softplus(log_pi)

            # --- Spatial loss (audio) ---
            C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
            L_spatial = frobenius_loss(C_pred, C_obs).mean()

            # --- Feature pooling ---
            daz = az_grid.unsqueeze(0).unsqueeze(0) - mu_az.view(B, self.K, 1, 1)
            daz = torch.min(torch.abs(daz), 2 * torch.pi - torch.abs(daz))
            del_ = el_grid.unsqueeze(0).unsqueeze(0) - mu_el.view(B, self.K, 1, 1)
            dist2 = daz ** 2 + del_ ** 2
            weights = torch.exp(-dist2 / (2 * 0.15 ** 2))
            weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)
            z = torch.einsum('bchw,bkhw->bkc', feature_field, weights)

            # --- Classification ---
            class_logits = self.classifier(z)
            if target_classes is not None:
                L_cls = F.cross_entropy(
                    class_logits.view(-1, self.num_classes),
                    target_classes.view(-1)
                )
            else:
                # If no targets, maximize entropy (encourage confident predictions)
                probs = F.softmax(class_logits, dim=-1)
                L_cls = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            total_loss = L_spatial + self.lambda_cls * L_cls
            total_loss.backward()
            optimizer.step()

            # Clamp elevation
            mu_az.data.clamp_(0, 2 * torch.pi)
            mu_el.data.clamp_(-0.5 * torch.pi, 0.5 * torch.pi)

        return mu_az.detach(), mu_el.detach(), F.softplus(log_pi).detach(), \
               class_logits, total_loss.detach()
