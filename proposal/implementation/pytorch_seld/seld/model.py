import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import YOLOFeatureExtractor, ImplicitFeaturePooling
from .gaussians import adam_fit
from .classifier import ClassificationHead
from .steering import covariance_from_gaussians, frobenius_loss


class SELDModel(nn.Module):
    def __init__(self, K=32, num_classes=14, yolo_channels=256):
        super().__init__()
        self.K = K
        self.visual_backbone = YOLOFeatureExtractor(out_channels=yolo_channels)
        self.feature_pooling = ImplicitFeaturePooling()
        self.classifier = ClassificationHead(in_features=yolo_channels, num_classes=num_classes)

    def forward(self, image, C_obs, mu_az_init=None, mu_el_init=None,
                log_pi_init=None, fit_iters=300, fit_lr=1e-2):
        B = image.shape[0]
        device = image.device

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

        mu_az, mu_el, pi = adam_fit(mu_az, mu_el, log_pi, C_obs, max_iters=fit_iters, lr=fit_lr)

        C_pred = covariance_from_gaussians(mu_az, mu_el, pi)
        fit_loss = frobenius_loss(C_pred, C_obs)

        H_v = self.visual_backbone(image)
        z = self.feature_pooling(H_v, mu_az, mu_el, sigma=0.15)
        class_logits = self.classifier(z)

        return class_logits, mu_az, mu_el, pi, fit_loss
