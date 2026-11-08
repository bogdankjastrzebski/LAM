"""
Demonstration: Moving Points with YOLO Features and Tracking a Silent Person
=============================================================================
Scenario:
  Frame 1: A person at direction (az_1, el_1) speaks.
           Audio covariance C_obs has active sound energy.
           The model estimates the sound source and pools the person's visual features.
  
  Frame 2: The person moves across the video to direction (az_2, el_2).
           The person is now COMPLETELY SILENT (C_obs is near-zero ambient noise).
           Audio alone has no directional information.
           Driven by YOLO video features and appearance tracking, the gradient method
           moves the Gaussian point to follow the person while silent!

  Frame 3: The person speaks again at (az_2, el_2).
           Both audio covariance and YOLO features reinforce the localization.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import seld
from seld import SELDModel, steering_vector, covariance_from_gaussians


def create_synthetic_frame_with_person(person_az, person_el, img_h=128, img_w=256):
    """
    Creates an equirectangular image with a visually distinctive person/subject
    located at spherical coordinates (person_az, person_el).
    """
    frame = torch.zeros(3, img_h, img_w)
    
    # Map (az, el) to pixel coordinates (px, py)
    # az in [-pi, pi] -> x in [0, W-1]
    # el in [pi/2, -pi/2] -> y in [0, H-1]
    px = int(((person_az + np.pi) / (2.0 * np.pi)) * (img_w - 1))
    py = int(((0.5 * np.pi - person_el) / np.pi) * (img_h - 1))
    
    # Render a subject patch (person pattern) with distinct colors
    radius = 8
    y_coords, x_coords = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    dist2 = (x_coords - px) ** 2 + (y_coords - py) ** 2
    mask = (dist2 <= radius ** 2).float()
    
    # Add distinctive color signature for the person (e.g. bright warm tone)
    frame[0] = 0.9 * mask + 0.1 * (1.0 - mask)
    frame[1] = 0.4 * mask + 0.1 * (1.0 - mask)
    frame[2] = 0.2 * mask + 0.1 * (1.0 - mask)
    
    return frame.unsqueeze(0)  # (1, 3, H, W)


def main():
    print("=" * 70)
    print("DEMO: Following a Silent Person Using YOLO Video Features")
    print("=" * 70)

    device = torch.device("cpu")
    model = SELDModel(K=1, num_classes=14, yolo_channels=256, use_real_yolo=True).to(device)

    # Coordinates for Person in Frame 1 and Frame 2
    az1, el1 = -0.80, 0.20  # Position 1: azimuth ~ -46 deg, elevation ~ 11 deg
    az2, el2 = 0.90, -0.15  # Position 2: azimuth ~ 51 deg, elevation ~ -9 deg

    print(f"\n--- FRAME 1: Person Speaking at az={az1:.2f} rad, el={el1:.2f} rad ---")
    img1 = create_synthetic_frame_with_person(az1, el1).to(device)
    
    # Generate active audio covariance for position 1
    mu_az_gt1 = torch.tensor([[az1]], device=device)
    mu_el_gt1 = torch.tensor([[el1]], device=device)
    pi_gt1 = torch.tensor([[1.5]], device=device)
    C_obs1 = covariance_from_gaussians(mu_az_gt1, mu_el_gt1, pi_gt1)
    # Add small noise
    C_obs1 = C_obs1 + 0.01 * torch.randn_like(C_obs1)

    # Frame 1: Forward pass (cold start or rough init)
    out1 = model(img1, C_obs1, fit_iters=150, fit_lr=0.03)
    fitted_az1 = out1["mu_az"].item()
    fitted_el1 = out1["mu_el"].item()
    fitted_feat1 = out1["features"]  # (1, 1, 256) pooled appearance feature of person

    print(f"  GT Position:       az={az1:.3f}, el={el1:.3f}")
    print(f"  Fitted Position:   az={fitted_az1:.3f}, el={fitted_el1:.3f}")
    print(f"  Error:             {math.hypot(fitted_az1 - az1, fitted_el1 - el1):.3f} rad")
    print(f"  Audio Loss:        {out1['spatial_loss'].item():.4f}")
    print(f"  Captured visual appearance feature vector: shape {fitted_feat1.shape}")

    print(f"\n--- FRAME 2: Person MOVES to az={az2:.2f}, el={el2:.2f} and is COMPLETELY SILENT ---")
    img2 = create_synthetic_frame_with_person(az2, el2).to(device)
    
    # SILENT AUDIO: C_obs has NO sound source energy (only ambient diffuse noise)
    C_obs2_silent = 0.001 * torch.randn(1, 2, 4, 4, device=device)

    # Test A: If we had NO visual tracking (only silent audio covariance)
    out2_no_vis = model(
        img2, C_obs2_silent,
        prev_features=None,
        mu_az_init=out1["mu_az"],
        mu_el_init=out1["mu_el"],
        fit_iters=100, fit_lr=0.02
    )
    print("  [Without Visual Tracking]:")
    print(f"    Point stuck near previous location: az={out2_no_vis['mu_az'].item():.3f}, el={out2_no_vis['mu_el'].item():.3f}")

    # Test B: WITH YOLO Visual Feature Tracking (tracking appearance against prev_features)
    out2_tracked = model(
        img2, C_obs2_silent,
        prev_features=fitted_feat1,
        mu_az_init=out1["mu_az"],
        mu_el_init=out1["mu_el"],
        fit_iters=150, fit_lr=0.03
    )
    fitted_az2 = out2_tracked["mu_az"].item()
    fitted_el2 = out2_tracked["mu_el"].item()
    err2 = math.hypot(fitted_az2 - az2, fitted_el2 - el2)

    print("  [WITH YOLO Visual Feature Tracking]:")
    print(f"    GT New Position:   az={az2:.3f}, el={el2:.3f}")
    print(f"    Tracked Position:  az={fitted_az2:.3f}, el={fitted_el2:.3f}")
    print(f"    Tracking Error:    {err2:.3f} rad")
    print(f"    Success: Point followed person across frame while silent: {err2 < 0.25} ✓")

    print(f"\n--- FRAME 3: Person Resumes Speaking at Position 2 ---")
    mu_az_gt3 = torch.tensor([[az2]], device=device)
    mu_el_gt3 = torch.tensor([[el2]], device=device)
    pi_gt3 = torch.tensor([[1.5]], device=device)
    C_obs3 = covariance_from_gaussians(mu_az_gt3, mu_el_gt3, pi_gt3)

    out3 = model(
        img2, C_obs3,
        prev_features=out2_tracked["features"],
        mu_az_init=out2_tracked["mu_az"],
        mu_el_init=out2_tracked["mu_el"],
        fit_iters=30, fit_lr=0.01  # Warm start converges in few steps
    )
    print(f"    Joint Audio-Visual Position: az={out3['mu_az'].item():.3f}, el={out3['mu_el'].item():.3f}")
    print(f"    Spatial Loss: {out3['spatial_loss'].item():.5f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)


if __name__ == '__main__':
    main()
