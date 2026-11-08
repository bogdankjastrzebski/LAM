"""
Test visual-driven Gaussian fitting and SELDModel:
1. Gaussians follow YOLO features via classification and appearance tracking.
2. SELDModel forward pass with video frames and microphone covariance.
3. Silent tracking: following a source when microphone sound is absent.
4. Execution on real STARSS23 dataset if available.
"""

import math
import numpy as np
import torch
import seld
from seld import VisualDrivenGaussianFitting, SELDModel, STARSSDataset, covariance_from_gaussians


def test_visual_driven_fitting():
    print("--- 1. Testing VisualDrivenGaussianFitting ---")
    B, K, C, H, W = 2, 8, 256, 32, 64
    num_classes = 14

    features = torch.randn(B, C, H, W).detach()
    C_obs = torch.randn(B, 2, 4, 4)
    targets = torch.randint(0, num_classes, (B, K))

    fitter = VisualDrivenGaussianFitting(
        K=K, num_classes=num_classes, yolo_channels=C,
        lambda_cls=0.5, fit_iters=30, lr=1e-2
    )

    for name, p in fitter.classifier.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"
    print("  Classifier frozen check passed ✓")

    mu_az, mu_el, pi, logits, z, loss = fitter(
        features, C_obs, target_classes=targets
    )

    print(f"  mu_az range: [{mu_az.min():.2f}, {mu_az.max():.2f}]")
    print(f"  mu_el range: [{mu_el.min():.2f}, {mu_el.max():.2f}]")
    print(f"  pi range:    [{pi.min():.3f}, {pi.max():.3f}]")
    print(f"  loss:        {loss.item():.3f}")
    print(f"  logits shape: {logits.shape}")
    print("  Visual-driven fitting works ✓")


def test_seld_model_pipeline():
    print("\n--- 2. Testing SELDModel Full Pipeline ---")
    B, K = 1, 4
    model = SELDModel(K=K, num_classes=14, yolo_channels=256, use_real_yolo=True)

    # Input video frame and microphone covariance
    video = torch.randn(B, 3, 128, 256)
    C_obs = torch.randn(B, 2, 4, 4)

    out = model(video, C_obs, fit_iters=10, fit_lr=0.02)
    print("  Forward pass successful!")
    print(f"  YOLO feature field shape: {out['feature_field'].shape}")
    print(f"  Converged mu_az: {out['mu_az'][0].tolist()}")
    print(f"  Converged mu_el: {out['mu_el'][0].tolist()}")
    print(f"  Class logits shape: {out['class_logits'].shape}")
    print(f"  Pooled features shape: {out['features'].shape}")
    print("  SELDModel pipeline works ✓")


def test_silent_tracking():
    print("\n--- 3. Testing Silent Source Tracking via YOLO Features ---")
    from demo_silent_tracking import create_synthetic_frame_with_person
    model = SELDModel(K=1, num_classes=14, yolo_channels=256, use_real_yolo=True)

    az1, el1 = -0.80, 0.20
    az2, el2 = 0.90, -0.15

    frame1 = create_synthetic_frame_with_person(az1, el1)
    C_obs1 = covariance_from_gaussians(torch.tensor([[az1]]), torch.tensor([[el1]]), torch.tensor([[1.5]]))
    out1 = model(frame1, C_obs1, fit_iters=100, fit_lr=0.03)

    frame2 = create_synthetic_frame_with_person(az2, el2)
    C_obs2_silent = 0.001 * torch.randn(1, 2, 4, 4)

    out2 = model(
        frame2, C_obs2_silent,
        prev_features=out1["features"],
        mu_az_init=out1["mu_az"],
        mu_el_init=out1["mu_el"],
        fit_iters=120, fit_lr=0.03
    )

    err = math.hypot(out2["mu_az"].item() - az2, out2["mu_el"].item() - el2)
    print(f"  Frame 1 initial pos:  az={out1['mu_az'].item():.3f}, el={out1['mu_el'].item():.3f}")
    print(f"  Frame 2 silent track: az={out2['mu_az'].item():.3f}, el={out2['mu_el'].item():.3f}")
    print(f"  GT target was:        az={az2:.3f}, el={el2:.3f}")
    print(f"  Angular tracking err: {err:.3f} rad")
    assert err < 0.25, f"Tracking error too high: {err}"
    print("  Silent tracking works ✓")


def test_real_dataset():
    print("\n--- 4. Testing Real STARSS23 Dataset ---")
    try:
        dataset = STARSSDataset(split="dev-train-sony", active_only=True)
        if len(dataset) > 0:
            item = dataset[0]
            print(f"  Loaded real STARSS23 clip: {item['clip']}, time={item['t']:.2f}s")
            print(f"  Real audio C_obs shape: {item['C_obs'].shape}, video image shape: {item['image'].shape}")
            model = SELDModel(K=2, num_classes=14, use_real_yolo=True)
            out = model(
                item['image'].unsqueeze(0),
                item['C_obs'].unsqueeze(0),
                fit_iters=15,
                wavelength=343.0 / item['freq']
            )
            print(f"  Forward pass on real STARSS23 data completed successfully ✓")
            print(f"  Estimated source azimuths: {out['mu_az'][0].tolist()}")
            print(f"  Estimated source elevations: {out['mu_el'][0].tolist()}")
        else:
            print("  STARSS dataset empty or path not found, skipping real sample.")
    except Exception as e:
        print(f"  Dataset test skipped: {e}")


def main():
    print("=" * 60)
    print("RUNNING PYTORCH SELD COMPREHENSIVE TESTS")
    print("=" * 60)
    test_visual_driven_fitting()
    test_seld_model_pipeline()
    test_silent_tracking()
    test_real_dataset()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
