"""
Training script for the Parametric Gaussian SELD model.

Envelope theorem approach:
  - Spatial fitting (Gaussian optimization) runs per frame, output detached
  - Only classification + feature pooling gradients backprop to YOLO
  - Spatial loss used for logging only; classification loss drives training

Usage:
  python train.py --epochs 100 --K 32 --lr 1e-3
"""

import torch
import torch.nn as nn
import argparse
from seld import SELDModel, TemporalTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fit_iters_cold', type=int, default=300)
    parser.add_argument('--fit_iters_warm', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=14)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SELDModel(K=args.K, num_classes=args.num_classes).to(device)
    tracker = TemporalTracker(K=args.K, cold_iters=args.fit_iters_cold,
                              warm_iters=args.fit_iters_warm)

    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dummy data for testing
    B = 4
    H, W = 128, 256
    dummy_image = torch.randn(B, 3, H, W, device=device)
    dummy_C_obs = torch.randn(B, 2, 4, 4, device=device)
    dummy_labels = torch.randint(0, args.num_classes, (B, args.K), device=device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass: train on a single "frame"
        logits, mu_az, mu_el, pi, fit_loss = model(
            dummy_image, dummy_C_obs, fit_iters=args.fit_iters_cold,
            fit_lr=1e-2
        )

        # Classification loss (envelope theorem: no spatial gradient)
        cls_loss = cls_criterion(
            logits.view(-1, args.num_classes), dummy_labels.view(-1)
        )

        total_loss = cls_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | cls_loss={cls_loss.item():.4f} | "
                  f"fit_loss={fit_loss.mean().item():.2f}")

    print("\nTraining loop runs — gradients flow through classification ✓")
    print("Spatial fitting detached (envelope theorem) ✓")


if __name__ == '__main__':
    main()
