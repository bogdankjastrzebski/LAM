# PyTorch SELD — Parametric Gaussian Sound Event Localization & Detection
#
# Python implementation of the direct Frobenius covariance matching method.
# Integrates YOLO visual features with 4-channel audio for classification.
#
# Package structure:
#   seld/steering.py    — steering vectors, mic geometry
#   seld/gaussians.py   — Gaussian fitting, Frobenius loss, Adam iteration
#   seld/optimizer.py   — cold-start / warm-start optimization loops
#   seld/features.py    — YOLO feature extraction + implicit pooling
#   seld/classifier.py  — classification head
#   seld/model.py       — top-level model combining all modules
#   train.py            — training loop (classification + spatial loss)
#
# Gradient flow:
#   YOLO backbone → H_v  ──→  feature pooling  ──→  classifier  →  L_cls
#                              ↑                           ↑ backprop OK
#   4ch audio → C_obs → Gaussian fitting → (μ_k, π_k)
#         ↑                                  ↑ envelope theorem: NO backprop
#         └────────────────── L_spatial ──────┘  (optimized in Julia or here)
#
# At training time, Gaussian parameters are fitted per frame and detached.
# Gradients flow only through feature pooling and classification, not
# through the spatial optimization (envelope theorem).
#
# Requirements: torch, torchvision, numpy
