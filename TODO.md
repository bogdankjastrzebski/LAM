# TODO: LAM/UpLAM Evaluation & Improvement

## 1. Baseline Evaluation
- [ ] Compare NN (LAM/UpLAM) vs APGD (200-iter) accuracy on test data
  - Load pre-trained checkpoints (`LAM.pth`, `UpLAM.pth`)
  - Run inference on held-out frames
  - Compute latent from NN forward pass
  - Compute latent from APGD (ground truth routine)
  - Compare: PSNR, SSIM, DoA error (azimuth/elevation RMSE), localization F-score
  - **Key question**: Does the NN actually learn something beyond a warm start, or is it just a fast approximation?

- [ ] Warm-start experiment
  - Run APGD with `x0 = NN_prediction` vs `x0 = zeros`
  - Measure convergence speed (iterations to rtol=1e-3)
  - If NN converges in ~10-20 iters from warm start vs ~200 from zeros, the NN is essentially a learned initializer
  - If final accuracy is similar, the NN + few APGD steps could replace pure NN inference

## 2. Performance Improvements
- [ ] Remove Python loops over 16 bands in encoder/decoder (batch eigh + einsum)
- [ ] Vectorize TV loss neighbor loop (single gather instead of 6 iterations)
- [ ] Move KDTree indices to GPU tensor
- [ ] Switch to float32/complex64 + AMP (currently float64/complex128 everywhere)
- [ ] Unify CDBPN real/imag branches (complex conv2d)
- [ ] Reduce CDBPN `num_stages` from 10 → 3-5
- [ ] Remove matplotlib rendering from validation loop (CPU bottleneck)

## 3. Accuracy Improvements
- [ ] Add LR scheduler (ReduceLROnPlateau or CosineAnnealing)
- [ ] Add gradient clipping
- [ ] Add weight decay to optimizer
- [ ] Add data augmentation (visibility noise, band masking, random 4-mic subset)
- [ ] Increase weight init for D (1e-5 → 1e-3) and tau (1e-7 → 1e-5)
- [ ] Increase TV weight (1e-5 is likely too small to matter)
- [ ] Add EMA (Exponential Moving Average) for inference weights
- [ ] Train longer (50 → 200 epochs) with cosine schedule

## 4. Hybrid Approach
- [ ] Implement NN → APGD refinement pipeline
  - NN forward pass → latent_x → warm-start APGD with 10-20 iterations
  - Compare accuracy vs pure NN and pure APGD
  - Profile runtime tradeoff

## 5. Architecture Questions
- [ ] Replace eigh with direct quadratic form: `latent = diag(Dᵴ · S · D)`
  - Avoid O(32³) eigendecomposition entirely
  - Test if PSD enforcement via eigh is actually needed
- [ ] Consider NNLS (non-negative least squares) as alternative to learned D+τ
- [ ] Evaluate if CDBPN + LAM co-training is better than pre-train LAM → freeze → train CDBPN

## 6. Prior-based Reconstruction (CDBPN alternative)
- **Idea**: Instead of learned upsampler, use analytical prior from dataset statistics
  - Linear conditional: `E[S_32|S_4] = μ + Σ_cross·Σ_4⁻¹·(S_4-μ_4)`
  - Low-rank projection onto learned covariance subspace
  - Dictionary learning: sparse combo of basis covariances
  - Spherical harmonics expansion + truncation
- **Open question**: Does CDBPN actually outperform these? (TODO #1 must be answered first)
- **If NN wins**: Likely because it learns non-linear structure specific to real acoustic data (room reflections, source directivity, noise patterns) that linear priors miss. The 4×4 → 32×32 mapping is highly non-linear in practice.
- **If NN doesn't win**: Replace CDBPN with analytical prior — simpler, faster, interpretable.
- **Hybrid**: Use prior as regularizer for the QP solver instead of target-based regularization. Encodes "what a valid 32×32 covariance looks like" rather than just penalizing deviation from ground truth.
