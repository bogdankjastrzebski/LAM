# SoundSources — Parametric Gaussian Sound Source Tracking

## Plan

### Goal
Replace the dense 484-pixel spatial map reconstruction (ADMM) with **K parametric 2D Gaussians** that model sound sources directly. Compare against existing solvers on the same microphone data.

### Week 1 (Today–Tomorrow)

| Step | Task | Status |
|------|------|--------|
| 1 | Copy SuperLam assets (steering matrices, grid, Hessians) | pending |
| 2 | Copy SuperLam solvers (ADMM, Chambolle-Pock, Subgradient) | pending |
| 3 | Implement Gaussian tracker: render K Gaussians onto grid, optimize via gradient descent | pending |
| 4 | Write comparison script: ADMM vs Gaussian on synthetic data | pending |
| 5 | Visualize: 2D unwrapped (azimuth×elevation) heatmaps | pending |
| 6 | Test Gaussian fitting on video frame data | pending |
| 7 | Next week: integrate neural feature extractors (swappable encoders) | — |

### Package Structure
```
SoundSources/
├── Project.toml
├── src/
│   ├── SoundSources.jl        # Main module
│   ├── conjugate_gradient.jl   # CG solver (from SuperLam)
│   ├── admm.jl                 # ADMM solver (from SuperLam)
│   ├── chambolle_pock.jl       # Chambolle-Pock solver (from SuperLam)
│   ├── subgradient.jl          # Subgradient solver (from SuperLam)
│   ├── gaussian_tracker.jl     # Gaussian Mixture tracker (NEW)
│   └── utils.jl                # Grid conversions, rendering
├── assets/                     # Copied from SuperLam
│   ├── R.jl                    # Grid points (3×484)
│   ├── A_em32.jl              # Steering 32-mic
│   ├── A_mic.jl               # Steering 4-mic
│   ├── H_em32.jl              # Hessian 32-mic
│   ├── H_mic.jl               # Hessian 4-mic
│   └── assets_loader.jl
├── scripts/
│   └── compare.jl              # Main comparison: ADMM vs Gaussians
└── test/
    └── test_video.jl           # Video frame test
```

### Gaussian Tracker Design

**Parameters** (K ≈ 32):
- `μ_k` — spatial centroid in angular domain (azimuth, elevation)
- `σ_k` — isotropic spread
- `π_k` — intensity / activity weight

**Rendering**: `x_j = Σ_k π_k · N(θ_j | μ_k, σ_k²I)` where θ_j are grid angular coordinates.

**Optimization**: minimize the same objective as ADMM — `0.5·x(θ)ᵀHx(θ) − c_vecᵀx(θ) + λ₁‖x(θ)‖₁ + λ_tv‖Dx(θ)‖₁` — via gradient descent on θ = {μ_k, σ_k, π_k}. Gradients computed with Zygote.jl autodiff.

### Running

```bash
# Start Julia daemon (fast recompilation)
julia --project=. -e 'using Revise; using SoundSources'

# Run comparison
julia --project=. scripts/compare.jl
```
