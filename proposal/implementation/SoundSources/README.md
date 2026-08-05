# SoundSources — Parametric Gaussian Sound Source Tracking

## Plan

### Goal
Replace the dense 484-pixel spatial map reconstruction (ADMM) with **K parametric 2D Gaussians** that model sound sources directly. Compare against existing solvers on the same microphone data.

### Week 1 (Today–Tomorrow)

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

### Running

```bash
# Start Julia daemon (fast recompilation)
julia --project=. -e 'using Revise; using SoundSources'

# Run comparison
julia --project=. scripts/compare.jl
```
