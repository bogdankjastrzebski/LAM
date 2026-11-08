"""
Fast CG-based solver for the acoustic camera problem.

Replaces APGD (200 iters, O(M²N) each) with CG (~20 iters, O(N²) each).
Precomputes H = |Aᴴ·A|² and c = diag(Aᴴ·S·A) once, then solves:

    min  xᵴHx - 2cᵴx + λγ||x||₁ + λ(1-γ)||x||²₂   s.t. x ≥ 0
"""

import numpy as np
from numpy.linalg import norm


def precompute(A):
    """Precompute H and constant terms (independent of S, x)."""
    AtA = A.conj().T @ A                  # (N, N)
    H = np.abs(AtA) ** 2                   # |Aᴴ·A|²  →  quadratic form
    return H, AtA


def compute_c(S, AtA):
    """c = diag(Aᴴ·S·A) — per-frame projection."""
    return np.real(np.diag(AtA @ S @ AtA.conj().T))


def solve_cg(S, A, H, AtA, lambda_=1e-3, gamma=0.5, x0=None,
             maxit=50, rtol=1e-3, verbose=False):
    """
    Conjugate gradient with soft-thresholding for L1-regularized QP.

    Solves: min  xᵴHx - 2cᵴx + λγ||x||₁ + λ(1-γ)||x||²₂   s.t. x ≥ 0
    """
    M, N = A.shape
    c = compute_c(S, AtA)

    if x0 is None:
        x0 = np.zeros(N)
    x = x0.copy()
    x = np.clip(x, 0, None)

    # Quadratic gradient: g = 2(Hx - c)
    g = 2 * (H @ x - c) + 2 * lambda_ * (1 - gamma) * x
    d = -g.copy()
    g_norm_sq = g @ g

    for k in range(1, maxit + 1):
        # CG direction: H·d
        Hd = H @ d
        alpha = g_norm_sq / (d @ Hd + 1e-12)

        # Line search with soft-thresholding (L1 + non-negativity)
        x_new = x + alpha * d
        threshold = lambda_ * gamma * alpha
        x_new = np.clip(x_new - threshold, 0, None)

        # Update residual
        g_old = g.copy()
        g = 2 * (H @ x_new - c) + 2 * lambda_ * (1 - gamma) * x_new
        g_new_norm_sq = g @ g

        # Check convergence
        if g_new_norm_sq < rtol ** 2:
            if verbose:
                print(f"CG converged at iter {k}, ||g||={np.sqrt(g_new_norm_sq):.2e}")
            break

        # Polak-Ribière beta
        beta = max(0, (g @ (g - g_old)) / g_norm_sq)
        d = -g + beta * d
        g_norm_sq = g_new_norm_sq
        x = x_new

    return x


# ─── Usage ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from util.utils import steering_operator, get_field

    A = steering_operator()               # (32, 484)
    H, AtA = precompute(A)                # precompute once

    # Example: solve for one visibility matrix S
    S = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
    S = (S + S.conj().T) / 2              # make Hermitian

    x = solve_cg(S, A, H, AtA, lambda_=1e-3, gamma=0.5, maxit=50, verbose=True)
    print(f"Solution: {x.shape}, sum={x.sum():.4f}, nnz={(x > 0).sum()}")
