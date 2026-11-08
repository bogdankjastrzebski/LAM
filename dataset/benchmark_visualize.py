"""
benchmark_visualize.py - Run Chambolle-Pock & Subgradient on synthetic HDF5 data
and visualize results using the project's draw_map() style.

Saves spherical acoustic maps (true vs recreated) as PNGs.
"""

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import skimage.util as skutil
import scipy.constants as constants
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from trainer.utils import cart2eq, wrapped_rad2deg, cmap_from_list, get_field as get_field_trainer

# ---------------------------------------------------------------------------
# Grid and array geometry (exact copy from util/utils.py)
# ---------------------------------------------------------------------------

_EIGENMIKE_ = {
    "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042], "5": [32, 0, 0.042],
    "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042], "10": [125, 315, 0.042],
    "11": [90, 291, 0.042], "12": [55, 315, 0.042], "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
    "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],"25": [148, 180, 0.042],
    "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
    "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}


def _deg2rad(d):
    return {m: [np.radians(c[0]), np.radians(c[1]), c[2]] for m, c in d.items()}


def _polar2cart(d):
    return {m: [c[2]*np.sin(c[0])*np.cos(c[1]),
                c[2]*np.sin(c[0])*np.sin(c[1]),
                c[2]*np.cos(c[0])] for m, c in d.items()}


def get_xyz():
    return np.array([[c for c in _polar2cart(_deg2rad(_EIGENMIKE_))[ch]] for ch in _EIGENMIKE_]).T


def fibonacci(N):
    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2*n + 1) / N_px)
    lon = (4*np.pi*n) / (1 + np.sqrt(5))
    return np.stack([np.sin(colat)*np.cos(lon),
                     np.sin(colat)*np.sin(lon),
                     np.cos(colat)], axis=0)


def get_field():
    freq, bw = (skutil
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0
    dev_xyz = get_xyz()
    sh_order = 10
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))
    return R[:, R_mask]


def steering_operator(XYZ, R, freq):
    wl = constants.speed_of_sound / freq
    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * XYZ.T) @ R)


def mel_frequencies(n_mels, fmin, fmax):
    mels = np.linspace(0, 2595*np.log10(1 + fmax/700), n_mels+2)
    mels = mels[1:-1]
    return 700*(10**(mels/2595) - 1)


# ---------------------------------------------------------------------------
# Solvers (Python implementations matching Julia)
# ---------------------------------------------------------------------------

def solve_chambolle_pock(A, C, max_iters=2000, λ_1=0.0, λ_tv=0.01):
    N = A.shape[1]
    H = np.abs(A.conj().T @ A)**2
    c_vec = np.array([np.real(A[:, i].conj().T @ C @ A[:, i]) for i in range(N)])
    
    L = np.linalg.norm(H, 2)
    τ = 1.0 / L
    σ = 0.9 / (4.0 * τ)
    
    x = np.zeros(N)
    x_bar = np.zeros(N)
    
    # 1D difference operator
    D = np.zeros((N-1, N))
    for i in range(N-1):
        D[i, i] = -1
        D[i, i+1] = 1
    
    for k in range(max_iters):
        y_new = σ * (D @ x_bar)
        y = np.clip(y_new, -λ_tv, λ_tv)
        
        x_old = x.copy()
        g_smooth = H @ x - c_vec
        v = x - τ * (g_smooth + D.T @ y)
        x = np.maximum(0.0, v - τ * λ_1)
        x_bar = 2.0 * x - x_old
    
    return x


def solve_subgradient(A, C, max_iters=5000, λ_1=0.0, λ_tv=0.01, α=None):
    if α is None:
        α = lambda k: 10.0 * 0.95**k
    
    N = A.shape[1]
    H = np.abs(A.conj().T @ A)**2
    c_vec = np.array([np.real(A[:, i].conj().T @ C @ A[:, i]) for i in range(N)])
    
    x = np.zeros(N)
    
    for k in range(1, max_iters + 1):
        g_smooth = H @ x - c_vec
        
        g_tv = np.zeros(N)
        for i in range(N-1):
            s = np.sign(x[i] - x[i+1])
            g_tv[i] += s
            g_tv[i+1] -= s
        
        g_l1 = np.ones(N)
        d = g_smooth + λ_tv * g_tv + λ_1 * g_l1
        
        x = np.maximum(0.0, x - α(k) * d / max(np.linalg.norm(d), 1e-10))
    
    return x


def solve_admm(A, C, max_iters=500, λ_1=0.0, λ_tv=0.01, ρ=1.0, outer_tol=1e-4, cg_tol_factor=0.5):
    N = A.shape[1]
    H = np.abs(A.conj().T @ A)**2
    c_vec = np.array([np.real(A[:, i].conj().T @ C @ A[:, i]) for i in range(N)])
    
    # 1D difference operator
    D = np.zeros((N-1, N))
    for i in range(N-1):
        D[i, i] = -1
        D[i, i+1] = 1
    DtD = D.T @ D
    
    # Strongly convex ADMM Hessian
    M = H + ρ * (np.eye(N) + DtD)
    
    t = time.time()

    # State initialization
    x = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N-1)
    u1 = np.zeros(N)
    u2 = np.zeros(N-1)
    
    primal_res_old = 1.0
    
    for k in range(1, max_iters + 1):
        if k == 1:
            tol_k = 1.0
        else:
            tol_k = cg_tol_factor**k
        
        # X-update (inner CG)
        b_k = c_vec + ρ * (z1 - u1) + ρ * (D.T @ (z2 - u2))
        
        r = b_k - M @ x
        p = r.copy()
        rs_old = r @ r
        for _ in range(200):
            Mp = M @ p
            alpha = rs_old / (p @ Mp)
            x += alpha * p
            r -= alpha * Mp
            rs_new = r @ r
            if np.sqrt(rs_new) < tol_k:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        # Z-update (proximal operators)
        v1 = x + u1
        z1_new = np.maximum(0.0, np.sign(v1) * np.maximum(np.abs(v1) - λ_1 / ρ, 0.0))
        
        v2 = D @ x + u2
        z2_new = np.sign(v2) * np.maximum(np.abs(v2) - λ_tv / ρ, 0.0)
        
        # Dual update
        u1 += x - z1_new
        u2 += D @ x - z2_new
        
        primal_res = max(np.linalg.norm(x - z1_new), np.linalg.norm(D @ x - z2_new))
        primal_res_old = primal_res
        
        if primal_res < outer_tol:
            z1 = z1_new
            z2 = z2_new
            break
        
        z1 = z1_new
        z2 = z2_new
    
    print(f"TIME: {time.time() - t}")

    return z1


# ---------------------------------------------------------------------------
# Visualization using project's draw_map style
# ---------------------------------------------------------------------------

def draw_map_single(I, R, ax=None, show_labels=False, title=""):
    if ax is None:
        fig, ax = plt.subplots()
    
    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])
    
    bm = basemap.Basemap(projection='mill',
                         llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                         llcrnrlon=R_az_min, urcrnrlon=R_az_max,
                         resolution='c', ax=ax)
    
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                     color='w', dashes=[1, 0], labels=[1, 0, 0, 1],
                     labelstyle='+/-', textcolor='#565656', zorder=0, linewidth=2)
    bm.drawmeridians(np.linspace(R_az_min, R_az_max, 7),
                     color='w', dashes=[1, 0], labels=[1, 0, 0, 1],
                     labelstyle='+/-', textcolor='#565656', zorder=0, linewidth=2)
    
    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)
    
    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)
    
    I_norm = I / (I.max() + 1e-10)
    N_px = len(I)
    colors = plt.cm.viridis(I_norm)
    mycmap = cmap_from_list('mycmap', colors, N=N_px)
    
    ax.tripcolor(triangulation, np.arange(N_px), cmap=mycmap,
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.1)
    
    ax.set_title(title)
    return ax


def plot_comparison(x_true, x_cp, x_sg, x_admm, R, sample_idx, band_idx, freq, dur, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    draw_map_single(x_true, R, ax=axes[0, 0], show_labels=True,
                    title=f"Ground Truth\nSample {sample_idx}, Band {band_idx} ({freq:.0f}Hz)")
    draw_map_single(x_cp, R, ax=axes[0, 1], show_labels=True,
                    title=f"Chambolle-Pock\nrel_err={np.linalg.norm(x_cp-x_true)/np.linalg.norm(x_true):.4f}")
    draw_map_single(x_sg, R, ax=axes[1, 0], show_labels=True,
                    title=f"Subgradient\nrel_err={np.linalg.norm(x_sg-x_true)/np.linalg.norm(x_true):.4f}")
    draw_map_single(x_admm, R, ax=axes[1, 1], show_labels=True,
                    title=f"ADMM\nrel_err={np.linalg.norm(x_admm-x_true)/np.linalg.norm(x_true):.4f}")
    
    plt.tight_layout()
    path = os.path.join(outdir, f"viz_sample{sample_idx:02d}_band{band_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hdf_path = "../dataset/synthetic_data/synthetic.hdf"
    outdir = "./benchmark_viz"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Loading: {hdf_path}")
    with h5py.File(hdf_path, "r") as f:
        em32_all = f["em32"][:]
        apgd_all = f["apgd"][:]
        dur_all = f["dur"][:]
    
    # Python writes: (n_samples, n_bands, 32, 32) - same order when read in Python
    n_samples = em32_all.shape[0]
    n_bands = em32_all.shape[1]
    
    R = get_field()
    xyz = get_xyz()
    
    # Use band 10 for better conditioning (~1751 Hz)
    band_target = 10
    freq_bands = mel_frequencies(n_bands, 50, 4500)
    freq = freq_bands[band_target]
    
    A = steering_operator(xyz, R, freq)
    print(f"Band {band_target}: {freq:.0f} Hz, A shape: {A.shape}")
    
    n_test = 4
    for i in range(n_test):
        print(f"\n--- Sample {i+1} ---")
        C = em32_all[i, band_target, :, :]
        x_true = apgd_all[i, band_target, :]
        dur = dur_all[i]
        
        print("  Running Chambolle-Pock (2000 iters)...")
        x_cp = solve_chambolle_pock(A, C, max_iters=100, λ_1=0.5, λ_tv=0.0)
        
        print("  Running Subgradient (5000 iters)...")
        x_sg = solve_subgradient(A, C, max_iters=100, λ_1=0.5, λ_tv=0.0, α=lambda k: 10.0 * 0.95**k)
        
        print("  Running ADMM (500 iters)...")
        x_admm = solve_admm(A, C, max_iters=1000, λ_1=0.0, λ_tv=0.0, ρ=75.0)
        
        err_cp = np.linalg.norm(x_cp - x_true) / np.linalg.norm(x_true)
        err_sg = np.linalg.norm(x_sg - x_true) / np.linalg.norm(x_true)
        err_admm = np.linalg.norm(x_admm - x_true) / np.linalg.norm(x_true)
        print(f"  CP error: {err_cp:.4f}, SG error: {err_sg:.4f}, ADMM error: {err_admm:.4f}")
        
        plot_comparison(x_true, x_cp, x_sg, x_admm, R, i+1, band_target, freq, dur, outdir)

        t = time.time()
        for _ in range(10):
            x_admm = solve_admm(A, C, max_iters=10, λ_1=0.5, λ_tv=10.0, ρ=20)
        print(f"Time: {time.time() - t}")

        
    print(f"\nDone. Visualizations in {outdir}/")


if __name__ == "__main__":
    main()
