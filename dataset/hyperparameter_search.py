"""
hyperparameter_search.py - Grid search CP, SG, ADMM on real LOCATA HDF5.

Tests multiple hyperparameter combinations at low iteration counts (~100-200).
Visualizes all results using the project's draw_map() style.

Output: benchmark_viz/hp_search_*.png
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from trainer.utils import cart2eq, wrapped_rad2deg, cmap_from_list

# ---------------------------------------------------------------------------
# Grid / geometry (copied from project)
# ---------------------------------------------------------------------------
_EIGENMIKE_ = {
    "1":[69,0,0.042],"2":[90,32,0.042],"3":[111,0,0.042],"4":[90,328,0.042],"5":[32,0,0.042],
    "6":[55,45,0.042],"7":[90,69,0.042],"8":[125,45,0.042],"9":[148,0,0.042],"10":[125,315,0.042],
    "11":[90,291,0.042],"12":[55,315,0.042],"13":[21,91,0.042],"14":[58,90,0.042],"15":[121,90,0.042],
    "16":[159,89,0.042],"17":[69,180,0.042],"18":[90,212,0.042],"19":[111,180,0.042],"20":[90,148,0.042],
    "21":[32,180,0.042],"22":[55,225,0.042],"23":[90,249,0.042],"24":[125,225,0.042],"25":[148,180,0.042],
    "26":[125,135,0.042],"27":[90,111,0.042],"28":[55,135,0.042],"29":[21,269,0.042],"30":[58,270,0.042],
    "31":[122,270,0.042],"32":[159,271,0.042]}

def _deg2rad(d): return {m: [np.radians(c[0]), np.radians(c[1]), c[2]] for m, c in d.items()}
def _polar2cart(d): return {m: [c[2]*np.sin(c[0])*np.cos(c[1]), c[2]*np.sin(c[0])*np.sin(c[1]), c[2]*np.cos(c[0])] for m, c in d.items()}
def get_xyz(): return np.array([[c for c in _polar2cart(_deg2rad(_EIGENMIKE_))[ch]] for ch in _EIGENMIKE_]).T

def fibonacci(N):
    N_px = 4*(N+1)**2; n = np.arange(N_px)
    colat = np.arccos(1-(2*n+1)/N_px); lon = (4*np.pi*n)/(1+np.sqrt(5))
    return np.stack([np.sin(colat)*np.cos(lon), np.sin(colat)*np.sin(lon), np.cos(colat)], axis=0)

def get_field():
    freq, _ = (skutil.view_as_windows(np.linspace(1500,4500,10),(2,),1).mean(axis=-1)), 50.0
    R = fibonacci(10)
    return R[:, np.abs(R[2,:]) < np.sin(np.deg2rad(90))]

def steering_operator(XYZ, R, freq):
    wl = constants.speed_of_sound / freq; scale = 2*np.pi/wl
    return np.exp((-1j*scale*XYZ.T) @ R)

def mel_frequencies(n, fmin, fmax):
    m = np.linspace(0, 2595*np.log10(1+fmax/700), n+2)[1:-1]
    return 700*(10**(m/2595)-1)

# ---------------------------------------------------------------------------
# Solvers (fast, low-iter versions)
# ---------------------------------------------------------------------------
def solve_chambolle_pock(A, C, max_iters=100, λ_tv=0.1):
    N = A.shape[1]; H = np.abs(A.conj().T@A)**2
    c_vec = np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    L = np.linalg.norm(H,2); τ = 1.0/L; σ = 0.9/(4.0*τ)
    x = np.zeros(N); x_bar = np.zeros(N)
    D = np.zeros((N-1,N)); D[np.arange(N-1), np.arange(N-1)] = -1; D[np.arange(N-1), np.arange(N-1)+1] = 1
    for k in range(max_iters):
        y = np.clip(σ*(D@x_bar), -λ_tv, λ_tv)
        x_old = x.copy(); v = x - τ*(H@x - c_vec + D.T@y)
        x = np.maximum(0.0, v); x_bar = 2.0*x - x_old
    return x

def solve_subgradient(A, C, max_iters=200, λ_tv=0.1, α_fn=None):
    if α_fn is None: α_fn = lambda k: 10.0*0.95**k
    N = A.shape[1]; H = np.abs(A.conj().T@A)**2
    c_vec = np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    x = np.zeros(N)
    for k in range(1, max_iters+1):
        g_smooth = H@x - c_vec
        g_tv = np.zeros(N)
        for i in range(N-1):
            s = np.sign(x[i]-x[i+1]); g_tv[i] += s; g_tv[i+1] -= s
        d = g_smooth + λ_tv*g_tv
        x = np.maximum(0.0, x - α_fn(k)*d/max(np.linalg.norm(d), 1e-10))
    return x

def solve_admm(A, C, max_iters=100, λ_tv=0.1, ρ=1.0, cg_iters=5):
    N = A.shape[1]; H = np.abs(A.conj().T@A)**2
    c_vec = np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    D = np.zeros((N-1,N)); D[np.arange(N-1), np.arange(N-1)] = -1; D[np.arange(N-1), np.arange(N-1)+1] = 1
    M = H + ρ*(np.eye(N) + D.T@D)
    x = np.zeros(N); z1 = np.zeros(N); z2 = np.zeros(N-1); u1 = np.zeros(N); u2 = np.zeros(N-1)
    for k in range(1, max_iters+1):
        tol_k = 0.5**k if k > 1 else 1.0
        b_k = c_vec + ρ*(z1-u1) + ρ*D.T@(z2-u2)
        r = b_k - M@x; p = r.copy(); rs_old = r@r
        for _ in range(cg_iters):
            Mp = M@p; a = rs_old/(p@Mp); x += a*p; r -= a*Mp
            rs_new = r@r
            if np.sqrt(rs_new) < tol_k: break
            p = r + (rs_new/rs_old)*p; rs_old = rs_new
        v1 = x+u1; z1_new = np.maximum(0.0, np.sign(v1)*np.maximum(np.abs(v1),0.0))
        v2 = D@x+u2; z2_new = np.sign(v2)*np.maximum(np.abs(v2)-λ_tv/ρ, 0.0)
        u1 += x-z1_new; u2 += D@x-z2_new; z1 = z1_new; z2 = z2_new
    return z1

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_map_single(I, R, ax, title=""):
    _, R_el, R_az = cart2eq(*R); R_el, R_az = wrapped_rad2deg(R_el, R_az)
    bm = basemap.Basemap(projection='mill', llcrnrlat=np.min(R_el), urcrnrlat=np.max(R_el),
                         llcrnrlon=np.min(R_az), urcrnrlon=np.max(R_az), resolution='c', ax=ax)
    bm.drawparallels(np.linspace(np.min(R_el), np.max(R_el), 3), color='w', dashes=[1,0], labels=[1,0,0,1], linewidth=1)
    bm.drawmeridians(np.linspace(np.min(R_az), np.max(R_az), 5), color='w', dashes=[1,0], labels=[1,0,0,1], linewidth=1)
    R_x, R_y = bm(R_az, R_el); triang = tri.Triangulation(R_x, R_y)
    I_n = I/(I.max()+1e-10); colors = plt.cm.viridis(I_n)
    ax.tripcolor(triang, np.arange(len(I)), cmap=cmap_from_list('m', colors, N=len(I)),
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.05)
    ax.set_title(title, fontsize=8)

# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------
def main():
    hdf_path = "../dataset/data_hdf/locata.hdf"
    outdir = "./benchmark_viz/hp_search"
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading: {hdf_path}")
    with h5py.File(hdf_path, "r") as f:
        em32_all = f["em32"][:]
        apgd_all = f["apgd"][:]
        dur_all = f["dur"][:]

    print(f"em32: {em32_all.shape}, apgd: {apgd_all.shape}")

    R = get_field(); xyz = get_xyz()
    # Use same frequency as APGD: max(bands) + 500 ≈ 5000 Hz
    freq = 5000.0
    A = steering_operator(xyz, R, freq)
    print(f"Steering at {freq:.0f} Hz (matching APGD), A: {A.shape}")

    band = 10
    sample_idx = 4
    C = em32_all[sample_idx, band, :, :]
    x_true = apgd_all[sample_idx, band, :]
    print(f"Sample {sample_idx}: dur={dur_all[sample_idx, 0]:.3f}s, x_true max={x_true.max():.3f}")

    # ---- Hyperparameter grids ----
    cp_params = [(100, tv) for tv in [0.01, 0.1, 1, 10, 100]]
    sg_params = [(200, tv, "10.0*0.95**k") for tv in [0.01, 0.1, 1, 10, 100]]
    admm_params = [(100, tv, ρ) for tv in [0.1, 1, 10] for ρ in [0.1, 1, 10, 100]]

    # ---- Chambolle-Pock grid ----
    print("\n--- Chambolle-Pock ---")
    n_cp = len(cp_params); fig_cp, axes_cp = plt.subplots(1, n_cp+1, figsize=(4*(n_cp+1), 4))
    draw_map_single(x_true, R, axes_cp[0], f"Ground Truth\nmax={x_true.max():.2f}")
    for j, (iters, tv) in enumerate(cp_params):
        x_cp = solve_chambolle_pock(A, C, max_iters=iters, λ_tv=tv)
        err = np.linalg.norm(x_cp-x_true)/np.linalg.norm(x_true)
        draw_map_single(x_cp, R, axes_cp[j+1], f"CP it={iters} tv={tv}\nerr={err:.3f}")
        print(f"  CP it={iters} tv={tv} -> err={err:.4f}")
    plt.tight_layout(); fig_cp.savefig(os.path.join(outdir, "hp_chambolle_pock.png"), dpi=150); plt.close()

    # ---- Subgradient grid ----
    print("\n--- Subgradient ---")
    n_sg = len(sg_params); fig_sg, axes_sg = plt.subplots(1, n_sg+1, figsize=(4*(n_sg+1), 4))
    draw_map_single(x_true, R, axes_sg[0], f"Ground Truth\nmax={x_true.max():.2f}")
    for j, (iters, tv, sched) in enumerate(sg_params):
        α_fn = eval(f"lambda k: {sched}")
        x_sg = solve_subgradient(A, C, max_iters=iters, λ_tv=tv, α_fn=α_fn)
        err = np.linalg.norm(x_sg-x_true)/np.linalg.norm(x_true)
        draw_map_single(x_sg, R, axes_sg[j+1], f"SG it={iters} tv={tv}\nerr={err:.3f}")
        print(f"  SG it={iters} tv={tv} -> err={err:.4f}")
    plt.tight_layout(); fig_sg.savefig(os.path.join(outdir, "hp_subgradient.png"), dpi=150); plt.close()

    # ---- ADMM grid ----
    print("\n--- ADMM ---")
    n_admm = len(admm_params); cols = 5
    rows = (n_admm + 1 + cols - 1) // cols
    fig_admm, axes_admm = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes_admm = axes_admm.flatten()
    draw_map_single(x_true, R, axes_admm[0], f"Ground Truth\nmax={x_true.max():.2f}")
    for j, (iters, tv, ρ) in enumerate(admm_params):
        x_admm = solve_admm(A, C, max_iters=iters, λ_tv=tv, ρ=ρ)
        err = np.linalg.norm(x_admm-x_true)/np.linalg.norm(x_true)
        draw_map_single(x_admm, R, axes_admm[j+1], f"ADMM it={iters} tv={tv} ρ={ρ}\nerr={err:.3f}")
        print(f"  ADMM it={iters} tv={tv} ρ={ρ} -> err={err:.4f}")
    for j in range(n_admm+1, len(axes_admm)):
        axes_admm[j].set_visible(False)
    plt.tight_layout(); fig_admm.savefig(os.path.join(outdir, "hp_admm.png"), dpi=150); plt.close()

    # ---- Best-of-each comparison ----
    print("\n--- Best comparison ---")
    fig_best, axes_best = plt.subplots(1, 4, figsize=(16, 4))
    draw_map_single(x_true, R, axes_best[0], "Ground Truth")

    x_cp = solve_chambolle_pock(A, C, max_iters=200, λ_tv=0.1)
    x_sg = solve_subgradient(A, C, max_iters=500, λ_tv=0.1, α_fn=lambda k: 10.0*0.95**k)
    x_admm = solve_admm(A, C, max_iters=200, λ_tv=0.1, ρ=10.0, cg_iters=10)

    for ax, x_rec, name in [(axes_best[1], x_cp, "Chambolle-Pock"), (axes_best[2], x_sg, "Subgradient"), (axes_best[3], x_admm, "ADMM")]:
        err = np.linalg.norm(x_rec-x_true)/np.linalg.norm(x_true)
        draw_map_single(x_rec, R, ax, f"{name}\nerr={err:.4f}")
        print(f"  {name}: err={err:.4f}")
    plt.tight_layout(); fig_best.savefig(os.path.join(outdir, "hp_best_comparison.png"), dpi=150); plt.close()

    print(f"\nDone. Results in {outdir}/")


if __name__ == "__main__":
    main()
