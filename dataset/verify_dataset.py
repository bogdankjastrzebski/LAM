"""
verify_dataset.py - CLI visualization tool for LAM HDF5 datasets.

Self-contained: copies only the needed math from trainer/utils.py
(cartesian conversion, Fibonacci grid, custom colormap).
No neural network dependencies.

All plots are saved to disk as PNGs (no GUI pop-ups).

Usage:
    python verify_dataset.py --hdf path/to/dataset.hdf --outdir ./viz
    python verify_dataset.py --hdf path/to/dataset.hdf --samples 5 --band 3
"""

import os
import argparse
import math
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors
import mpl_toolkits.basemap as basemap
import scipy.constants as constants
import skimage.util as skutil

# ---------------------------------------------------------------------------
# Extracted from trainer/utils.py - only the math we need
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


def _deg2rad(coords_dict):
    return {m: [math.radians(c[0]), math.radians(c[1]), c[2]] for m, c in coords_dict.items()}


def _polar2cart(coords_dict):
    return {
        m: [c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0])]
        for m, c in coords_dict.items()
    }


def get_xyz():
    mic_coords = _polar2cart(_deg2rad(_EIGENMIKE_))
    return [[coord for coord in mic_coords[ch]] for ch in mic_coords]


def cart2eq(x, y, z):
    """Cartesian to equatorial (r, lat, lon) in radians."""
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / np.clip(r, 1e-10, None))
    lon = np.arctan2(y, x)
    return r, lat, lon


def wrapped_rad2deg(lat_r, lon_r):
    """Equatorial [rad] -> [deg], longitude wrapped to [-180, 180)."""
    lat_d = np.rad2deg(lat_r)
    lon_d = np.rad2deg(lon_r)
    lon_d = ((lon_d + 180) % 360) - 180
    return lat_d, lon_d


def fibonacci(N, direction=None, FoV=None):
    """Fibonacci-sampled spherical grid."""
    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    
    x = np.sin(colat) * np.cos(lon)
    y = np.sin(colat) * np.sin(lon)
    z = np.cos(colat)
    XYZ = np.stack([x, y, z], axis=0)
    
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]
    
    return XYZ


def get_field(min_freq=1500, max_freq=4500, nbands=10):
    """Generate the Fibonacci spherical grid used for acoustic imaging."""
    freq, bw = (skutil
            .view_as_windows(np.linspace(min_freq, max_freq, nbands), (2,), 1)
            .mean(axis=-1)), 50.0
    
    xyz = get_xyz()
    dev_xyz = np.array(xyz).T
    wl_min = constants.speed_of_sound / (freq.max() + 500)
    sh_order = 10
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))
    R = R[:, R_mask]
    return R


def cmap_from_list(name, colors, N=256, gamma=1.0):
    """Create custom colormap from a list of colors."""
    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(np.linspace(0, 1, len(colors)), colors):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))
    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def draw_map_single(I, R, ax=None, show_labels=False, title=""):
    """Render acoustic intensity map using Basemap Miller projection + triangular mesh.
    
    Matches the style of draw_map() from trainer/utils.py.
    """
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


def plot_ground_truth_map(apgd_sample, R, sample_idx, band_idx, outdir):
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_map_single(apgd_sample, R, ax=ax, show_labels=True,
                    title=f"Ground Truth - Sample {sample_idx}, Band {band_idx}")
    plt.tight_layout()
    path = os.path.join(outdir, f"gt_map_sample{sample_idx:04d}_band{band_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_covariance_matrix(cov_matrix, sample_idx, band_idx, ch_label, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(np.abs(cov_matrix), cmap="magma", aspect="auto")
    axes[0].set_title(f"|Covariance| - {ch_label}")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Channel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(np.real(cov_matrix), cmap="RdBu_r", aspect="auto")
    axes[1].set_title(f"Re(Covariance) - {ch_label}")
    axes[1].set_xlabel("Channel")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(np.angle(cov_matrix), cmap="twilight", aspect="auto")
    axes[2].set_title(f"Phase(Covariance) - {ch_label}")
    axes[2].set_xlabel("Channel")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    fig.suptitle(f"Covariance Matrix - Sample {sample_idx}, Band {band_idx}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, f"cov_{ch_label}_sample{sample_idx:04d}_band{band_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_eigenvalue_spectrum(cov_matrix, sample_idx, band_idx, ch_label, outdir):
    S_D = np.linalg.eigvalsh(cov_matrix)
    S_D_sorted = np.sort(S_D)[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(S_D_sorted, "o-", markersize=4)
    ax.set_xlabel("Eigenvalue Index (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalue Spectrum - {ch_label}\nSample {sample_idx}, Band {band_idx}")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(outdir, f"eigspec_{ch_label}_sample{sample_idx:04d}_band{band_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_intensity_histogram(apgd_sample, sample_idx, band_idx, outdir):
    intensity = np.abs(apgd_sample)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(intensity.flatten(), bins=100, log=True, color="steelblue", edgecolor="white")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Intensity Distribution\nSample {sample_idx}, Band {band_idx}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(outdir, f"hist_sample{sample_idx:04d}_band{band_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_grid(hdf_path, outdir, n_samples=4, band=0):
    with h5py.File(hdf_path, "r") as f:
        apgd_all = f["apgd"][:]
        em32_all = f["em32"][:]
        dur_all = f["dur"][:]
    
    R = get_field()
    n_samples = min(n_samples, len(apgd_all))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 10))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_samples):
        draw_map_single(apgd_all[i, band], R, ax=axes[0, i], show_labels=False,
                        title=f"Sample {i}\nDur={dur_all[i]:.3f}s")
        
        cov = em32_all[i, band]
        im = axes[1, i].imshow(np.abs(cov), cmap="magma", aspect="auto")
        axes[1, i].set_title(f"EM32 Cov |Band {band}|")
        axes[1, i].set_xlabel("Channel")
        if i == 0:
            axes[1, i].set_ylabel("Channel")
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    fig.suptitle(f"Dataset Summary - {os.path.basename(hdf_path)} (Band {band})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "summary_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify and visualize LAM HDF5 datasets")
    parser.add_argument("--hdf", type=str, required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--outdir", type=str, default="./viz_output", help="Output directory for plots")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument("--band", type=int, default=3, help="Frequency band index (0-14)")
    parser.add_argument("--summary", action="store_true", help="Generate summary grid plot")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Loading: {args.hdf}")
    with h5py.File(args.hdf, "r") as f:
        print(f"  Keys: {list(f.keys())}")
        print(f"  em32 shape:  {f['em32'].shape}  dtype={f['em32'].dtype}")
        print(f"  mic shape:   {f['mic'].shape}  dtype={f['mic'].dtype}")
        print(f"  apgd shape:  {f['apgd'].shape}  dtype={f['apgd'].dtype}")
        print(f"  dur shape:   {f['dur'].shape}  dtype={f['dur'].dtype}")
        print(f"  sr:          {f.attrs.get('sr', 'N/A')}")
        
        n_total = len(f["apgd"])
        n_bands = f["apgd"].shape[1]
        
        R = get_field()
        print(f"  Grid points: {R.shape[1]}")
        
        n_vis = min(args.samples, n_total)
        band = min(args.band, n_bands - 1)
        
        print(f"\nVisualizing {n_vis} samples at band {band}...")
        
        for i in range(n_vis):
            print(f"\n--- Sample {i} ---")
            
            apgd_sample = f["apgd"][i, band]
            plot_ground_truth_map(apgd_sample, R, i, band, args.outdir)
            plot_intensity_histogram(apgd_sample, i, band, args.outdir)
            
            em32_cov = f["em32"][i, band]
            plot_covariance_matrix(em32_cov, i, band, "em32", args.outdir)
            plot_eigenvalue_spectrum(em32_cov, i, band, "em32", args.outdir)
            
            mic_cov = f["mic"][i, band]
            plot_covariance_matrix(mic_cov, i, band, "mic", args.outdir)
            plot_eigenvalue_spectrum(mic_cov, i, band, "mic", args.outdir)
        
        if args.summary:
            print(f"\nGenerating summary grid...")
            plot_summary_grid(args.hdf, args.outdir, n_samples=args.samples, band=band)
        
        print(f"\nDone. All plots saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
