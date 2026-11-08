"""
generate_assets.py - Generate solver assets for Julia LAM solver.

Produces the exact same steering matrix A, spherical grid R, neighbor indices,
and precomputed Hessian H that the Python pipeline uses. Saves as .npy files
that Julia can load via NPZ.jl.

Output: superlam/assets/
"""

import os
import json
import math
import numpy as np
import scipy.constants as constants
import skimage.util as skutil

# ---------------------------------------------------------------------------
# Grid and array geometry - EXACT copy from util/utils.py
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
    return {m: [math.radians(c[0]), math.radians(c[1]), c[2]] for m, c in d.items()}


def _polar2cart(d):
    return {m: [c[2]*math.sin(c[0])*math.cos(c[1]),
                c[2]*math.sin(c[0])*math.sin(c[1]),
                c[2]*math.cos(c[0])] for m, c in d.items()}


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
    """Exact copy from util/utils.py:get_field()"""
    freq, bw = (skutil
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0
    dev_xyz = get_xyz()
    wl_min = constants.speed_of_sound / (freq.max() + 500)
    sh_order = 10
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))
    return R[:, R_mask]


def steering_operator(XYZ, R):
    """Exact copy from util/utils.py:steering_operator()
    
    Uses freq.max() + 500 for wavelength, matching the dataset pipeline.
    """
    freq, bw = (skutil
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0
    wl = constants.speed_of_sound / (freq.max() + 500)
    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * XYZ.T) @ R)


def compute_neighbors(R, k=6):
    """k-nearest neighbors on the sphere via angular distance."""
    dots = R.T @ R
    np.fill_diagonal(dots, -2.0)
    N_px = R.shape[1]
    neighbors = np.zeros((N_px, k), dtype=np.int64)
    for i in range(N_px):
        neighbors[i] = np.argsort(dots[i])[::-1][:k]
    return neighbors


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(outdir, exist_ok=True)
    
    # Spherical grid
    R = get_field()
    N_px = R.shape[1]
    print(f"Grid: {N_px} visible points")
    
    # Array geometries
    xyz_em32 = get_xyz()
    xyz_mic = xyz_em32[:, [5, 9, 25, 21]]
    
    # Steering matrices (single A, same for all bands - matches dataset pipeline)
    print("Computing steering matrices...")
    A_em32 = steering_operator(xyz_em32, R)
    A_mic = steering_operator(xyz_mic, R)
    
    # Precompute Hessian: H_ij = |a_i^H a_j|^2
    print("Computing Hessians...")
    H_em32 = np.abs(A_em32.conj().T @ A_em32) ** 2
    H_mic = np.abs(A_mic.conj().T @ A_mic) ** 2
    
    # Neighbor indices for graph TV
    print("Computing neighbors (k=6)...")
    neighbors = compute_neighbors(R, k=6)
    
    # Save
    print(f"Saving to {outdir}/...")
    np.save(os.path.join(outdir, "R.npy"), R)
    np.save(os.path.join(outdir, "neighbors.npy"), neighbors)
    np.save(os.path.join(outdir, "A_em32.npy"), A_em32)
    np.save(os.path.join(outdir, "A_mic.npy"), A_mic)
    np.save(os.path.join(outdir, "H_em32.npy"), H_em32)
    np.save(os.path.join(outdir, "H_mic.npy"), H_mic)
    
    # Wavelength info for Julia
    freq_bands_linear = skutil.view_as_windows(np.linspace(1500, 4500, 10), (2,), 1).mean(axis=-1)
    wl = constants.speed_of_sound / (freq_bands_linear.max() + 500)
    
    meta = {
        "N_px": N_px,
        "N_mics_em32": 32,
        "N_mics_mic": 4,
        "wavelength_m": wl,
        "steering_freq_hz": float(freq_bands_linear.max() + 500),
        "neighbor_k": 6,
        "grid": "Fibonacci N=10, upper hemisphere mask",
        "mic_subsample_indices": [5, 9, 25, 21],
        "files": {
            "R.npy": f"(3, {N_px}) float64 - spherical grid points",
            "neighbors.npy": f"({N_px}, 6) int64 - k-NN indices for graph TV",
            "A_em32.npy": f"(32, {N_px}) complex128 - steering matrix",
            "A_mic.npy": f"(4, {N_px}) complex128 - 4-ch steering matrix",
            "H_em32.npy": f"({N_px}, {N_px}) float64 - |A'A|^2 Hessian",
            "H_mic.npy": f"({N_px}, {N_px}) float64 - |A'A|^2 Hessian (4-ch)",
        }
    }
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    for name, shape, dtype in [
        ("R", R.shape, R.dtype),
        ("neighbors", neighbors.shape, neighbors.dtype),
        ("A_em32", A_em32.shape, A_em32.dtype),
        ("A_mic", A_mic.shape, A_mic.dtype),
        ("H_em32", H_em32.shape, H_em32.dtype),
        ("H_mic", H_mic.shape, H_mic.dtype),
    ]:
        print(f"  {name}.npy: {shape} {dtype}")
    
    print(f"\nDone. Assets in {outdir}/")


if __name__ == "__main__":
    main()
