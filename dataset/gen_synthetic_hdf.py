"""
gen_synthetic_hdf.py - Generate synthetic LAM HDF5 dataset without external RIRs.

Creates mock covariance matrices and ground truth acoustic intensity maps
that match the exact structure of the real pipeline output.

Usage:
    python gen_synthetic_hdf.py --n_samples 360 --n_pixels 440 --save_path ./synthetic_data
    python gen_synthetic_hdf.py --n_samples 100 --seed 42 --save_path ./synthetic_data
"""

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Grid helpers (mirrors util.utils)
# ---------------------------------------------------------------------------

def fibonacci(N=10):
    """Fibonacci-sampled spherical grid."""
    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)
    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    
    x = np.sin(colat) * np.cos(lon)
    y = np.sin(colat) * np.sin(lon)
    z = np.cos(colat)
    
    XYZ = np.stack([x, y, z], axis=0)
    mask = np.abs(XYZ[2, :]) < np.sin(np.deg2rad(90))
    return XYZ[:, mask]


def get_xyz():
    """Eigenmike 32 microphone Cartesian coordinates."""
    _EIGENMIKE_ = {
        "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042], "5": [32, 0, 0.042],
        "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042], "10": [125, 315, 0.042],
        "11": [90, 291, 0.042], "12": [55, 315, 0.042], "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
        "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
        "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],"25": [148, 180, 0.042],
        "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
        "31": [122, 270, 0.042], "32": [159, 271, 0.042],
    }
    
    def pol2cart(colat_deg, lon_deg, r):
        colat = np.deg2rad(colat_deg)
        lon = np.deg2rad(lon_deg)
        x = r * np.sin(colat) * np.cos(lon)
        y = r * np.sin(colat) * np.sin(lon)
        z = r * np.cos(colat)
        return x, y, z
    
    xyz = []
    for ch in _EIGENMIKE_.values():
        xyz.append(pol2cart(ch[0], ch[1], ch[2]))
    return np.array(xyz).T


def steering_operator(XYZ, R, freq=2000):
    """Steering matrix A = exp(-j * 2*pi/wl * P^T * R)."""
    import scipy.constants as constants
    wl = constants.speed_of_sound / freq
    scale = 2 * np.pi / wl
    return np.exp((-1j * scale * XYZ.T) @ R)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_sample(R, A_em32, A_mic, rng, n_sources=2, noise_level=0.05):
    """Generate one synthetic sample: ground truth + covariance matrices.
    
    Places 1-3 acoustic point sources on the sphere, computes the resulting
    covariance matrices for both EM32 and 4-channel mic arrays.
    """
    N_px = R.shape[1]
    N_mics_em32 = A_em32.shape[0]
    N_mics_mic = A_mic.shape[0]
    
    # Ground truth: sparse acoustic blobs on the sphere
    x_true = np.zeros(N_px, dtype=np.float64)
    
    n_sources = rng.integers(1, n_sources + 1)
    for _ in range(n_sources):
        # Pick a random grid point as source center
        center_idx = rng.integers(0, N_px)
        center = R[:, center_idx]
        
        # Compute angular distances to all grid points
        dots = R.T @ center  # cos(theta) for unit vectors
        angles = np.arccos(np.clip(dots, -1, 1))
        
        # Create a Gaussian blob (width ~25 degrees - broad enough for solver convergence)
        sigma = np.deg2rad(25)
        blob = np.exp(-0.5 * (angles / sigma) ** 2)
        amplitude = rng.uniform(1.0, 10.0)
        x_true += amplitude * blob
    
    x_true = np.clip(x_true, 0, None)
    
    # Covariance: C = A * diag(x_true) * A^H + noise
    C_em32 = A_em32 @ np.diag(x_true) @ A_em32.conj().T
    noise_em32 = noise_level * (rng.standard_normal((N_mics_em32, N_mics_em32)) + 1j * rng.standard_normal((N_mics_em32, N_mics_em32)))
    C_em32 = C_em32 + noise_em32
    # Make Hermitian
    C_em32 = 0.5 * (C_em32 + C_em32.conj().T)
    
    C_mic = A_mic @ np.diag(x_true) @ A_mic.conj().T
    noise_mic = noise_level * (rng.standard_normal((N_mics_mic, N_mics_mic)) + 1j * rng.standard_normal((N_mics_mic, N_mics_mic)))
    C_mic = C_mic + noise_mic
    C_mic = 0.5 * (C_mic + C_mic.conj().T)
    
    return x_true, C_em32, C_mic


def generate_synthetic_dataset(n_samples=360, n_bands=15, save_path="./synthetic_data",
                                seed=42, n_frames_per_sample=10, noise_level=0.05):
    """Generate a full synthetic HDF5 dataset matching the real pipeline structure.
    
    HDF5 structure:
        em32: (n_total_frames, n_bands, 32, 32)  complex128
        mic:  (n_total_frames, n_bands, 4, 4)     complex128
        apgd: (n_total_frames, n_bands, N_px)     float64
        dur:  (n_total_frames,)                    float64
        sr:   attribute (sample rate = 24000)
    """
    os.makedirs(save_path, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    
    # Spherical grid (N=10 -> 441 points, masked to ~300 visible)
    R = fibonacci(N=10)
    N_px = R.shape[1]
    print(f"Spherical grid: {N_px} visible points")
    
    # Array geometries
    xyz_em32 = get_xyz()  # (3, 32)
    xyz_mic = xyz_em32[:, [5, 9, 25, 21]]  # 4-channel subsample
    
    # Frequency bands (log-spaced, 50Hz - 4500Hz, matching real pipeline)
    freqs = np.logspace(np.log10(50), np.log10(4500), n_bands)
    print(f"Frequency bands: {freqs.astype(int)} Hz")
    
    # Precompute steering matrices for each band
    A_em32_bands = []
    A_mic_bands = []
    for f in freqs:
        A_em32_bands.append(steering_operator(xyz_em32, R, freq=f))
        A_mic_bands.append(steering_operator(xyz_mic, R, freq=f))
    
    # Log-spaced durations (matching real pipeline: 20ms - 200ms)
    T_logs = np.round(np.logspace(np.log10(20e-3), np.log10(200e-3), 9), 3)
    
    total_frames = n_samples * n_frames_per_sample
    print(f"Generating {n_samples} samples x {n_frames_per_sample} frames = {total_frames} total frames")
    print(f"Per-band shapes: em32=(32,32), mic=(4,4), apgd=({N_px},)")
    
    # Pre-allocate arrays
    em32_all = np.zeros((total_frames, n_bands, 32, 32), dtype=np.complex128)
    mic_all = np.zeros((total_frames, n_bands, 4, 4), dtype=np.complex128)
    apgd_all = np.zeros((total_frames, n_bands, N_px), dtype=np.float64)
    dur_all = np.zeros(total_frames, dtype=np.float64)
    
    frame_idx = 0
    for sample in tqdm(range(n_samples), desc="Samples"):
        T_sti = float(rng.choice(T_logs))
        
        for frame in range(n_frames_per_sample):
            for band in range(n_bands):
                A_em32 = A_em32_bands[band]
                A_mic = A_mic_bands[band]
                
                x_true, C_em32, C_mic = generate_synthetic_sample(
                    R, A_em32, A_mic, rng, n_sources=2, noise_level=noise_level
                )
                
                em32_all[frame_idx, band] = C_em32
                mic_all[frame_idx, band] = C_mic
                apgd_all[frame_idx, band] = x_true
            
            dur_all[frame_idx] = T_sti
            frame_idx += 1
    
    # Save to HDF5
    hdf_path = os.path.join(save_path, "synthetic.hdf")
    print(f"\nSaving to {hdf_path}...")
    with h5py.File(hdf_path, "w") as f:
        f.create_dataset("em32", data=em32_all)
        f.create_dataset("mic", data=mic_all)
        f.create_dataset("apgd", data=apgd_all)
        f.create_dataset("dur", data=dur_all)
        f.attrs["sr"] = 24000
        
        print(f"  em32: {em32_all.shape} {em32_all.dtype}")
        print(f"  mic:  {mic_all.shape} {mic_all.dtype}")
        print(f"  apgd: {apgd_all.shape} {apgd_all.dtype}")
        print(f"  dur:  {dur_all.shape} {dur_all.dtype}")
    
    print(f"\nDone! Dataset saved to {hdf_path}")
    return hdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LAM HDF5 dataset")
    parser.add_argument("--n_samples", type=int, default=360, help="Number of audio samples")
    parser.add_argument("--n_bands", type=int, default=15, help="Number of frequency bands")
    parser.add_argument("--n_frames", type=int, default=10, help="Frames per sample")
    parser.add_argument("--save_path", type=str, default="./synthetic_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        n_samples=args.n_samples,
        n_bands=args.n_bands,
        n_frames_per_sample=args.n_frames,
        save_path=args.save_path,
        seed=args.seed,
        noise_level=args.noise,
    )


if __name__ == "__main__":
    main()
