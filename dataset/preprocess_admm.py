"""preprocess_admm.py - Replace APGD labels with ADMM solver output.

Precomputes constant matrices (A, H, M, D) once, only recomputes c_vec per frame.
"""
import os, sys, numpy as np, h5py, time
from tqdm import tqdm
import scipy.constants as constants

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

_EIGENMIKE_ = {"1":[69,0,0.042],"2":[90,32,0.042],"3":[111,0,0.042],"4":[90,328,0.042],"5":[32,0,0.042],"6":[55,45,0.042],"7":[90,69,0.042],"8":[125,45,0.042],"9":[148,0,0.042],"10":[125,315,0.042],"11":[90,291,0.042],"12":[55,315,0.042],"13":[21,91,0.042],"14":[58,90,0.042],"15":[121,90,0.042],"16":[159,89,0.042],"17":[69,180,0.042],"18":[90,212,0.042],"19":[111,180,0.042],"20":[90,148,0.042],"21":[32,180,0.042],"22":[55,225,0.042],"23":[90,249,0.042],"24":[125,225,0.042],"25":[148,180,0.042],"26":[125,135,0.042],"27":[90,111,0.042],"28":[55,135,0.042],"29":[21,269,0.042],"30":[58,270,0.042],"31":[122,270,0.042],"32":[159,271,0.042]}
def _deg2rad(d): return {m: [np.radians(c[0]), np.radians(c[1]), c[2]] for m, c in d.items()}
def _polar2cart(d): return {m: [c[2]*np.sin(c[0])*np.cos(c[1]), c[2]*np.sin(c[0])*np.sin(c[1]), c[2]*np.cos(c[0])] for m, c in d.items()}
def get_xyz(): return np.array([[c for c in _polar2cart(_deg2rad(_EIGENMIKE_))[ch]] for ch in _EIGENMIKE_]).T
def get_field():
    N=10; N_px=4*(N+1)**2; n=np.arange(N_px)
    colat=np.arccos(1-(2*n+1)/N_px); lon=(4*np.pi*n)/(1+np.sqrt(5))
    R=np.stack([np.sin(colat)*np.cos(lon), np.sin(colat)*np.sin(lon), np.cos(colat)], axis=0)
    return R[:, np.abs(R[2,:])<np.sin(np.deg2rad(90))]
def steering_operator(XYZ, R, freq):
    wl=constants.speed_of_sound/freq; return np.exp((-1j*2*np.pi/wl*XYZ.T)@R)

# Precomputed components
class ADMMSolver:
    def __init__(self, A, rho=10):
        self.N = A.shape[1]; self.A = A; self.rho = rho
        self.H = np.abs(A.conj().T @ A) ** 2
        self.D = np.zeros((self.N-1, self.N))
        self.D[np.arange(self.N-1), np.arange(self.N-1)] = -1
        self.D[np.arange(self.N-1), np.arange(self.N-1)+1] = 1
        self.M = self.H + rho * (np.eye(self.N) + self.D.T @ self.D)

    def solve(self, C, it=100, tv=1):
        cv = np.array([np.real(self.A[:, i].conj().T @ C @ self.A[:, i]) for i in range(self.N)])
        x = np.zeros(self.N); z1 = np.zeros(self.N); z2 = np.zeros(self.N-1)
        u1 = np.zeros(self.N); u2 = np.zeros(self.N-1)
        rho = self.rho; D = self.D; M = self.M; N = self.N
        for k in range(1, it+1):
            tk = 0.5**k if k > 1 else 1.0
            bk = cv + rho*(z1 - u1) + rho*D.T @ (z2 - u2)
            r = bk - M @ x; p = r.copy(); rs = r @ r
            for _ in range(5):
                Mp = M @ p; a = rs / (p @ Mp); x += a * p; r -= a * Mp
                rn = r @ r
                if np.sqrt(rn) < tk: break
                p = r + (rn / rs) * p; rs = rn
            z1 = np.maximum(0, x + u1)
            v2 = D @ x + u2; z2 = np.sign(v2) * np.maximum(np.abs(v2) - tv/rho, 0)
            u1 += x - z1; u2 += D @ x - z2
        return z1


def main():
    R = get_field(); xyz = get_xyz(); A_mat = steering_operator(xyz, R, 5000.0)

    src = "data_hdf/locata.hdf"; dst = "data_hdf/locata_admm.hdf"
    print(f"Loading {src}...")
    with h5py.File(src, "r") as f:
        em32 = f["em32"][:]; mic = f["mic"][:]; dur = f["dur"][:]; sr = f.attrs["sr"]

    n_samples, n_bands = em32.shape[0], em32.shape[1]
    apgd_new = np.zeros((n_samples, n_bands, 484), dtype=np.float64)

    # Precompute once
    print("Precomputing ADMM matrices...")
    solver = ADMMSolver(A_mat, rho=10)

    t0 = time.time()
    for i in tqdm(range(n_samples), desc="ADMM"):
        for b in range(n_bands):
            C = em32[i, b]
            apgd_new[i, b] = solver.solve(C, it=100, tv=1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Saving {dst}...")
    with h5py.File(dst, "w") as f:
        f.create_dataset("em32", data=em32)
        f.create_dataset("mic", data=mic)
        f.create_dataset("apgd", data=apgd_new)
        f.create_dataset("dur", data=dur)
        f.attrs["sr"] = sr
    print(f"Saved: {dst}")

if __name__ == "__main__":
    main()
