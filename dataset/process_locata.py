"""process_locata.py - Batch ADMM processing on LOCATA (with & without warm-start)."""
import os, sys, numpy as np, h5py, time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scipy.constants as constants

_EIGENMIKE_ = {"1":[69,0,0.042],"2":[90,32,0.042],"3":[111,0,0.042],"4":[90,328,0.042],"5":[32,0,0.042],"6":[55,45,0.042],"7":[90,69,0.042],"8":[125,45,0.042],"9":[148,0,0.042],"10":[125,315,0.042],"11":[90,291,0.042],"12":[55,315,0.042],"13":[21,91,0.042],"14":[58,90,0.042],"15":[121,90,0.042],"16":[159,89,0.042],"17":[69,180,0.042],"18":[90,212,0.042],"19":[111,180,0.042],"20":[90,148,0.042],"21":[32,180,0.042],"22":[55,225,0.042],"23":[90,249,0.042],"24":[125,225,0.042],"25":[148,180,0.042],"26":[125,135,0.042],"27":[90,111,0.042],"28":[55,135,0.042],"29":[21,269,0.042],"30":[58,270,0.042],"31":[122,270,0.042],"32":[159,271,0.042]}
def _deg2rad(d): return {m: [np.radians(c[0]), np.radians(c[1]), c[2]] for m, c in d.items()}
def _polar2cart(d): return {m: [c[2]*np.sin(c[0])*np.cos(c[1]), c[2]*np.sin(c[0])*np.sin(c[1]), c[2]*np.cos(c[0])] for m, c in d.items()}
def get_xyz(): return np.array([[c for c in _polar2cart(_deg2rad(_EIGENMIKE_))[ch]] for ch in _EIGENMIKE_]).T
def get_field():
    N=10; n=np.arange(4*(N+1)**2)
    colat=np.arccos(1-(2*n+1)/(4*(N+1)**2)); lon=(4*np.pi*n)/(1+np.sqrt(5))
    R=np.stack([np.sin(colat)*np.cos(lon), np.sin(colat)*np.sin(lon), np.cos(colat)], axis=0)
    return R[:, np.abs(R[2,:])<np.sin(np.deg2rad(90))]
def steering_operator(XYZ, R, freq):
    wl=constants.speed_of_sound/freq; return np.exp((-1j*2*np.pi/wl*XYZ.T)@R)

class ADMMSolver:
    def __init__(self, A, rho=100):
        self.N = A.shape[1]; self.A = A; self.rho = rho
        self.H = np.abs(A.conj().T @ A)**2
        self.D = np.zeros((self.N-1, self.N))
        self.D[np.arange(self.N-1), np.arange(self.N-1)] = -1
        self.D[np.arange(self.N-1), np.arange(self.N-1)+1] = 1
        self.M = self.H + rho*(np.eye(self.N) + self.D.T @ self.D)

    def solve(self, C, it=10, tv=1, cg_in=10, x0=None):
        cv = np.array([np.real(self.A[:,i].conj().T@C@self.A[:,i]) for i in range(self.N)])
        x = np.zeros(self.N) if x0 is None else x0.copy()
        z1=np.zeros(self.N); z2=np.zeros(self.N-1); u1=np.zeros(self.N); u2=np.zeros(self.N-1)
        for k in range(1, it+1):
            tk = 0.5**k if k>1 else 1.0
            bk = cv + self.rho*(z1-u1) + self.rho*self.D.T@(z2-u2)
            r=bk-self.M@x; p=r.copy(); rs=r@r
            for _ in range(cg_in):
                Mp=self.M@p; a=rs/(p@Mp); x+=a*p; r-=a*Mp; rn=r@r
                if np.sqrt(rn)<tk: break
                p=r+(rn/rs)*p; rs=rn
            z1=np.maximum(0, x+u1)
            v2=self.D@x+u2; z2=np.sign(v2)*np.maximum(np.abs(v2)-tv/self.rho, 0)
            u1+=x-z1; u2+=self.D@x-z2
        return z1


def process(src, dst, warm_start=False):
    print(f"\n{'='*60}")
    print(f"Processing: {'warm-start' if warm_start else 'cold-start'}")
    print(f"{'='*60}")

    with h5py.File(src, "r") as f:
        em32 = f["em32"][:]; mic = f["mic"][:]; dur = f["dur"][:]; sr = f.attrs["sr"]

    n_frames, n_bands = em32.shape[0], em32.shape[1]
    apgd_new = np.zeros((n_frames, n_bands, 484), dtype=np.float64)

    R = get_field(); xyz = get_xyz(); A = steering_operator(xyz, R, 5000.0)
    solver = ADMMSolver(A, rho=100)

    t0 = time.time()
    x_prev = None
    prev_dur = None

    for i in tqdm(range(n_frames), desc="Frames"):
        cur_dur = dur[i, 0]
        new_clip = (prev_dur is not None and abs(cur_dur - prev_dur) > 1e-6)

        for b in range(n_bands):
            C = em32[i, b]
            if warm_start and not new_clip and x_prev is not None:
                x_prev = solver.solve(C, it=10, tv=1, cg_in=10, x0=x_prev)
            else:
                x_prev = solver.solve(C, it=10, tv=1, cg_in=10, x0=None)
            apgd_new[i, b] = x_prev

        prev_dur = cur_dur

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
    return apgd_new


def main():
    src = "data_hdf/locata.hdf"

    # Cold start
    apgd_cold = process(src, "data_hdf/locata_cold.hdf", warm_start=False)

    # Warm start
    apgd_warm = process(src, "data_hdf/locata_warm.hdf", warm_start=True)

    # Compare
    diff = np.abs(apgd_cold - apgd_warm)
    print(f"\nCold vs Warm difference: mean={diff.mean():.6f} max={diff.max():.6f}")
    print(f"Identical frames: {(diff.max(axis=(1,2))<1e-10).sum()}/{apgd_cold.shape[0]}")

    # Compare both to APGD reference
    with h5py.File(src, "r") as f:
        apgd_ref = f["apgd"][:]
    nrm = np.linalg.norm(apgd_ref.reshape(-1, 484), axis=1) + 1e-10
    err_cold = np.linalg.norm(apgd_cold.reshape(-1, 484) - apgd_ref.reshape(-1, 484), axis=1) / nrm
    err_warm = np.linalg.norm(apgd_warm.reshape(-1, 484) - apgd_ref.reshape(-1, 484), axis=1) / nrm
    print(f"\nRelative error vs APGD reference:")
    print(f"  Cold: mean={err_cold.mean():.4f} median={np.median(err_cold):.4f}")
    print(f"  Warm: mean={err_warm.mean():.4f} median={np.median(err_warm):.4f}")


if __name__ == "__main__":
    main()
