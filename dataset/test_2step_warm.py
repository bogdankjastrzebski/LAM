"""test_2step_warm.py - ADMM with 2 steps per frame, full-state warm-start."""
import os, sys, numpy as np, h5py, time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_locata import ADMMSolver, get_field, get_xyz, steering_operator

class ADMMState:
    pass  # will hold x, z1, z2, u1, u2 per band

def solve_full_warm(solver, C, state, it=2, tv=1, cg_in=10):
    """ADMM with full state warm-start. Updates state in-place, returns z1."""
    cv = np.array([np.real(solver.A[:,i].conj().T@C@solver.A[:,i]) for i in range(solver.N)])
    x, z1, z2, u1, u2 = state.x, state.z1, state.z2, state.u1, state.u2
    rho, D, M, N = solver.rho, solver.D, solver.M, solver.N

    for k in range(1, it+1):
        tk = 0.5**k if k>1 else 1.0
        bk = cv + rho*(z1-u1) + rho*D.T@(z2-u2)
        r=bk-M@x; p=r.copy(); rs=r@r
        for _ in range(cg_in):
            Mp=M@p; a=rs/(p@Mp); x+=a*p; r-=a*Mp; rn=r@r
            if np.sqrt(rn)<tk: break
            p=r+(rn/rs)*p; rs=rn
        z1_new=np.maximum(0, x+u1)
        v2=D@x+u2; z2_new=np.sign(v2)*np.maximum(np.abs(v2)-tv/rho, 0)
        u1+=x-z1_new; u2+=D@x-z2_new
        z1=z1_new; z2=z2_new

    state.x, state.z1, state.z2, state.u1, state.u2 = x, z1, z2, u1, u2
    return z1

def main():
    src="data_hdf/locata.hdf"
    with h5py.File(src,"r") as f:
        em32=f["em32"][:]; mic=f["mic"][:]; dur=f["dur"][:]; sr=f.attrs["sr"]

    n_frames, n_bands = em32.shape[0], em32.shape[1]
    apgd_new = np.zeros((n_frames, n_bands, 484))

    R=get_field(); xyz=get_xyz(); A=steering_operator(xyz,R,5000.0)
    solver=ADMMSolver(A,rho=100)

    # Per-band state tracking (full warm-start)
    states = [None]*n_bands

    t0=time.time()
    prev_dur=None

    for i in tqdm(range(n_frames), desc="2-step warm"):
        cur_dur = dur[i,0]
        new_clip = (prev_dur is not None and abs(cur_dur-prev_dur)>1e-6)

        for b in range(n_bands):
            C = em32[i,b]

            if new_clip or states[b] is None:
                # Reset full state
                s = ADMMState()
                s.x = np.zeros(484); s.z1 = np.zeros(484)
                s.z2 = np.zeros(483); s.u1 = np.zeros(484); s.u2 = np.zeros(483)
                states[b] = s

            apgd_new[i,b] = solve_full_warm(solver, C, states[b], it=2, tv=1, cg_in=10)

        prev_dur = cur_dur

    elapsed=time.time()-t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    dst="data_hdf/locata_warm2.hdf"
    with h5py.File(dst,"w") as f:
        f.create_dataset("em32",data=em32); f.create_dataset("mic",data=mic)
        f.create_dataset("apgd",data=apgd_new); f.create_dataset("dur",data=dur)
        f.attrs["sr"]=sr
    print(f"Saved: {dst}")

    # Compare with cold 10-step, warm 10-step, and APGD reference
    apgd_warm10 = h5py.File("data_hdf/locata_warm.hdf","r")["apgd"][:]
    apgd_cold   = h5py.File("data_hdf/locata_cold.hdf","r")["apgd"][:]
    apgd_ref    = h5py.File(src,"r")["apgd"][:]

    for name, arr in [("cold-10it",apgd_cold),("warm-10it",apgd_warm10),("warm-2it",apgd_new)]:
        nrm = np.linalg.norm(apgd_ref.reshape(-1,484), axis=1)+1e-10
        err = np.linalg.norm(arr.reshape(-1,484)-apgd_ref.reshape(-1,484), axis=1)/nrm
        print(f"  {name:>12s}: mean={err.mean():.4f} median={np.median(err):.4f}")

if __name__=="__main__":
    main()
