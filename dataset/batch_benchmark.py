"""batch_benchmark.py - Run solvers on multiple LOCATA samples with best params."""
import os, sys, numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import skimage.util as skutil
import scipy.constants as constants

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from trainer.utils import cart2eq, wrapped_rad2deg, cmap_from_list

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

# Solvers (fast versions with best params)
def solve_cp(A, C, it=100, tv=0.01):
    N=A.shape[1]; H=np.abs(A.conj().T@A)**2
    cv=np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    L=np.linalg.norm(H,2); τ=1/L; σ=0.9/(4*τ)
    x=np.zeros(N); xb=np.zeros(N)
    D=np.zeros((N-1,N)); D[np.arange(N-1),np.arange(N-1)]=-1; D[np.arange(N-1),np.arange(N-1)+1]=1
    for _ in range(it):
        y=np.clip(σ*(D@xb),-tv,tv); xo=x.copy(); v=x-τ*(H@x-cv+D.T@y); x=np.maximum(0,v); xb=2*x-xo
    return x

def solve_sg(A, C, it=200, tv=0.1):
    N=A.shape[1]; H=np.abs(A.conj().T@A)**2
    cv=np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    x=np.zeros(N)
    for k in range(1,it+1):
        g=H@x-cv; gt=np.zeros(N)
        for i in range(N-1): s=np.sign(x[i]-x[i+1]); gt[i]+=s; gt[i+1]-=s
        d=g+tv*gt; x=np.maximum(0,x-10*0.95**k*d/max(np.linalg.norm(d),1e-10))
    return x

def solve_admm(A, C, it=100, tv=1, rho=10):
    N=A.shape[1]; H=np.abs(A.conj().T@A)**2
    cv=np.array([np.real(A[:,i].conj().T@C@A[:,i]) for i in range(N)])
    D=np.zeros((N-1,N)); D[np.arange(N-1),np.arange(N-1)]=-1; D[np.arange(N-1),np.arange(N-1)+1]=1
    M=H+rho*(np.eye(N)+D.T@D)
    x=np.zeros(N); z1=np.zeros(N); z2=np.zeros(N-1); u1=np.zeros(N); u2=np.zeros(N-1)
    for k in range(1,it+1):
        tk=0.5**k if k>1 else 1.0
        bk=cv+rho*(z1-u1)+rho*D.T@(z2-u2); r=bk-M@x; p=r.copy(); rs=r@r
        for _ in range(5):
            Mp=M@p; a=rs/(p@Mp); x+=a*p; r-=a*Mp; rn=r@r
            if np.sqrt(rn)<tk: break
            p=r+(rn/rs)*p; rs=rn
        z1=np.maximum(0,x+u1); z2=np.sign(D@x+u2)*np.maximum(np.abs(D@x+u2)-tv/rho,0)
        u1+=x-z1; u2+=D@x-z2
    return z1

def draw_map_single(I, R, ax, title=""):
    _, Re, Ra=cart2eq(*R); Re, Ra=wrapped_rad2deg(Re, Ra)
    bm=basemap.Basemap(projection='mill', llcrnrlat=np.min(Re), urcrnrlat=np.max(Re),
                       llcrnrlon=np.min(Ra), urcrnrlon=np.max(Ra), resolution='c',ax=ax)
    bm.drawparallels(np.linspace(np.min(Re),np.max(Re),3),color='w',dashes=[1,0],labels=[1,0,0,1],linewidth=1)
    bm.drawmeridians(np.linspace(np.min(Ra),np.max(Ra),5),color='w',dashes=[1,0],labels=[1,0,0,1],linewidth=1)
    Rx,Ry=bm(Ra,Re); tria=tri.Triangulation(Rx,Ry)
    In=I/(I.max()+1e-10); colors=plt.cm.viridis(In)
    ax.tripcolor(tria,np.arange(len(I)),cmap=cmap_from_list('m',colors,N=len(I)),shading='gouraud',alpha=0.9,edgecolors='w',linewidth=0.05)
    ax.set_title(title,fontsize=8)

def main():
    with h5py.File("../dataset/data_hdf/locata.hdf","r") as f:
        em32=f["em32"][:]; apgd=f["apgd"][:]; dur=f["dur"][:]

    R=get_field(); xyz=get_xyz(); A=steering_operator(xyz,R,5000.0)
    band=10; n_samples=12; step=max(1,em32.shape[0]//n_samples)

    outdir="./benchmark_viz/batch"; os.makedirs(outdir,exist_ok=True)

    print(f"Testing {n_samples} samples at band={band} with best params")
    print(f"CP: tv=0.01 100it | SG: tv=0.1 200it | ADMM: tv=1 rho=10 100it\n")

    results={"CP":[],"SG":[],"ADMM":[]}
    for idx in range(0, min(em32.shape[0], n_samples*step), step):
        C=em32[idx,band]; xt=apgd[idx,band]
        x_cp=solve_cp(A,C); x_sg=solve_sg(A,C); x_ad=solve_admm(A,C)
        ec=np.linalg.norm(x_cp-xt)/np.linalg.norm(xt)
        es=np.linalg.norm(x_sg-xt)/np.linalg.norm(xt)
        ea=np.linalg.norm(x_ad-xt)/np.linalg.norm(xt)
        results["CP"].append(ec); results["SG"].append(es); results["ADMM"].append(ea)
        print(f"  sample {idx:4d}: CP={ec:.3f}  SG={es:.3f}  ADMM={ea:.3f}")

    # Summary stats
    print("\n--- Aggregate ---")
    for name in ["CP","SG","ADMM"]:
        arr=np.array(results[name]); print(f"  {name:>6s}: mean={arr.mean():.3f}  median={np.median(arr):.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

    # Visualization: first 8 samples in 2x4 grid
    n_show=min(8,n_samples); fig,axes=plt.subplots(4,n_show,figsize=(3*n_show,12))
    for j,idx in enumerate(range(0, min(em32.shape[0], n_show*step), step)):
        C=em32[idx,band]; xt=apgd[idx,band]
        x_cp=solve_cp(A,C); x_sg=solve_sg(A,C); x_ad=solve_admm(A,C)
        draw_map_single(xt,R,axes[0,j],f"GT #{idx}" if j==0 else f"#{idx}")
        draw_map_single(x_cp,R,axes[1,j],f"CP" if j==0 else "")
        draw_map_single(x_sg,R,axes[2,j],f"SG" if j==0 else "")
        draw_map_single(x_ad,R,axes[3,j],f"ADMM" if j==0 else "")
    plt.tight_layout(); fig.savefig(os.path.join(outdir,"batch_grid.png"),dpi=150); plt.close()
    print(f"\nSaved: {outdir}/batch_grid.png")

if __name__=="__main__": main()
