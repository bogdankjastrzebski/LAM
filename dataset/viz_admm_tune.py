"""visualize tuning results for admm with rho=100 vs rho=10"""
import numpy as np, h5py, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import scipy.constants as constants
from trainer.utils import cart2eq, wrapped_rad2deg, cmap_from_list

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
from tune_admm import ADMMSolver

with h5py.File("data_hdf/locata.hdf","r") as f: em32=f["em32"][:]; apgd=f["apgd"][:]
R=get_field(); xyz=get_xyz(); A=steering_operator(xyz,R,5000.0)
band=10

def draw(ax, I, title):
    _,Re,Ra=cart2eq(*R); Re,Ra=wrapped_rad2deg(Re, Ra)
    bm=basemap.Basemap(projection='mill', llcrnrlat=np.min(Re), urcrnrlat=np.max(Re), llcrnrlon=np.min(Ra), urcrnrlon=np.max(Ra), resolution='c', ax=ax)
    bm.drawparallels(np.linspace(np.min(Re),np.max(Re),3),color='w',dashes=[1,0],labels=[1,0,0,1],linewidth=1)
    bm.drawmeridians(np.linspace(np.min(Ra),np.max(Ra),5),color='w',dashes=[1,0],labels=[1,0,0,1],linewidth=1)
    Rx,Ry=bm(Ra,Re); tria=tri.Triangulation(Rx,Ry)
    In=I/(I.max()+1e-10); colors=plt.cm.viridis(In)
    ax.tripcolor(tria,np.arange(len(I)),cmap=cmap_from_list('m',colors,N=len(I)),shading='gouraud',alpha=0.9,edgecolors='w',linewidth=0.05)
    ax.set_title(title,fontsize=9)

solvers = [ADMMSolver(A, rho=100), ADMMSolver(A, rho=10)]
n_samples=6; step=max(1,em32.shape[0]//n_samples)
fig,axes=plt.subplots(4, n_samples, figsize=(3*n_samples, 12))

for j,idx in enumerate(range(0, min(em32.shape[0], n_samples*step), step)):
    C=em32[idx,band]; xt=apgd[idx,band]
    x_fast = solvers[0].solve(C, it=10, tv=1, cg_in=10)
    x_10   = solvers[1].solve(C, it=10, tv=1, cg_in=10)
    x_100  = solvers[1].solve(C, it=100, tv=1, cg_in=10)
    draw(axes[0,j], xt, f"GT #{idx}" if j==0 else f"#{idx}")
    e1=np.linalg.norm(x_fast-xt)/np.linalg.norm(xt)
    e2=np.linalg.norm(x_10-xt)/np.linalg.norm(xt)
    e3=np.linalg.norm(x_100-xt)/np.linalg.norm(xt)
    draw(axes[1,j], x_fast, f"rho=100 10it err={e1:.2f}" if j==0 else "")
    draw(axes[2,j], x_10,   f"rho=10  10it err={e2:.2f}" if j==0 else "")
    draw(axes[3,j], x_100,  f"rho=10 100it err={e3:.2f}" if j==0 else "")

plt.tight_layout()
fig.savefig("benchmark_viz/admm_rho_comparison.png", dpi=150)
print("Saved: benchmark_viz/admm_rho_comparison.png")
