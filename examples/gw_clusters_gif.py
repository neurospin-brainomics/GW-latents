"""Animation for the README: entropic GW matching two clouds of clusters as eps decreases.

    python examples/gw_clusters_gif.py      # writes assets/gw_clusters.gif
"""
import numpy as np, ot, matplotlib, imageio.v2 as imageio, io, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
warnings.filterwarnings("ignore")
rng = np.random.default_rng(3)
BG="#fbf8f3"; INK="#1d1b19"; INK2="#5f5a52"; GRID="#e8e2d8"
COLS=["#3fb58f","#6f86d6","#ec7d6a"]
# --- source: 3 clusters, different sizes and spreads, non-equilateral layout
sizes=[34,22,14]; centers=np.array([[0,0],[4.2,0.6],[1.3,3.4]]); spreads=[0.55,0.45,0.35]
X=np.vstack([c+s*rng.normal(size=(n,2)) for c,s,n in zip(centers,spreads,sizes)])
lab=np.repeat(np.arange(3),sizes); n=len(X)
# --- target: same structure, rotated + mirrored + rescaled + noise, shuffled order
th=2.1; R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])@np.diag([1,-1])
Y=1.6*X@R.T+rng.normal(scale=0.12,size=X.shape)+np.array([10,2])
perm=rng.permutation(n); Y=Y[perm]; labY=lab[perm]
def D(Z):
    d=np.sqrt(((Z[:,None]-Z[None])**2).sum(-1)); return d/np.sqrt((d**2).mean())
DX,DY=D(X),D(Y); p=q=np.ones(n)/n
eps_c=1.0                                     # empirical: above it the plan stays ≈ uniform
rels=np.geomspace(3.0,0.05,36)
plans=[]; T0=None
for r in rels:
    T=ot.gromov.entropic_gromov_wasserstein(DX,DY,p,q,"square_loss",epsilon=r*eps_c,max_iter=500,tol=1e-9,G0=T0)
    plans.append(T); T0=T
CX=np.array([to_rgb(c) for c in COLS])[lab]
orderY=np.argsort(labY,kind="stable"); CH=sum((np.array(sizes)/n)**2)
def frame(r,T):
    fig=plt.figure(figsize=(12,5),dpi=100,facecolor=BG)
    fig.text(0.03,0.925,"Gromov–Wasserstein matches two clouds using only their internal distances",fontsize=15,weight="bold",color=INK)
    fig.text(0.03,0.865,"cloud B is cloud A rotated, mirrored, rescaled, noised and shuffled: no shared axes, no point labels",fontsize=10.5,color=INK2)
    a1=fig.add_axes([0.03,0.13,0.28,0.62]); a2=fig.add_axes([0.35,0.13,0.28,0.62]); a3=fig.add_axes([0.71,0.13,0.24,0.62])
    for a in (a1,a2): a.set_facecolor(BG); a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]; a.set_aspect("equal")
    for x_,t_ in [(0.03,"cloud A"),(0.35,"cloud B, coloured by where its mass comes from"),(0.71,"coupling T  (rows A, columns B)")]:
        fig.text(x_,0.79,t_,fontsize=11.5,color=INK2)
    a1.scatter(*X.T,s=38,c=CX,edgecolor=BG,lw=0.8)
    w=T/T.sum(0,keepdims=True); CY=np.clip(w.T@CX,0,1)
    a2.scatter(*Y.T,s=38,c=CY,edgecolor=BG,lw=0.8)
    Tm=(T*n*n)[np.argsort(lab,kind="stable")][:,orderY]
    a3.imshow(Tm,cmap="Blues",vmin=0,vmax=VMAX,interpolation="nearest")
    a3.set_xticks([]); a3.set_yticks([])
    for s_ in a3.spines.values(): s_.set_color(GRID)
    acc=(T*(lab[:,None]==labY[None])).sum()
    # ε slider
    sl=fig.add_axes([0.03,0.045,0.30,0.03]); sl.set_xscale("log"); sl.set_xlim(rels.max()*1.1,rels.min()/1.1)
    sl.axhline(0,color=GRID,lw=3); sl.axvline(1,color=INK2,lw=1,ls=(0,(2,2))); sl.scatter([r],[0],s=90,color=INK,zorder=3)
    sl.set_ylim(-1,1); sl.axis("off"); sl.text(1,0.9,"ε$_c$",ha="center",va="bottom",fontsize=9,color=INK2)
    fig.text(0.35,0.048,f"ε = {r:.2g} ε$_c$",fontsize=12,weight="bold",color=INK,va="center")
    fig.text(0.47,0.048,f"mass sent to the right cluster: {acc:.0%}   (chance {CH:.0%})",fontsize=11,color=INK2,va="center")
    b=io.BytesIO(); fig.savefig(b,format="png",facecolor=BG); plt.close(fig); b.seek(0); return imageio.imread(b)
VMAX=4.0
frames=[frame(r,T) for r,T in zip(rels,plans)]
frames=[frames[0]]*6+frames+[frames[-1]]*14
import os; os.makedirs("assets", exist_ok=True)
imageio.mimsave("assets/gw_clusters.gif",frames,duration=0.12,loop=0)
print("ok", [round((T*(lab[:,None]==labY[None])).sum(),2) for T in plans[::5]])
