"""Figures on the Swiss-roll toy: the plots used for the real analyses, on data anyone can generate.

    pip install -e ".[figures]"
    python examples/toy_plots.py            # ~20 s on CPU, writes toy_figures/*.png
"""
from pathlib import Path

import numpy as np

import latent_gw as lg
from latent_gw import plots

out = Path("toy_figures"); out.mkdir(exist_ok=True)

# same toy as examples/toy_swiss_roll.py: a Swiss roll (3-D) and its unrolled version (flat, 5-D, other axes)
rng = np.random.default_rng(0)
n = 300
t = 1.5 * np.pi * (1 + 2 * rng.random(n)); h = 12 * rng.random(n)
X = np.c_[t * np.cos(t), h, t * np.sin(t)] + 0.1 * rng.normal(size=(n, 3))
s = 0.5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
Y = np.c_[s, h, np.zeros((n, 3))] @ Q + 0.1 * rng.normal(size=(n, 5))

metrics = {"Euclidean": "euclidean", "geodesic (k=10)": "geodesic_k10"}
grid = (1.5, 0.3, 0.1, 0.03)
kw = dict(l2=False, dev="cpu")                    # l2=False: projecting a roll on the sphere would distort it

# 1. scores and retrieval across eps, against same-spectrum surrogates
res = {name: lg.compare(X, Y, metric=m, grid=grid, surrogate="pcperm", B=3, **kw) for name, m in metrics.items()}
plots.scores_vs_eps(res, title="Swiss roll vs unrolled: GW score beyond the spectrum").savefig(out / "1_scores_vs_eps.png", dpi=150)
plots.retrieval_vs_eps(res, title="Same-sample retrieval, n·tr(T)").savefig(out / "2_retrieval.png", dpi=150)

# 2. transport plans at the finest eps, pooled in blocks, samples ordered along the unrolled sheet
order = plots.spectral_order(Y)
T = {name: plots.transport_plans(X, Y, m, grid=grid, **kw)[0.03] for name, m in metrics.items()}
plots.transport_plans_figure(T, order=order, n_blocks=30,
                             title="Transport plans at 0.03 ε_c (diagonal = same sample)").savefig(out / "3_transport_plans.png", dpi=150)

# 3. spectra of the two clouds
plots.spectra({"rolled (3-D)": X, "unrolled (5-D)": Y}).savefig(out / "4_spectra.png", dpi=150)

print("written:", ", ".join(sorted(p.name for p in out.glob("*.png"))))
