"""Figures for latent_gw results (needs matplotlib: pip install -e ".[figures]").

    import latent_gw as lg
    from latent_gw import plots
    res = {m: lg.compare(X, Y, metric=m) for m in ("euclidean", "geodesic_k20")}
    plots.scores_vs_eps(res).savefig("scores.png")
    T = {m: plots.transport_plans(X, Y, m, grid=(1.5, 0.03))[0.03] for m in res}
    plots.transport_plans_figure(T, order=plots.spectral_order(Y)).savefig("plans.png")

Every figure function returns a matplotlib Figure. Transport plans are shown pooled in blocks of samples,
which is also what keeps subject-level plans from being displayed when the data are restricted.
"""
from __future__ import annotations

import numpy as np
import torch

from .compare import prep_region
from .solver import eps_c, solve

DT = torch.float64
BG, INK, MUTED, GRID = "#fbf8f3", "#1d1b19", "#8a837a", "#e8e2d8"
PALETTE = ["#3fb58f", "#6f86d6", "#ec7d6a", "#b07cc6", "#d9a441", "#5aa9c9"]   # one colour per metric, fixed order


def style():
    """Light, legible defaults used by every figure of the package."""
    import matplotlib as mpl
    mpl.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False})


def _colors(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------------- data helpers
def transport_plans(X, Y, metric="euclidean", grid=(1.5, 0.7, 0.3, 0.1, 0.03), l2=True, dev=None,
                    delta=1e-3, max_outer=3000, tol=1e-8):
    """GW plans {eps/eps_c: T (n x m numpy array)} for the real pair, computed exactly as in compare / path
    (downward continuation from the slightly tilted product coupling)."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    px, py = prep_region(np.asarray(X, float), metric, dev, l2), prep_region(np.asarray(Y, float), metric, dev, l2)
    n = px["n"]; ec = eps_c(px, py)
    T = torch.full((n, n), 1.0 / n ** 2, dtype=DT, device=dev)
    P = torch.outer(px["u1"], py["u1"]); T = (T * (1 + delta * P / P.abs().max())).clamp_min(1e-300); T = T / T.sum()
    out = {}
    for rel in sorted({float(g) for g in grid}, reverse=True):
        T, *_ = solve(px["D"], py["D"], rel * ec, T, max_outer, tol)
        out[rel] = T.cpu().numpy()
    return out


def spectral_order(Y, k=15):
    """1-D ordering of the samples along the manifold of Y: Fiedler vector of its symmetric kNN graph
    (falls back to the first principal component if the graph is disconnected)."""
    from scipy.spatial import cKDTree
    Y = np.asarray(Y, float); n = len(Y)
    _, nb = cKDTree(Y).query(Y, k + 1)
    A = np.zeros((n, n)); A[np.repeat(np.arange(n), k), nb[:, 1:].ravel()] = 1; A = np.maximum(A, A.T)
    d = A.sum(1); L = np.eye(n) - A / np.sqrt(np.outer(d, d))
    w, U = np.linalg.eigh(L)
    if w[1] < 1e-8:
        return np.argsort(np.linalg.svd(Y - Y.mean(0), full_matrices=False)[0][:, 0])
    return np.argsort(U[:, 1] / np.sqrt(d))


def pool_blocks(T, order=None, n_blocks=40):
    """Pool a square plan into n_blocks x n_blocks blocks (rows and columns in the same sample order) and return
    the enrichment over chance (1 = uniform coupling)."""
    T = np.asarray(T, float); n = len(T)
    order = np.arange(n) if order is None else np.asarray(order)
    M = T[np.ix_(order, order)] * n                                   # rows sum to 1
    edges = np.linspace(0, n, min(n_blocks, n) + 1).astype(int)
    out = np.add.reduceat(np.add.reduceat(M, edges[:-1], axis=0), edges[:-1], axis=1)
    size = np.diff(edges)
    return out / (size[:, None] * size[None, :] / n)


# --------------------------------------------------------------------------------------------- figures
def scores_vs_eps(results, title="GW score against same-spectrum surrogates"):
    """results: {name: DataFrame from lg.compare}. Left: rA of the real pair (line) and of the surrogates (dashed).
    Right: contrast = real - surrogates, with the z value written at each point."""
    import matplotlib.pyplot as plt
    style(); col = _colors(results)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for name, r in results.items():
        r = r.sort_values("rel", ascending=False); x = r.rel.values
        a1.plot(x, r.rA, "-o", color=col[name], lw=2.2, ms=6, label=f"{name}")
        a1.plot(x, r.rA_surr, "--", color=col[name], lw=1.5, alpha=.8)
        a2.plot(x, r.contrast, "-o", color=col[name], lw=2.2, ms=6, label=name)
        for xi, ci, zi in zip(x, r.contrast, r.z):
            if xi < 1 and np.isfinite(zi):
                a2.annotate(f"z={zi:.0f}" if abs(zi) >= 10 else f"z={zi:.1f}", (xi, ci), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8.5, color=col[name])
    for a in (a1, a2):
        a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("ε / ε$_c$  (finer →)")
        xs = sorted({float(v) for r in results.values() for v in r.rel}); a.set_xticks(xs); a.set_xticklabels([f"{v:g}" for v in xs])
        a.minorticks_off()
    a1.set_ylabel("$r_A$"); a1.set_title("real pair (solid) vs surrogates (dashed)", loc="left")
    a2.axhline(0, color=MUTED, lw=1); a2.set_ylabel("contrast = real − surrogates"); a2.set_title("beyond the spectrum", loc="left")
    a2.legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    return fig


def retrieval_vs_eps(results, title="Same-sample retrieval"):
    """n·tr(T) of the real pair (solid) and of the surrogates (dashed) across eps; x1 = chance."""
    import matplotlib.pyplot as plt
    style(); col = _colors(results)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for name, r in results.items():
        r = r.sort_values("rel", ascending=False)
        ax.plot(r.rel, r.ntrT, "-o", color=col[name], lw=2.2, ms=6, label=name)
        if "ntrT_surr" in r:
            ax.plot(r.rel, r.ntrT_surr, "--", color=col[name], lw=1.3, alpha=.7)
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_yscale("log"); ax.minorticks_off()
    from matplotlib.ticker import FixedLocator, FuncFormatter
    lo, hi = ax.get_ylim(); ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.annotate("chance", (1, 1), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                ha="right", va="bottom", color=MUTED, fontsize=9)
    xs = sorted({float(v) for r in results.values() for v in r.rel}); ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel("ε / ε$_c$  (finer →)"); ax.set_ylabel("n·tr(T)  (×1 = chance)"); ax.legend()
    ax.set_title(title, loc="left")
    return fig


def transport_plans_figure(plans, order=None, n_blocks=40, title="Transport plans, pooled in blocks of samples"):
    """plans: {name: square plan}. Heatmaps of the block-pooled plan (enrichment over chance, log scale);
    a diagonal band means that GW sends each sample close to itself. order: e.g. spectral_order(Y)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    style()
    pooled = {k: pool_blocks(T, order, n_blocks) for k, T in plans.items()}
    vmax = max(np.nanmax(p) for p in pooled.values()); vmax = max(vmax, 2.0)
    fig, axes = plt.subplots(1, len(pooled), figsize=(4.2 * len(pooled) + 0.8, 4.4), constrained_layout=True, squeeze=False)
    for ax, (name, B) in zip(axes[0], pooled.items()):
        im = ax.imshow(np.clip(B, 1 / vmax, None), cmap="magma", norm=LogNorm(1 / vmax, vmax), interpolation="nearest")
        ax.set_title(name, loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel("samples of Y (ordered)"); ax.set_ylabel("samples of X (same order)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=.85, label="mass / chance")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    return fig


def spectra(clouds, n_show=None, title="Variance spectra"):
    """clouds: {name: (n, d) array}. Normalised covariance eigenvalues (share of variance per principal axis)."""
    import matplotlib.pyplot as plt
    style(); col = _colors(clouds)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for name, X in clouds.items():
        X = np.asarray(X, float); ev = np.sort(np.linalg.eigvalsh(np.cov((X - X.mean(0)).T)))[::-1]
        ev = np.clip(ev, 0, None); ev = ev / ev.sum(); ev = ev[:n_show] if n_show else ev
        ax.plot(np.arange(1, len(ev) + 1), ev, "-o", color=col[name], lw=2, ms=5, label=name)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("principal axis"); ax.set_ylabel("share of variance"); ax.legend(); ax.set_title(title, loc="left")
    return fig


def score_matrix(M, title="Scores between spaces", label="score", cmap="Purples"):
    """M: square DataFrame (e.g. lg.cka_matrix output); diagonal hidden."""
    import matplotlib.pyplot as plt
    style()
    A = np.asarray(M, float).copy(); np.fill_diagonal(A, np.nan)
    fig, ax = plt.subplots(figsize=(0.32 * len(A) + 3, 0.32 * len(A) + 2.2), constrained_layout=True)
    im = ax.imshow(A, cmap=cmap, interpolation="nearest"); ax.grid(False)
    ax.set_xticks(range(len(A))); ax.set_yticks(range(len(A)))
    ax.set_xticklabels(list(M.columns), rotation=90, fontsize=8); ax.set_yticklabels(list(M.index), fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.8, label=label); ax.set_title(title, loc="left")
    return fig
