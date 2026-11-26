#!/usr/bin/env python
"""
222 -- Champollion slide figures from the real runs (Jean Zay, CPU). AGGREGATED OUTPUTS ONLY.

Reads 215 run directories (map_shard_*.csv, sujets_positions.npy, plans/<X>__<Y>/rel<ε>.npz [+ substitut0/]) and the aligned
region latents, and writes four 16:9 figures plus the CSV of every number they show:

  P1_plan_blocks.png      GW plan of one pair, subjects sorted along the target region's manifold (spectral order) and
                          pooled into B×B blocks of ≈ n/B subjects: a diagonal band = GW sends each subject to its own
                          neighbourhood. One column per run (metric), top = real regions, bottom = same-spectrum surrogate.
  P2_neighbourhoods.png   neighbourhood recovery: mass that the plan sends from subject i into the r nearest neighbours of i
                          in the target region (i itself included), divided by its chance value r/n.  r = 1 is n·tr(T).
                          Median over the saved pairs of each group (homologues / others), real vs surrogate.
  P3_cka_vs_gw.png        per region pair: debiased linear CKA vs GW shape contrast at ε/ε_c = --rel, Spearman ρ per metric.
  P5_retrieval_eps_<run>.png   n·tr(T) of every region pair at every ε < ε_c, by pair type, vs the surrogates of all pairs.
  P6_retrieval_repro_<run>.png the top pairs followed across ε, and n·tr(T) of each pair in the reference run vs other runs
                          (other k, other metric, other subject sample): is ×20+ reproducible?
  P4_typicality.png       subject typicality (mean distance to the others, as 217) left vs right for a few homologues,
                          as a decile × decile table of counts ÷ independence (cells < --min-cell subjects blanked).

Privacy: no figure or CSV holds a subject-level value. P1 pools ≥ n/B subjects per block; P4 shows densities of ranks with
small cells removed. Plans and latents are read in place and never copied. Still check with V. Frouin before showing outside.

    python 222_champ_figs_jz.py --out $METHODO/figs_222 \
        --run geo20=$METHODO/champ_sub_geo20 --run orc20=$METHODO/champ_sub_orc20 [--run euc=$METHODO/champ_sub_euc] \
        [--map euc=$METHODO/champ_map_pcperm --map geo20=$METHODO/champ_map_geo20 --map orc20=$METHODO/champ_map_orc20]
Run names starting with euc / geo / orc get the slide colours of the metric. --map defaults to the --run directories.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ_style as S
from champ_style import EUC, GEO, ORC, INK, INK2, MUTED, NULL, GRID, BG, BAND

SIDE = re.compile(r"^(?P<base>.+)_(?P<side>left|right)$")
base = lambda r: SIDE.match(r)["base"] if SIDE.match(r) else r
side = lambda r: SIDE.match(r)["side"] if SIDE.match(r) else ""
tok = lambda r: set(base(r).split("-"))


def pair_type(x, y):
    if base(x) == base(y): return "homologue"
    if tok(x) & tok(y): return "shared sulcus"
    return "other"


def metric_of(name):
    n = name.lower()
    return "euclidean" if n.startswith("euc") else "geodesic" if n.startswith("geo") else "orc" if n.startswith("orc") else "other"


def colour(name): return S.METRIC_COL.get(metric_of(name), INK2)


def pretty(name):
    m = metric_of(name); k = re.search(r"(\d+)", name)
    lab = S.METRIC_NAME.get(m, name)
    return f"{lab}  k={k[1]}" if (k and m != "euclidean") else lab


def short(r):
    return f"{base(r)} ({'L' if side(r) == 'left' else 'R' if side(r) == 'right' else '?'})"


# ---------------------------------------------------------------------------------------------- data access
def load_plan(fn):
    """row-stochastic view: P[i] = n·T[i, :] (sums to 1). Returns (n, dict(idx, val) or dense, diag)"""
    z = np.load(fn)
    n = int(z["n"])
    if "plan" in z:
        Pm = z["plan"].astype(np.float64) / n
        return n, dict(dense=Pm), np.diag(Pm).copy()
    return n, dict(idx=z["idx"].astype(np.int64), val=z["val"].astype(np.float64) / n), z["diag"].astype(np.float64) / n


def dense(n, P):
    if "dense" in P: return P["dense"]
    M = np.zeros((n, n)); np.put_along_axis(M, P["idx"], P["val"], axis=1); return M


def region_latents(aligned, name, idx):
    return np.asarray(np.load(Path(aligned) / f"{name}.npy", mmap_mode="r")[idx], dtype=np.float64)


def l2(X):
    X = X - X.mean(0); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def spectral_order(Y, k=15):
    """1-D ordering along the manifold: Fiedler vector of the symmetric kNN graph (falls back to PC1 if disconnected)"""
    n = len(Y); _, nb = cKDTree(Y).query(Y, k + 1)
    A = np.zeros((n, n)); r = np.repeat(np.arange(n), k); A[r, nb[:, 1:].ravel()] = 1; A = np.maximum(A, A.T)
    d = A.sum(1); Lp = np.eye(n) - A / np.sqrt(np.outer(d, d))
    w, U = np.linalg.eigh(Lp)
    if w[1] < 1e-8:                                                   # disconnected graph
        return np.argsort(np.linalg.svd(Y - Y.mean(0), full_matrices=False)[0][:, 0])
    return np.argsort(U[:, 1] / np.sqrt(d))


def plan_dirs(run, rel):
    for d in sorted((Path(run) / "plans").glob("*__*")):
        f = d / f"rel{rel:g}.npz"
        if f.exists():
            x, y = d.name.split("__"); yield x, y, f, (d / "substitut0" / f"rel{rel:g}.npz")


# ---------------------------------------------------------------------------------------------- P1
def blocks(P, n, order, B):
    """B×B pooled mass, rows/cols in the same subject order, as enrichment over chance (1 = uniform)"""
    M = dense(n, P)[np.ix_(order, order)]
    edges = np.linspace(0, n, B + 1).astype(int)
    out = np.add.reduceat(np.add.reduceat(M, edges[:-1], axis=0), edges[:-1], axis=1)
    size = np.diff(edges)
    return out / (size[:, None] * size[None, :] / n)                  # chance mass of a block = |a|·|b| / n (rows sum to 1)


def fig_plans(a, runs, out, rows_csv):
    want = [tuple(p.split(":")) for p in a.plan_pair.split(",")]
    for (px, py) in want:
        cols = []
        for name, run in runs:
            hit = [(x, y, f, fs) for x, y, f, fs in plan_dirs(run, a.rel) if {x, y} == {px, py}]
            if hit: cols.append((name, run, *hit[0]))
        if not cols:
            print(f"P1: no saved plan for {px}:{py} at rel {a.rel:g} in any run — skipped"); continue
        fig = plt.figure(figsize=(S.W, S.H))
        x, y = cols[0][2], cols[0][3]
        S.headline(fig, "GW sends each subject to its own neighbourhood",
                   f"{short(x)} → {short(y)}  ·  ε = {a.rel:g} ε$_c$  ·  subjects ordered along the target manifold, {a.bins}×{a.bins} blocks")
        nc = len(cols); hh = 0.31; w = hh * S.H / S.W; gap = 0.075
        vmax = a.vmax
        norm = LogNorm(vmin=1 / vmax, vmax=vmax)
        for c, (name, run, x, y, f, fs) in enumerate(cols):
            idx = np.load(Path(run) / "sujets_positions.npy")
            Y = l2(region_latents(a.aligned, y, idx)); order = spectral_order(Y)
            for r_, (fn, lab) in enumerate([(f, "real"), (fs, "surrogate")]):
                if not fn.exists(): continue
                n, P, dg = load_plan(fn)
                Mb = blocks(P, n, order, a.bins)
                ax = fig.add_axes([0.2 + c * (w + gap), 0.43 - r_ * 0.38, w, hh])
                ax.imshow(np.clip(Mb, 1 / vmax, vmax), cmap=S.BLUES, norm=norm, interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(colour(name) if r_ == 0 else NULL); sp.set_linewidth(3 if r_ == 0 else 1.5)
                band = np.mean([Mb[i, j] for i in range(a.bins) for j in range(a.bins) if abs(i - j) <= 1])
                ax.text(0, 1.03, f"band ×{band:.1f} · n·tr(T) ×{dg.sum():.1f}", transform=ax.transAxes, fontsize=12.5,
                        color=INK if r_ == 0 else INK2, va="bottom", weight="semibold" if r_ == 0 else "normal")
                if r_ == 0: ax.text(0, 1.15, pretty(name), transform=ax.transAxes, fontsize=18, weight="semibold", va="bottom")
                if c == 0: fig.text(0.185, 0.43 - r_ * 0.38 + hh / 2, "real\nregions" if r_ == 0 else "same-spectrum\nsurrogate",
                                    fontsize=15, ha="right", va="center", color=INK if r_ == 0 else INK2)
                rows_csv.append(dict(figure="P1", run=name, src=x, tgt=y, kind=lab, rel=a.rel, bins=a.bins, diag_band=band, ntrT=float(dg.sum()),
                                     **{f"b{i}_{j}": Mb[i, j] for i in range(a.bins) for j in range(a.bins)} if a.csv_blocks else {}))
        cax = fig.add_axes([0.2 + nc * (w + gap), 0.12, 0.012, 0.6])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=S.BLUES), cax=cax); cb.outline.set_visible(False)
        cax.minorticks_off(); cb.set_ticks([1 / vmax, 1, vmax]); cb.set_ticklabels([f"×1/{vmax:g}", "chance", f"×{vmax:g}"]); cax.tick_params(labelsize=12)
        cax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        fig.text(0.04, 0.015, "Rows: source subjects; columns: the same subjects in the target region, same order. Band = mean over the 3 central diagonals.",
                 fontsize=11.5, color=MUTED)
        S.save(fig, out / f"P1_plan_blocks_{base(px)}_{side(px)[0]}__{base(py)}_{side(py)[0]}.png")


# ---------------------------------------------------------------------------------------------- P2
def recovery(P, n, dg, nb, radii):
    """mean over subjects of the plan mass inside N_r(i) (i + its r−1 nearest neighbours in the target), / (r/n)"""
    out = []
    if "dense" in P:
        M = P["dense"]
        for r in radii:
            out.append(np.take_along_axis(M, nb[:, :r], 1).sum(1).mean() / (r / n))
        return np.array(out)
    idx, val = P["idx"], P["val"]
    rank = np.full((n, n), np.iinfo(np.int32).max, dtype=np.int32)    # rank of j among i's target neighbours (0 = i itself)
    np.put_along_axis(rank, nb, np.broadcast_to(np.arange(nb.shape[1], dtype=np.int32), nb.shape), axis=1)
    rk = np.take_along_axis(rank, idx, 1); self_in = (idx == np.arange(n)[:, None])
    for r in radii:
        m = (val * ((rk < r) & ~self_in)).sum(1) + dg            # top-k entries other than i, + the stored diagonal
        out.append(m.mean() / (r / n))
    return np.array(out)


def fig_neighbourhoods(a, runs, out, rows_csv):
    radii = [int(r) for r in a.radii.split(",")]
    R = []
    for name, run in runs:
        idx = np.load(Path(run) / "sujets_positions.npy"); cacheY = {}
        for x, y, f, fs in plan_dirs(run, a.rel):
            if y not in cacheY:
                Y = l2(region_latents(a.aligned, y, idx)); _, nb = cKDTree(Y).query(Y, max(radii))
                nb[:, 0] = np.arange(len(Y)); cacheY[y] = nb            # guarantee the subject itself comes first
            for fn, kind in [(f, "real"), (fs, "surrogate")]:
                if not fn.exists(): continue
                n, P, dg = load_plan(fn)
                if n != len(idx): continue
                e = recovery(P, n, dg, cacheY[y], radii)
                for r, v in zip(radii, e):
                    R.append(dict(run=name, src=x, tgt=y, type=pair_type(x, y), kind=kind, r=r, enrichment=v))
        print(f"P2: {name}: {len({(d['src'], d['tgt']) for d in R if d['run'] == name})} pairs", flush=True)
    if not R: print("P2: no plans found — skipped"); return
    R = pd.DataFrame(R); R.to_csv(out / "P2_neighbourhoods_by_pair.csv", index=False)
    G = R.groupby(["run", "type", "kind", "r"]).enrichment.agg(med="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), n="size").reset_index()
    G.to_csv(out / "P2_neighbourhoods_summary.csv", index=False)
    groups = [t for t in ["homologue", "shared sulcus", "other"] if t in set(G.type)]
    titles = {"homologue": "L↔R homologues", "shared sulcus": "regions sharing a sulcus", "other": "other region pairs"}
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "GW recovers each subject's neighbourhood",
               f"Plan mass landing among the r nearest neighbours of the same subject (target region), ÷ chance r/n  ·  ε = {a.rel:g} ε$_c$")
    ng = len(groups); w = 0.84 / ng - 0.03
    ymax = max(G.q3.max() * 1.5, 3); ymin = min(G.q1.min() * 0.8, 0.8)
    for g_, t in enumerate(groups):
        ax = fig.add_axes([0.075 + g_ * (w + 0.04), 0.14, w, 0.56]); ends = []
        ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
        for name, _ in runs:
            for kind in ["surrogate", "real"]:
                d = G[(G.run == name) & (G.type == t) & (G.kind == kind)].sort_values("r")
                if d.empty: continue
                c = colour(name) if kind == "real" else NULL
                ax.fill_between(d.r, d.q1, d.q3, color=c, alpha=.13 if kind == "real" else .2, lw=0)
                ax.plot(d.r, d.med, color=c, lw=3.5 if kind == "real" else 2, ls="-" if kind == "real" else (0, (4, 3)),
                        marker="o" if kind == "real" else None, ms=8, mec=BG, mew=1.5)
                if kind == "real" and g_ == ng - 1:
                    ends.append([np.log(d.med.iloc[-1]), pretty(name), d.r.iloc[-1]])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ymin, ymax); ax.minorticks_off()
        ax.set_xticks(radii); ax.set_xticklabels([str(r) for r in radii], fontsize=12.5)
        yt = [v for v in [0.5, 1, 2, 3, 5, 10, 20, 50] if ymin <= v <= ymax]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt])
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        nn = int(R[(R.type == t)].groupby(["src", "tgt"]).ngroups)
        ax.set_title(f"{titles[t]}  ({nn} pair{'s' if nn > 1 else ''})", fontsize=17)
        if ends:                                                       # spread end labels so they never overlap
            ends.sort(); sep = 0.075 * np.log(ymax / ymin)
            for i in range(1, len(ends)): ends[i][0] = max(ends[i][0], ends[i - 1][0] + sep)
            for ly, lab, rx in ends: ax.text(rx * 1.15, np.exp(ly), lab, color=INK, fontsize=13, va="center")
        if g_ == 0: ax.set_ylabel("enrichment over chance")
        else: ax.set_yticklabels([])
        if g_ == ng - 1: ax.set_xlim(radii[0] * 0.8, radii[-1] * 4)
    fig.text(0.5, 0.065, "neighbourhood size r   (r = 1: the subject itself)", ha="center", fontsize=15)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=3.5, marker="o", ms=8, label="real regions (median, IQR band)"),
                        Line2D([], [], color=NULL, lw=2, ls=(0, (4, 3)), label="same-spectrum surrogate")],
               loc="upper left", bbox_to_anchor=(0.07, 0.855), ncol=2, fontsize=13.5)
    fig.text(0.04, 0.015, "Neighbours: Euclidean kNN on L2-normalised target latents, same for every metric. Top-k plans: a lower bound.",
             fontsize=11.5, color=MUTED)
    S.save(fig, out / "P2_neighbourhoods.png")


# ---------------------------------------------------------------------------------------------- P3
def fig_cka_gw(a, maps, out, rows_csv):
    T = []
    for name, d in maps:
        fs = sorted(Path(d).glob("map_shard_*.csv"))
        if not fs: print(f"P3: no map_shard in {d}"); continue
        D = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["src", "tgt", "rel"])
        rel = min(D.rel.unique(), key=lambda r: abs(r - a.rel)); G = D[D.rel == rel].copy()
        G["type"] = [pair_type(x, y) for x, y in zip(G.src, G.tgt)]; G["run"] = name
        T.append(G[["run", "src", "tgt", "type", "cka", "contrast", "ntrT"]])
    if not T: return
    T = pd.concat(T); T.to_csv(out / "P3_cka_vs_gw_by_pair.csv", index=False)
    nm = len(maps); w = 0.84 / nm - 0.035
    rhos = {n_: spearmanr(T[T.run == n_].cka, T[T.run == n_].contrast, nan_policy="omit") for n_, _ in maps if (T.run == n_).any()}
    neg = np.mean([r[0] for r in rhos.values()]) < 0
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "GW and CKA read different things" if neg else "GW shape contrast vs linear similarity (CKA)",
               f"One dot per region pair  ·  x: debiased linear CKA  ·  y: GW shape contrast at ε = {a.rel:g} ε$_c$  ·  Spearman ρ per metric")
    lo = max(T.cka[T.cka > 0].min() * 0.8, 1e-3); hi = T.cka.max() * 1.25
    ylo, yhi = np.nanpercentile(T.contrast, 0.5), np.nanpercentile(T.contrast, 99.5); pad = 0.08 * (yhi - ylo)
    for c, (name, _) in enumerate(maps):
        G = T[T.run == name]
        if G.empty: continue
        ax = fig.add_axes([0.075 + c * (w + 0.035), 0.15, w, 0.58])
        ax.axhline(0, color=INK2, lw=0.8)
        o = G[G.type == "other"]; s_ = G[G.type == "shared sulcus"]; h = G[G.type == "homologue"]
        ax.scatter(o.cka.clip(lo), o.contrast, s=22, color=NULL, alpha=.55, lw=0, label="other pairs")
        ax.scatter(s_.cka.clip(lo), s_.contrast, s=70, facecolor="none", edgecolor=INK, lw=1.6, label="shared sulcus / overlapping boxes")
        ax.scatter(h.cka.clip(lo), h.contrast, s=95, color=colour(name), edgecolor=BG, lw=1.5, zorder=4, label="L↔R homologues")
        # binned median trend
        bins = np.geomspace(lo, hi, 8); mid = np.sqrt(bins[1:] * bins[:-1])
        med = [G.contrast[(G.cka >= b0) & (G.cka < b1)].median() if ((G.cka >= b0) & (G.cka < b1)).sum() >= 5 else np.nan for b0, b1 in zip(bins[:-1], bins[1:])]
        ax.plot(mid, med, color=INK, lw=2.2, alpha=.7, zorder=3)
        r, p = rhos[name]
        ax.text(0.97, 0.96, f"ρ = {r:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=20, weight="bold")
        ax.text(0.97, 0.86, f"p = {p:.1g}  ·  {len(G)} pairs", transform=ax.transAxes, ha="right", va="top", fontsize=12.5, color=INK2)
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(pretty(name), fontsize=18); ax.set_xlabel("CKA (log)", fontsize=14)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        if c == 0: ax.set_ylabel("GW shape contrast")
        else: ax.set_yticklabels([])
        rows_csv.append(dict(figure="P3", run=name, rel=a.rel, spearman=r, p=p, n_pairs=len(G)))
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_[::-1], l_[::-1], loc="lower left", bbox_to_anchor=(0.07, 0.005), ncol=3, fontsize=13)
    S.save(fig, out / "P3_cka_vs_gw.png")


# ---------------------------------------------------------------------------------------------- P4
def typicality(X):
    X = X - X.mean(0); sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)); return D.mean(1)


def fig_typicality(a, out, rows_csv):
    A = Path(a.aligned); names = [b for b in a.typ_regions.split(",") if (A / f"{b}_left.npy").exists() and (A / f"{b}_right.npy").exists()]
    if not names: print("P4: none of --typ-regions found"); return
    N = np.load(A / f"{names[0]}_left.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(1).permutation(N)[: a.n_typ]) if a.n_typ < N else np.arange(N)   # same draw as 217
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "Typicality is shared across hemispheres", f"{len(idx)} subjects binned by typicality decile, left × right  ·  colour = count ÷ count if independent")
    k = len(names); w = min(0.26, 0.84 / k - 0.04)
    for c, b in enumerate(names):
        eL = typicality(region_latents(A, f"{b}_left", idx)); eR = typicality(region_latents(A, f"{b}_right", idx))
        rL, rR = (pd.Series(eL).rank().values - 0.5) / len(idx), (pd.Series(eR).rank().values - 0.5) / len(idx)
        rho = spearmanr(eL, eR)[0]
        hgt = min(w * S.W / S.H, 0.56)
        ax = fig.add_axes([0.07 + c * (w + 0.05), 0.15, hgt * S.H / S.W, hgt])
        B = a.typ_bins; qL = np.minimum((rL * B).astype(int), B - 1); qR = np.minimum((rR * B).astype(int), B - 1)
        C = np.zeros((B, B)); np.add.at(C, (qR, qL), 1)
        Cm = np.where(C >= a.min_cell, C, np.nan) / (len(idx) / B ** 2)          # enrichment over independence
        im = ax.imshow(Cm, origin="lower", cmap=S.DIVERGE.reversed(), norm=TwoSlopeNorm(1, 0, max(3, np.nanmax(Cm))), extent=(0, 1, 0, 1))
        ax.text(0.5 / B, 0.5 / B, f"×{Cm[0, 0]:.1f}", ha="left", va="bottom", fontsize=12, color=INK, transform=ax.transAxes)
        ax.text(1 - 0.5 / B, 1 - 0.5 / B, f"×{Cm[-1, -1]:.1f}", ha="right", va="top", fontsize=12, color=INK, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0.08, 0.92]); ax.set_xticklabels(["typical", "atypical"], fontsize=12.5)
        ax.set_yticks([0.08, 0.92]); ax.set_yticklabels(["typical", "atypical"], fontsize=12.5, rotation=90, va="center"); ax.tick_params(length=0)
        ax.set_title(f"{b}   ρ = {rho:.2f}", fontsize=18); ax.set_xlabel("left", fontsize=14); ax.set_ylabel("right" if c == 0 else "", fontsize=14)

        rows_csv.append(dict(figure="P4", region=b, n=len(idx), spearman=rho, corner_typical=Cm[0, 0], corner_atypical=Cm[-1, -1]))
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cax.tick_params(labelsize=12)
    cb.set_label("× independence", fontsize=13)
    fig.text(0.07, 0.02, f"Rank bins hold ≈ {len(idx) // a.typ_bins ** 2} subjects each; bins with < {a.min_cell} subjects blanked.", fontsize=11.5, color=MUTED)
    S.save(fig, out / "P4_typicality.png")


# ---------------------------------------------------------------------------------------------- P5 / P6
GROUP_COL = {"overlap": GEO, "homologue": ORC, "shared, cross": INK, "other": NULL}
GROUP_LAB = {"overlap": "shared sulcus, same hemisphere (overlapping boxes)", "homologue": "L↔R homologues",
             "shared, cross": "shared sulcus, cross hemisphere", "other": "other pairs"}
GROUPS = ["overlap", "homologue", "shared, cross", "other"]


def group5(x, y):
    t = pair_type(x, y)
    if t == "shared sulcus": return "overlap" if side(x) == side(y) else "shared, cross"
    return t


def spread(vals, sep, top=None):
    """push sorted label positions apart by at least sep (same units), optionally keeping them below top"""
    o = np.argsort(vals); v = np.array(vals, float)[o]
    for i in range(1, len(v)): v[i] = max(v[i], v[i - 1] + sep)
    if top is not None and len(v) and v[-1] > top: v -= v[-1] - top
    out = np.empty_like(v); out[o] = v; return out


def pair_label(x, y):
    if side(x) == side(y) and side(x): return f"{base(x)} ↔ {base(y)} ({side(x)[0].upper()})"
    return f"{short(x)} ↔ {short(y)}"


def load_maps(maps):
    T = []
    for name, d in maps:
        fs = sorted(Path(d).glob("map_shard_*.csv"))
        if not fs: continue
        D = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["src", "tgt", "rel"])
        D["type"] = [group5(x, y) for x, y in zip(D.src, D.tgt)]; D["run"] = name
        T.append(D[["run", "src", "tgt", "type", "rel", "ntrT", "ntrT_surr", "top1", "top1_surr"]])
    return pd.concat(T) if T else None


def fig_retrieval_eps(a, maps, out, rows_csv):
    """P5: n·tr(T) of every region pair, by pair type, at every ε below ε_c, against the surrogates of all pairs"""
    T = load_maps(maps)
    if T is None: return
    ref = a.retrieval_run or next((n_ for n_, _ in maps if metric_of(n_) != "euclidean"), maps[0][0])
    G = T[T.run == ref]; rels = sorted([r for r in G.rel.unique() if r < 1], reverse=True)
    G.to_csv(out / f"P5_retrieval_{ref}_by_pair.csv", index=False)
    groups = GROUPS; cats = groups + ["surrogate"]
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "The same overlapping pairs stand out at every ε",
               f"n·tr(T), one dot per region pair ({G.groupby(['src', 'tgt']).ngroups})  ·  {pretty(ref)}  ·  dashed line: surrogates' 95th percentile")
    nr = len(rels); w = 0.88 / nr - 0.02
    ylo = max(min(G.ntrT.min(), G.ntrT_surr.min()) * 0.7, 0.02); yhi = max(G.ntrT.max(), 3) * 2.2
    rng = np.random.default_rng(0)
    for c, rel in enumerate(rels):
        g = G[G.rel == rel]; ax = fig.add_axes([0.075 + c * (w + 0.02), 0.11, w, 0.62])
        ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
        q95 = g.ntrT_surr.quantile(.95); ax.axhline(q95, color=INK2, lw=1.4, ls=(0, (6, 3)))
        for k_, t in enumerate(cats):
            v = g.ntrT_surr.values if t == "surrogate" else g[g.type == t].ntrT.values
            if not len(v): continue
            xj = k_ + rng.uniform(-0.28, 0.28, len(v))
            if t == "surrogate":
                ax.scatter(xj, v, s=12, facecolor="none", edgecolor=NULL, lw=0.7)
            elif t == "shared, cross":
                ax.scatter(xj, v, s=40, facecolor="none", edgecolor=INK, lw=1.2, zorder=3)
            else:
                ax.scatter(xj, v, s=60 if t != "other" else 14, color=GROUP_COL[t], alpha=.9 if t != "other" else .45, edgecolor=BG, lw=0.8, zorder=3)
            ax.plot([k_ - 0.34, k_ + 0.34], [np.median(v)] * 2, color=INK, lw=2.5, zorder=4)
            rows_csv.append(dict(figure="P5", run=ref, rel=rel, group=t, n=len(v), median=float(np.median(v)), q95_surr=float(q95),
                                 frac_above_q95=float((v > q95).mean())))
        top = g.nlargest(a.n_label, "ntrT")
        if len(top):
            ly = spread(np.log(top.ntrT.values), 0.055 * np.log(yhi / ylo))
            for (_, r), yy in zip(top.iterrows(), ly):
                ax.annotate(f"×{r.ntrT:.0f}", (cats.index(r.type), r.ntrT), xytext=(cats.index(r.type) + 0.45, np.exp(yy)), fontsize=12,
                            va="center", color=INK, weight="semibold", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=0, shrinkB=4))
        ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.minorticks_off()
        yt = [v for v in [0.1, 0.3, 1, 3, 10, 30, 100] if ylo <= v <= yhi]; ax.set_yticks(yt)
        ax.set_yticklabels([f"×{v:g}" for v in yt] if c == 0 else [])
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(["overlap", "homol.", "cross", "other", "surr."], fontsize=11.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.set_title(f"ε = {rel:g} ε$_c$", fontsize=18)
        n_sig = int((g[g.type == "overlap"].ntrT > q95).sum()); n_sh = int((g.type == "overlap").sum())
        ax.text(0.97, 0.03, f"overlap > q95: {n_sig}/{n_sh}", transform=ax.transAxes, fontsize=12, color=INK2, va="bottom", ha="right")
        if c == 0: ax.set_ylabel("n·tr(T)   (×1 = chance)")
    S.save(fig, out / f"P5_retrieval_eps_{ref}.png")


def fig_retrieval_repro(a, maps, out, rows_csv):
    """P6: the top pairs across ε (left) and across runs: other k, other metric, other subject sample (right)"""
    T = load_maps(maps)
    if T is None: return
    ref = a.retrieval_run or next((n_ for n_, _ in maps if metric_of(n_) != "euclidean"), maps[0][0])
    G = T[T.run == ref]; rels = sorted(G.rel.unique(), reverse=True)
    rel0 = min(G.rel.unique(), key=lambda r: abs(r - a.rel))
    top = G[G.rel == rel0].nlargest(a.n_top, "ntrT")[["src", "tgt", "type"]].values.tolist()
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "Is ×20 a lucky draw?",
               f"Left: top {len(top)} pairs at ε = {rel0:g} ε$_c$ followed across ε ({pretty(ref)})  ·  right: n·tr(T) per pair, {ref} vs other runs")
    ax = fig.add_axes([0.075, 0.13, 0.3, 0.58])
    S_ = G.groupby("rel").ntrT_surr.quantile([.05, .95]).unstack()
    ax.fill_between(S_.index, S_[0.05], S_[0.95], color=NULL, alpha=.25, lw=0)
    ax.text(S_.index.min(), S_[0.95].iloc[0] * 1.1, "surrogates 5–95%", fontsize=11.5, color=INK2, ha="right", va="bottom")
    ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
    ends = []
    for x, y, t in top:
        d = G[(G.src == x) & (G.tgt == y)].sort_values("rel", ascending=False)
        ax.plot(d.rel, d.ntrT, color=GROUP_COL[t], lw=2.6, marker="o", ms=7, mec=BG, mew=1.2, alpha=.95)
        ends.append([np.log(d.ntrT.iloc[-1]), pair_label(x, y), d.rel.iloc[-1]])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xlim(max(rels) * 1.3, min(rels) / 1.25); ax.set_xticks(rels); ax.set_xticklabels([f"{r:g}" for r in rels])
    ylo, yhi = ax.get_ylim(); ylo = min(ylo, 0.5)
    yt = [v for v in [0.3, 1, 3, 10, 30, 100] if ylo <= v <= yhi * 1.5]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt])
    ax.set_xlabel("ε / ε$_c$   (finer →)", fontsize=14); ax.set_ylabel("n·tr(T)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_ylim(ylo, yhi)
    ly = spread([e[0] for e in ends], 0.06 * np.log(yhi / ylo), top=np.log(yhi) - 0.03 * np.log(yhi / ylo))
    for (y0, lab, rx), yy in zip(ends, ly):
        ax.annotate(lab, (rx, np.exp(y0)), xytext=(min(rels) / 1.6, np.exp(yy)), fontsize=10.5, va="center", ha="left", color=INK,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3))
    others = [n_ for n_, _ in maps if n_ != ref][:4]
    if others:
        nc = 2 if len(others) > 1 else 1; nrw = int(np.ceil(len(others) / nc)); ph = (0.58 - 0.1 * (nrw - 1)) / nrw; pw = min(0.15, ph * S.H / S.W)
        R0 = G[G.rel == rel0].set_index(["src", "tgt"])
        for k_, o in enumerate(others):
            O = T[(T.run == o)]; ro = min(O.rel.unique(), key=lambda r: abs(r - rel0)); O = O[O.rel == ro].set_index(["src", "tgt"])
            J = R0.join(O[["ntrT"]], rsuffix="_o", how="inner").dropna(subset=["ntrT", "ntrT_o"])
            if J.empty: continue
            ax2 = fig.add_axes([0.62 + (k_ % nc) * (pw + 0.055), 0.13 + (nrw - 1 - k_ // nc) * (ph + 0.1), pw, ph])
            lo_, hi_ = max(min(J.ntrT.min(), J.ntrT_o.min()) * 0.7, 0.02), max(J.ntrT.max(), J.ntrT_o.max()) * 1.6
            ax2.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=0.8, ls=(0, (2, 3)))
            for t in ["other", "shared, cross", "homologue", "overlap"]:
                j = J[J.type == t]
                if t == "shared, cross": ax2.scatter(j.ntrT, j.ntrT_o, s=30, facecolor="none", edgecolor=INK, lw=1, zorder=3)
                else: ax2.scatter(j.ntrT, j.ntrT_o, s=12 if t == "other" else 40, color=GROUP_COL[t], alpha=.45 if t == "other" else .9, lw=0.6, edgecolor=BG, zorder=3)
            tops = [(x, y) for x, y, _ in top if (x, y) in J.index]
            if tops: ax2.scatter(J.loc[tops].ntrT, J.loc[tops].ntrT_o, s=120, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
            rho = spearmanr(J.ntrT, J.ntrT_o)[0]
            keep = J.loc[tops] if tops else J.iloc[:0]
            ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(lo_, hi_); ax2.set_ylim(lo_, hi_); ax2.minorticks_off()
            tk = [v for v in [0.1, 1, 10, 100] if lo_ <= v <= hi_]
            ax2.set_xticks(tk); ax2.set_yticks(tk); ax2.set_xticklabels([f"×{v:g}" for v in tk], fontsize=11); ax2.set_yticklabels([f"×{v:g}" for v in tk], fontsize=11)
            ax2.text(0.04, 0.96, f"ρ = {rho:.2f} (all)", transform=ax2.transAxes, fontsize=13, weight="bold", va="top")
            if len(keep): ax2.text(0.04, 0.8, f"top-{len(keep)} ≥ ×{keep.ntrT_o.min():.0f}", transform=ax2.transAxes, fontsize=11.5, color=INK2, va="top")
            if k_ // nc == nrw - 1: ax2.set_xlabel(f"{ref}", fontsize=12)
            rows_csv.append(dict(figure="P6", ref=ref, other=o, rel=rel0, spearman=rho, n_pairs=len(J),
                                 top_min_other=float(keep.ntrT_o.min()) if len(keep) else np.nan))
            ax2.set_title(f"vs {o}", fontsize=14, loc="left")
    fig.legend(handles=[Line2D([], [], color=GROUP_COL[t], marker="o", ls="", ms=9, label=lab) for t, lab in
                        [("overlap", "overlap (same hemi.)"), ("homologue", "L↔R homologue"), ("other", "other")]]
                       + [Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=8, label="shared sulcus, cross hemi."),
                          Line2D([], [], marker="o", ls="", mfc="none", mec=INK, mew=1.6, ms=13, label=f"top {len(top)} pairs")],
               loc="upper left", bbox_to_anchor=(0.07, 0.86), ncol=5, fontsize=12.5, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.04, 0.015, "Right: one dot per region pair present in both runs, same ε / ε_c; dotted line = identical value; ρ = Spearman over all common pairs.",
             fontsize=11.5, color=MUTED)
    S.save(fig, out / f"P6_retrieval_repro_{ref}.png")


# ----------------------------------------------------------------------------------------------
def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", "."), "aligned_champ"))
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR of a 215 run with saved plans (repeatable, order = column order)")
    ap.add_argument("--map", action="append", default=[], help="NAME=DIR of a 215 map (map_shard_*.csv) for P3; default: the --run dirs")
    ap.add_argument("--rel", type=float, default=0.1)
    ap.add_argument("--plan-pair", default="SFint-FCMant_left:SFint-SR_left,SCall_left:SCall_right",
                    help="pairs X:Y shown in P1 (one figure each)")
    ap.add_argument("--bins", type=int, default=40, help="P1 blocks per side (≥ n/bins subjects per block)")
    ap.add_argument("--vmax", type=float, default=8.0, help="P1 colour range ×1/vmax … ×vmax")
    ap.add_argument("--radii", default="1,2,5,10,20,50")
    ap.add_argument("--typ-regions", default="SCall,FPO-SCu-ScCal,SFint-SR")
    ap.add_argument("--n-typ", type=int, default=4000); ap.add_argument("--typ-bins", type=int, default=10); ap.add_argument("--min-cell", type=int, default=5)
    ap.add_argument("--csv-blocks", action="store_true", help="also write P1 block values (aggregates) to the CSV")
    ap.add_argument("--retrieval-run", default=None, help="P5/P6 reference run (default: first non-Euclidean --map)")
    ap.add_argument("--n-label", type=int, default=3); ap.add_argument("--n-top", type=int, default=8)
    ap.add_argument("--only", default="P1,P2,P3,P4,P5,P6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S.apply()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [tuple(r.split("=", 1)) for r in a.run]
    maps = [tuple(m.split("=", 1)) for m in a.map] or runs
    rows = []; only = set(a.only.split(","))
    if "P1" in only and runs: fig_plans(a, runs, out, rows)
    if "P2" in only and runs: fig_neighbourhoods(a, runs, out, rows)
    if "P3" in only and maps: fig_cka_gw(a, maps, out, rows)
    if "P4" in only: fig_typicality(a, out, rows)
    if "P5" in only and maps: fig_retrieval_eps(a, maps, out, rows)
    if "P6" in only and maps: fig_retrieval_repro(a, maps, out, rows)
    pd.DataFrame(rows).to_csv(out / "figures_222_numbers.csv", index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
