#!/usr/bin/env python
"""
224 -- The GW-on-Champollion story, slide by slide (16:9 PNGs). Exhaustive on purpose: pick what you need.
AGGREGATED OUTPUTS ONLY (pair-level statistics; transport plans pooled into blocks of ≥ n/bins subjects).

    0_cka/            linear CKA: 30×30 matrix, by pair type, per homologue
    1_<euc run>/      for each main metric (Euclidean, geodesic, ORC — one folder each), in the same order:
    2_<geo run>/        scores      a contrast matrix · b residual after the common factor · c contrast by pair type, every ε
    3_<orc run>/                    d contrast vs ε by type · e homologue residuals · f fake-homologue null
                        retrieval   g n·tr(T) matrix · h n·tr(T) by type, every ε · i rank of the true partner
                                    j neighbourhood recovery
                        transport   k one pair, every ε (real / surrogate) · l showcase pairs at --rel
    4_compare/        the main metrics side by side (matrices, null, rank curves, transport, pair-by-pair scatter)
    5_orc_iterations/ ORC Ricci-flow steps side by side (transport per step × ε, scores vs steps)
    6_k_sweep/        k for geodesic and ORC (scores vs k, transport vs k)
    numbers_224.csv   every number shown

Runs: --run NAME=DIR[,DIR...]  (a 215 output dir; several dirs are merged, e.g. a full map + a subset run with saved plans).
NAME encodes the metric: euc | geo<k> | orc<k> [ _it<steps> ] [ _<tag> ], e.g. euc, geo20, orc20, orc20_it3, geo20_s2.
Main runs default to euc / geo20 / orc20 (override with --main). Plans are found in <dir>/plans/<X>__<Y>/rel<ε>.npz.

    python 224_gw_story.py --out $METHODO/story_224 \
        --run euc=$METHODO/champ_map_pcperm,$METHODO/champ_sub_euc \
        --run geo20=$METHODO/champ_map_geo20,$METHODO/champ_sub_geo20 \
        --run orc20=$METHODO/champ_map_orc20,$METHODO/champ_sub_orc20 \
        --run orc20_it3=$METHODO/champ_sub_orc20_it3 --run orc20_it5=$METHODO/champ_sub_orc20_it5 \
        --run geo10=... --run geo40=... --run orc10=... --run orc40=...
CPU, a few minutes. Needs champ_style.py and 222_champ_figs_jz.py next to it.
"""
import argparse, importlib.util, re, sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import champ_style as S
    _spec = importlib.util.spec_from_file_location("m222", HERE / "222_champ_figs_jz.py"); m222 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m222)
except (ImportError, FileNotFoundError) as e:
    sys.exit(f"champ_style.py and 222_champ_figs_jz.py must sit next to this script ({e})")
from champ_style import EUC, GEO, ORC, INK, INK2, MUTED, NULL, GRID, BG, BAND

base, side, l2 = m222.base, m222.side, m222.l2
from matplotlib.colors import LinearSegmentedColormap
DARK = LinearSegmentedColormap.from_list("dark", ["#0b1420", "#13284a", "#1c4f8f", "#2a78d6", "#7fb2ee", "#e8f2fd"])
PLAN_CMAP = DARK
GROUPS = ["overlap", "homologue", "shared, cross", "other"]
GLAB = {"overlap": "shared sulcus, same hemisphere", "homologue": "L↔R homologues", "shared, cross": "shared sulcus, cross hemisphere", "other": "other pairs"}
GSHORT = {"overlap": "overlap", "homologue": "homol.", "shared, cross": "cross", "other": "other", "surrogate": "surr."}
GCOL = {"overlap": GEO, "homologue": ORC, "shared, cross": INK, "other": NULL}
NAME_RE = re.compile(r"^(?P<fam>euc|geo|orc)(?P<k>\d+)?(?:_it(?P<it>\d+))?(?:_(?P<tag>[A-Za-z]\w*))?$")
FAM_COL = {"euc": EUC, "geo": GEO, "orc": ORC}
FAM_NAME = {"euc": "Euclidean", "geo": "Geodesic", "orc": "Ollivier–Ricci"}
ROWS = []                                                              # numbers_224.csv
SCORE = "contrast"                                                     # column used as the GW score (--score)
SL = "contrast"; ST = "GW shape contrast"                              # its axis label / title wording
FIT_ALL = False
BASELINE = "local"                                                     # "local" (simple, default) | "rank1" (common factor a_X a_Y)


# ============================================================================================ runs
class Run:
    def __init__(self, name, dirs):
        self.name, self.dirs = name, [Path(d) for d in dirs]
        m = NAME_RE.match(name)
        self.fam = m["fam"] if m else "geo"; self.k = int(m["k"]) if m and m["k"] else None
        self.it = int(m["it"]) if m and m["it"] else 1; self.tag = m["tag"] if m else name
        self.col = FAM_COL.get(self.fam, INK2)
        fs = [f for d in self.dirs for f in sorted(d.glob("map_shard_*.csv"))]
        self.map = (pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["src", "tgt", "rel"]) if fs else pd.DataFrame())
        if len(self.map):
            self.map["type"] = [group5(x, y) for x, y in zip(self.map.src, self.map.tgt)]
            if SCORE != "contrast":
                if SCORE not in self.map: sys.exit(f"--score {SCORE}: no such column in {self.dirs}")
                self.map["contrast"] = self.map[SCORE]                         # every figure below reads 'contrast'
        self.plans = {}                                                # (x, y) -> {rel: (real, surr, positions)}
        for d in self.dirs:
            pos = d / "sujets_positions.npy"
            for pdir in sorted((d / "plans").glob("*__*")):
                x, y = pdir.name.split("__")
                for f in pdir.glob("rel*.npz"):
                    rel = float(f.stem[3:]); self.plans.setdefault((x, y), {}).setdefault(rel, (f, pdir / "substitut0" / f.name, pos))

    @property
    def label(self):
        s = FAM_NAME.get(self.fam, self.name)
        if self.k and self.fam != "euc": s += f" k={self.k}"
        if self.it > 1: s += f" · {self.it} flow steps"
        if self.tag: s += f" · {self.tag}"
        return s

    def at(self, rel):
        if not len(self.map): return pd.DataFrame(columns=["src", "tgt", "rel", "type", "contrast", "cka", "ntrT", "ntrT_surr"])
        r = min(self.map.rel.unique(), key=lambda v: abs(v - rel)); return self.map[self.map.rel == r]

    def rels(self, below=1.0):
        return sorted([r for r in self.map.rel.unique() if r < below], reverse=True) if len(self.map) else []


def group5(x, y):
    if base(x) == base(y): return "homologue"
    if set(base(x).split("-")) & set(base(y).split("-")): return "overlap" if side(x) == side(y) else "shared, cross"
    return "other"


def pair_label(x, y):
    if side(x) == side(y): return f"{base(x)} ↔ {base(y)} ({side(x)[0].upper()})"
    if base(x) == base(y): return f"{base(x)} L↔R"
    return f"{base(x)} ({side(x)[0].upper()}) ↔ {base(y)} ({side(y)[0].upper()})"


def region_order(runs):
    regs = sorted({r for R in runs if len(R.map) for r in pd.concat([R.map.src, R.map.tgt]).unique()})
    bases_ = sorted({base(r) for r in regs})
    return [f"{b}_left" for b in bases_ if f"{b}_left" in regs] + [f"{b}_right" for b in bases_ if f"{b}_right" in regs]


def matrix(df, col, order):
    ix = {r: i for i, r in enumerate(order)}; M = np.full((len(order), len(order)), np.nan)
    for x, y, v in zip(df.src, df.tgt, df[col]):
        if x in ix and y in ix: M[ix[x], ix[y]] = M[ix[y], ix[x]] = v
    return M


def new_fig(title, sub=None):
    fig = plt.figure(figsize=(S.W, S.H)); S.headline(fig, title, sub); return fig


NOTES = False


def foot(fig, txt):
    if NOTES: fig.text(0.04, 0.015, txt, fontsize=11, color=MUTED)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True); S.save(fig, path)


# ============================================================================================ common factor & nulls
def fit_rank1(M, mask, iters=200):
    a = np.sqrt(np.nanmean(np.abs(M[mask]))) * np.ones(len(M))
    W = mask & ~np.isnan(M); Mz = np.where(W, M, 0)
    for _ in range(iters):
        den = (W * a[None, :] ** 2).sum(1); a = np.where(den > 0, (Mz * a[None, :]).sum(1) / np.maximum(den, 1e-30), a)
    return a


def residual_analysis(df, order, n_null=1000, seed=0):
    if BASELINE == "local": return residual_local(df, order, n_null, seed)
    return residual_rank1(df, order, n_null, seed)


def residual_local(df, order, n_null=1000, seed=0):
    """SIMPLE TEST (default).
    statistic: mean score over the true left↔right homologues;
    null: same mean over random left↔right pairings of the same regions (derangements, never a true homologue);
    per pair: residual = s(X,Y) − b(X,Y), b = ½[mean_{Y'≠Y, same side as Y} s(X,Y') + mean_{X'≠X, same side as X} s(X',Y)]
    (how much X matches Y better than X matches Y's other neighbours-in-hemisphere, and vice versa)."""
    M = matrix(df, "contrast", order); n = len(order); ix = {r: i for i, r in enumerate(order)}
    bases_ = [base(r) for r in order if side(r) == "left" and f"{base(r)}_right" in ix]
    L = np.array([ix[f"{b}_left"] for b in bases_]); R = np.array([ix[f"{b}_right"] for b in bases_])
    sd = np.array([side(r) for r in order]); B = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            rj = [k for k in range(n) if sd[k] == sd[j] and k not in (i, j)]; ri = [k for k in range(n) if sd[k] == sd[i] and k not in (i, j)]
            B[i, j] = 0.5 * (np.nanmean(M[i, rj]) + np.nanmean(M[ri, j]))
    full = M - B; np.fill_diagonal(full, np.nan)
    hres = full[L, R]; obs = np.nanmean(M[L, R])
    g = np.random.default_rng(seed); null = []
    for _ in range(n_null):
        p = g.permutation(len(L))
        while np.any(p == np.arange(len(L))): p = g.permutation(len(L))
        v = M[L, R[p]]
        if np.isfinite(v).sum() >= max(len(L) - 3, 3): null.append(np.nanmean(v))
    null = np.array(null)
    if len(null) < max(20, n_null // 2): null = np.array([])
    ok = ~np.isnan(hres)
    return dict(bases=bases_, hres=hres, a=None, resid=full, null=null, obs=obs, stat="mean score of homologues",
                p=(1 + (null >= obs).sum()) / (1 + len(null)) if len(null) else np.nan,
                n_pos=int((hres[ok] > 0).sum()), n_hom=int(ok.sum()),
                p_sign=binomtest(int((hres[ok] > 0).sum()), int(ok.sum()), 0.5, alternative="greater").pvalue if ok.sum() else np.nan)


def residual_rank1(df, order, n_null=1000, seed=0):
    """rank-1 common factor c(X,Y) ≈ a_X a_Y fitted on non-homologue pairs; homologue residuals; derangement null
    (fake homologue pairings, factor refitted without them each time)"""
    M = matrix(df, "contrast", order); n = len(order); ix = {r: i for i, r in enumerate(order)}
    bases_ = [base(r) for r in order if side(r) == "left" and f"{base(r)}_right" in ix]
    L = np.array([ix[f"{b}_left"] for b in bases_]); R = np.array([ix[f"{b}_right"] for b in bases_])
    off = ~np.eye(n, dtype=bool)
    if not FIT_ALL:                                                    # the common factor describes unrelated regions:
        for i in range(n):                                             # leave out every pair sharing a sulcus (overlapping boxes)
            for j in range(n):
                if i != j and base(order[i]) != base(order[j]) and set(base(order[i]).split("-")) & set(base(order[j]).split("-")): off[i, j] = False

    def resid_for(Li, Ri):
        mask = off.copy(); mask[Li, Ri] = mask[Ri, Li] = False
        a = fit_rank1(M, mask); return M[Li, Ri] - a[Li] * a[Ri], a
    hres, a = resid_for(L, R)
    full = M - np.outer(a, a); np.fill_diagonal(full, np.nan)
    g = np.random.default_rng(seed); null = []
    for _ in range(n_null):
        p = g.permutation(len(L))
        while np.any(p == np.arange(len(L))): p = g.permutation(len(L))
        r, _ = resid_for(L, R[p])
        if np.isfinite(r).sum() >= max(len(L) - 3, 3): null.append(np.nanmean(r))      # skip draws whose fake pairs are mostly absent
    null = np.array(null); obs = np.nanmean(hres)
    if len(null) < max(20, n_null // 2): null = np.array([])                                           # subset map: too few fake pairs to build the null
    ok = ~np.isnan(hres)
    return dict(bases=bases_, hres=hres, a=a, resid=full, null=null, obs=obs, p=(1 + (null >= obs).sum()) / (1 + len(null)) if len(null) else np.nan,
                n_pos=int((hres[ok] > 0).sum()), n_hom=int(ok.sum()),
                p_sign=binomtest(int((hres[ok] > 0).sum()), int(ok.sum()), 0.5, alternative="greater").pvalue if ok.sum() else np.nan)


# ============================================================================================ plans
_ORDER = {}


def subject_order(aligned, target, posfile):
    key = (target, str(posfile))
    if key not in _ORDER:
        idx = np.load(posfile); _ORDER[key] = m222.spectral_order(l2(m222.region_latents(aligned, target, idx)))
    return _ORDER[key]


def plan_blocks(f, aligned, target, posfile, bins):
    n, P, dg = m222.load_plan(f)
    Mb = m222.blocks(P, n, subject_order(aligned, target, posfile), bins)
    band = float(np.mean([Mb[i, j] for i in range(bins) for j in range(bins) if abs(i - j) <= 1]))
    return Mb, float(dg.sum()), band


def partner_rank(f):
    """rank of the true partner (the subject itself) in each row of n·T; inf = beyond the saved top-k"""
    n, P, dg = m222.load_plan(f)
    if "dense" in P:
        M = P["dense"]; return n, 1 + (M > dg[:, None]).sum(1).astype(float)
    idx, val = P["idx"], P["val"]; self_in = idx == np.arange(n)[:, None]
    rank = 1 + ((val > dg[:, None]) & ~self_in).sum(1).astype(float)
    rank[(~self_in.any(1)) & (dg < val[:, -1])] = np.inf
    return n, rank


# ============================================================================================ drawing blocks
def draw_matrix(ax, M, order, cmap, norm, outline=True, labels=True, fs=9.5):
    n = len(order); nl = sum(side(r) == "left" for r in order)
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_facecolor("#ecebe7")
    for v in (nl - 0.5,): ax.axhline(v, color=INK, lw=1.4); ax.axvline(v, color=INK, lw=1.4)
    if outline:
        ix = {r: i for i, r in enumerate(order)}
        for r in order:
            if side(r) == "left" and f"{base(r)}_right" in ix:
                i, j = ix[r], ix[f"{base(r)}_right"]
                for (u, v) in ((i, j), (j, i)): ax.add_patch(Rectangle((v - 0.5, u - 0.5), 1, 1, fill=False, ec=ORC, lw=1.6))
    if labels:
        lab = [base(r) for r in order]
        ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=fs); ax.set_xticks([])
        ax.text(nl / 2 - 0.5, -1.2, "LEFT", ha="center", fontsize=12, weight="bold", color=INK2)
        ax.text(nl + (n - nl) / 2 - 0.5, -1.2, "RIGHT", ha="center", fontsize=12, weight="bold", color=INK2)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)


def colorbar(fig, rect, cmap, norm, label, ticks=None, ticklabels=None):
    cax = fig.add_axes(rect); cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax); cb.outline.set_visible(False)
    cax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if ticks is not None: cb.set_ticks(ticks)
    if ticklabels is not None: cb.set_ticklabels(ticklabels)
    cax.tick_params(labelsize=11.5); cb.set_label(label, fontsize=12.5); return cb


def strips(ax, df, col, surr=None, logy=False, seed=0, med=True):
    rng = np.random.default_rng(seed); cats = GROUPS + (["surrogate"] if surr is not None else [])
    for k, t in enumerate(cats):
        v = (surr if t == "surrogate" else df[df.type == t][col]).dropna().values
        if not len(v): continue
        xj = k + rng.uniform(-0.28, 0.28, len(v))
        if t == "surrogate": ax.scatter(xj, v, s=11, facecolor="none", edgecolor=NULL, lw=0.7)
        elif t == "shared, cross": ax.scatter(xj, v, s=36, facecolor="none", edgecolor=INK, lw=1.1, zorder=3)
        else: ax.scatter(xj, v, s=55 if t != "other" else 13, color=GCOL[t], alpha=.9 if t != "other" else .45, edgecolor=BG, lw=0.7, zorder=3)
        if med: ax.plot([k - 0.34, k + 0.34], [np.median(v)] * 2, color=INK, lw=2.5, zorder=4)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([GSHORT[c] for c in cats], fontsize=11.5, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(cats) - 0.4)
    if logy: ax.set_yscale("log"); ax.minorticks_off()
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)


def type_legend(fig, y=0.855, surrogate=False):
    h = [Line2D([], [], marker="o", ls="", color=GCOL[t], ms=9, label=GLAB[t]) if t != "shared, cross" else
         Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=8, label=GLAB[t]) for t in GROUPS]
    if surrogate: h.append(Line2D([], [], marker="o", ls="", mfc="none", mec=NULL, ms=7, label="same-spectrum surrogates"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.04, y), ncol=len(h), fontsize=12, handletextpad=0.3, columnspacing=1.1)


def plan_panels(fig, cells, aligned, bins, vmax, top=0.73, bottom=0.08, left=0.17, right=0.9, row_labels=None, stats=True, gapy=0.075):
    """cells: list of rows, each a list of (title, file or None, target, posfile, is_surrogate). Square panels, shared colour scale."""
    nr, nc = len(cells), max(len(r) for r in cells)
    gapx = 0.018
    side_h = min((top - bottom - gapy * (nr - 1)) / nr, ((right - left) - gapx * (nc - 1)) / nc * S.W / S.H)
    side_w = side_h * S.H / S.W
    x0 = left + ((right - left) - (nc * side_w + (nc - 1) * gapx)) / 2
    norm = LogNorm(vmin=1 / vmax, vmax=vmax); out = []
    for i, row in enumerate(cells):
        y = top - (i + 1) * side_h - i * gapy
        if row_labels: fig.text(x0 - 0.012, y + side_h / 2, row_labels[i], ha="right", va="center", fontsize=14, color=INK if i == 0 else INK2)
        for j, (title, f, target, pos, surr) in enumerate(row):
            if title is None and f is None and target is None: continue          # padding cell
            ax = fig.add_axes([x0 + j * (side_w + gapx), y, side_w, side_h]); ax.set_xticks([]); ax.set_yticks([])
            if title: ax.set_title(title.replace(" ↔ ", "\n↔ ") if (nc > 3 and len(title) > 18) else title, fontsize=12.5 if nc > 3 else 15, loc="left", pad=20 if stats else 6)
            if f is None or not Path(f).exists():
                ax.text(0.5, 0.5, "not saved", ha="center", va="center", color=MUTED, fontsize=12, transform=ax.transAxes)
                for sp in ax.spines.values(): sp.set_color(GRID)
                continue
            Mb, nt, band = plan_blocks(f, aligned, target, pos, bins)
            ax.imshow(np.clip(Mb, 1 / vmax, vmax), cmap=PLAN_CMAP, norm=norm, interpolation="nearest")
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(NULL if surr else INK2); sp.set_linewidth(1)
            if stats: ax.text(0, 1.015, f"n·tr(T) ×{nt:.1f} · band ×{band:.1f}" if nc <= 3 else f"tr ×{nt:.1f} · band ×{band:.1f}", transform=ax.transAxes,
                              fontsize=11 if nc <= 3 else 9.5, color=INK2 if surr else INK, va="bottom")
            out.append(dict(title=title, file=str(f), surrogate=surr, ntrT=nt, band=band))
    colorbar(fig, [min(x0 + nc * (side_w + gapx) + 0.01, 0.95), bottom + 0.05, 0.011, top - bottom - 0.1], PLAN_CMAP, norm,
             "plan mass ÷ chance", [1 / vmax, 1, vmax], [f"×1/{vmax:g}", "×1", f"×{vmax:g}"])
    return out


TRANSPORT_FOOT = ("Rows: subjects in the source region; columns: the same subjects in the target region, both ordered along the target's manifold\n"
                  "(spectral order) and pooled in blocks. Diagonal = each subject sent to itself or its neighbourhood. Colour shared by all panels.")


# ============================================================================================ 0. CKA
def fig_cka(runs, order, out, rel):
    R = max(runs, key=lambda r: len(r.map)); df = R.at(rel)
    if "cka" not in df: return
    M = matrix(df, "cka", order)
    fig = new_fig("Linear similarity (CKA) between regions", "30 regions  ·  orange: L↔R homologues")
    ax = fig.add_axes([0.16, 0.035, 0.6, 0.74])
    norm = PowerNorm(0.5, vmin=0, vmax=np.nanmax(M)); draw_matrix(ax, M, order, S.BLUES, norm, fs=9.5)
    colorbar(fig, [0.8, 0.15, 0.012, 0.5], S.BLUES, norm, "CKA")
    savefig(fig, out / "0_cka" / "a_cka_matrix.png")

    fig = new_fig("CKA follows anatomy", "one dot per region pair")
    ax = fig.add_axes([0.1, 0.13, 0.55, 0.6]); strips(ax, df, "cka", logy=True)
    ax.set_ylabel("CKA (log)")
    meds = df.groupby("type").cka.median()
    for k, t in enumerate(GROUPS):
        if t in meds: ax.text(k, meds[t] * 1.25, f"{meds[t]:.2f}", ha="center", fontsize=13, weight="bold")
    p = mannwhitneyu(df[df.type == "homologue"].cka, df[df.type == "other"].cka, alternative="greater").pvalue if (df.type == "homologue").any() else np.nan
    fig.text(0.7, 0.6, "homologues vs other pairs", fontsize=15, weight="semibold")
    fig.text(0.7, 0.55, f"Mann–Whitney p = {p:.1g}", fontsize=14, color=INK2)
    fig.text(0.7, 0.45, "Overlapping boxes share voxels:\nhigh CKA there is expected.", fontsize=13.5, color=INK2, va="top", linespacing=1.35)
    savefig(fig, out / "0_cka" / "b_cka_by_type.png")
    for t in GROUPS: ROWS.append(dict(fig="0b", run=R.name, group=t, median_cka=float(meds.get(t, np.nan))))

    h = df[df.type == "homologue"].assign(b=lambda d: [base(x) for x in d.src]).sort_values("cka")
    fig = new_fig("CKA of each left ↔ right homologue pair", "dotted: median of other pairs")
    ax = fig.add_axes([0.3, 0.1, 0.62, 0.66]); y = np.arange(len(h))
    ax.axvline(df[df.type == "other"].cka.median(), color=INK2, lw=1, ls=(0, (2, 3)))
    ax.hlines(y, 0, h.cka, color=ORC, lw=4, alpha=.3); ax.plot(h.cka, y, "o", color=ORC, ms=13, mec=BG, mew=2)
    for yi, v in zip(y, h.cka): ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=12.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(h.b, fontsize=14); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, h.cka.max() * 1.2); ax.set_xlabel("CKA")
    savefig(fig, out / "0_cka" / "c_cka_homologues.png")


# ============================================================================================ 1-3. one metric
def contrast_lim(runs, rel):
    v = np.concatenate([np.abs(R.at(rel).contrast.dropna().values) for R in runs if len(R.map) and "contrast" in R.map])
    return float(np.percentile(v, 98)) if len(v) else 0.01


def fig_metric(R, order, out, a, vlim, showcase, ra):
    d = out / a.folder[R.name]; rel = a.rel; df = R.at(rel); rels = R.rels()
    tag = f"{R.label}  ·  ε = {rel:g} ε$_c$"
    if not len(df): return
    # a / b matrices
    norm = TwoSlopeNorm(0, -vlim, vlim)
    for key, M, ttl in [("a_contrast_matrix", matrix(df, "contrast", order), f"{ST} between regions"),
                        ("b_residual_matrix", ra["resid"], "GW score above the baseline of its two regions")]:
        fig = new_fig(f"{ttl} — {R.label}", f"ε = {rel:g} ε$_c$  ·  orange: homologues")
        ax = fig.add_axes([0.16, 0.035, 0.6, 0.74]); draw_matrix(ax, M, order, S.DIVERGE.reversed(), norm, fs=9.5)
        colorbar(fig, [0.8, 0.15, 0.012, 0.5], S.DIVERGE.reversed(), norm, SL)
        savefig(fig, d / f"{key}.png")
    # c contrast by type every ε
    fig = new_fig(f"Shape contrast by pair type — {R.label}", "one dot per region pair")
    w = 0.86 / max(len(rels), 1) - 0.02
    lo, hi = np.nanpercentile(R.map[R.map.rel < 1].contrast, [0.5, 99.5]); pad = 0.1 * (hi - lo); hi += 2 * pad
    for c, r_ in enumerate(rels):
        ax = fig.add_axes([0.08 + c * (w + 0.02), 0.13, w, 0.58]); g = R.at(r_)
        ax.axhline(0, color=INK2, lw=0.8); strips(ax, g, "contrast"); ax.set_ylim(lo - pad, hi + pad); ax.set_title(f"ε = {r_:g} ε$_c$", fontsize=16)
        if c: ax.set_yticklabels([])
        else: ax.set_ylabel(SL)
        p = mannwhitneyu(g[g.type == "homologue"].contrast, g[g.type == "other"].contrast, alternative="greater").pvalue if (g.type == "homologue").sum() > 2 else np.nan
        ax.text(0.97, 0.97, f"homol. > other\np = {p:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=INK2)
        ROWS.append(dict(fig="1c", run=R.name, rel=r_, mw_p_homol_vs_other=p, **{f"med_{t}": float(g[g.type == t].contrast.median()) for t in GROUPS}))
    type_legend(fig); savefig(fig, d / "c_contrast_by_type.png")
    # d score vs ε: per type (left) and difference from unrelated pairs (right)
    fig_score_vs_eps(R, d / "d_contrast_vs_eps.png", order)
    # e homologue residuals
    fig_homologue_residuals(R, ra, rel, d / "e_homologue_residuals.png")
    # f fake-homologue null
    if len(ra["null"]) == 0:
        print(f"  {R.name}: map covers too few pairs for the fake-homologue null (subset run) — figure f skipped, sign test only")
    else:
      fig = new_fig(f"True vs fake homologues — {R.label}", f"mean score of the 15 homologues vs {len(ra['null'])} random L↔R pairings")
      ax = fig.add_axes([0.1, 0.14, 0.62, 0.56])
      ax.hist(ra["null"], bins=40, color=NULL, alpha=.7, edgecolor=BG)
      ax.axvline(ra["obs"], color=R.col, lw=4); ax.axvline(np.percentile(ra["null"], 95), color=INK2, lw=1.2, ls=(0, (4, 3)))
      ax.text(ra["obs"], ax.get_ylim()[1] * 0.95, "  true homologues", color=INK, fontsize=14, weight="bold", va="top")
      ax.text(np.percentile(ra["null"], 95), ax.get_ylim()[1] * 0.75, "  95th pct", color=INK2, fontsize=12, va="top")
      ax.set_xlabel("mean homologue residual"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
      fig.text(0.76, 0.5, fmt_p(ra['p'], len(ra['null'])), fontsize=32, weight="bold"); fig.text(0.76, 0.43, "fake-homologue null", fontsize=14, color=INK2)
      savefig(fig, d / "f_fake_homologue_null.png")
    ROWS.append(dict(fig="1e", run=R.name, rel=rel, n_pos=ra["n_pos"], n_hom=ra["n_hom"], p_sign=ra["p_sign"], obs=ra["obs"],
                     null_med=float(np.median(ra["null"])) if len(ra["null"]) else np.nan, null_q95=float(np.percentile(ra["null"], 95)) if len(ra["null"]) else np.nan, p_fake=ra["p"]))
    # g n·tr(T) matrix
    if "ntrT" in df:
        M = matrix(df, "ntrT", order); vm = max(np.nanpercentile(M, 99), 4)
        nrm = LogNorm(1, vm)
        fig = new_fig(f"Same-subject retrieval between regions — {R.label}", f"n·tr(T), ×1 = chance  ·  ε = {rel:g} ε$_c$")
        ax = fig.add_axes([0.16, 0.035, 0.6, 0.74]); draw_matrix(ax, np.clip(M, 1, vm), order, S.BLUES, nrm, fs=9.5)
        tk = [v for v in [1, 2, 3, 5, 10, 20, 30, 50, 100] if v <= vm]
        colorbar(fig, [0.8, 0.15, 0.012, 0.5], S.BLUES, nrm, "n·tr(T)  (≤ ×1 shown white)", tk, [f"×{v:g}" for v in tk])
        savefig(fig, d / "g_retrieval_matrix.png")
        # h n·tr(T) by type every ε
        fig = new_fig(f"Same-subject retrieval by pair type — {R.label}", "n·tr(T) per region pair  ·  dashed: surrogate q95")
        ylo = max(min(R.map[R.map.rel < 1].ntrT.min(), R.map.ntrT_surr.min()) * 0.7, 0.02); yhi = max(R.map.ntrT.max(), 3) * 2
        for c, r_ in enumerate(rels):
            g = R.at(r_); ax = fig.add_axes([0.08 + c * (w + 0.02), 0.13, w, 0.58])
            ax.axhline(1, color=INK2, lw=0.8, ls=(0, (2, 3))); q95 = g.ntrT_surr.quantile(.95); ax.axhline(q95, color=INK2, lw=1.3, ls=(0, (6, 3)))
            strips(ax, g, "ntrT", surr=g.ntrT_surr, logy=True); ax.set_ylim(ylo, yhi); ax.set_title(f"ε = {r_:g} ε$_c$", fontsize=16)
            yt = [v for v in [0.1, 0.3, 1, 3, 10, 30, 100] if ylo <= v <= yhi]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt] if c == 0 else [])
            for t in ["overlap", "homologue"]:
                v = g[g.type == t].ntrT
                ROWS.append(dict(fig="1h", run=R.name, rel=r_, group=t, median=float(v.median()), frac_above_q95=float((v > q95).mean())))
            nh = int((g[g.type == "homologue"].ntrT > q95).sum()); ntot = int((g.type == "homologue").sum())
            ax.text(0.97, 0.03, f"homol. > q95: {nh}/{ntot}", transform=ax.transAxes, ha="right", fontsize=11.5, color=INK2)
            if c == 0: ax.set_ylabel("n·tr(T)")
        type_legend(fig, surrogate=True); savefig(fig, d / "h_retrieval_by_type.png")
    # i rank of true partner / j neighbourhoods (need plans)
    pairs = [p for p in R.plans if rel_close(R.plans[p], rel)]
    if pairs:
        rr = [1, 2, 5, 10, 20, 50]; recs = []
        for (x, y) in pairs:
            f, fs, pos = rel_close(R.plans[(x, y)], rel)
            for ff, kind in [(f, "real"), (fs, "surrogate")]:
                if not Path(ff).exists(): continue
                n, rk = partner_rank(ff)
                for r in rr: recs.append(dict(src=x, tgt=y, type=group5(x, y), kind=kind, r=r, enr=float((rk <= r).mean() / (r / n))))
        Rk = pd.DataFrame(recs); Rk.to_csv(d / "i_partner_rank_by_pair.csv", index=False)
        fig = new_fig(f"Where does the true partner land? — {R.label}",
                      f"P(rank ≤ r) ÷ chance  ·  ε = {rel:g} ε$_c$")
        gs = [t for t in GROUPS if (Rk.type == t).any()]; w2 = 0.86 / len(gs) - 0.03
        ymax = max(Rk.enr.max() * 1.4, 3)
        for c, t in enumerate(gs):
            ax = fig.add_axes([0.08 + c * (w2 + 0.03), 0.13, w2, 0.56]); ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
            for kind in ["surrogate", "real"]:
                q = Rk[(Rk.type == t) & (Rk.kind == kind)].groupby("r").enr.quantile([.25, .5, .75]).unstack()
                if q.empty: continue
                c_ = R.col if kind == "real" else NULL
                ax.fill_between(q.index, q[0.25], q[0.75], color=c_, alpha=.15, lw=0)
                ax.plot(q.index, q[0.5], color=c_, lw=3.5 if kind == "real" else 2, marker="o" if kind == "real" else None, ms=8, mec=BG,
                        ls="-" if kind == "real" else (0, (4, 3)))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.5, ymax); ax.set_xticks(rr); ax.set_xticklabels(rr, fontsize=12)
            yt = [v for v in [0.5, 1, 2, 5, 10, 20, 50, 100] if v <= ymax]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt] if c == 0 else [])
            ax.set_title(f"{GLAB[t]} ({Rk[Rk.type == t].groupby(['src', 'tgt']).ngroups})", fontsize=14); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        fig.text(0.5, 0.05, "r (top-r of the subject's row in the plan)", ha="center", fontsize=14)
        fig.legend(handles=[Line2D([], [], color=R.col, lw=3.5, marker="o", label="real regions"), Line2D([], [], color=NULL, lw=2, ls=(0, (4, 3)), label="surrogate")],
                   loc="upper left", bbox_to_anchor=(0.04, 0.855), ncol=2, fontsize=13)
        savefig(fig, d / "i_partner_rank.png")
        neighbourhood_fig(R, pairs, rel, a, d / "j_neighbourhoods.png")
    # k / l transport
    for (x, y) in showcase:
        if (x, y) not in R.plans: continue
        rs = sorted(R.plans[(x, y)], reverse=True)
        cells = [[(f"ε = {r_:g} ε$_c$", R.plans[(x, y)][r_][0], y, R.plans[(x, y)][r_][2], False) for r_ in rs],
                 [(None, R.plans[(x, y)][r_][1], y, R.plans[(x, y)][r_][2], True) for r_ in rs]]
        fig = new_fig(f"GW transport plan across ε — {R.label}", f"{pair_label(x, y)}  ·  {group_name(x, y)}")
        out_ = plan_panels(fig, cells, a.aligned, a.bins, a.vmax, row_labels=["real\nregions", "same-spectrum\nsurrogate"])
        foot(fig, TRANSPORT_FOOT); savefig(fig, d / f"k_transport_eps_{slug(x, y)}.png")
        for o in out_: ROWS.append(dict(fig="1k", run=R.name, pair=pair_label(x, y), **o))
    sc = [(x, y) for (x, y) in showcase if (x, y) in R.plans and rel_close(R.plans[(x, y)], rel)]
    if sc:
        cells = [[(pair_label(x, y), rel_close(R.plans[(x, y)], rel)[0], y, rel_close(R.plans[(x, y)], rel)[2], False) for x, y in sc],
                 [(None, rel_close(R.plans[(x, y)], rel)[1], y, rel_close(R.plans[(x, y)], rel)[2], True) for x, y in sc]]
        fig = new_fig(f"GW transport plans — {R.label}", f"Showcase pairs at ε = {rel:g} ε$_c$")
        plan_panels(fig, cells, a.aligned, a.bins, a.vmax, row_labels=["real\nregions", "same-spectrum\nsurrogate"])
        foot(fig, TRANSPORT_FOOT); savefig(fig, d / "l_transport_pairs.png")


def shades(runs, sweep):
    """one colour per run: the family colour; along a sweep, light → dark steps of it"""
    from matplotlib.colors import to_rgb
    out = {}
    for fam in {R.fam for R in runs}:
        rs = [R for R in runs if R.fam == fam]
        for j, R in enumerate(rs):
            c = np.array(to_rgb(R.col))
            if sweep and len(rs) > 1:
                t = j / (len(rs) - 1); c = (1 - (0.55 - 0.85 * t)) * c if t > 0.65 else c + (1 - c) * (0.55 - 0.85 * t)
            out[R.name] = tuple(np.clip(c, 0, 1))
    return out


def short_lab(R, sweep):
    if sweep == "it": return f"{R.it} step" + ("s" if R.it > 1 else "")
    if sweep == "k": return f"k={R.k}"
    return FAM_NAME.get(R.fam, R.name)


def group_name(x, y): return GLAB[group5(x, y)]
def slug(x, y): return f"{base(x)}_{side(x)[0]}__{base(y)}_{side(y)[0]}"


def rel_close(d, rel, tol=1e-6):
    if not d: return None
    r = min(d, key=lambda v: abs(v - rel)); return d[r] if abs(r - rel) < max(tol, 0.2 * rel) else None


def neighbourhood_fig(R, pairs, rel, a, path):
    radii = [1, 2, 5, 10, 20, 50]; recs = []; cache = {}
    for (x, y) in pairs:
        f, fs, pos = rel_close(R.plans[(x, y)], rel)
        if (y, str(pos)) not in cache:
            Y = l2(m222.region_latents(a.aligned, y, np.load(pos))); _, nb = cKDTree(Y).query(Y, max(radii)); nb[:, 0] = np.arange(len(Y)); cache[(y, str(pos))] = nb
        for ff, kind in [(f, "real"), (fs, "surrogate")]:
            if not Path(ff).exists(): continue
            n, P, dg = m222.load_plan(ff)
            for r, v in zip(radii, m222.recovery(P, n, dg, cache[(y, str(pos))], radii)): recs.append(dict(type=group5(x, y), kind=kind, r=r, enr=v))
    D = pd.DataFrame(recs)
    if D.empty: return
    fig = new_fig(f"Neighbourhood recovery — {R.label}", f"mass in the r nearest neighbours ÷ chance  ·  ε = {rel:g} ε$_c$")
    gs = [t for t in GROUPS if (D.type == t).any()]; w2 = 0.86 / len(gs) - 0.03; ymax = max(D.enr.max() * 1.4, 3)
    for c, t in enumerate(gs):
        ax = fig.add_axes([0.08 + c * (w2 + 0.03), 0.13, w2, 0.56]); ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
        for kind in ["surrogate", "real"]:
            q = D[(D.type == t) & (D.kind == kind)].groupby("r").enr.quantile([.25, .5, .75]).unstack()
            if q.empty: continue
            c_ = R.col if kind == "real" else NULL
            ax.fill_between(q.index, q[0.25], q[0.75], color=c_, alpha=.15, lw=0)
            ax.plot(q.index, q[0.5], color=c_, lw=3.5 if kind == "real" else 2, marker="o" if kind == "real" else None, ms=8, mec=BG, ls="-" if kind == "real" else (0, (4, 3)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.5, ymax); ax.set_xticks(radii); ax.set_xticklabels(radii, fontsize=12)
        yt = [v for v in [0.5, 1, 2, 5, 10, 20, 50] if v <= ymax]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt] if c == 0 else [])
        ax.set_title(GLAB[t], fontsize=14); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    fig.text(0.5, 0.05, "neighbourhood size r  (r = 1: the subject itself)", ha="center", fontsize=14)
    savefig(fig, path)


def fmt_p(p, n_draws=None):
    if p is None or not np.isfinite(p): return "n/a"
    if n_draws and p <= 1.0 / (n_draws + 1) + 1e-12: return f"p < {1.0 / n_draws:.0e}".replace("e-0", "e-") if 1.0 / n_draws < 1e-3 else f"p ≤ {1.0 / (n_draws + 1):.3f}"
    return f"p = {p:.2g}" if p >= 1e-3 else f"p = {p:.1e}"


def end_labels(ax, ends, frac=0.06, dx=1.12, fs=13):
    """ends: [(x, y, text, colour)] at the right end of lines; spread vertically so they never overlap"""
    if not ends: return
    lo, hi = ax.get_ylim(); sep = frac * (hi - lo)
    ys = m222.spread([e[1] for e in ends], sep)
    for (x, y, t, c), yy in zip(ends, ys):
        ax.annotate(t, (x, y), xytext=(x / dx, yy), fontsize=fs, va="center", ha="left", color=INK,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9, shrinkA=0, shrinkB=2), annotation_clip=False)


def fig_score_vs_eps(R, path, order, n_boot=500):
    rels = sorted(R.map.rel.unique(), reverse=True); fine = [r for r in rels if r < 1]
    fig = new_fig(f"{ST} across ε — {R.label}", "left: each pair type  ·  right: residual above the baseline of the two regions (median, 90% bootstrap band)")
    ix = {r: i for i, r in enumerate(order)}; RES = {}
    for r in rels:
        if r < 1:
            M = residual_analysis(R.at(r), order, 0)["resid"]
            RES[r] = R.at(r).assign(res=[M[ix[x], ix[y]] if x in ix and y in ix else np.nan for x, y in zip(R.at(r).src, R.at(r).tgt)])
    g = np.random.default_rng(0)
    for c, mode in enumerate(["abs", "diff"]):
        ax = fig.add_axes([0.07 + c * 0.48, 0.13, 0.31, 0.6]); ends = []
        if mode == "abs": ax.axvspan(1, max(rels) * 1.6, color=BAND, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        xs = rels if mode == "abs" else fine
        for t in (GROUPS if mode == "abs" else ["homologue", "overlap", "shared, cross", "other"]):
            med, lo, hi = [], [], []
            for r in xs:
                v = R.map[(R.map.type == t) & (R.map.rel == r)].contrast.dropna().values
                if not len(v): med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                if mode == "abs":
                    med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
                else:
                    v = RES[r][RES[r].type == t].res.dropna().values
                    bs = [np.median(g.choice(v, len(v))) for _ in range(n_boot)]
                    med.append(np.median(v)); lo.append(np.percentile(bs, 5)); hi.append(np.percentile(bs, 95))
            col = GCOL[t] if t != "shared, cross" else INK2
            ax.fill_between(xs, lo, hi, color=col, alpha=.16, lw=0)
            ax.plot(xs, med, color=col, lw=3.2 if t in ("homologue", "overlap") else 2.2, marker="o", ms=8, mec=BG, mew=1.4,
                    ls=(0, (4, 2)) if t == "shared, cross" else "-")
            short = {"overlap": "overlap", "homologue": "homologues", "shared, cross": "shared, cross", "other": "unrelated"}[t]
            if np.isfinite(med[-1]): ends.append((xs[-1], med[-1], short, col))
        ax.set_xscale("log"); ax.set_xlim(max(xs) * 1.3, min(xs) / 1.4); ax.minorticks_off()
        ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_xlabel("ε / ε$_c$   (finer →)", fontsize=14)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.set_title(SL if mode == "abs" else "residual above baseline", fontsize=15)
        end_labels(ax, ends, dx=1.15, fs=12)
    savefig(fig, path)


def fig_homologue_residuals(R, ra, rel, path):
    o = np.argsort(np.nan_to_num(ra["hres"], nan=-9)); y = np.arange(len(o)); hv = ra["hres"][o]; names = np.array(ra["bases"])[o]
    sub = f"{ra['n_pos']}/{ra['n_hom']} above zero (sign test {fmt_p(ra['p_sign'])})"
    if np.isfinite(ra["p"]): sub += f"  ·  fake-homologue null {fmt_p(ra['p'], len(ra['null']))}"
    fig = new_fig(f"Homologue residual above baseline — {R.label}", sub + f"  ·  ε = {rel:g} ε$_c$")
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.68])
    for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    col = [R.col if v > 0 else NULL for v in hv]
    ax.hlines(y, 0, hv, color=col, lw=5, alpha=.45, zorder=2); ax.scatter(hv, y, s=190, c=col, edgecolor=BG, lw=2, zorder=3)
    span = np.nanmax(np.abs(hv)) if np.isfinite(hv).any() else 1
    for yi, v in zip(y, hv):
        if np.isfinite(v): ax.text(v + (0.04 if v >= 0 else -0.04) * span, yi, f"{v:+.3f}" if span > 0.01 else f"{v:+.4f}", va="center",
                                   ha="left" if v >= 0 else "right", fontsize=11.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=14); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_xlim(min(0, np.nanmin(hv)) - 0.25 * span, np.nanmax(hv) + 0.3 * span)
    ax.set_xlabel(f"residual  ({SL} − baseline)"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    savefig(fig, path)


# ============================================================================================ 4-6 comparisons
def fig_compare(runs, order, out, a, vlim, showcase, RA, d, title_word, sweep):
    cols = shades(runs, sweep)
    rel = a.rel
    # matrices side by side
    for key, getter, ttl in [("a_contrast_matrices", lambda R: matrix(R.at(rel), "contrast", order), ST),
                             ("b_residual_matrices", lambda R: RA[R.name]["resid"], "GW score above the baseline of its two regions")]:
        fig = new_fig(f"{ttl}: {title_word}", f"ε = {rel:g} ε$_c$")
        nc = len(runs); wm = min(0.82 / nc - 0.02, 0.62 * S.H / S.W); norm = TwoSlopeNorm(0, -vlim, vlim)
        for c, R in enumerate(runs):
            ax = fig.add_axes([0.04 + c * (wm + 0.02), 0.1, wm, wm * S.W / S.H]); draw_matrix(ax, getter(R), order, S.DIVERGE.reversed(), norm, labels=False)
            ax.set_title(R.label, fontsize=15, loc="left", color=INK)
            ra = RA[R.name]; ax.text(0, -0.06, f"homologues > baseline: {ra['n_pos']}/{ra['n_hom']} · " + (f"fake-null p = {ra['p']:.2g}" if np.isfinite(ra["p"]) else f"sign p = {ra['p_sign']:.2g}"), transform=ax.transAxes, fontsize=11.5, color=INK2, va="top")
        colorbar(fig, [0.9, 0.2, 0.01, 0.45], S.DIVERGE.reversed(), norm, SL)
        savefig(fig, d / f"{key}.png")
    # null comparison
    fig = new_fig("True vs fake homologues", f"{title_word}  ·  ε = {rel:g} ε$_c$")
    ax = fig.add_axes([0.3, 0.14, 0.52, 0.6])
    for i, R in enumerate(runs[::-1]):
        ra = RA[R.name]
        if len(ra["null"]):
            med, q95 = np.median(ra["null"]), np.percentile(ra["null"], 95)
            ax.plot([np.percentile(ra["null"], 5), q95], [i, i], color=NULL, lw=14, alpha=.45); ax.plot(med, i, "o", color=NULL, ms=12, mec=BG, mew=2)
        ax.plot(ra["obs"], i, "D", color=cols[R.name], ms=20, mec=BG, mew=3, zorder=5)
        ptxt = f"p = {ra['p']:.2g}" if np.isfinite(ra["p"]) else f"sign p = {ra['p_sign']:.2g}"
        ax.text(1.02, i, f"{ptxt}   {ra['n_pos']}/{ra['n_hom']}", transform=ax.get_yaxis_transform(), va="center", fontsize=15,
                weight="bold" if (ra["p"] if np.isfinite(ra["p"]) else ra["p_sign"]) < 0.05 else "normal")
    ax.axvline(0, color=INK2, lw=1, ls=(0, (2, 3))); ax.set_yticks(range(len(runs))); ax.set_yticklabels([R.label for R in runs[::-1]], fontsize=15)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("homologue residual"); ax.set_ylim(-0.6, len(runs) - 0.4)
    fig.legend(handles=[Line2D([], [], marker="D", ls="", color=INK2, ms=12, label="true homologues"),
                        Line2D([], [], color=NULL, lw=10, alpha=.45, label="fake homologues 5–95% (dot: median)")], loc="upper left", bbox_to_anchor=(0.3, 0.84), ncol=2, fontsize=13)
    savefig(fig, d / "c_homologue_null.png")
    # scores vs a parameter (iterations / k), when meaningful
    xs = [R.it if sweep == "it" else (R.k or 0) for R in runs]
    if sweep and len(set(xs)) > 1:
        fig = new_fig(f"Scores along the sweep: {title_word}", f"ε = {rel:g} ε$_c$")
        panels = [("homologues > baseline", [RA[R.name]["n_pos"] for R in runs], None, (0, 15.8)),
                  ("mean homol. residual", [RA[R.name]["obs"] for R in runs], [np.percentile(RA[R.name]["null"], 95) if len(RA[R.name]["null"]) else np.nan for R in runs], "zero"),
                  ("n·tr(T), homologues", [R.at(rel)[R.at(rel).type == "homologue"].ntrT.median() for R in runs], None, "one"),
                  ("n·tr(T), overlap", [R.at(rel)[R.at(rel).type == "overlap"].ntrT.median() for R in runs], None, "one")]
        for c, (lab, v, nullv, yl) in enumerate(panels):
            ax = fig.add_axes([0.07 + c * 0.235, 0.15, 0.19, 0.55])
            if nullv and all(np.isfinite(nullv)): ax.plot(xs, nullv, color=NULL, lw=2, ls=(0, (4, 3)), marker="o"); ax.text(xs[-1], nullv[-1], " null q95", fontsize=11, color=INK2, va="center")
            ax.plot(xs, v, color=runs[0].col, lw=3.5, marker="o", ms=11, mec=BG, mew=2)
            ax.set_xticks(xs); ax.set_title(lab, fontsize=14); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
            ax.set_xlabel("Ricci-flow steps" if sweep == "it" else "k")
            vv = [x for x in list(v) + list(nullv or []) if np.isfinite(x)]
            if yl == "zero" and vv: ax.set_ylim(min(0, min(vv)) * 1.1, max(vv) * 1.25); ax.axhline(0, color=INK2, lw=0.8)
            elif yl == "one" and vv: ax.set_ylim(min(0.9, min(vv) * 0.9), max(vv) * 1.2); ax.axhline(1, color=INK2, lw=0.8, ls=(0, (2, 3)))
            elif isinstance(yl, tuple): ax.set_ylim(*yl)
            if c == 0: ax.axhline(12, color=INK2, lw=1, ls=(0, (2, 3))); ax.text(xs[0], 12.2, "sign test p<0.05", fontsize=10.5, color=INK2)
        savefig(fig, d / "d_scores_along_sweep.png")
    # transport: runs × (real, surrogate) per showcase pair, at rel; and runs × ε (real only)
    for (x, y) in showcase:
        have = [R for R in runs if (x, y) in R.plans and rel_close(R.plans[(x, y)], rel)]
        if len(have) < 2: continue
        lab = (lambda R: short_lab(R, sweep)) if sweep else (lambda R: R.label)
        cells = [[(lab(R), rel_close(R.plans[(x, y)], rel)[0], y, rel_close(R.plans[(x, y)], rel)[2], False) for R in have],
                 [(None, rel_close(R.plans[(x, y)], rel)[1], y, rel_close(R.plans[(x, y)], rel)[2], True) for R in have]]
        fig = new_fig(f"Same pair, {title_word}", f"{pair_label(x, y)}  ·  {group_name(x, y)}  ·  ε = {rel:g} ε$_c$")
        plan_panels(fig, cells, a.aligned, a.bins, a.vmax, row_labels=["real\nregions", "same-spectrum\nsurrogate"])
        foot(fig, TRANSPORT_FOOT); savefig(fig, d / f"e_transport_{slug(x, y)}.png")
        rs = sorted({r for R in have for r in R.plans[(x, y)]}, reverse=True)
        cells = [[(f"ε = {r_:g} ε$_c$" if i == 0 else None, R.plans[(x, y)].get(r_, (None,))[0], y, R.plans[(x, y)].get(r_, (None, None, None))[2], False)
                  for r_ in rs] for i, R in enumerate(have)]
        fig = new_fig(f"Transport across ε: {title_word}", f"{pair_label(x, y)}  ·  {group_name(x, y)}  ·  real regions only")
        plan_panels(fig, cells, a.aligned, a.bins, a.vmax, row_labels=[lab(R).replace(" · ", "\n") for R in have], left=0.2, top=0.76)
        foot(fig, TRANSPORT_FOOT); savefig(fig, d / f"f_transport_eps_{slug(x, y)}.png")
    # rank curves per run (overlay)
    recs = []
    for R in runs:
        for (x, y), dd in R.plans.items():
            e = rel_close(dd, rel)
            if not e: continue
            for ff, kind in [(e[0], "real"), (e[1], "surrogate")]:
                if not Path(ff).exists(): continue
                n, rk = partner_rank(ff)
                for r in [1, 2, 5, 10, 20, 50]: recs.append(dict(run=R.name, type=group5(x, y), kind=kind, r=r, enr=float((rk <= r).mean() / (r / n))))
    D = pd.DataFrame(recs)
    if len(D):
        fig = new_fig(f"Rank of the true partner: {title_word}", f"P(rank ≤ r) ÷ chance  ·  ε = {rel:g} ε$_c$  ·  dashed: surrogates")
        gs = [t for t in GROUPS if (D.type == t).any()]; w2 = 0.86 / len(gs) - 0.03; ymax = max(D.enr.max() * 1.4, 3)
        for c, t in enumerate(gs):
            ax = fig.add_axes([0.08 + c * (w2 + 0.03), 0.13, w2, 0.56]); ax.axhline(1, color=INK2, lw=1, ls=(0, (2, 3)))
            ends = []
            for j, R in enumerate(runs):
                for kind in ["surrogate", "real"]:
                    q = D[(D.run == R.name) & (D.type == t) & (D.kind == kind)].groupby("r").enr.median()
                    if q.empty: continue
                    ax.plot(q.index, q.values, color=cols[R.name] if kind == "real" else NULL, lw=3 if kind == "real" else 1.3,
                            alpha=1 if kind == "real" else .6, ls="-" if kind == "real" else (0, (4, 3)),
                            marker="o" if kind == "real" else None, ms=7, mec=BG)
                    if kind == "real": ends.append([np.log(q.values[-1]), short_lab(R, sweep), q.index[-1]])
            ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.5, ymax); ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels([1, 2, 5, 10, 20, 50], fontsize=12)
            yt = [v for v in [0.5, 1, 2, 5, 10, 20, 50, 100] if v <= ymax]; ax.set_yticks(yt); ax.set_yticklabels([f"×{v:g}" for v in yt] if c == 0 else [])
            ax.set_title(GLAB[t], fontsize=14); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
            if c == len(gs) - 1 and ends:
                ly = m222.spread([e[0] for e in ends], 0.07 * np.log(ymax / 0.5)) if hasattr(m222, "spread") else [e[0] for e in ends]
                for (y0, lab, rx), yy in zip(ends, ly): ax.text(rx * 1.15, np.exp(yy), lab, fontsize=11, va="center")
                ax.set_xlim(0.8, 400)
        fig.text(0.5, 0.05, "r", ha="center", fontsize=14)
        savefig(fig, d / "g_partner_rank.png")
    # pair-by-pair n·tr(T) scatter vs the first run
    if len(runs) > 1 and all("ntrT" in R.map for R in runs):
        ref = runs[0]; others = runs[1:]
        fig = new_fig(f"Pair by pair, n·tr(T): {title_word}", f"ε = {rel:g} ε$_c$")
        nc = len(others); pw = min(0.3, 0.86 / nc - 0.05, 0.62 * S.H / S.W)
        for c, R in enumerate(others):
            J = ref.at(rel).set_index(["src", "tgt"])[["ntrT", "type"]].join(R.at(rel).set_index(["src", "tgt"])[["ntrT"]], rsuffix="_o", how="inner").dropna()
            if J.empty: continue
            ax = fig.add_axes([0.07 + c * (pw + 0.05), 0.1, pw, pw * S.W / S.H])
            lo_, hi_ = max(min(J.ntrT.min(), J.ntrT_o.min()) * 0.7, 0.02), max(J.ntrT.max(), J.ntrT_o.max()) * 1.6
            ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=0.8, ls=(0, (2, 3)))
            for t in ["other", "shared, cross", "homologue", "overlap"]:
                j = J[J.type == t]
                if t == "shared, cross": ax.scatter(j.ntrT, j.ntrT_o, s=28, facecolor="none", edgecolor=INK, lw=1)
                else: ax.scatter(j.ntrT, j.ntrT_o, s=12 if t == "other" else 42, color=GCOL[t], alpha=.45 if t == "other" else .9, edgecolor=BG, lw=.6, zorder=3)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_); ax.minorticks_off()
            tk = [v for v in [0.1, 1, 10, 100] if lo_ <= v <= hi_]; ax.set_xticks(tk); ax.set_yticks(tk)
            ax.set_xticklabels([f"×{v:g}" for v in tk], fontsize=11); ax.set_yticklabels([f"×{v:g}" for v in tk], fontsize=11)
            ax.set_title(R.label, fontsize=14, loc="left"); ax.set_xlabel(ref.label, fontsize=12)
        type_legend(fig); savefig(fig, d / "h_ntrT_pair_by_pair.png")


# ============================================================================================ result-dense figures
def fig_per_homologue(runs, RA, out, rel):
    """15 rows (homologues) × one dot per metric: residual above the factor, and n·tr(T)"""
    cols = {R.name: R.col for R in runs}
    ref = next((R for R in runs if R.fam == "geo"), runs[0]); b0 = RA[ref.name]["bases"]
    o = np.argsort(np.nan_to_num(RA[ref.name]["hres"], nan=-9)); bases_ = [b0[i] for i in o]
    for key, ttl, getter, logx in [
            ("i_per_homologue_residual", "Each homologue: residual above baseline",
             lambda R: dict(zip(RA[R.name]["bases"], RA[R.name]["hres"])), False),
            ("j_per_homologue_ntrT", "Each homologue: same-subject retrieval n·tr(T)",
             lambda R: {base(x): v for x, y, v in zip(R.at(rel).src, R.at(rel).tgt, R.at(rel).ntrT) if base(x) == base(y)}, True)]:
        fig = new_fig(ttl, f"ε = {rel:g} ε$_c$")
        ax = fig.add_axes([0.27, 0.11, 0.6, 0.66]); y = np.arange(len(bases_))
        ax.axvline(1 if logx else 0, color=INK2, lw=1)
        for j, R in enumerate(runs):
            d = getter(R); v = np.array([d.get(b, np.nan) for b in bases_]); off = (j - (len(runs) - 1) / 2) * 0.22
            ax.scatter(v, y + off, s=95, color=cols[R.name], edgecolor=BG, lw=1.5, zorder=3, label=R.label)
            for b, val in zip(bases_, v): ROWS.append(dict(fig=key, run=R.name, region=b, value=val))
        for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
        ax.set_yticks(y); ax.set_yticklabels(bases_, fontsize=13.5); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(y) - 0.4)
        if logx: ax.set_xscale("log"); ax.minorticks_off(); xt = [0.3, 0.5, 1, 2, 3, 5, 10]; ax.set_xticks(xt); ax.set_xticklabels([f"×{v:g}" for v in xt])
        ax.set_xlabel("n·tr(T)" if logx else "residual (score − baseline)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=len(runs), fontsize=13, handletextpad=0.2)
        savefig(fig, out / "4_compare" / f"{key}.png")


def fig_overlap(runs, order, out, rel):
    """region pairs that share a sulcus in the same hemisphere: CKA, GW contrast and n·tr(T), every metric"""
    ref = max(runs, key=lambda R: len(R.map)); df = ref.at(rel); ov = df[df.type == "overlap"].sort_values("cka")
    if ov.empty: return
    labs = [pair_label(x, y) for x, y in zip(ov.src, ov.tgt)]; y = np.arange(len(ov))
    others = {R.name: R.at(rel)[R.at(rel).type == "other"] for R in runs}
    fig = new_fig("Overlapping regions: linear similarity, shape, retrieval", f"pairs sharing a sulcus in one hemisphere  ·  ε = {rel:g} ε$_c$  ·  dotted: median of unrelated pairs")
    panels = [("CKA", "cka", False), (ST, "contrast", False), ("n·tr(T)", "ntrT", True)]
    for c, (ttl, col, logx) in enumerate(panels):
        ax = fig.add_axes([0.33 + c * 0.22, 0.1, 0.19, 0.64])
        for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=BAND, lw=0, zorder=0)
        runs_c = [ref] if col == "cka" else runs
        for j, R in enumerate(runs_c):
            g = R.at(rel).set_index(["src", "tgt"])
            if col not in g: continue
            v = np.array([g[col].get((x, yy), np.nan) for x, yy in zip(ov.src, ov.tgt)]); off = 0 if col == "cka" else (j - (len(runs_c) - 1) / 2) * 0.22
            ax.scatter(v, y + off, s=80, color=GEO if col == "cka" else R.col, edgecolor=BG, lw=1.2, zorder=3, label=None if col == "cka" else R.label)
            ax.axvline(others[R.name][col].median(), color=R.col if col != "cka" else INK2, lw=1, ls=(0, (2, 3)))
            for (x, yy), val in zip(zip(ov.src, ov.tgt), v): ROWS.append(dict(fig="overlap", run=R.name, pair=pair_label(x, yy), measure=col, value=val))
        ax.set_title(ttl, fontsize=15); ax.set_ylim(-0.6, len(y) - 0.4)
        if logx: ax.set_xscale("log"); ax.minorticks_off(); xt = [1, 3, 10, 30]; ax.set_xticks(xt); ax.set_xticklabels([f"×{v:g}" for v in xt])
        ax.set_yticks(y); ax.set_yticklabels(labs if c == 0 else [], fontsize=12); ax.tick_params(axis="y", length=0)
        if c: ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    h = [Line2D([], [], marker="o", ls="", color=R.col, ms=10, label=R.label) for R in runs]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.98, 0.86), ncol=len(runs), fontsize=13)
    savefig(fig, out / "4_compare" / "k_overlap_pairs.png")
    for R in runs:
        g = R.at(rel)
        if "contrast" in g:
            print(f"  overlap pairs, {R.name}: median contrast {g[g.type == 'overlap'].contrast.median():+.4f} (other {g[g.type == 'other'].contrast.median():+.4f}, "
                  f"homol. {g[g.type == 'homologue'].contrast.median():+.4f}); median n·tr(T) {g[g.type == 'overlap'].ntrT.median():.1f}")


def fig_all_plans(R, out, a, per_page=10):
    """every pair with a saved plan at --rel, real regions only, 2 × 5 per page (homologues and overlapping pairs first)"""
    pairs = sorted([p for p in R.plans if rel_close(R.plans[p], a.rel)], key=lambda p: (GROUPS.index(group5(*p)), base(p[0]), base(p[1])))
    if not pairs: return
    pages = [pairs[i:i + per_page] for i in range(0, len(pairs), per_page)]
    for k, pg in enumerate(pages):
        rows = [pg[i:i + 5] for i in range(0, len(pg), 5)]
        cells = [[(pair_label(x, y), rel_close(R.plans[(x, y)], a.rel)[0], y, rel_close(R.plans[(x, y)], a.rel)[2], False)
                  for x, y in row] + [(None, None, None, None, False)] * (5 - len(row)) for row in rows]
        if len(cells) == 1: cells.append([(None, None, None, None, False)] * 5)           # keep the panel size of a full page
        fig = new_fig(f"Transport plans — {R.label}", f"ε = {a.rel:g} ε$_c$  ·  real regions  ·  page {k + 1}/{len(pages)}")
        plan_panels(fig, cells, a.aligned, a.bins, a.vmax, top=0.72, bottom=0.04, left=0.06, right=0.92, gapy=0.11)
        savefig(fig, out / a.folder[R.name] / f"m_transport_all_pairs_p{k + 1:02d}.png")


def fig_signal_vs_eps(runs, order, out, n_null):
    """homologue signal at every ε below ε_c, one line per metric"""
    recs = []
    for R in runs:
        for rel in R.rels():
            ra = residual_analysis(R.at(rel), order, n_null)
            recs.append(dict(run=R.name, rel=rel, n_pos=ra["n_pos"], n_hom=ra["n_hom"], obs=ra["obs"], p_sign=ra["p_sign"], p_fake=ra["p"],
                             q95=np.percentile(ra["null"], 95) if len(ra["null"]) else np.nan, q50=np.median(ra["null"]) if len(ra["null"]) else np.nan,
                             n_draws=len(ra["null"])))
            print(f"  signal vs ε: {R.name} ε={rel:g}: {ra['n_pos']}/{ra['n_hom']}, fake p = {ra['p']:.3g}", flush=True)
    D = pd.DataFrame(recs)
    if D.empty: return
    D["excess"] = D.obs - D.q95
    D.to_csv(out / "4_compare" / "l_signal_vs_eps.csv", index=False)
    fig = new_fig("Where in ε does the homologue signal live?", f"{SL}  ·  every ε below ε$_c$")
    panels = [("n_pos", "homologues above baseline"), ("excess", "residual − null 95th pct"), ("p_fake", "fake-homologue p")]
    xs = sorted(D.rel.unique(), reverse=True)
    for c, (k, lab) in enumerate(panels):
        ax = fig.add_axes([0.07 + c * 0.315, 0.13, 0.25, 0.58])
        if k == "n_pos":
            ax.axhspan(12, 15.5, color=BAND, lw=0); ax.text(xs[-1], 12.25, "sign test p < 0.05", fontsize=11, color=INK2, va="bottom", ha="right")
            ax.axhline(7.5, color=INK2, lw=0.8, ls=(0, (2, 3))); ax.text(xs[-1], 7.2, "chance", fontsize=11, color=INK2, va="top", ha="right")
            ax.set_ylim(0, 15.5); ax.set_yticks([0, 5, 10, 15])
        if k == "excess":
            ax.axhspan(0, 1, color=BAND, lw=0, transform=ax.get_yaxis_transform()) if False else None
            ax.axhline(0, color=INK2, lw=1)
        if k == "p_fake":
            ax.set_yscale("log"); ax.axhline(0.05, color=INK2, lw=1, ls=(0, (2, 3))); ax.text(xs[0], 0.06, "0.05", fontsize=11, color=INK2)
            floor = 1.0 / (D.n_draws.max() + 1); ax.axhline(floor, color=INK2, lw=0.8); ax.text(xs[-1], floor * 0.75, "resolution limit", fontsize=11, color=INK2, va="top", ha="right")
            ax.set_ylim(floor / 2.5, 1.3)
        for R in runs:
            d_ = D[D.run == R.name].sort_values("rel", ascending=False)
            ax.plot(d_.rel, d_[k], color=R.col, lw=3.2, marker="o", ms=10, mec=BG, mew=1.6, label=R.label, zorder=3)
        ax.set_xscale("log"); ax.set_xlim(max(xs) * 1.25, min(xs) / 1.25); ax.minorticks_off()
        ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_xlabel("ε / ε$_c$   (finer →)", fontsize=13.5)
        ax.set_title(lab, fontsize=15); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    h = [Line2D([], [], color=R.col, lw=3.2, marker="o", ms=9, mec=BG, label=R.label) for R in runs]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 0.865), ncol=len(runs), fontsize=13.5)
    savefig(fig, out / "4_compare" / "l_signal_vs_eps.png")


# ============================================================================================ main
def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", "."), "aligned_champ"),
                    help="the aligned latents the runs used (for the subject ordering of the transport plans)")
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR[,DIR...]  (repeatable)")
    ap.add_argument("--main", default="euc,geo20,orc20", help="the three main runs, in story order")
    ap.add_argument("--rel", type=float, default=0.1)
    ap.add_argument("--pairs", default="", help="showcase pairs X:Y,... (default: chosen from the saved plans)")
    ap.add_argument("--n-hom-show", type=int, default=3)
    ap.add_argument("--bins", type=int, default=40); ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--common-pairs", action="store_true",
                    help="restrict every run to the region pairs present in ALL runs (fair comparison between full maps and subset runs)")
    ap.add_argument("--score", default="contrast", help="GW score column: contrast (real − surrogate, default) | rA | r (raw score, no surrogate)")
    ap.add_argument("--plan-cmap", choices=["dark", "light"], default="dark", help="transport plans: dark (low mass = dark) or light")
    ap.add_argument("--n-null-eps", type=int, default=300, help="fake-homologue draws per ε for the signal-vs-ε figure")
    ap.add_argument("--notes", action="store_true", help="add the explanatory footnotes under the figures")
    ap.add_argument("--baseline", choices=["local", "rank1"], default="local", help="local: simple test (default); rank1: common factor a_X a_Y")
    ap.add_argument("--fit-all", action="store_true", help="fit the common factor on all non-homologue pairs (default: also leave out shared-sulcus pairs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); S.apply()
    global NOTES, FIT_ALL, SCORE, SL, ST, PLAN_CMAP, BASELINE; NOTES, FIT_ALL, BASELINE = a.notes, a.fit_all, a.baseline
    SCORE = a.score; PLAN_CMAP = DARK if a.plan_cmap == "dark" else S.BLUES
    if SCORE != "contrast": SL = f"raw GW score ({SCORE})"; ST = "Raw GW score (no surrogate)"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = {n: Run(n, d.split(",")) for n, d in (r.split("=", 1) for r in a.run)}
    if a.common_pairs:
        sets = [set(zip(R.map.src, R.map.tgt)) for R in runs.values() if len(R.map)]
        common = set.intersection(*sets) if sets else set()
        for R in runs.values():
            if len(R.map): R.map = R.map[[p in common for p in zip(R.map.src, R.map.tgt)]]
        print(f"--common-pairs: {len(common)} pairs kept in every run")
    for R in runs.values(): print(f"{R.name:12s} {R.label:34s} pairs in map: {R.map.groupby(['src', 'tgt']).ngroups if len(R.map) else 0:4d}   pairs with plans: {len(R.plans)}")
    main_ = [runs[n] for n in a.main.split(",") if n in runs and len(runs[n].map)]
    for n in a.main.split(","):
        if n in runs and not len(runs[n].map): print(f"  {n}: no map_shard_*.csv found in {[str(d) for d in runs[n].dirs]} — left out")
    order = region_order(list(runs.values()))
    # showcase pairs
    if a.pairs:
        showcase = [tuple(p.split(":")) for p in a.pairs.split(",")]
    else:
        common = set.intersection(*[set(R.plans) for R in main_ if R.plans]) if any(R.plans for R in main_) else set()
        if not common: common = set().union(*[set(R.plans) for R in runs.values()])
        ref = next((R for R in main_ if R.fam == "geo"), main_[0] if main_ else next(iter(runs.values())))
        g = ref.at(a.rel).set_index(["src", "tgt"])
        score = lambda p: g.ntrT.get(p, 0) if "ntrT" in g else 0
        by = lambda t: sorted([p for p in common if group5(*p) == t], key=score, reverse=True)
        showcase = by("overlap")[:1] + by("homologue")[:a.n_hom_show] + by("other")[len(by("other")) // 2:][:1]
    print("showcase:", ", ".join(pair_label(*p) for p in showcase))
    a.folder = {}
    for i, R in enumerate(main_): a.folder[R.name] = f"{i + 1}_{R.name}"
    for R in runs.values(): a.folder.setdefault(R.name, f"x_{R.name}")
    vlim = contrast_lim(list(runs.values()), a.rel)
    RA = {}
    for R in runs.values():
        if len(R.map) and "contrast" in R.map:
            RA[R.name] = residual_analysis(R.at(a.rel), order, a.n_null)
            npairs = R.at(a.rel).groupby(["src", "tgt"]).ngroups
            print(f"  {R.name}: {npairs} pairs, homologues > baseline {RA[R.name]['n_pos']}/{RA[R.name]['n_hom']} (sign p = {RA[R.name]['p_sign']:.3g}), "
                  f"fake-null p = {RA[R.name]['p']:.3g}" + ("  [subset map: factor fitted on few pairs]" if npairs < 300 else ""), flush=True)
    fig_cka(list(runs.values()), order, out, a.rel)
    for R in runs.values():
        if R.name in RA: fig_metric(R, order, out, a, vlim, showcase, RA[R.name]); print(f"  figures {R.name} done", flush=True)
    main_ = [R for R in main_ if R.name in RA]
    for R in runs.values():
        if R.plans and R.name in RA: fig_all_plans(R, out, a)
    if main_:
        fig_overlap(main_, order, out, a.rel)
        fig_signal_vs_eps([R for R in main_ if R.at(a.rel).groupby(["src", "tgt"]).ngroups >= 300], order, out, a.n_null_eps)
        if len(main_) > 1: fig_per_homologue(main_, RA, out, a.rel)
    if len(main_) > 1: fig_compare(main_, order, out, a, vlim, showcase, RA, out / "4_compare", "Euclidean vs geodesic vs ORC", None)
    its = sorted([R for R in runs.values() if R.fam == "orc" and R.name in RA and not R.tag], key=lambda R: (R.k or 0, R.it))
    ks = {R.k for R in its if R.it > 1}
    its = [R for R in its if R.k in ks]
    if len(its) > 1: fig_compare(its, order, out, a, vlim, showcase, RA, out / "5_orc_iterations", "Ricci-flow steps", "it")
    for fam in ["geo", "orc"]:
        sw = sorted([R for R in runs.values() if R.fam == fam and R.it == 1 and not R.tag and R.name in RA], key=lambda R: R.k or 0)
        if len(sw) > 1: fig_compare(sw, order, out, a, vlim, showcase, RA, out / "6_k_sweep" / fam, f"{FAM_NAME[fam]}, k sweep", "k")
    pd.DataFrame(ROWS).to_csv(out / "numbers_224.csv", index=False)
    n = len(list(out.rglob("*.png"))); print(f"{n} figures in {out}")


if __name__ == "__main__":
    main()
