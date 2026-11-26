#!/usr/bin/env python
"""
230 -- CKA figures in the style of 226 (16:9, slide-ready). Aggregates only.

  C1_cka_matrix.png      30×30 CKA between regions (left block, then right block; homologues outlined)
  C2_cka_by_type.png     one dot per region pair, by type, with the mean (log scale)
  C3_cka_homologues.png  CKA of each homologue vs its baseline; same test as GW (residual, fake-homologue p)
  C4_cka_vs_gw.png       per pair: CKA vs geodesic GW contrast — the two measures rank pairs differently

    python 230_cka_figs.py --run geo20=$METHODO/champ_map_215_geo20 --out $METHODO/cka_230
(the cka column is the same in every 215 map: linear CKA on L2-normalised latents, independent of the GW metric)
--standby: no Jean Zay needed, draws C2 from the type means only.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ_style as S

BG, INK, INK2, MUTED = "#fbf8f3", "#1d1b19", "#5b5751", "#9b958c"
NULLC, NULLBAND, GRID, STRIPE = "#cfc8bd", "#ece6dc", "#ebe5dc", "#f4efe7"
CKA = "#b07cc6"                                   # CKA gets its own hue (not a GW metric colour)
HOM = "#ec7d6a"
TYPES = [("overlap", "same sulcus,\nsame hemisphere"), ("homologue", "L↔R\nhomologues"), ("shared, cross", "same sulcus,\ncross hemisphere"),
         ("other", "unrelated\npairs")]
CMAP = LinearSegmentedColormap.from_list("cka", [BG, "#ead9f0", "#cfa9dd", "#b07cc6", "#7d4f97", "#4a2a5e"])


def style():
    S.apply()
    matplotlib.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": MUTED,
                                "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def headline(fig, title, sub):
    fig.text(0.045, 0.955, title, fontsize=25, weight="bold", va="top")
    fig.text(0.045, 0.88, sub, fontsize=15, va="top", color=INK2)


def c2(df, out, standby_means=None):
    fig = plt.figure(figsize=(S.W, S.H))
    headline(fig, "CKA follows anatomical overlap", "linear CKA between two regions, same subjects  ·  bar: mean  ·  log scale")
    ax = fig.add_axes([0.1, 0.16, 0.6, 0.58]); g = np.random.default_rng(0); means = {}
    for k, (t, lab) in enumerate(TYPES):
        col = HOM if t == "homologue" else CKA
        if df is not None:
            v = df[df.type == t].cka.clip(lower=1e-3).values
            ax.scatter(k + g.uniform(-0.22, 0.22, len(v)), v, s=14 if t == "other" else 55, color=NULLC if t == "other" else col,
                       alpha=.6 if t == "other" else .9, edgecolor=BG, lw=.6, zorder=3)
            m = v.mean()
        else:
            m = standby_means[t]
        means[t] = m; ax.plot([k - 0.32, k + 0.32], [m, m], color=INK, lw=3, zorder=4)
        ax.text(k + 0.36, m, f"{m:.2f}" if m >= 0.1 else f"{m:.3f}", fontsize=17, weight="bold", va="center", color=INK)
    ax.set_yscale("log"); ax.minorticks_off(); yt = [0.01, 0.03, 0.1, 0.3, 1]; ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_ylim(0.008, 1.1); ax.set_xlim(-0.5, len(TYPES) - 0.2)
    ax.set_xticks(range(len(TYPES))); ax.set_xticklabels([l for _, l in TYPES], fontsize=13.5); ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_ylabel("CKA", fontsize=14)
    fig.text(0.74, 0.62, "Overlapping boxes share voxels:\nhigh CKA there is expected.", fontsize=14, color=INK2, va="top", linespacing=1.4)
    fig.text(0.74, 0.45, f"Homologues: ≈ {means['homologue'] / means['other']:.0f}× unrelated pairs.", fontsize=14, color=INK, va="top", weight="semibold")
    fig.savefig(out / "C2_cka_by_type.png", dpi=200); plt.close(fig)


def c4_standby(out):
    """type means/medians from the logs: CKA (mean) vs geodesic GW contrast (median), ε = 0.1 ε_c"""
    T = [("overlap", "same sulcus,\nsame hemisphere"), ("homologue", "L↔R\nhomologues"), ("other", "unrelated\npairs")]
    ck = {"overlap": 0.61, "homologue": 0.18, "other": 0.028}; gw = {"overlap": -0.0003, "homologue": 0.0104, "other": 0.0046}
    _c4_bars(out, T, ck, gw, "mean", "median")


def _c4_bars(out, T, ck, gw, ck_stat, gw_stat):
    fig = plt.figure(figsize=(S.W, S.H))
    headline(fig, "CKA and geodesic GW rank the pairs differently",
             f"left: linear CKA ({ck_stat})  ·  right: geodesic GW, beyond the spectrum ({gw_stat})  ·  ε = 0.1 ε$_c$")
    GEOC = "#6f86d6"
    for c, (vals, col, lab, fmt) in enumerate([(ck, CKA, "CKA", "{:.2f}"), (gw, GEOC, "geodesic GW contrast", "{:+.4f}")]):
        ax = fig.add_axes([0.08 + c * 0.47, 0.16, 0.38, 0.56]); x = np.arange(len(T))
        v = [vals[t] for t, _ in T]
        for xi, (t, _), vv in zip(x, T, v):
            hl = t == "homologue"
            ax.bar(xi, vv, width=0.6, color=(HOM if hl else col), alpha=1 if hl else .85, lw=0)
            ax.text(xi, vv + (0.02 if c == 0 else 0.0004) * (1 if vv >= 0 else -1), fmt.format(vv), ha="center",
                    va="bottom" if vv >= 0 else "top", fontsize=16, weight="bold")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in T], fontsize=13.5); ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=11.5)
        lo, hi = min(0, min(v)), max(v); pad = 0.18 * (hi - lo); ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        fig.text(0.08 + c * 0.47, 0.765, lab, fontsize=19, weight="bold")
    fig.text(0.08, 0.05, "Overlapping regions: highest CKA, no shape excess.   Homologues: moderate CKA, the strongest shared shape.",
             fontsize=14, color=INK, weight="semibold")
    fig.savefig(out / "C4_cka_vs_gw.png", dpi=200); plt.close(fig)


def c4_full(R, Rg, order, rel, out, m224):
    """per pair: CKA vs geodesic GW contrast, coloured by type; medians per type"""
    d = R.at(rel)[["src", "tgt", "type", "cka"]].merge(Rg.at(rel)[["src", "tgt", "contrast"]], on=["src", "tgt"])
    fig = plt.figure(figsize=(S.W, S.H))
    from scipy.stats import spearmanr
    rho = spearmanr(d.cka, d.contrast)[0]
    headline(fig, "CKA and geodesic GW read the pairs differently",
             f"one dot per region pair  ·  x: linear CKA (log)  ·  y: geodesic GW contrast  ·  ε = {rel:g} ε$_c$  ·  Spearman ρ = {rho:+.2f}")
    ax = fig.add_axes([0.09, 0.13, 0.6, 0.62]); ax.axhline(0, color=MUTED, lw=1)
    cols = {"other": NULLC, "shared, cross": "#a79c8e", "overlap": CKA, "homologue": HOM}
    for t in ["other", "shared, cross", "overlap", "homologue"]:
        j = d[d.type == t]
        ax.scatter(j.cka.clip(lower=1e-3), j.contrast, s=14 if t == "other" else 80, color=cols[t], alpha=.55 if t == "other" else .95,
                   edgecolor=BG, lw=1, zorder=3 if t != "other" else 2)
        if len(j): ax.scatter(j.cka.median(), j.contrast.median(), s=420, marker="X", color=cols[t] if t != "other" else MUTED, edgecolor=INK, lw=1.5, zorder=5)
    ax.set_xscale("log"); ax.minorticks_off(); xt = [0.01, 0.03, 0.1, 0.3, 1]; ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_xlabel("CKA", fontsize=14); ax.set_ylabel("geodesic GW contrast", fontsize=14); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    labs = {"overlap": "same sulcus, same hemisphere", "homologue": "L↔R homologues", "shared, cross": "same sulcus, cross hemisphere", "other": "unrelated pairs"}
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=cols[t], ms=10, label=labs[t]) for t in ["homologue", "overlap", "shared, cross", "other"]]
    h.append(Line2D([], [], marker="X", ls="", color=MUTED, mec=INK, ms=14, label="median of each type"))
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.72, 0.45), fontsize=13, frameon=False)
    fig.savefig(out / "C4_cka_vs_gw.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="NAME=DIR of any full 215 map (for the cka column)")
    ap.add_argument("--rel", type=float, default=0.1); ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--gw-run", default=None, help="NAME=DIR of the geodesic 215 map for C4 (default: --run)")
    ap.add_argument("--standby", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); style(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.standby:
        c2(None, out, {"overlap": 0.61, "homologue": 0.18, "shared, cross": 0.12, "other": 0.028}); c4_standby(out)
        print("standby C2, C4 written"); return
    _s = importlib.util.spec_from_file_location("m224", HERE / "224_gw_story.py"); m224 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m224)
    n, d = a.run.split("=", 1); R = m224.Run(n, d.split(",")); order = m224.region_order([R])
    df = R.at(a.rel)
    # C1 matrix
    M = m224.matrix(df, "cka", order); nl = sum(m224.side(r) == "left" for r in order); nn = len(order)
    fig = plt.figure(figsize=(S.W, S.H)); headline(fig, "CKA between the 30 regions", "left regions, then right regions  ·  outlined: left↔right homologues")
    ax = fig.add_axes([0.2, 0.03, 0.56, 0.76]); norm = PowerNorm(0.5, vmin=0, vmax=np.nanmax(M))
    ax.imshow(np.ma.masked_invalid(M), cmap=CMAP, norm=norm, interpolation="nearest"); ax.set_facecolor(NULLBAND)
    ax.axhline(nl - 0.5, color=INK, lw=1.4); ax.axvline(nl - 0.5, color=INK, lw=1.4)
    ix = {r: i for i, r in enumerate(order)}
    for r in order:
        if m224.side(r) == "left" and f"{m224.base(r)}_right" in ix:
            i, j = ix[r], ix[f"{m224.base(r)}_right"]
            for u, v in ((i, j), (j, i)): ax.add_patch(Rectangle((v - .5, u - .5), 1, 1, fill=False, ec=HOM, lw=2))
    ax.set_yticks(range(nn)); ax.set_yticklabels([m224.base(r) for r in order], fontsize=9.5); ax.set_xticks([]); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(nl / 2 - .5, -1.1, "LEFT", ha="center", fontsize=13, weight="bold", color=INK2); ax.text(nl + (nn - nl) / 2 - .5, -1.1, "RIGHT", ha="center", fontsize=13, weight="bold", color=INK2)
    cax = fig.add_axes([0.79, 0.2, 0.012, 0.45]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.outline.set_visible(False); cb.set_label("CKA", fontsize=13); cax.tick_params(labelsize=11.5)
    fig.savefig(out / "C1_cka_matrix.png", dpi=200); plt.close(fig)
    # C2
    c2(df, out)
    # C3 homologues vs baseline, same test as GW
    R.map["contrast"] = R.map["cka"]; ra = m224.residual_analysis(R.at(a.rel), order, a.n_null)
    h = df[df.type == "homologue"].assign(b=lambda x: [m224.base(s) for s in x.src]).set_index("b")
    o = np.argsort(np.nan_to_num(ra["hres"], nan=-9)); bases = [ra["bases"][i] for i in o]; res = ra["hres"][o]
    base_lvl = np.array([h.cka.get(b, np.nan) for b in bases]) - res
    fig = plt.figure(figsize=(S.W, S.H))
    p = ra["p"]; ptxt = f"p < {1 / len(ra['null']):.3f}".rstrip("0") if p <= 1 / (len(ra["null"]) + 1) + 1e-12 else f"p = {p:.2g}"
    headline(fig, f"{ra['n_pos']}/{ra['n_hom']} homologues have a higher CKA than their baseline",
             f"dot: CKA of the left↔right pair  ·  bar end: baseline (its two regions with the other regions)  ·  fake-homologue {ptxt}")
    ax = fig.add_axes([0.3, 0.08, 0.6, 0.7]); y = np.arange(len(bases))
    for yi in y[::2]: ax.axhspan(yi - .5, yi + .5, color=STRIPE, lw=0, zorder=0)
    cka_v = np.array([h.cka.get(b, np.nan) for b in bases])
    ax.hlines(y, base_lvl, cka_v, color=[HOM if r > 0 else NULLC for r in res], lw=4, alpha=.55, zorder=2)
    ax.scatter(base_lvl, y, s=70, marker="|", color=INK2, lw=2.5, zorder=3)
    ax.scatter(cka_v, y, s=170, color=[HOM if r > 0 else BG for r in res], edgecolor=HOM, lw=2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([b.replace("_", " ") for b in bases], fontsize=13.5); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_ylim(-.6, len(y) - .4); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("CKA", fontsize=14)
    fig.savefig(out / "C3_cka_homologues.png", dpi=200); plt.close(fig)
    if a.gw_run:
        gn, gd = a.gw_run.split("=", 1); Rg = m224.Run(gn, gd.split(","))
    else:
        Rg = m224.Run(n, d.split(","))
    c4_full(R, Rg, order, a.rel, out, m224)
    print(f"CKA: homologues > baseline {ra['n_pos']}/{ra['n_hom']}, sign p = {ra['p_sign']:.3g}, fake-homologue p = {ra['p']:.3g}")
    print("written", out)


if __name__ == "__main__":
    main()
