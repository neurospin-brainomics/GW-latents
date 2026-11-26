#!/usr/bin/env python
"""
226 -- The four key figures of the GW-on-Champollion story (16:9, slide-ready). Aggregates only.

  K1_homologues_vs_surrogate.png  each homologue pair, geodesic and ORC: real − surrogate GW score, with the
                                  surrogates' ±2 sd band (z > 2 = the real pair beats its own surrogates)
  K1b_overlap_vs_surrogate.png    same for the pairs of regions that share a sulcus in one hemisphere (overlapping boxes)
  K2_with_surrogate.png           contrast (real − surrogate): Euclidean fails, geodesic / ORC single out homologues
  K3_without_surrogate.png        raw score (r_A): every metric singles out homologues, Euclidean included
  K4_spectrum.png                 why: raw Euclidean score follows spectral similarity; geodesic does not
  K5_retrieval.png                n·tr(T) of each overlapping pair, next to the homologue and unrelated means, every metric
  K7_across_eps.png               homologues above baseline at every ε, with and without surrogate, every metric
  K6_transport_<pair>.png         transport plan across ε, real vs surrogate (needs saved plans: add the champ_sub_* dir to --run)

    python 226_key_figures.py --aligned $METHODO/aligned_champ --out $METHODO/key_226 \
        --run euc=$METHODO/champ_map_215_pcperm --run geo20=$METHODO/champ_map_215_geo20 --run orc20=$METHODO/champ_map_215_orc20
CPU, ~2 min. Needs champ_style.py, 222_champ_figs_jz.py, 224_gw_story.py, 225_spectra.py (and Inter-*.ttf) next to it.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ_style as S
_l = lambda n, f: (lambda s: (s, s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[2])(importlib.util.spec_from_file_location(n, HERE / f))
m224 = _l("m224", "224_gw_story.py"); m225 = _l("m225", "225_spectra.py")

# ---- palette: soft but saturated enough to read on a projector (validated all-pairs, always direct-labelled)
C = dict(euc="#3fb58f", geo="#6f86d6", orc="#ec7d6a")
BG, INK, INK2, MUTED = "#fbf8f3", "#1d1b19", "#5b5751", "#9b958c"
NULLC, NULLBAND, GRID, STRIPE = "#cfc8bd", "#ece6dc", "#ebe5dc", "#f4efe7"
TINT = {k: matplotlib.colors.to_hex(0.18 * np.array(matplotlib.colors.to_rgb(v)) + 0.82 * np.array(matplotlib.colors.to_rgb(BG))) for k, v in C.items()}
ZTHR = 2.0
NAME = dict(euc="Euclidean", geo="Geodesic", orc="Ollivier–Ricci")


def style():
    S.apply()
    matplotlib.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": MUTED,
                                "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def headline(fig, title, sub):
    fig.text(0.045, 0.955, title, fontsize=25, weight="bold", va="top", color=INK)
    fig.text(0.045, 0.88, sub, fontsize=15, va="top", color=INK2)


def fam(name): return name[:3]


def pstr(p, n=None):
    if not np.isfinite(p): return "n/a"
    if n and p <= 1 / (n + 1) + 1e-12: return f"p < {1 / n:.3f}".rstrip("0")
    return f"p = {p:.2g}"


def residual_table(R, rel, order, score, n_null):
    R.map["contrast"] = R.map[score]
    df = R.at(rel); ra = m224.residual_analysis(df, order, n_null)
    ix = {r: i for i, r in enumerate(order)}
    df = df.assign(res=[ra["resid"][ix[x], ix[y]] if x in ix and y in ix else np.nan for x, y in zip(df.src, df.tgt)])
    return df, ra


# ===================================================================================== K1
def k1(runs, rel, out, B, group="homologue"):
    """real − surrogate, pair by pair, for one pair type (homologue or overlap), geodesic and ORC"""
    rr = [R for R in runs if fam(R.name) in ("geo", "orc")]
    key = lambda x, y: m224.base(x) if group == "homologue" else m224.pair_label(x, y)
    ref = rr[0].at(rel); h = ref[ref.type == group]
    if h.empty: print(f"K1: no {group} pairs"); return []
    order_b = list(h.assign(b=[key(x, y) for x, y in zip(h.src, h.tgt)]).sort_values("contrast").b); y = np.arange(len(order_b))
    fig = plt.figure(figsize=(S.W, S.H)); rows = []; counts = []
    left = 0.25 if group == "homologue" else 0.31; w = 0.33 if group == "homologue" else 0.3; step = w + 0.05
    for c, R in enumerate(rr):
        d = R.at(rel); d = d[d.type == group]
        d = d.assign(b=[key(x, yy) for x, yy in zip(d.src, d.tgt)]).set_index("b").reindex(order_b)
        sd = (d.contrast / (d.z * np.sqrt(1 + 1 / B))).abs().where(lambda s_: np.isfinite(s_))
        col = C[fam(R.name)]
        ax = fig.add_axes([left + c * step, 0.1, w, 0.6])
        for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=STRIPE, lw=0, zorder=0)
        lim = 1.25 * max(np.nanmax(np.abs(d.contrast)), np.nanmedian(2 * sd))      # robust: one noisy surrogate set must not set the scale
        sdc = np.minimum(sd, lim / 2)
        ax.barh(y, 4 * sdc, left=-2 * sdc, height=0.62, color=NULLBAND, lw=0, zorder=1)
        ax.axvline(0, color=MUTED, lw=1, zorder=2)
        sig = (d.z > ZTHR).values
        ax.hlines(y, 0, d.contrast, color=col, lw=4, alpha=.5, zorder=3)
        ax.scatter(d.contrast[sig], y[sig], s=180, color=col, edgecolor=BG, lw=2, zorder=4)
        ax.scatter(d.contrast[~sig], y[~sig], s=150, facecolor=BG, edgecolor=col, lw=2.2, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([b.replace("_", " ") for b in order_b] if c == 0 else [], fontsize=13.5 if group == "homologue" else 12)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.set_ylim(-0.6, len(y) - 0.4)
        ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlabel("real − surrogate", fontsize=14)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.tick_params(axis="x", labelsize=12.5)
        lo_ = min(-lim * 0.6, np.nanmin(d.contrast) * 1.25); ax.set_xlim(lo_, lim)
        fig.text(left + c * step, 0.755, R.label, fontsize=20, weight="bold", color=INK)
        fig.text(left + c * step, 0.715, f"{int(sig.sum())}/{len(sig)} pairs with z > {ZTHR:g}", fontsize=16, weight="bold", color=col)
        counts.append(sig.mean()); rows += [dict(fig=f"K1_{group}", run=R.name, pair=b, contrast=v, z=z) for b, v, z in zip(order_b, d.contrast, d.z)]
    what = "homologous pairs" if group == "homologue" else "overlapping pairs"
    title = (f"Most {what} beat their own surrogates" if min(counts) > 0.5 else
             f"Few {what} beat their own surrogates" if max(counts) < 0.5 else f"{what.capitalize()} versus their own surrogates")
    unit = "left↔right pair" if group == "homologue" else "pair sharing a sulcus"
    headline(fig, title, f"real − surrogate GW score per {unit}  ·  grey band: ±2 sd of the {B} surrogates  ·  ε = {rel:g} ε$_c$")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=11, label=f"z > {ZTHR:g}"),
                        Line2D([], [], marker="o", ls="", mfc=BG, mec=INK2, mew=2, ms=10, label=f"z ≤ {ZTHR:g}"),
                        Line2D([], [], color=NULLBAND, lw=12, label="surrogates ±2 sd")], loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=1, fontsize=12.5)
    S.save(fig, out / ("K1_homologues_vs_surrogate.png" if group == "homologue" else "K1b_overlap_vs_surrogate.png"))
    return rows


# ===================================================================================== K2 / K3
def k23(runs, rel, order, score, out, fname, title, sub, n_null):
    fig = plt.figure(figsize=(S.W, S.H))
    g = np.random.default_rng(0); rows = []
    tabs = [residual_table(R, rel, order, score, n_null) for R in runs]
    cats = [("other", "unrelated\npairs"), ("overlap", "overlapping\nregions"), ("homologue", "L↔R\nhomologues")]
    for c, (R, (df, ra)) in enumerate(zip(runs, tabs)):
        col = C[fam(R.name)]; ok = np.isfinite(ra["p"]) and ra["p"] < 0.05
        x0 = 0.085 + c * 0.31; ax = fig.add_axes([x0, 0.12, 0.25, 0.55])
        vals = {t: df[df.type == t].res.dropna().values for t, _ in cats}
        allv = np.concatenate(list(vals.values()))
        lim = np.nanmax(np.abs(allv)) * 1.02                                  # full range: every point is drawn
        for k, (t, _) in enumerate(cats):
            v = vals[t]
            if not len(v): continue
            vc = np.clip(v, -lim, lim); out_ = np.abs(v) > lim; xj = k + g.uniform(-0.18, 0.18, len(v))
            if t == "other":
                vp = ax.violinplot([vc], positions=[k], widths=0.8, showextrema=False)
                for b_ in vp["bodies"]: b_.set_facecolor(NULLBAND); b_.set_edgecolor("none"); b_.set_alpha(1)
                ax.scatter(xj, vc, s=6, color=NULLC, lw=0, zorder=2)
            elif t == "overlap":
                ax.scatter(xj[~out_], vc[~out_], s=70, color="#a79c8e", edgecolor=BG, lw=1.2, zorder=3)
            else:
                kw = dict(color=col, edgecolor=BG) if ok else dict(color=BG, edgecolor=col)
                ax.scatter(xj[~out_], vc[~out_], s=150, lw=2, zorder=4, **kw)
            m = np.mean(v); se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.plot([k - 0.3, k + 0.3], [np.clip(m, -lim, lim)] * 2, color=INK if t != "other" else INK2, lw=2.6, zorder=5)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        ax.set_xlim(-0.55, 2.55); ax.set_ylim(-lim * 1.08, lim * 1.08)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in cats], fontsize=12.5); ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=11)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        if c == 0: ax.set_ylabel("residual above baseline", fontsize=13.5)
        fig.text(x0, 0.785, NAME[fam(R.name)], fontsize=20, weight="bold")
        fig.text(x0 + 0.25, 0.785, f"{ra['n_pos']}/{ra['n_hom']}", fontsize=24, weight="bold", color=col if ok else MUTED, ha="right")
        fig.text(x0 + 0.25, 0.75, "homologues > 0  ·  " + pstr(ra["p"], len(ra["null"])), fontsize=11.5, color=INK2, ha="right", va="top")
        rows.append(dict(fig=fname, run=R.name, score=score, n_pos=ra["n_pos"], n_hom=ra["n_hom"], p_fake=ra["p"], p_sign=ra["p_sign"],
                         mean_overlap=float(np.mean(vals["overlap"])) if len(vals["overlap"]) else np.nan))
    fig.text(0.045, 0.025, f"bar: mean  ·  p: true vs {n_null} random left↔right pairings", fontsize=11, color=MUTED)
    win = [NAME[fam(r["run"])] for r in rows if np.isfinite(r["p_fake"]) and r["p_fake"] < 0.05 and r["n_pos"] >= 2 * r["n_hom"] / 3]
    verdict = ("homologues stand out for every metric" if len(win) == len(rows) else
               f"homologues stand out consistently for {' and '.join(win)}" if win else "homologues do not stand out consistently")
    headline(fig, f"{title}: {verdict}", sub)
    S.save(fig, out / fname); return rows


# ===================================================================================== K4
def k4(runs, rel, aligned, out, n_sub, seed, n_bins=5):
    """raw score vs spectral similarity, binned: one line per metric (scores z-scored within metric)"""
    A = Path(aligned); regs = sorted(p.stem for p in A.glob("*.npy") if m224.side(p.stem))
    N = np.load(A / f"{regs[0]}.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(seed).permutation(N)[:n_sub]) if n_sub < N else np.arange(N)
    spec = {r: m225.spectrum(np.asarray(np.load(A / f"{r}.npy", mmap_mode="r")[idx], dtype=np.float64)) for r in regs}
    fig = plt.figure(figsize=(S.W, S.H))
    headline(fig, "Euclidean GW rewards similar spectra; geodesic GW does not",
             f"raw GW score r$_A$ (standardised within metric) for region pairs grouped by how similar their variance spectra are  ·  ε = {rel:g} ε$_c$")
    ax = fig.add_axes([0.1, 0.17, 0.62, 0.56]); rows = []; ends = []
    for R in runs:
        d = R.at(rel).copy(); d["sd"] = [np.abs(spec[x] - spec[y]).sum() for x, y in zip(d.src, d.tgt)]
        z = (d.rA - d.rA.mean()) / d.rA.std(); q = pd.qcut(d.sd, n_bins, labels=False)
        mu = z.groupby(q).mean(); se = z.groupby(q).std() / np.sqrt(z.groupby(q).size())
        col = C[fam(R.name)]; rho = spearmanr(d.sd, d.rA)[0]; xs = np.arange(n_bins)
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=.16, lw=0)
        ax.plot(xs, mu, color=col, lw=4, marker="o", ms=13, mec=BG, mew=2.2, zorder=3)
        ends.append((mu.iloc[-1], f"{NAME[fam(R.name)]}   ρ = {rho:+.2f}", col)); rows.append(dict(fig="K4", run=R.name, spearman=rho))
    ys = m224.m222.spread([e[0] for e in ends], 0.09)
    for (y0, lab, col), yy in zip(ends, ys):
        ax.annotate(lab, (n_bins - 1, y0), xytext=(n_bins - 0.7, yy), fontsize=16, weight="bold", color=col, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.2), annotation_clip=False)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(["most\nsimilar"] + [""] * (n_bins - 2) + ["least\nsimilar"], fontsize=13.5)
    ax.set_xlabel("spectral similarity of the two regions (quintiles)", fontsize=14); ax.set_ylabel("raw GW score (z)", fontsize=14)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlim(-0.3, n_bins - 0.7 + 0.05)
    fig.text(0.1, 0.015, "ρ: Spearman between spectral distance and raw score over all 435 pairs  ·  band: 95% CI of the mean", fontsize=11, color=MUTED)
    S.save(fig, out / "K4_spectrum.png"); return rows


# ===================================================================================== K5 retrieval
def k5(runs, rel, out):
    ref = next(R for R in runs if fam(R.name) == "geo"); d0 = ref.at(rel)
    ov = d0[d0.type == "overlap"].sort_values("ntrT", ascending=False)
    labels = [m224.pair_label(x, y) for x, y in zip(ov.src, ov.tgt)] + ["L↔R homologues (mean of 15)", "unrelated pairs (mean)"]
    fig = plt.figure(figsize=(S.W, S.H))
    nhi = int((ov.ntrT >= 3).sum())
    headline(fig, f"GW finds the same subject in {nhi} of {len(ov)} overlapping pairs",
             f"n·tr(T): mass the plan puts on the same subject (×1 = chance, threshold ×3, geodesic)  ·  ε = {rel:g} ε$_c$")
    ax = fig.add_axes([0.34, 0.1, 0.6, 0.63]); n = len(labels); y = np.arange(n)[::-1]
    for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=STRIPE, lw=0, zorder=0)
    ax.axhline(y[len(ov)] + 0.5, color=MUTED, lw=1)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    rows = []
    for j, R in enumerate(runs):
        d = R.at(rel).set_index(["src", "tgt"]); col = C[fam(R.name)]; off = (j - (len(runs) - 1) / 2) * 0.22
        v = [d.ntrT.get((x, yy), np.nan) for x, yy in zip(ov.src, ov.tgt)]
        v += [R.at(rel)[R.at(rel).type == "homologue"].ntrT.mean(), R.at(rel)[R.at(rel).type == "other"].ntrT.mean()]
        ax.scatter(v, y + off, s=150, color=col, edgecolor=BG, lw=1.8, zorder=3, label=NAME[fam(R.name)])
        rows += [dict(fig="K5", run=R.name, pair=l, ntrT=float(vv)) for l, vv in zip(labels, v)]
    ax.set_xscale("log"); ax.minorticks_off(); xt = [0.3, 1, 3, 10, 30, 100, 300]
    xmax = max(60, np.nanmax([r["ntrT"] for r in rows]) * 1.5); xt = [t for t in xt if t <= xmax]
    ax.set_xticks(xt); ax.set_xticklabels([f"×{t:g}" for t in xt]); ax.set_xlim(0.2, max(60, np.nanmax([r["ntrT"] for r in rows]) * 1.5))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12.5); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, n - 0.4); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlabel("n·tr(T)  (log)", fontsize=14)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(runs), fontsize=13, frameon=False, handletextpad=0.2)
    S.save(fig, out / "K5_retrieval.png"); return rows


# ===================================================================================== K6 transport
def k6(runs, pair, aligned, out, bins=40, vmax=8.0):
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    pref = sorted(runs, key=lambda R: ["geo", "orc", "euc"].index(fam(R.name)) if fam(R.name) in ("geo", "orc", "euc") else 9)
    R = next((R for R in pref if pair in R.plans), None)
    if R is None: print(f"K6: no saved plan for {pair[0]}:{pair[1]} — skipped"); return []
    cm = LinearSegmentedColormap.from_list("night", ["#1b1f2e", "#2c3a63", "#4f68b8", "#8fa4ea", "#d9e1fb", "#fbf8f3"])
    rels = sorted(R.plans[pair], reverse=True); norm = LogNorm(1 / vmax, vmax)
    fig = plt.figure(figsize=(S.W, S.H))
    headline(fig, "The transport plan emerges from noise as ε decreases",
             f"{m224.pair_label(*pair)}  ·  {R.label}  ·  subjects ordered along the manifold, {bins}×{bins} blocks")
    nc = len(rels); side = 0.3; w = side * S.H / S.W; gap = 0.025; x0 = 0.15
    rows = []
    for c, r in enumerate(rels):
        f, fs, pos = R.plans[pair][r]
        for k, (ff, lab) in enumerate([(f, "real regions"), (fs, "same-spectrum surrogate")]):
            ax = fig.add_axes([x0 + c * (w + gap), 0.43 - k * 0.37, w, side]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if not Path(ff).exists(): continue
            Mb, nt, band = m224.plan_blocks(ff, aligned, pair[1], pos, bins)
            ax.imshow(np.clip(Mb, 1 / vmax, vmax), cmap=cm, norm=norm, interpolation="nearest")
            ax.text(0.0, 1.03, f"×{nt:.1f}", transform=ax.transAxes, fontsize=17 if k == 0 else 13, weight="bold" if k == 0 else "normal",
                    color=C[fam(R.name)] if k == 0 else MUTED, va="bottom")
            if k == 0: ax.text(1.0, 1.03, f"ε = {r:g} ε$_c$", transform=ax.transAxes, fontsize=13.5, color=INK2, va="bottom", ha="right")
            if c == 0: fig.text(x0 - 0.012, 0.43 - k * 0.37 + side / 2, lab.replace(" ", "\n", 1), fontsize=14, ha="right", va="center",
                                color=INK if k == 0 else INK2)
            rows.append(dict(fig="K6", rel=r, kind=lab, ntrT=nt, band=band))
    cax = fig.add_axes([x0 + nc * (w + gap) + 0.01, 0.12, 0.011, 0.55])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax); cb.outline.set_visible(False)
    cax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); cb.set_ticks([1 / vmax, 1, vmax]); cb.set_ticklabels(["×1/8", "chance", "×8"])
    cax.tick_params(labelsize=12); fig.text(0.045, 0.025, "×: n·tr(T), mass on the same subject relative to chance", fontsize=11, color=MUTED)
    S.save(fig, out / f"K6_transport_{pair[0]}__{pair[1]}.png"); return rows


# ===================================================================================== K7 across ε
def k7(runs, order, out, n_null):
    rels = sorted({r for R in runs for r in R.rels()}, reverse=True); recs = []
    for score, lab in [("contrast_orig", "with surrogate"), ("rA", "without surrogate")]:
        for R in runs:
            R.map["contrast"] = R.map[score]
            for rel in rels:
                ra = m224.residual_analysis(R.at(rel), order, n_null)
                recs.append(dict(score=lab, run=R.name, rel=rel, n_pos=ra["n_pos"], n_hom=ra["n_hom"], p=ra["p"]))
                print(f"  K7 {lab:18s} {R.name:6s} ε={rel:g}: {ra['n_pos']}/{ra['n_hom']}, p = {ra['p']:.3g}", flush=True)
        for R in runs: R.map["contrast"] = R.map["contrast_orig"]
    D = pd.DataFrame(recs); D.to_csv(out / "K7_across_eps.csv", index=False)
    fig = plt.figure(figsize=(S.W, S.H))
    headline(fig, "The homologue signal across ε", f"homologue pairs above baseline (of 15)  ·  filled: fake-homologue p < 0.05  ·  {n_null} re-pairings per point")
    for c, lab in enumerate(["with surrogate", "without surrogate"]):
        ax = fig.add_axes([0.07 + c * 0.47, 0.13, 0.4, 0.58]); ends = []
        ax.axhspan(12, 15.5, color=STRIPE, lw=0); ax.axhline(7.5, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(rels[-1], 7.2, "chance", fontsize=11.5, color=MUTED, ha="right", va="top")
        ax.text(rels[0], 12.2, "sign test p < 0.05", fontsize=11.5, color=MUTED, va="bottom")
        for R in runs:
            d = D[(D.score == lab) & (D.run == R.name)].sort_values("rel", ascending=False); col = C[fam(R.name)]
            ax.plot(d.rel, d.n_pos, color=col, lw=3.5, zorder=3)
            sig = (d.p < 0.05).values
            ax.scatter(d.rel[sig], d.n_pos[sig], s=150, color=col, edgecolor=BG, lw=2, zorder=4)
            ax.scatter(d.rel[~sig], d.n_pos[~sig], s=130, color=BG, edgecolor=col, lw=2.4, zorder=4)
            ends.append((d.n_pos.iloc[-1], NAME[fam(R.name)], col))
        ys = m224.m222.spread([e[0] for e in ends], 0.9)
        for (y0, t, col), yy in zip(ends, ys): ax.text(rels[-1] / 1.15, yy, t, fontsize=14, weight="bold", color=col, va="center")
        ax.set_xscale("log"); ax.set_xlim(rels[0] * 1.3, rels[-1] / 3.2); ax.minorticks_off()
        ax.set_xticks(rels); ax.set_xticklabels([f"{r:g}" for r in rels]); ax.set_xlabel("ε / ε$_c$   (finer →)", fontsize=14)
        ax.set_ylim(0, 15.5); ax.set_yticks([0, 5, 10, 15]); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        fig.text(0.07 + c * 0.47, 0.765, lab.capitalize(), fontsize=20, weight="bold")
    S.save(fig, out / "K7_across_eps.png")
    return recs


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", ""), "aligned_champ"))
    ap.add_argument("--run", action="append", default=[], help="euc=DIR geo20=DIR orc20=DIR (full 435-pair maps)")
    ap.add_argument("--rel", type=float, default=0.1); ap.add_argument("--surr-b", type=int, default=4, help="number of surrogates per pair in 215 (SURRB)")
    ap.add_argument("--n-null", type=int, default=1000); ap.add_argument("--n-sub", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plan-pair", default="SFint-FCMant_left:SFint-SR_left", help="K6: pairs X:Y,... (plans must be saved in one of the --run dirs)")
    ap.add_argument("--all-eps", action="store_true", help="K1–K5 at every ε < ε_c of the maps (one sub-folder each)")
    ap.add_argument("--n-null-eps", type=int, default=300, help="re-pairings per point in K7")
    ap.add_argument("--z-thr", type=float, default=2.0, help="z threshold for K1 (with B=4 surrogates, z ~ t(3) under the null: 2.35 one-sided 5%%)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); style()
    global ZTHR; ZTHR = a.z_thr; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [m224.Run(n, d.split(",")) for n, d in (r.split("=", 1) for r in a.run)]
    for R in runs: R.map["contrast_orig"] = R.map["contrast"]
    order = m224.region_order(runs)
    rels = sorted({r for R in runs for r in R.rels()}, reverse=True) if a.all_eps else [a.rel]
    rows = []
    for rel in rels:
        o = out / f"eps_{rel:g}" if len(rels) > 1 else out; o.mkdir(parents=True, exist_ok=True)
        print(f"ε = {rel:g} ε_c -> {o}", flush=True)
        rows += [dict(r, rel=rel) for r in k1(runs, rel, o, a.surr_b, "homologue") + k1(runs, rel, o, a.surr_b, "overlap")]
        for R in runs: R.map["contrast"] = R.map["contrast_orig"]
        rows += [dict(r, rel=rel) for r in k23(runs, rel, order, "contrast_orig", o, "K2_with_surrogate.png", "With the surrogate",
                 f"score = real − surrogate, residual above baseline  ·  ε = {rel:g} ε$_c$", a.n_null)]
        rows += [dict(r, rel=rel) for r in k23(runs, rel, order, "rA", o, "K3_without_surrogate.png", "Without the surrogate",
                 f"score = raw r$_A$, residual above baseline  ·  ε = {rel:g} ε$_c$", a.n_null)]
        for R in runs: R.map["contrast"] = R.map["contrast_orig"]
        rows += [dict(r, rel=rel) for r in k4(runs, rel, a.aligned, o, a.n_sub, a.seed)]
        rows += [dict(r, rel=rel) for r in k5(runs, rel, o)]
    rows += [dict(r, fig="K7") for r in k7(runs, order, out, a.n_null_eps)]
    for p in a.plan_pair.split(","):
        if p: rows += k6(runs, tuple(p.split(":")), a.aligned, out)
    pd.DataFrame(rows).to_csv(out / "key_226.csv", index=False)
    for r in rows: print(r)


if __name__ == "__main__":
    main()
