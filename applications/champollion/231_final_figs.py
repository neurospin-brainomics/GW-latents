#!/usr/bin/env python
"""
231 -- The result figures of the talk, computed from the 215 maps with the SIMPLE test (default of 224):
       statistic = mean score of the 15 true homologues; null = random left↔right pairings of the same regions.
Aggregates only. Prints every number shown (copy the console to Claude).

  F1_raw_vs_contrast.png   homologues above baseline (of 15) + p, raw score vs contrast, three metrics, at --rel
  F2_contrast_across_eps.png  mean contrast of homologues − mean contrast of unrelated pairs, every ε, three metrics
  F3_spectrum.png          homologue spectral distance vs random pairings; ρ(spectral distance, score) raw / contrast
  F4_robustness.png        contrast, homologues above baseline for every run given (variants included)
  numbers_231.csv / console

    python 231_final_figs.py --aligned $METHODO/aligned_champ --out $METHODO/final_231 \
      --run euc=$METHODO/champ_map_215_pcperm --run geo20=$METHODO/champ_map_215_geo20 --run orc20=$METHODO/champ_map_215_orc20 \
      --variant geo10=$METHODO/champ_sub_geo10 --variant geo40=$METHODO/champ_sub_geo40 --variant geo80=$METHODO/champ_sub_geo80 \
      --variant geo160=$METHODO/champ_sub_geo160 --variant geo20_s2=$METHODO/champ_sub_geo20_s2 --variant orc10=$METHODO/champ_sub_orc10 \
      --variant orc40=$METHODO/champ_sub_orc40 --variant orc20_it3=$METHODO/champ_sub_orc20_it3 --variant orc20_it5=$METHODO/champ_sub_orc20_it5
Needs champ_style.py, 222_champ_figs_jz.py, 224_gw_story.py, 225_spectra.py, 226_key_figures.py next to it.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ_style as S
_l = lambda n, f: (lambda s: (s, s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[2])(importlib.util.spec_from_file_location(n, HERE / f))
m224 = _l("m224", "224_gw_story.py"); m225 = _l("m225", "225_spectra.py"); m226 = _l("m226", "226_key_figures.py")
C, NAME, BG, INK, INK2, MUTED = m226.C, m226.NAME, m226.BG, m226.INK, m226.INK2, m226.MUTED
NULLC, NULLBAND, GRID, STRIPE = m226.NULLC, m226.NULLBAND, m226.GRID, m226.STRIPE
fam = lambda n: n[:3]
ROWS = []


def pstr(p, n):
    if not np.isfinite(p): return "n/a"
    return f"p < {1 / n:.3f}".rstrip("0") if p <= 1 / (n + 1) + 1e-12 else f"p = {p:.2g}"


def test(R, rel, order, col, n_null):
    R.map["contrast"] = R.map[col]; ra = m224.residual_analysis(R.at(rel), order, n_null)
    R.map["contrast"] = R.map["contrast_orig"]; return ra


def f1(runs, rel, order, out, n_null):
    fig = plt.figure(figsize=(S.W, S.H))
    m226.headline(fig, "Beyond the spectrum, homologues stand out consistently only with manifold metrics",
                  f"homologue pairs above baseline (of 15)  ·  p: mean of true homologues vs {n_null} random left↔right pairings  ·  ε = {rel:g} ε$_c$")
    for c, (lab, col) in enumerate([("Raw GW score", "rA"), ("Contrast (beyond the spectrum)", "contrast_orig")]):
        ax = fig.add_axes([0.16 + c * 0.43, 0.12, 0.37, 0.6]); y = np.arange(len(runs))[::-1]
        for yi, R in zip(y, runs):
            ra = test(R, rel, order, col, n_null); k = fam(R.name); ok = np.isfinite(ra["p"]) and ra["p"] < 0.05
            ax.barh(yi, ra["n_pos"], height=0.55, color=C[k] if ok else NULLBAND, edgecolor=C[k], lw=0 if ok else 2.2)
            ax.text(ra["n_pos"] + 0.25, yi + 0.06, f"{ra['n_pos']}/{ra['n_hom']}", fontsize=19, weight="bold", va="center", color=INK if ok else MUTED)
            ax.text(ra["n_pos"] + 0.25, yi - 0.2, pstr(ra["p"], len(ra["null"])), fontsize=11.5, va="center", color=INK2)
            ROWS.append(dict(fig="F1", run=R.name, score=col, rel=rel, n_pos=ra["n_pos"], n_hom=ra["n_hom"], p=ra["p"], obs=ra["obs"]))
        ax.set_yticks(y); ax.set_yticklabels([NAME[fam(R.name)] for R in runs] if c == 0 else [], fontsize=15)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 17.5); ax.set_ylim(-0.75, len(runs) - 0.2); ax.set_xticks([0, 5, 10, 15]); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        fig.text(0.16 + c * 0.43, 0.765, lab, fontsize=19, weight="bold")
    S.save(fig, out / f"F1_raw_vs_contrast_eps{rel:g}.png")


def f2(runs, out):
    """one panel per metric: homologue − unrelated contrast across ε, 90 % bootstrap band; filled dot = band excludes 0"""
    g = np.random.default_rng(0); res = []
    for R in runs:
        rels = R.rels(); d, lo, hi = [], [], []
        for r in rels:
            x = R.at(r); h = x[x.type == "homologue"].contrast_orig.dropna().values; o = x[x.type == "other"].contrast_orig.dropna().values
            d.append(h.mean() - o.mean()); bs = [g.choice(h, len(h)).mean() - g.choice(o, len(o)).mean() for _ in range(1000)]
            lo.append(np.percentile(bs, 5)); hi.append(np.percentile(bs, 95))
            ROWS.append(dict(fig="F2", run=R.name, rel=r, diff=d[-1], lo=lo[-1], hi=hi[-1]))
        res.append((R, np.array(rels), np.array(d), np.array(lo), np.array(hi)))
    fig = plt.figure(figsize=(S.W, S.H))
    m226.headline(fig, "Manifold metrics keep a reliable homologue excess across ε",
                  "homologues − unrelated pairs, contrast score  ·  band: 90 % bootstrap  ·  filled dot: band above 0")
    ymin = min(0, min(r[3].min() for r in res)); ymax = max(r[4].max() for r in res); pad = 0.08 * (ymax - ymin)
    n = len(res); w = 0.84 / n
    for c, (R, rels, d, lo, hi) in enumerate(res):
        k = fam(R.name); ax = fig.add_axes([0.08 + c * w, 0.13, w - 0.035, 0.56])
        ax.axhspan(ymin - pad, 0, color=NULLBAND, alpha=.55, lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=1.4, zorder=1)
        ax.fill_between(rels, lo, hi, color=C[k], alpha=.25, lw=0, zorder=2)
        ax.plot(rels, d, color=C[k], lw=3.5, zorder=3)
        ok = lo > 0
        ax.scatter(rels[ok], d[ok], s=170, color=C[k], edgecolor=BG, lw=2, zorder=4)
        ax.scatter(rels[~ok], d[~ok], s=170, color=BG, edgecolor=C[k], lw=2.5, zorder=4)
        ax.set_xscale("log"); ax.set_xlim(rels.max() * 1.6, rels.min() / 1.6); ax.minorticks_off()
        ax.set_xticks(rels); ax.set_xticklabels([f"{e:g}" for e in rels], fontsize=13)
        ax.set_ylim(ymin - pad, ymax + pad); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        if c: ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        else: ax.set_ylabel("homologue excess (contrast)", fontsize=14)
        ax.set_xlabel("ε / ε$_c$  (finer →)", fontsize=13)
        fig.text(0.08 + c * w, 0.745, NAME[k], fontsize=19, weight="bold", color=C[k])
        fig.text(0.08 + c * w, 0.71, f"band above 0 at {ok.sum()}/{len(ok)} ε", fontsize=13.5, color=INK2)
    S.save(fig, out / "F2_contrast_across_eps.png")


def f3(runs, rel, aligned, out, n_sub, seed, n_null=2000):
    A = Path(aligned); regs = sorted(p.stem for p in A.glob("*.npy") if m224.side(p.stem))
    N = np.load(A / f"{regs[0]}.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(seed).permutation(N)[:n_sub]) if n_sub < N else np.arange(N)
    spec = {r: m225.spectrum(np.asarray(np.load(A / f"{r}.npy", mmap_mode="r")[idx], dtype=np.float64)) for r in regs}
    dist = lambda x, y: float(np.abs(spec[x] - spec[y]).sum())
    bases_ = sorted({m224.base(r) for r in regs if f"{m224.base(r)}_left" in spec and f"{m224.base(r)}_right" in spec})
    obs = np.mean([dist(f"{b}_left", f"{b}_right") for b in bases_]); g = np.random.default_rng(0); null = []
    for _ in range(n_null):
        p = g.permutation(len(bases_))
        while np.any(p == np.arange(len(bases_))): p = g.permutation(len(bases_))
        null.append(np.mean([dist(f"{bases_[i]}_left", f"{bases_[p[i]]}_right") for i in range(len(bases_))]))
    null = np.array(null); p_spec = (1 + (null <= obs).sum()) / (1 + n_null)
    ROWS.append(dict(fig="F3", what="spectral distance", obs=obs, null_med=np.median(null), null_q05=np.percentile(null, 5), p=p_spec))
    fig = plt.figure(figsize=(S.W, S.H))
    m226.headline(fig, "Homologues share their spectrum; only Euclidean GW follows it",
                  f"all 435 region pairs  ·  ε = {rel:g} ε$_c$")
    ax = fig.add_axes([0.16, 0.15, 0.29, 0.55])
    ax.barh([1], [np.median(null)], height=0.45, color=NULLBAND, lw=0); ax.barh([0], [obs], height=0.45, color=C["orc"], lw=0)
    q05 = np.percentile(null, 5); ax.plot([q05, q05], [0.7, 1.3], color=MUTED, lw=1.5, ls=(0, (3, 2)))
    ax.text(q05, 1.33, "5th pct", fontsize=11, color=MUTED, ha="center", va="bottom")
    ax.text(np.median(null) + 0.004, 1, f"{np.median(null):.3f}", va="center", fontsize=15, color=INK2)
    ax.text(obs + 0.004, 0, f"{obs:.3f}", va="center", fontsize=17, weight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true\nhomologues", "random L↔R\npairings (median)"], fontsize=14)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.set_xlim(0, max(np.median(null), obs) * 1.3); ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("spectral distance (L1)  →  less similar", fontsize=13.5); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    fig.text(0.16, 0.765, "Spectral distance", fontsize=19, weight="bold"); fig.text(0.45, 0.765, pstr(p_spec, n_null), fontsize=15, color=INK2, ha="right")
    ax = fig.add_axes([0.57, 0.15, 0.38, 0.55]); w = 0.36
    for i, R in enumerate(runs):
        d = R.at(rel).copy(); d["sd"] = [dist(x, y) for x, y in zip(d.src, d.tgt)]; k = fam(R.name)
        for j, col in enumerate(["rA", "contrast_orig"]):
            val = -spearmanr(d.sd, d[col], nan_policy="omit")[0]; xx = i + (j - 0.5) * w   # ρ(spectral similarity, score)
            ax.bar(xx, val, width=w * 0.92, color=C[k] if j == 0 else BG, edgecolor=C[k], lw=0 if j == 0 else 2.4)
            ax.text(xx, val + (0.02 if val >= 0 else -0.02), f"{val:+.2f}".replace("-", "−"), ha="center", va="bottom" if val >= 0 else "top", fontsize=14,
                    weight="bold" if abs(val) > 0.1 else "normal", color=INK if abs(val) > 0.1 else MUTED)
            ROWS.append(dict(fig="F3", run=R.name, score=col, rel=rel, spearman=val))
    ax.axhline(0, color=MUTED, lw=1); vmax = max(0.3, max(abs(r["spearman"]) for r in ROWS if r.get("fig") == "F3" and r.get("rel") == rel)); ax.set_ylim(-vmax * 1.3, vmax * 1.3); ax.set_xticks(range(len(runs))); ax.set_xticklabels([NAME[fam(R.name)] for R in runs], fontsize=14)
    ax.tick_params(axis="x", length=0); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_ylabel("Spearman ρ\n(similar spectra ↔ higher score)", fontsize=12.5)
    fig.text(0.57, 0.765, "Does GW follow the spectrum?", fontsize=19, weight="bold")
    ax.legend(handles=[matplotlib.patches.Patch(color=INK2, label="raw score"), matplotlib.patches.Patch(facecolor=BG, edgecolor=INK2, lw=2, label="contrast")],
              loc="upper right", fontsize=12.5, frameon=False, ncol=1)
    S.save(fig, out / f"F3_spectrum_eps{rel:g}.png")


def f4(allruns, rel, order, out, n_null):
    lab = lambda R: (NAME[fam(R.name)] + (f"  k={R.k}" if R.k and fam(R.name) != "euc" else "") + (f" · {R.it} Ricci-flow steps" if R.it > 1 else "")
                     + (" · new subjects" if R.tag and R.tag.startswith("s") else ""))
    rows = []
    for R in allruns:
        if not len(R.at(rel)): print(f"  F4: {R.name} has no ε = {rel:g} ε_c, skipped"); continue
        ra = test(R, rel, order, "contrast_orig", n_null); full = R.at(rel).groupby(["src", "tgt"]).ngroups >= 300
        rows.append((lab(R), ra["n_pos"], fam(R.name), not full)); ROWS.append(dict(fig="F4", run=R.name, rel=rel, n_pos=ra["n_pos"], full_map=full, p=ra["p"]))
    fig = plt.figure(figsize=(S.W, S.H))
    m226.headline(fig, "Every variant keeps a majority of homologues above baseline", f"contrast  ·  homologue pairs above baseline (of 15)  ·  ε = {rel:g} ε$_c$  ·  hollow: 115-pair subset")
    ax = fig.add_axes([0.33, 0.08, 0.6, 0.68]); y = np.arange(len(rows))[::-1]
    for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color=STRIPE, lw=0, zorder=0)
    for yi, (l, n, k, sub) in zip(y, rows):
        ax.plot([0, n], [yi, yi], color=C[k], lw=4, alpha=.35)
        ax.scatter(n, yi, s=170, color=BG if sub else C[k], edgecolor=C[k], lw=2.5, zorder=4)
        ax.text(n + 0.35, yi, str(n), ha="left", va="center", fontsize=14, weight="bold", color=INK, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=14); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 16.2); ax.set_ylim(-0.6, len(rows) - 0.1); ax.set_xticks([0, 5, 10, 15]); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    S.save(fig, out / f"F4_robustness_eps{rel:g}.png")


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", ""), "aligned_champ"))
    ap.add_argument("--run", action="append", default=[], help="main full maps: euc=DIR geo20=DIR orc20=DIR")
    ap.add_argument("--variant", action="append", default=[], help="extra runs for F4 (NAME=DIR)")
    ap.add_argument("--rel", type=float, default=0.1); ap.add_argument("--spectrum-rels", default="0.1")
    ap.add_argument("--n-null", type=int, default=1000); ap.add_argument("--n-sub", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--baseline", choices=["local", "rank1"], default="local")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); m226.style(); m224.BASELINE = a.baseline
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [m224.Run(n, d.split(",")) for n, d in (r.split("=", 1) for r in a.run)]
    var = [m224.Run(n, d.split(",")) for n, d in (r.split("=", 1) for r in a.variant)]
    for R in [R for R in runs + var if not len(R.map)]: print(f"  {R.name}: no map_shard_*.csv in {[str(d) for d in R.dirs]} — skipped")
    runs = [R for R in runs if len(R.map)]; var = [R for R in var if len(R.map)]
    for R in runs + var: R.map["contrast_orig"] = R.map["contrast"]
    order = m224.region_order(runs)
    f1(runs, a.rel, order, out, a.n_null); f2(runs, out)
    for r in [float(x) for x in a.spectrum_rels.split(",") if x]: f3(runs, r, a.aligned, out, a.n_sub, a.seed)
    order_v = m224.region_order(runs + var)
    by = {R.name: R for R in runs + var}
    ordered = [by[n] for n in ["euc", "geo10", "geo20", "geo20_s2", "geo40", "geo80", "geo160", "orc10", "orc20", "orc40", "orc20_it3", "orc20_it5"] if n in by]
    ordered += [R for R in runs + var if R not in ordered]
    f4(ordered, a.rel, order_v, out, a.n_null)
    D = pd.DataFrame(ROWS); D.to_csv(out / f"numbers_231_eps{a.rel:g}.csv", index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 200): print(D.to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
