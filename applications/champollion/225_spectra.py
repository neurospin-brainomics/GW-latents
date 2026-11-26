#!/usr/bin/env python
"""
225 -- Are homologous regions similar in their variance spectrum? And is that what the raw GW score sees?

Per region: eigenvalues of the latent covariance (same subjects as the runs), normalised to sum to 1.
Spectral distance between two regions = L1 distance between their normalised spectra (0 = identical profiles).

  S1_homologue_spectra.png   the 15 homologue pairs: left vs right spectra (cumulative variance), one panel each
  S2_spectral_distance.png   spectral distance by pair type + true vs fake homologues (random L↔R re-pairings)
  S3_spectrum_vs_gw.png      per region pair: spectral distance vs GW score, raw (rA) and contrast, for each metric
  spectra_225.csv            the numbers

If the raw Euclidean GW effect is a spectrum effect, S2 shows homologues closer in spectrum, and S3 shows the raw
Euclidean score tracking spectral distance while the contrast does not.  Aggregates only.

    python 225_spectra.py --aligned $METHODO/aligned_champ --out $METHODO/spectra_225 \
        --run euc=$METHODO/champ_map_215_pcperm --run geo20=$METHODO/champ_map_215_geo20 --run orc20=$METHODO/champ_map_215_orc20
CPU, < 1 min. Needs champ_style.py, 222_champ_figs_jz.py and 224_gw_story.py next to it.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champ_style as S
from champ_style import INK, INK2, MUTED, NULL, GRID, BG, BAND, GEO, ORC, EUC
_s = importlib.util.spec_from_file_location("m224", HERE / "224_gw_story.py"); m224 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m224)
base, side, group5, GCOL, GLAB = m224.base, m224.side, m224.group5, m224.GCOL, m224.GLAB


def spectrum(X):
    X = X - X.mean(0); lam = np.clip(np.linalg.eigvalsh(np.cov(X.T))[::-1], 0, None); return lam / lam.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR of a 215 map, as for 224 (for S3)")
    ap.add_argument("--rel", type=float, default=0.1); ap.add_argument("--n-sub", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-null", type=int, default=2000)
    a = ap.parse_args(); S.apply(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = Path(a.aligned); regs = sorted(p.stem for p in A.glob("*.npy") if side(p.stem))
    N = np.load(A / f"{regs[0]}.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(a.seed).permutation(N)[: a.n_sub]) if a.n_sub < N else np.arange(N)
    spec = {r: spectrum(np.asarray(np.load(A / f"{r}.npy", mmap_mode="r")[idx], dtype=np.float64)) for r in regs}
    pr = {r: 1 / (spec[r] ** 2).sum() for r in regs}
    rows = [dict(src=x, tgt=y, type=group5(x, y), spec_dist=float(np.abs(spec[x] - spec[y]).sum()))
            for i, x in enumerate(regs) for y in regs[i + 1:]]
    D = pd.DataFrame(rows)
    bases_ = sorted({base(r) for r in regs if f"{base(r)}_left" in spec and f"{base(r)}_right" in spec})
    dist = lambda x, y: float(np.abs(spec[x] - spec[y]).sum())
    obs = np.mean([dist(f"{b}_left", f"{b}_right") for b in bases_])
    g = np.random.default_rng(0); null = []
    for _ in range(a.n_null):
        p = g.permutation(len(bases_))
        while np.any(p == np.arange(len(bases_))): p = g.permutation(len(bases_))
        null.append(np.mean([dist(f"{bases_[i]}_left", f"{bases_[p[i]]}_right") for i in range(len(bases_))]))
    null = np.array(null); p_fake = (1 + (null <= obs).sum()) / (1 + len(null))
    p_mw = mannwhitneyu(D[D.type == "homologue"].spec_dist, D[D.type == "other"].spec_dist, alternative="less").pvalue
    print(f"spectral distance: homologues {obs:.3f} (mean) vs fake re-pairings median {np.median(null):.3f}, q05 {np.percentile(null, 5):.3f}; "
          f"p = {p_fake:.3g}; Mann–Whitney homol. < other p = {p_mw:.2g}")

    # S1 homologue spectra
    fig = plt.figure(figsize=(S.W, S.H)); S.headline(fig, "Variance spectra of homologous regions", "cumulative share of variance  ·  left (solid) vs right (dashed)")
    nc = 5; nr = int(np.ceil(len(bases_) / nc)); k = np.arange(1, len(spec[regs[0]]) + 1)
    for i, b in enumerate(bases_):
        ax = fig.add_axes([0.05 + (i % nc) * 0.188, 0.08 + (nr - 1 - i // nc) * (0.7 / nr), 0.16, 0.7 / nr - 0.07])
        ax.plot(k, np.cumsum(spec[f"{b}_left"]), color=GEO, lw=2.4); ax.plot(k, np.cumsum(spec[f"{b}_right"]), color=ORC, lw=2.4, ls=(0, (4, 2)))
        ax.set_ylim(0, 1.02); ax.set_xlim(1, k[-1]); ax.set_xticks([1, 16, 32] if i // nc == nr - 1 else []); ax.set_yticks([0, 0.5, 1] if i % nc == 0 else [])
        ax.tick_params(labelsize=10); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.set_title(f"{b.replace('_', ' ')}   L1 {dist(f'{b}_left', f'{b}_right'):.2f}", fontsize=11, loc="left")
    fig.legend(handles=[Line2D([], [], color=GEO, lw=2.4, label="left"), Line2D([], [], color=ORC, lw=2.4, ls=(0, (4, 2)), label="right")],
               loc="upper right", bbox_to_anchor=(0.97, 0.93), ncol=2, fontsize=13)
    S.save(fig, out / "S1_homologue_spectra.png")

    # S2 distance by type + fake null
    fig = plt.figure(figsize=(S.W, S.H))
    S.headline(fig, "Are homologues closer in spectrum than other pairs?", f"L1 distance between normalised covariance spectra  ·  {len(idx)} subjects")
    ax = fig.add_axes([0.07, 0.13, 0.42, 0.6]); m224.strips(ax, D, "spec_dist"); ax.set_ylabel("spectral distance (L1)")
    ax.set_title(f"by pair type  ·  homol. < other: Mann–Whitney {m224.fmt_p(p_mw)}", fontsize=14)
    ax = fig.add_axes([0.58, 0.13, 0.36, 0.6])
    ax.hist(null, bins=40, color=NULL, edgecolor=BG, alpha=.9); ax.axvline(obs, color=ORC, lw=4); ax.axvline(np.percentile(null, 5), color=INK2, lw=1.2, ls=(0, (4, 3)))
    left = obs < np.median(null)
    ax.text(obs, ax.get_ylim()[1] * 0.5, " true\n homologues " if left else " true\n homologues", fontsize=13, weight="bold", va="center",
            ha="right" if left else "left", rotation=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("mean spectral distance of the 15 pairings")
    ax.set_title(f"true vs {len(null)} fake re-pairings  ·  {m224.fmt_p(p_fake, len(null))}", fontsize=14)
    S.save(fig, out / "S2_spectral_distance.png")

    # S3 spectrum vs GW score
    runs = [m224.Run(n_, d_.split(",")) for n_, d_ in (r.split("=", 1) for r in a.run)]
    recs = []
    if runs:
        fig = plt.figure(figsize=(S.W, S.H))
        S.headline(fig, "Does the GW score track spectral similarity?", f"one dot per region pair  ·  ε = {a.rel:g} ε$_c$  ·  ρ: Spearman (spectral distance vs score)")
        for c, R in enumerate(runs):
            G = R.at(a.rel).merge(D[["src", "tgt", "spec_dist"]], on=["src", "tgt"])
            for r_, (col, lab) in enumerate([("rA", "raw GW score (rA)"), ("contrast", "contrast (− surrogate)")]):
                if col not in G: continue
                ax = fig.add_axes([0.08 + c * 0.31, 0.43 - r_ * 0.35, 0.25, 0.27])
                for t in ["other", "shared, cross", "overlap", "homologue"]:
                    j = G[G.type == t]
                    if t == "shared, cross": ax.scatter(j.spec_dist, j[col], s=22, facecolor="none", edgecolor=INK, lw=0.8)
                    else: ax.scatter(j.spec_dist, j[col], s=10 if t == "other" else 40, color=GCOL[t], alpha=.5 if t == "other" else .95, edgecolor=BG, lw=.5, zorder=3)
                rho, p = spearmanr(G.spec_dist, G[col], nan_policy="omit")
                ax.text(0.97, 0.95, f"ρ = {rho:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=14, weight="bold")
                ax.tick_params(labelsize=10); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
                if r_ == 0: ax.set_title(R.label, fontsize=15, loc="left")
                if c == 0: ax.set_ylabel(lab, fontsize=12.5)
                if r_ == 1: ax.set_xlabel("spectral distance (L1)", fontsize=12.5)
                recs.append(dict(run=R.name, score=col, spearman=rho, p=p))
                print(f"  {R.name:8s} {col:9s}: Spearman(spectral distance, score) = {rho:+.2f} (p = {p:.2g})")
        m224.type_legend(fig, y=0.865)
        S.save(fig, out / "S3_spectrum_vs_gw.png")
    D.assign(pr_src=[pr[x] for x in D.src], pr_tgt=[pr[y] for y in D.tgt]).to_csv(out / "spectra_225.csv", index=False)
    pd.DataFrame(recs + [dict(run="spectrum", score="homologue fake-null", spearman=np.nan, p=p_fake)]).to_csv(out / "spectra_225_summary.csv", index=False)


if __name__ == "__main__":
    main()
