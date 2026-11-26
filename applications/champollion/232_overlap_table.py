"""
232 -- are high CKA, same-subject retrieval and strong contrast found in the SAME region pairs?
Prints one table per ε: every overlapping pair (and the 15 homologues), with
  cka (linear, metric-independent), and for each metric: contrast, residual above baseline, n·tr(T).
Aggregates and region names only.

    python 232_overlap_table.py --run euc=$EUC --run geo20=$GEO --run orc20=$ORC --rels 0.1,0.03 | tee $OUT/overlap_table.log
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_l = lambda n, f: (lambda s: (s, s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[2])(importlib.util.spec_from_file_location(n, HERE / f))
m224 = _l("m224", "224_gw_story.py"); m226 = _l("m226", "226_key_figures.py")

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True); ap.add_argument("--rels", default="0.1,0.03")
ap.add_argument("--n-null", type=int, default=200)
a = ap.parse_args()
runs = [m224.Run(n, d.split(",")) for n, d in (r.split("=", 1) for r in a.run)]
runs = [R for R in runs if len(R.map)]
for R in runs: R.map["contrast_orig"] = R.map["contrast"]
order = m224.region_order(runs)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)

for rel in [float(x) for x in a.rels.split(",")]:
    T = None
    for R in runs:
        df, _ = m226.residual_table(R, rel, order, "contrast_orig", a.n_null)
        df = df[df.type.isin(["overlap", "homologue"])][["src", "tgt", "type", "cka", "contrast_orig", "res", "ntrT"]]
        df = df.rename(columns={"contrast_orig": f"con_{R.name}", "res": f"res_{R.name}", "ntrT": f"ntrT_{R.name}"})
        T = df if T is None else T.merge(df.drop(columns=["cka", "type"]), on=["src", "tgt"])
    T.insert(0, "pair", [m224.pair_label(x, y) for x, y in zip(T.src, T.tgt)]); T = T.drop(columns=["src", "tgt"])
    T = T.sort_values(["type", "cka"], ascending=[False, False])
    num = T.select_dtypes("number").columns
    print(f"\n==== ε = {rel:g} ε_c ====")
    print(T.to_string(index=False, formatters={c: ("{:.3f}".format if c == "cka" else "{:.1f}".format if c.startswith("ntrT") else "{:+.4f}".format) for c in num}))
    ov = T[T.type == "overlap"]
    for R in runs:
        n = R.name
        print(f"  overlap, {n:6s}: Spearman(cka, contrast) = {spearmanr(ov.cka, ov[f'con_{n}'])[0]:+.2f}   "
              f"Spearman(cka, n·tr(T)) = {spearmanr(ov.cka, ov[f'ntrT_{n}'])[0]:+.2f}   "
              f"Spearman(contrast, n·tr(T)) = {spearmanr(ov[f'con_{n}'], ov[f'ntrT_{n}'])[0]:+.2f}   "
              f"residual > 0: {(ov[f'res_{n}'] > 0).sum()}/{len(ov)}   n·tr(T) ≥ 3: {(ov[f'ntrT_{n}'] >= 3).sum()}/{len(ov)}")
