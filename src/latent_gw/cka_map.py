"""Debiased linear CKA between every pair of spaces, plus an optional test of expected pairs.

All spaces must list the SAME samples in the SAME order (CKA uses the pairing). Same preprocessing as the
Champollion maps (215_champ_map.py): random subsample of n_sub samples (seed), optional L2 normalisation of the
points (centre, unit norm), debiased linear CKA (unbiased HSIC). With the defaults (n_sub=2000, seed=1, L2) the
values equal the `cka` column of the maps.

Command line (aggregates only: one CKA value per pair of spaces):

    python -m latent_gw.cka_map --inputs $ALIGNED --out $OUT/cka              # every *.npy in the folder
    python -m latent_gw.cka_map --inputs a.npy,b.npy,c.npy --out $OUT/cka
    python -m latent_gw.cka_map --inputs $ALIGNED --glob '*_left.npy,*_right.npy' --out $OUT/cka --homologues

Outputs: cka_pairs.csv (src, tgt, cka), cka_matrix.csv, and with --homologues: cka_homologues.csv (residual per
homologue pair) and cka_test.txt (mean CKA of homologues vs random left/right re-pairings, sign test).
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .cka import gram0, hs
from .inference import pair_test
from .preprocessing import l2_points


def subsample_index(N, n_sub=2000, seed=1):
    """Same subsample as 215_champ_map.py: sorted prefix of a seeded permutation (all samples if n_sub <= 0 or >= N)."""
    return np.sort(np.random.default_rng(seed).permutation(N)[:n_sub]) if 0 < n_sub < N else np.arange(N)


def cka_matrix(arrays, l2=True, dev="cpu", progress=True):
    """arrays: dict name -> (n, d) array, same samples in the same order. Returns a DataFrame (names x names).
    Each Gram matrix and its self-HSIC is computed once (memory: one n x n float64 matrix per space)."""
    names = list(arrays)
    G, self_hs = {}, {}
    for nm in tqdm(names, desc="Gram matrices", disable=not progress, leave=False):
        X = np.asarray(arrays[nm], dtype=np.float64)
        G[nm] = gram0(l2_points(X) if l2 else X, dev); self_hs[nm] = hs(G[nm], G[nm])
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    pairs = list(itertools.combinations(names, 2))
    for a, b in tqdm(pairs, desc="CKA pairs", disable=not progress, leave=False):
        den = np.sqrt(self_hs[a] * self_hs[b])
        M.loc[a, b] = M.loc[b, a] = hs(G[a], G[b]) / den if den > 0 else np.nan
    return M


def homologue_test(M, n_null=1000, seed=0, pattern=r"_(left|right)$"):
    """Left/right homologues (names <base>_left / <base>_right): baseline and residual per pair, mean CKA of the
    homologues vs random left/right re-pairings, sign test (inference.pair_test; candidates on the same side)."""
    names = list(M.index); ix = {r: i for i, r in enumerate(names)}
    base = lambda r: re.sub(pattern, "", r)
    side = lambda r: (re.search(pattern, r) or [None, None])[1]
    bases = [base(r) for r in names if side(r) == "left" and f"{base(r)}_right" in ix]
    A = [ix[f"{b}_left"] for b in bases]; B = [ix[f"{b}_right"] for b in bases]
    left = [i for i, r in enumerate(names) if side(r) == "left"]; right = [i for i, r in enumerate(names) if side(r) == "right"]
    t = pair_test(M.values, A, B, cand_a=left, cand_b=right, n_null=n_null, seed=seed)
    per_pair = pd.DataFrame(dict(base=bases, cka=M.values[A, B], residual=t["residuals"]))
    return t, per_pair


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="folder of .npy files, or comma-separated list of .npy files")
    ap.add_argument("--glob", default="*.npy", help="file patterns inside the --inputs folder, comma-separated (e.g. '*_left.npy,*_right.npy')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sub", type=int, default=2000, help="samples used (<= 0: all)")
    ap.add_argument("--seed", type=int, default=1, help="subsample seed (1 = same subjects as the Champollion maps)")
    ap.add_argument("--no-l2", action="store_true", help="skip the L2 normalisation of the points")
    ap.add_argument("--homologues", action="store_true", help="test <base>_left / <base>_right pairs")
    ap.add_argument("--n-null", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Path(a.inputs)
    files = sorted({f for g in a.glob.split(",") for f in src.glob(g)}) if src.is_dir() else [Path(f) for f in a.inputs.split(",") if f]
    if len(files) < 2:
        raise SystemExit("need at least two .npy files")
    N = np.load(files[0], mmap_mode="r").shape[0]
    for f in files:
        if np.load(f, mmap_mode="r").shape[0] != N:
            raise SystemExit(f"{f.name}: {np.load(f, mmap_mode='r').shape[0]} rows, expected {N} (same samples everywhere)")
    idx = subsample_index(N, a.n_sub, a.seed)
    arrays = {f.stem: np.asarray(np.load(f, mmap_mode="r")[idx], dtype=np.float64) for f in files}
    print(f"{len(arrays)} spaces, {len(idx)} samples, L2 = {not a.no_l2}, device = {dev}", flush=True)
    M = cka_matrix(arrays, l2=not a.no_l2, dev=dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "cka_matrix.csv")
    iu = np.triu_indices(len(M), 1)
    pd.DataFrame(dict(src=M.index[iu[0]], tgt=M.columns[iu[1]], cka=M.values[iu])).to_csv(out / "cka_pairs.csv", index=False)
    msg = [f"written: {out}/cka_matrix.csv, cka_pairs.csv"]
    if a.homologues:
        t, per = homologue_test(M, a.n_null)
        per.to_csv(out / "cka_homologues.csv", index=False)
        txt = (f"CKA, left/right homologues: mean {t['obs']:.4f}, p = {t['p']:.4g} vs {a.n_null} random re-pairings; "
               f"{t['n_pos']}/{t['n']} above their baseline (sign test p = {t['p_sign']:.3g})")
        (out / "cka_test.txt").write_text(txt + "\n"); msg = [txt] + msg + [f"         {out}/cka_homologues.csv, cka_test.txt"]
    print("\n".join(msg))


if __name__ == "__main__":
    main()
