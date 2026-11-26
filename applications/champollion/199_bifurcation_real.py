#!/usr/bin/env python
"""
199 -- La bifurcation en ε_c sur nos vraies paires : balayage fin de ε par continuation, points bruts vs L2.

Théorie (191-193) :  au couplage produit T0, la jacobienne de l'itération de point fixe a pour rayon spectral ε_c/ε.
  * L2 (κ ≈ 0,002) : fourche nette attendue EXACTEMENT en ε = ε_c :
        - au-dessus : T ≈ T0, paramètres d'ordre m_ij ≈ 0, taux de contraction ≈ ε_c/ε (-> 1 en ε_c) ;
        - en dessous : m_11 ≠ 0, croissance en racine (m² ∝ 1 − ε/ε_c près du seuil), deux branches ± ;
        - modes suivants : seuils ε_ij = 4 λ_i λ'_j (écart λ1/λ2 petit chez nous -> seuils rapprochés).
  * brut (κ ~ 1-3) : forçage d'excentricité -> bifurcation imparfaite et/ou décalée ; pas de pic net attendu.
Paramètres d'ordre : m_ij = n · u_iᵀ T v_j, u_i, v_j vecteurs propres de J D J (branche de signe commune).
Deux continuations à chaud : « descente » depuis T0(1 + δ u1v1ᵀ) à 3 ε_c, et « montée » depuis un plan franchement
du côté m_11 < 0 à 0,15 ε_c. Fourche nette : les deux branches (±) se rejoignent en m = 0 à ε_c, sans hystérésis.
Fourche imparfaite : la branche opposée disparaît plus bas (col-nœud) et saute sur l'autre.

Ne sort que des scalaires : sweep.csv (une ligne par ε, graine, variante, métrique), meta.csv, summary.txt.
    python 199_bifurcation_real.py --src NTV3_MONDO --tgt BH_MEANTOK --metrics euclidean,geodesic_k20 \
        --variants l2,brut --out $METHODO/bif_199_ntv3_bh
GPU conseillé (n = 2000 ; ~50 ε × 2 graines × 4 combinaisons). Voir run_199_gpu.slurm.
"""
import argparse
import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r178", HERE / "178_r_null_test.py")
r178 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r178)
DT = torch.float64


@torch.no_grad()
def solve(Dx, Dy, eps, T, max_outer, tol, max_inner=2000, tol_inner=1e-9):
    """Même schéma que 178.gw_solve (convention POT), avec l'historique des écarts pour le taux de contraction."""
    n, m = T.shape
    lp = torch.full((n,), -math.log(n), dtype=DT, device=T.device); lq = torch.full((m,), -math.log(m), dtype=DT, device=T.device)
    f = torch.zeros(n, dtype=DT, device=T.device); g = torch.zeros(m, dtype=DT, device=T.device)
    hist, delta, conv = [], float("inf"), False
    for it in range(1, max_outer + 1):
        K = (4.0 / eps) * (Dx @ T @ Dy)
        tol_it = max(tol_inner, 0.1 * delta) if math.isfinite(delta) else 1e-3
        for j in range(1, max_inner + 1):
            f = lp - torch.logsumexp(K + g[None, :], 1)
            g = lq - torch.logsumexp(K + f[:, None], 0)
            if j % 10 == 0:
                err = float((torch.exp(torch.logsumexp(K + f[:, None] + g[None, :], 1)) - 1.0 / n).abs().sum())
                if not math.isfinite(err) or err < tol_it:
                    break
        Tn = torch.exp(K + f[:, None] + g[None, :])
        delta = float((Tn - T).abs().sum()); hist.append(delta); T = Tn
        if delta < tol:
            conv = True; break
    k = min(20, len(hist) - 1)
    rate = (hist[-1] / hist[-1 - k]) ** (1 / k) if k > 0 and hist[-1 - k] > 0 and hist[-1] > 0 else float("nan")
    return T, it, conv, float(rate)


def modes(p, q, kmax=3):
    n = p["n"]
    wx, Ux = torch.linalg.eigh(p["A"]); wy, Uy = torch.linalg.eigh(q["A"])
    lx, ly = (wx / n).cpu().numpy(), (wy / n).cpu().numpy()
    sign = "neg" if p["ln"] * q["ln"] >= p["lp"] * q["lp"] else "pos"
    ix = list(range(kmax)) if sign == "neg" else list(range(len(lx) - 1, len(lx) - 1 - kmax, -1))
    iy = list(range(kmax)) if sign == "neg" else list(range(len(ly) - 1, len(ly) - 1 - kmax, -1))
    pairs = sorted(((lx[i] * ly[j], a, b) for a, i in enumerate(ix) for b, j in enumerate(iy)), reverse=True)
    return sign, Ux[:, ix], Uy[:, iy], lx[ix], ly[iy], pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", "."), "aligned"))
    ap.add_argument("--src", default="NTV3_MONDO"); ap.add_argument("--tgt", default="BH_MEANTOK")
    ap.add_argument("--metrics", default="euclidean,geodesic_k20")
    ap.add_argument("--variants", default="l2,brut")
    ap.add_argument("--n-sub", type=int, default=2000); ap.add_argument("--subsample", default="perm", choices=["perm", "prefix"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hi", type=float, default=3.0); ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--npts", type=int, default=36); ap.add_argument("--dense", default="0.7,1.3,25", help="zone resserrée autour de ε_c : a,b,n")
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--max-outer", type=int, default=5000); ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = Path(a.aligned)
    N = np.load(A / f"{a.src}.npy", mmap_mode="r").shape[0]
    idx = (np.arange(a.n_sub) if a.subsample == "prefix" else np.sort(np.random.default_rng(a.seed).permutation(N)[:a.n_sub])) if 0 < a.n_sub < N else np.arange(N)
    Xs = np.asarray(np.load(A / f"{a.src}.npy", mmap_mode="r")[idx], dtype=np.float64)
    Xt = np.asarray(np.load(A / f"{a.tgt}.npy", mmap_mode="r")[idx], dtype=np.float64)
    da, db, dn = a.dense.split(",")
    grid = np.unique(np.round(np.r_[np.geomspace(a.hi, a.lo, a.npts), np.linspace(float(da), float(db), int(dn))], 5))[::-1]
    rows, meta = [], []
    for var in a.variants.split(","):
        r178.L2 = (var == "l2")
        for met in a.metrics.split(","):
            t0 = time.time()
            p = r178.describe(r178.dist_from_emb(Xs, met, dev)); q = r178.describe(r178.dist_from_emb(Xt, met, dev))
            ec = r178.eps_c(p, q)
            sign, U, V, lx, ly, pairs = modes(p, q)
            n = p["n"]
            e_x, e_y = p["e"], q["e"]
            v_sign = torch.sign((e_x @ U[:, 0]) * (e_y @ V[:, 0])) or 1.0
            V = V * v_sign                                    # signe de m_11 : celui que favorise le forçage
            kappa = float(e_x.std() * e_y.std()) / (ec / 4)
            top = pairs[:5]
            M = dict(variant=var, metric=met, n=n, eps_c=ec, branch=sign, kappa=kappa,
                     align=float(abs(e_x @ U[:, 0]) / e_x.norm() * abs(e_y @ V[:, 0]) / e_y.norm()),
                     **{f"eps_{a_ + 1}{b_ + 1}_over_c": 4 * pv / ec for pv, a_, b_ in top},
                     gap_x=float(lx[0] / lx[1]), gap_y=float(ly[0] / ly[1]))
            meta.append(M); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M.items()}, flush=True)
            T0 = torch.full((n, n), 1.0 / n ** 2, dtype=DT, device=dev)
            P = torch.outer(U[:, 0], V[:, 0])
            for sd, gr in (("descente", grid), ("montée", grid[::-1])):
                if sd == "descente":     # depuis le produit (+δ) à ε grand : suit la branche que choisit le système
                    T = (T0 * (1 + a.delta * P / P.abs().max())).clamp_min(1e-300)
                else:                    # depuis un plan franchement du côté m11 < 0 à ε petit : suit la branche opposée en montant
                    T = (T0 * (1 - 0.5 * P / P.abs().max())).clamp_min(1e-300)
                T = T / T.sum()
                for rel in gr:
                    eps = rel * ec
                    ts = time.time()
                    T, it, conv, rate = solve(p["D"], q["D"], eps, T, a.max_outer, a.tol)
                    mm = (n * (U.T @ T @ V)).cpu().numpy()
                    st = r178.plan_stats(p, q, T)
                    row = dict(variant=var, metric=met, seed_init=sd, rel=float(rel), eps=eps, n_outer=it, converged=conv, rate=rate,
                               rate_theory_product=1.0 / rel, **{f"m{i + 1}{j + 1}": float(mm[i, j]) for i in range(3) for j in range(3)},
                               r_off=st["r_off"], r_ecc=st["r_ecc"], r_rest=st["r_rest"], H_rel=st["H_rel"], nT2=st["nT2"],
                               secs=time.time() - ts)
                    rows.append(row)
                    print(f"{var:4s} {met:13s} {sd} ε/ε_c={rel:.4f} it={it:5d} taux={rate:.3f} (théorie produit {1 / rel:.3f}) "
                          f"m11={mm[0, 0]:+.3f} m12={mm[0, 1]:+.3f} m21={mm[1, 0]:+.3f} m22={mm[1, 1]:+.3f} r_off={st['r_off']:.4f}", flush=True)
                    pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
            print(f"{var} {met} fini en {time.time() - t0:.0f} s", flush=True)
            pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    # résumé : seuil observé = plus grand ε/ε_c où les deux graines se séparent (|m11(+δ) − m11(−δ)| > 0,05)
    S = pd.DataFrame(rows); L = []
    for (var, met), G in S.groupby(["variant", "metric"]):
        Gp = G[G.seed_init == "descente"].set_index("rel"); Gm = G[G.seed_init == "montée"].set_index("rel")
        both = (Gp.m11 > 0.02) & (Gm.m11.reindex(Gp.index) < -0.02)
        split = both[both].index.max() if both.any() else np.nan
        peak = Gp.n_outer.idxmax()
        above = Gp[Gp.index > 1.05]
        err_rate = float((above.rate - above.rate_theory_product).abs().median()) if len(above) else np.nan
        near = Gp[(Gp.index < 1) & (Gp.index > 0.5) & (Gp.m11.abs() > 0.02)]
        fit = np.nan
        if len(near) >= 3:
            c = np.polyfit(near.index.values, near.m11.values ** 2, 1); fit = -c[1] / c[0] if c[0] != 0 else np.nan
        L.append(f"[{var} {met}] deux branches ± coexistent jusqu'à ε/ε_c = {split:.3f} ; pic d'itérations à {peak:.3f} ({Gp.n_outer.max()} it.) ; "
                 f"|taux − ε_c/ε| médian au-dessus de 1,05 ε_c = {err_rate:.3f} ; zéro de l'ajustement m11² ∝ (ε* − ε) : ε*/ε_c = {fit:.3f}")
    s = "\n".join(L); print("\n" + s); (out / "summary.txt").write_text(s + "\n")
    print(f"\nécrit : {out}/sweep.csv, meta.csv, summary.txt (scalaires uniquement)")


if __name__ == "__main__":
    main()
