"""Compare two latent spaces: the per-pair computation of the Champollion maps (215_champ_map.py), as a function.

    import latent_gw as lg
    res = lg.compare(X, Y, metric="geodesic_k20", surrogate="pcperm", B=4)

For one pair (X, Y):
  - entropic GW by downward continuation in eps / eps_c (eps_c specific to the pair);
  - rA = <A_x, T A_y T^T> / (sigma sigma') and the part c11 carried by the first pair of modes;
  - the same for B surrogates of X (same spectrum, no correspondence between samples):
    contrast = rA(X) - mean rA(surrogates), z = contrast / (sd sqrt(1 + 1/B));
  - pairing (both clouds list the SAME samples): n·tr(T) (x1 = chance) and top-1 retrieval;
  - debiased linear CKA between X and Y.
Decomposition (plan_stats): r = r_ecc + rA, rA = r_diag + r_rest, r_off; contrast and z for each part
(columns c_<part>, z_<part>). l2=False keeps the norms (no L2 normalisation of the points).
One row per eps / eps_c. The first grid point above 1 serves as a control (expected ~0).
"""
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .cka import cka_u
from .distances import dist_from_emb
from .preprocessing import l2_points
from .scores import plan_stats
from .solver import describe, eps_c, solve
from .surrogates import surr_elliptic, surr_pcperm

DT = torch.float64
COMPS = ["r", "r_ecc", "rA", "r_diag", "r_rest", "r_off"]   # r = r_ecc + rA ; rA = r_diag + r_rest ; r_off: off-diagonal distances


def prep_region(X, metric, dev, l2=True):
    """Distances + descriptors of one cloud, plus the leading eigenpair of A = J D J and spectral summaries
    (lam1, gap = |lam1|/|lam2|, participation ratio). From 215_champ_map.py."""
    p = describe(dist_from_emb(X, metric, dev, l2=l2))
    w, U = torch.linalg.eigh(p["A"]); o = torch.argsort(w.abs(), descending=True); w, U = w[o], U[:, o]
    lam = (w / p["n"]).cpu().numpy(); a = np.abs(lam)
    p.update(w1=w[0], u1=U[:, 0], lam1=float(lam[0]), gap=float(a[0] / a[1]), pr=float(a.sum() ** 2 / (a ** 2).sum()))
    return p


def path(px, py, grid, dev, delta=1e-3, max_outer=3000, tol=1e-8):
    """Downward continuation in eps / eps_c (from 215_champ_map.py, plan saving removed).
    Starts from the product coupling slightly tilted along the leading modes (the product is a fixed point).
    One row per eps: r and its parts, rA, c11 (part of rA carried by the first pair of modes), n·tr(T), top-1."""
    n = px["n"]; s = px["sig"] * py["sig"]; ec = eps_c(px, py)
    T0 = torch.full((n, n), 1.0 / n ** 2, dtype=DT, device=dev)
    P = torch.outer(px["u1"], py["u1"]); T = (T0 * (1 + delta * P / P.abs().max())).clamp_min(1e-300); T = T / T.sum()
    out = []
    for rel in grid:
        T, it, conv, rate = solve(px["D"], py["D"], rel * ec, T, max_outer, tol)
        rA = float((px["A"] * (T @ py["A"] @ T.T)).sum() / s)
        c11 = float(px["w1"] * py["w1"] * (px["u1"] @ T @ py["u1"]) ** 2 / s)
        st = plan_stats(px, py, T)
        out.append(dict(rel=rel, eps_c=ec, rA=rA, c11=c11, ntrT=float(n * torch.trace(T)),
                        **{k: st[k] for k in COMPS if k != "rA"},
                        top1=float((T.argmax(1) == torch.arange(n, device=dev)).double().mean()), it=it, conv=conv))
    return out


def compare(X, Y, metric="euclidean", grid=(1.5, 0.7, 0.3, 0.1, 0.03), surrogate="elliptic", B=4, l2=True,
            seed=0, dev=None, delta=1e-3, max_outer=3000, tol=1e-8, progress=True):
    """One row per eps / eps_c for the pair (X, Y); the loop body of 215_champ_map.main.
    progress: tqdm bar over the steps (distances, then GW for the real pair and each surrogate)."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = sorted({float(g) for g in grid}, reverse=True)
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    gen = torch.Generator().manual_seed(seed)
    make = {"elliptic": surr_elliptic, "pcperm": surr_pcperm}[surrogate]
    kw = dict(delta=delta, max_outer=max_outer, tol=tol)
    with tqdm(total=2 * (B + 1), desc=f"{metric} ({surrogate}, B={B})", unit="step", disable=not progress, leave=False) as bar:
        def step(label, fn):
            bar.set_postfix_str(label); out = fn(); bar.update(); return out
        PY = prep_region(Y, metric, dev, l2)
        PX = step("distances X", lambda: prep_region(X, metric, dev, l2))
        PS = [step(f"distances surrogate {b + 1}", lambda: prep_region(make(X, gen, dev), metric, dev, l2)) for b in range(B)]
        obs = step("GW real pair", lambda: path(PX, PY, grid, dev, **kw))
        sur = [step(f"GW surrogate {b + 1}", lambda ps=ps: path(ps, PY, grid, dev, **kw)) for b, ps in enumerate(PS)]
    ck = cka_u(l2_points(X), l2_points(Y), dev) if l2 else cka_u(X, Y, dev)
    rows = []
    for g_i, o in enumerate(obs):
        R = np.array([s_[g_i]["rA"] for s_ in sur]); c = o["rA"] - R.mean() if len(R) else np.nan
        sd = R.std(ddof=1) if len(R) > 1 else np.nan
        z = c / (sd * np.sqrt(1 + 1 / len(R))) if sd and sd > 1e-9 * max(abs(o["rA"]), 1e-12) else np.nan
        dec = {}
        for k_ in COMPS:                                       # decomposition: contrast and z per part
            Rk = np.array([s_[g_i][k_] for s_ in sur]); ck_ = o[k_] - Rk.mean() if len(Rk) else np.nan
            sk = Rk.std(ddof=1) if len(Rk) > 1 else np.nan
            dec[f"{k_}_surr"] = float(Rk.mean()) if len(Rk) else np.nan; dec[f"c_{k_}"] = ck_
            dec[f"z_{k_}"] = ck_ / (sk * np.sqrt(1 + 1 / len(Rk))) if sk and sk > 1e-9 * max(abs(o[k_]), 1e-12) else np.nan
        rows.append(dict(metric=metric, surrogate=surrogate, **o, **{k_: v for k_, v in dec.items() if k_ != "rA_surr"},
                         rA_surr=float(R.mean()) if len(R) else np.nan, contrast=c, z=z,
                         ntrT_surr=float(np.mean([s_[g_i]["ntrT"] for s_ in sur])) if sur else np.nan, cka=ck))
    return pd.DataFrame(rows)
