"""Tests.

From 178_r_null_test.py (unchanged):

Monte-Carlo test against surrogates and pairing tests (from 178_r_null_test.py, unchanged).

mc_test        z = (obs - null mean) / null sd ; rank p-value per eps ; p of the max over eps (exact under
               exchangeability: family-wise control over the eps grid).
pairing_tests  n*tr(T) and off-diagonal Mantel against column permutations of the plan (no re-solving).

From 224_gw_story.residual_local (generalised from left/right regions to any spaces):

pair_test      expected pairs inside a matrix of scores between many spaces: baseline per pair, residuals,
               mean of the expected pairs vs random re-pairings (derangements), sign test.
"""
import numpy as np
import torch
from scipy.stats import binomtest

def mc_test(obs, sur):
    """obs: dict rel -> value ; sur: list (B) of dicts rel -> value.
    z descriptif = (obs - moyenne nulle) / sd nulle (substituts seuls) ;
    rank p per eps, and p of the max over eps: statistic max_eps z_pool, z_pool standardised over the B+1 values
    (symmetric function, hence exchangeable under H0: exact level, family-wise control over the grid)."""
    rels = sorted(obs)
    M = np.array([[obs[c] for c in rels]] + [[s[c] for c in rels] for s in sur])   # (B+1) x |eps|
    mu0, sd0 = M[1:].mean(0), M[1:].std(0, ddof=1) + 1e-15
    z = (M[0] - mu0) / sd0
    Zp = (M - M.mean(0)) / (M.std(0, ddof=1) + 1e-15)
    p = (1 + (M[1:] >= M[0]).sum(0)) / len(M)
    zmax = Zp.max(1)
    pmax = (1 + (zmax[1:] >= zmax[0]).sum()) / len(M)
    return rels, z, p, mu0, sd0, zmax[0], pmax, rels[int(np.argmax(z))]


def pairing_tests(px, py, T, n_perm, gen):
    """Pairing tests (equivariant procedure => null by permuting the columns of the plan, no re-solving)."""
    n = T.shape[0]
    dev = T.device
    obs_d = float(torch.diagonal(T).sum()) * n
    Dx, Dy = px["Doff"], py["Doff"]
    mant = float((Dx * Dy).sum())
    nd, nm = 0, 0
    for _ in range(n_perm):
        pi = torch.randperm(n, generator=gen).to(dev)
        nd += float(T[torch.arange(n, device=dev), pi].sum()) * n >= obs_d
        nm += float((Dx * Dy[pi][:, pi]).sum()) >= mant
    w = float(((Dx ** 2).sum() * (Dy ** 2).sum()).sqrt())
    return dict(diag_n=obs_d, p_diag=(1 + nd) / (1 + n_perm), mantel=mant / w, p_mantel=(1 + nm) / (1 + n_perm))


def pair_test(M, A, B, cand_a=None, cand_b=None, n_null=1000, seed=0):
    """M[i, j]: score between spaces i and j. Expected pairs (A[p], B[p]).
    baseline b(i, j) = 1/2 [ mean_{k in cand_b, k != i, j} M[i, k] + mean_{k in cand_a, k != i, j} M[k, j] ]
    (cand_a / cand_b: spaces on the same side as A / B, e.g. left and right hemispheres; default: all).
    Statistic: mean M over expected pairs; null: same mean over derangements of B (no expected pair kept).
    Returns obs, p, residuals, n_pos (pairs above baseline), n, p_sign, null."""
    M = np.asarray(M, dtype=float); n = len(M); A = np.asarray(A); B = np.asarray(B)
    cand_a = np.arange(n) if cand_a is None else np.asarray(cand_a)
    cand_b = np.arange(n) if cand_b is None else np.asarray(cand_b)
    res = np.empty(len(A))
    for p, (i, j) in enumerate(zip(A, B)):
        rj = [k for k in cand_b if k not in (i, j)]; ri = [k for k in cand_a if k not in (i, j)]
        res[p] = M[i, j] - 0.5 * (np.nanmean(M[i, rj]) + np.nanmean(M[ri, j]))
    obs = np.nanmean(M[A, B])
    g = np.random.default_rng(seed); null = []
    for _ in range(n_null):
        q = g.permutation(len(A))
        while np.any(q == np.arange(len(A))):
            q = g.permutation(len(A))
        null.append(np.nanmean(M[A, B[q]]))
    null = np.array(null); ok = np.isfinite(res); k = int((res[ok] > 0).sum())
    return dict(obs=float(obs), p=float((1 + (null >= obs).sum()) / (1 + len(null))), residuals=res,
                n_pos=k, n=int(ok.sum()), p_sign=float(binomtest(k, int(ok.sum()), 0.5, alternative="greater").pvalue),
                null=null)
