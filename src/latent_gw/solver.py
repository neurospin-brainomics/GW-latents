"""Distance descriptors, eps_c and the entropic GW solver (from 178_r_null_test.py, unchanged).

D is RMS-normalised (mean D^2 = 1); rho = mean(D), e = row means - rho, A = J D J.
eps_c = 4 max(lambda+_x lambda+_y, lambda-_x lambda-_y), lambda = extreme eigenvalues of A / n (POT convention):
above eps_c the entropic GW plan is the product coupling.
gw_solve: entropic GW, POT convention (E + eps KL), log-domain Sinkhorn inner loop, optional warm start T0.
solve: same solver, also returning the contraction rate (from 199_bifurcation_real.py).
run_path: the same procedure on the grid eps = c * eps_c, largest first, each plan warm-starting the next.
"""
import math

import torch

from .scores import plan_stats

DT = torch.float64

def describe(D):
    n = D.shape[0]
    rho = float(D.mean())
    sig = math.sqrt(max(1.0 - rho ** 2, 1e-15))
    e = D.mean(1) - rho
    A = D - rho - e[:, None] - e[None, :]                       # = J D J
    w = torch.linalg.eigvalsh(A)
    lam = w / n
    off = ~torch.eye(n, dtype=torch.bool, device=D.device)
    return dict(D=D, n=n, rho=rho, sig=sig, e=e, A=A, dA=torch.diagonal(A).clone(),
                sv=torch.sort(w.abs(), descending=True).values, es=torch.sort(e).values,
                alpha=float(2 * (e ** 2).mean() / sig ** 2), q=rho ** 2 / (n * sig ** 2),
                lp=max(float(lam.max()), 0.0), ln=min(float(lam.min()), 0.0),
                om=float(D[off].mean()), osd=float(D[off].std()), Doff=torch.where(off, D - float(D[off].mean()), 0.0))


def eps_c(px, py):
    return 4.0 * max(px["lp"] * py["lp"], px["ln"] * py["ln"])


@torch.no_grad()
def gw_solve(Dx, Dy, eps, T0=None, max_outer=1000, tol=1e-7, max_inner=2000, tol_inner=1e-8):
    """Entropic GW, POT convention (E + eps KL), same scheme as synth_gw_solver.entropic_gw,
    with a starting plan T0 (continuation in eps). T0=None -> product coupling (permutation-equivariant)."""
    n, m = Dx.shape[0], Dy.shape[0]
    lp = torch.full((n,), -math.log(n), dtype=DT, device=Dx.device)
    lq = torch.full((m,), -math.log(m), dtype=DT, device=Dx.device)
    T = torch.full((n, m), 1.0 / (n * m), dtype=DT, device=Dx.device) if T0 is None else T0.clone()
    f = torch.zeros(n, dtype=DT, device=Dx.device); g = torch.zeros(m, dtype=DT, device=Dx.device)
    reg = eps / 2.0
    delta, conv, it = float("inf"), False, 0
    for it in range(1, max_outer + 1):
        K = (2.0 / reg) * (Dx @ T @ Dy)
        tol_it = max(tol_inner, 0.1 * delta) if math.isfinite(delta) else 1e-3
        for j in range(1, max_inner + 1):
            f = lp - torch.logsumexp(K + g[None, :], 1)
            g = lq - torch.logsumexp(K + f[:, None], 0)
            if j % 10 == 0:
                err = float((torch.exp(torch.logsumexp(K + f[:, None] + g[None, :], 1)) - 1.0 / n).abs().sum())
                if not math.isfinite(err) or err < tol_it:
                    break
        Tn = torch.exp(K + f[:, None] + g[None, :])
        delta = float((Tn - T).abs().sum())
        T = Tn
        if delta < tol:
            conv = True
            break
    A = float((Dx * (T @ Dy @ T.T)).sum())
    F = float((Dx ** 2).mean() + (Dy ** 2).mean()) - 2 * A + eps * float((T * torch.log(T.clamp_min(1e-300) * n * m)).sum())
    return T, F, conv, it


def run_path(px, py, rels, path="cont", max_outer=1000, tol=1e-7, max_inner=2000):
    """Applies THE procedure (identical for observed and surrogates) on the grid eps = c · eps_c(pair)."""
    ec = eps_c(px, py)
    out, Tprev = {}, None
    for c in sorted(rels, reverse=True):
        eps = c * ec
        cands = []
        if path in ("cont", "best") and Tprev is not None:
            cands.append(("cont",) + gw_solve(px["D"], py["D"], eps, Tprev, max_outer, tol, max_inner))
        if path in ("cold", "best") or Tprev is None:
            cands.append(("cold",) + gw_solve(px["D"], py["D"], eps, None, max_outer, tol, max_inner))
        tag, T, F, conv, it = min(cands, key=lambda z: z[2])
        st = plan_stats(px, py, T)
        st.update(eps=eps, eps_c=ec, rel=c, F=F, converged=conv, n_outer=it, branch=tag)
        out[c] = (st, T)
        Tprev = T
    return out


@torch.no_grad()
def solve(Dx, Dy, eps, T, max_outer, tol, max_inner=2000, tol_inner=1e-9):
    """Same scheme as gw_solve (POT convention), keeping the history of updates to estimate the contraction rate.
    From 199_bifurcation_real.py; used by the per-pair maps (215). Returns (T, n_iterations, converged, rate)."""
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
