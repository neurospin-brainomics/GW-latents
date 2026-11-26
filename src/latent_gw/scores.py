"""Scores read from a plan (from 178_r_null_test.py, unchanged).

r = r_ecc + rA, with s = sigma_x sigma_y (sigma = std of the RMS-normalised distances):
    rA    = <A_x, T A_y T^T> / s           structure of the double-centred matrices A = J D J
    r_ecc = 2 e_x^T T e_y / s               row-mean (eccentricity) part
r equals the GW-cost gain of T over the product coupling, divided by s; r = 0 for the product coupling.
Also: r_off (excludes i=k or j=l), r_diag, n*tr(T) (diag_n, x1 = chance), top-1 retrieval, relative entropy.
ceiling: upper bound on r for ANY coupling (rearrangement + von Neumann), i.e. what spectra alone allow.
"""
import math

import torch

def ceiling(px, py):
    """Ceiling without resolution: r(T) <= sqrt(a a') c_mono + sqrt((1-a)(1-a')) R_spec for ANY coupling T
    (rearrangement for the eccentricity term; von Neumann + Birkhoff for <A_x, T A_y T^T>)."""
    c_mono = float((px["es"] * py["es"]).sum() / (px["es"].norm() * py["es"].norm()).clamp_min(1e-300))
    R_spec = float((px["sv"] * py["sv"]).sum() / (px["sv"].norm() * py["sv"].norm()))
    ax, ay = px["alpha"], py["alpha"]
    return dict(c_mono=c_mono, R_spec=R_spec, r_ceil=math.sqrt(ax * ay) * c_mono + math.sqrt((1 - ax) * (1 - ay)) * R_spec)


@torch.no_grad()
def plan_stats(px, py, T):
    n = T.shape[0]
    s = px["sig"] * py["sig"]
    TAT = T @ py["A"] @ T.T
    rA = float((px["A"] * TAT).sum()) / s
    r_ecc = 2.0 * float(px["e"] @ T @ py["e"]) / s
    r_diag = float((px["dA"][:, None] * py["dA"][None, :] * T ** 2).sum()) / s
    w = 1.0 - float((T.sum(1) ** 2).sum()) - float((T.sum(0) ** 2).sum()) + float((T ** 2).sum())
    r_off = float((px["Doff"] * (T @ py["Doff"] @ T.T)).sum()) / w / (px["osd"] * py["osd"])
    H = float(-(T * torch.log(T.clamp_min(1e-300))).sum())
    return dict(r=r_ecc + rA, r_off=r_off, r_ecc=r_ecc, r_diag=r_diag, r_rest=rA - r_diag,
                nT2=n * float((T ** 2).sum()), H_rel=H / math.log(n * n),
                diag_n=float(torch.diagonal(T).sum()) * n,
                top1=float((T.argmax(1) == torch.arange(n, device=T.device)).double().mean()))
