"""Preprocessing and distance matrices (from 178_r_null_test.py and synth_graph_metrics.py).

Pipeline, identical for real clouds and surrogates:
    l2_points (optional: centre, then every point on the unit sphere)  ->  distances  ->  rms (mean D^2 = 1).
"""
import numpy as np
import torch

DT = torch.float64


def rms(D):
    D = 0.5 * (D + D.T)
    D.fill_diagonal_(0)
    return D / torch.sqrt((D ** 2).mean())


def l2_points(X):
    if torch.is_tensor(X):
        Xc = X - X.mean(0, keepdim=True); return Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X = np.asarray(X, dtype=np.float64); Xc = X - X.mean(0)
    return Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
