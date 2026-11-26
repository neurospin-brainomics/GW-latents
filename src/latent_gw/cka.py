"""Linear CKA, a baseline that uses the sample pairing (row i of X is row i of Y).

linear_cka: plain linear CKA. cka_u: debiased (unbiased HSIC) linear CKA, as in the Champollion maps (215_champ_map.py).
"""
import numpy as np
import torch

DT = torch.float64


def linear_cka(X, Y):
    """||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F) with column-centred X, Y."""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def gram0(Z, dev="cpu"):
    """Centred linear Gram matrix with a zeroed diagonal (input of the unbiased HSIC estimator)."""
    Z = torch.as_tensor(Z, dtype=DT, device=dev); Z = Z - Z.mean(0); G = Z @ Z.T; G.fill_diagonal_(0); return G


def hs(K, L):
    """Unbiased HSIC estimator (Song et al., 2012) between two zero-diagonal Gram matrices."""
    n = K.shape[0]; k1, l1 = K.sum(0), L.sum(0)
    return float(((K * L).sum() + k1.sum() * l1.sum() / ((n - 1) * (n - 2)) - 2 * (k1 @ l1) / (n - 2)) / (n * (n - 3)))


def cka_u(X, Y, dev="cpu"):
    """Debiased linear CKA (unbiased HSIC estimator, zeroed Gram diagonals). From 215_champ_map.py."""
    K, L = gram0(X, dev), gram0(Y, dev); kl, kk, ll = hs(K, L), hs(K, K), hs(L, L)
    return kl / np.sqrt(kk * ll) if kk > 0 and ll > 0 else np.nan
