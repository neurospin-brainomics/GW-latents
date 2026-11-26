"""Same-spectrum surrogates (from 178_r_null_test.py, unchanged).

native        X' = U' S V^T with U' Haar on Stiefel(1-perp): exactly the same empirical covariance.
elliptic      native, then each row rescaled to the norm of a random original row: same norms.
spectral      from D only (any metric): A = J D J = W L W^T -> A' = W' L W'^T, W' Haar; same spectrum of J D J.
spectral_ecc  spectral + original eccentricities re-imposed (randomly permuted).
pcperm        PCA scores permuted independently per component (from 215_champ_map.py).
mds_embedding classical MDS of D o D, to recover a native cloud from Euclidean distances.
"""
import numpy as np
import torch

from .distances import dist_from_emb
from .preprocessing import rms

DT = torch.float64

def _stiefel_perp(n, k, gen, dev):
    W = torch.randn(n, k, generator=gen, dtype=DT).to(dev)
    W = W - W.mean(0, keepdim=True)                              # columns in 1^perp
    Q, R = torch.linalg.qr(W)
    return Q * torch.sign(torch.diagonal(R))[None, :]


def surr_native(X, gen, dev):
    X = X.to(dev, DT) if torch.is_tensor(X) else torch.as_tensor(X, dtype=DT, device=dev)
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    k = int((S > S[0] * 1e-10).sum())
    return _stiefel_perp(X.shape[0], k, gen, dev) @ (S[:k, None] * Vt[:k])


def surr_elliptic(X, gen, dev):
    X = X.to(dev, DT) if torch.is_tensor(X) else torch.as_tensor(X, dtype=DT, device=dev)
    Xc = X - X.mean(0, keepdim=True)
    Z = surr_native(X, gen, dev)
    nr = Xc.norm(dim=1)[torch.randperm(X.shape[0], generator=gen).to(dev)]
    return Z * (nr / Z.norm(dim=1).clamp_min(1e-300))[:, None]


def surr_spectral(D, gen, dev, cache):
    if "eig" not in cache:
        w, V = torch.linalg.eigh(D - D.mean(0, keepdim=True) - D.mean(1, keepdim=True) + D.mean())
        keep = w.abs() > w.abs().max() * 1e-10
        cache["eig"] = w[keep]
    w = cache["eig"]
    W = _stiefel_perp(D.shape[0], len(w), gen, dev)
    G = -0.5 * (W * w[None, :]) @ W.T
    d = torch.diagonal(G)
    return d[:, None] + d[None, :] - 2 * G


def surr_spectral_ecc(D, gen, dev, cache):
    """spectral, then eccentricities replaced by a random permutation of the original eccentricities
    (D'' = D' + d 1^T + 1 d^T, diagonal reset to 0: spectrum of J D J kept up to O(n^-1/2), alpha kept)."""
    Ds = surr_spectral(D, gen, dev, cache)
    rho = float(D.mean())
    e = D.mean(1) - rho
    es = Ds.mean(1) - float(Ds.mean())
    dlt = e[torch.randperm(D.shape[0], generator=gen).to(dev)] - es
    Ds = Ds + dlt[:, None] + dlt[None, :]
    Ds.fill_diagonal_(0)
    return Ds


def mds_embedding(D):
    """For a Euclidean D (not squared): classical MDS of D∘D -> centred native cloud (up to rotation), exact."""
    B = -0.5 * (lambda M: M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean())(D ** 2)
    w, V = torch.linalg.eigh(B)
    keep = w > w.max() * 1e-10
    neg = float(-w[w < 0].sum() / w.abs().sum())
    return V[:, keep] * torch.sqrt(w[keep])[None, :], neg


def surr_pcperm(X, gen, dev):
    """"Component marginals" surrogate (from 215_champ_map.py): PCA scores permuted independently column by column,
    then mapped back to the original space with the mean. Keeps the mean (offset, cone seen from the origin), the
    spectrum and the distribution of every component (skewness, edge along an axis); destroys the dependencies
    between components."""
    X = torch.as_tensor(np.asarray(X), dtype=DT, device=dev); mu = X.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X - mu, full_matrices=False); Z = U * S
    n = Z.shape[0]
    Zp = torch.stack([Z[torch.randperm(n, generator=gen).to(dev), k] for k in range(Z.shape[1])], 1)
    return (Zp @ Vt + mu).cpu().numpy()


def make_surrogate(kind, src, gen, dev, metric, cache, l2=False):
    if kind == "spectral":
        return rms(surr_spectral(src["D"], gen, dev, cache))
    if kind == "spectral_ecc":
        return rms(surr_spectral_ecc(src["D"], gen, dev, cache))
    if "X" not in src:
        raise SystemExit("native/elliptic surrogates need embeddings, or Euclidean distances")
    Z = {"native": surr_native, "elliptic": surr_elliptic, "pcperm": surr_pcperm}[kind](src["X"], gen, dev)
    return dist_from_emb(Z if metric == "euclidean" else Z.cpu().numpy(), metric, dev, l2=l2)
