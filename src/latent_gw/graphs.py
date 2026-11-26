"""kNN graphs and shortest paths (from synth_graph_metrics.py, unchanged).

Symmetric kNN graph (edge {i,j} if j in kNN(i) or i in kNN(j)), Euclidean edge weights. If disconnected:
k += 1 up to 2k + 40, then the components are bridged by a minimum spanning tree on their centroids, each
tree edge realised by the closest pair of real points.
"""
from __future__ import annotations

import os

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def n_jobs_default() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("N_JOBS")
    return int(v) if v else (os.cpu_count() or 1)


def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _sym_knn_graph(X, nbr_idx, nbr_dist, k):
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = nbr_idx[:, :k].reshape(-1)
    w = np.maximum(nbr_dist[:, :k].reshape(-1), 1e-12)
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)  # symmetric union (same weight on both sides)
    return G.tocsr()


def connected_knn_graph(X: np.ndarray, k: int):
    """Returns (symmetric csr G, k_final, n_comp_initial, n_bridges, nbr_idx)."""
    n = X.shape[0]
    k_cap = min(2 * k + 40, n - 1)
    nn = NearestNeighbors(n_neighbors=min(k_cap + 1, n)).fit(X)
    dist, idx = nn.kneighbors(X)
    # drop the point itself (robust to duplicates)
    nbr_idx = np.empty((n, idx.shape[1] - 1), dtype=np.int64)
    nbr_dist = np.empty((n, idx.shape[1] - 1))
    for i in range(n):
        keep = idx[i] != i
        ii, dd = idx[i][keep], dist[i][keep]
        nbr_idx[i] = ii[: nbr_idx.shape[1]]
        nbr_dist[i] = dd[: nbr_idx.shape[1]]

    k_cur = k
    G = _sym_knn_graph(X, nbr_idx, nbr_dist, k_cur)
    n_comp0, labels = connected_components(G, directed=False)
    n_comp = n_comp0
    while n_comp > 1 and k_cur < k_cap:
        k_cur += 1
        G = _sym_knn_graph(X, nbr_idx, nbr_dist, k_cur)
        n_comp, labels = connected_components(G, directed=False)

    n_bridges = 0
    if n_comp > 1:
        cents = np.stack([X[labels == c].mean(0) for c in range(n_comp)])
        Cc = cdist(cents, cents)
        T = minimum_spanning_tree(csr_matrix(Cc)).tocoo()
        G = G.tolil()
        for a, b in zip(T.row, T.col):
            ia = np.where(labels == a)[0]
            ib = np.where(labels == b)[0]
            Dab = cdist(X[ia], X[ib])
            u, v = np.unravel_index(np.argmin(Dab), Dab.shape)
            w = max(Dab[u, v], 1e-12)
            G[ia[u], ib[v]] = w
            G[ib[v], ia[u]] = w
            n_bridges += 1
        G = G.tocsr()
    return G, k_cur, n_comp0, n_bridges, nbr_idx[:, :k_cur]


def all_pairs_shortest_paths(G: csr_matrix, n_jobs: int) -> np.ndarray:
    n = G.shape[0]
    chunks = np.array_split(np.arange(n), max(1, min(n_jobs * 4, n)))

    def run(ix):
        return dijkstra(G, directed=False, indices=ix)

    parts = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=None)(delayed(run)(c) for c in chunks)
    D = np.vstack(parts)
    if not np.isfinite(D).all():
        raise RuntimeError("graph still disconnected after bridging (should not happen)")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D
