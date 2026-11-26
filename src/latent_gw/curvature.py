"""Ollivier–Ricci re-weighting and discrete Ricci flow (from ricci_flow.py, unchanged).

Iteration t, for every edge (i, j):
  m_i = 1/2 delta_i + 1/2 uniform on the k neighbours of i ;  W1_t(i, j) = W1(m_i, m_j) with cost c_t ;
  kappa_t(i, j) = 1 - W1_t(i, j) / d_t(i, j) ;  w_{t+1}(i, j) = d_t(i, j) (1 - eta kappa_t(i, j))   (Ni et al., 2019)
  then renormalisation sum w_{t+1} = sum w_0.
c_0 = Euclidean distance between points (iters = 1, eta = 1 gives the "orc_k<k>" metric);
for t >= 1, c_t = d_t = shortest paths on the graph weighted by w_t. W1 is exact: both measures are uniform
on 2k atoms (centre duplicated k times), solved by linear_sum_assignment.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

from . import graphs as sgm

def _w1_chunk(nbr_idx, edges, X=None, D=None):
    k = nbr_idx.shape[1]; out = np.empty(len(edges))
    sel = np.concatenate([np.zeros(k, dtype=np.int64), np.arange(1, k + 1)])
    for e, (i, j) in enumerate(edges):
        Ai = np.concatenate([[i], nbr_idx[i]]); Bj = np.concatenate([[j], nbr_idx[j]])
        C = cdist(X[Ai], X[Bj]) if D is None else D[np.ix_(Ai, Bj)]
        Cf = C[np.ix_(sel, sel)]; r, c = linear_sum_assignment(Cf)
        out[e] = Cf[r, c].sum() / (2 * k)
    return out


def orc_flow_distance(X, k, iters=1, eta=1.0, n_jobs=4):
    X = np.ascontiguousarray(X, dtype=np.float64); n = X.shape[0]
    G, k_fin, nc0, nb, nbr_idx = sgm.connected_knn_graph(X, k)
    U = G.tocoo(); m = U.row < U.col
    ei, ej, w = U.row[m], U.col[m], U.data[m].astype(np.float64)
    edges = np.stack([ei, ej], 1); w0_sum = w.sum()
    chunks = np.array_split(np.arange(len(edges)), max(1, min(len(edges), n_jobs * 8)))
    mk = lambda ww: coo_matrix((np.concatenate([ww, ww]), (np.concatenate([ei, ej]), np.concatenate([ej, ei]))), shape=(n, n)).tocsr()
    stats = dict(k_final=int(k_fin), n_comp0=int(nc0), n_bridges=int(nb), iters=int(iters), eta=float(eta))
    D = None
    for t in range(iters):
        if t == 0:
            d_edge = w.copy()                                    # Euclidean length of the edge (= d_0 on a kNN edge)
            W1 = np.concatenate(Parallel(n_jobs=n_jobs, backend="loky")(delayed(_w1_chunk)(nbr_idx, edges[c], X=X) for c in chunks))
        else:
            D = sgm.all_pairs_shortest_paths(mk(w), n_jobs); d_edge = D[ei, ej]
            W1 = np.concatenate(Parallel(n_jobs=n_jobs, backend="loky")(delayed(_w1_chunk)(nbr_idx, edges[c], D=D) for c in chunks))
        kappa = 1.0 - W1 / np.maximum(d_edge, 1e-12)
        w = np.maximum(d_edge * (1.0 - eta * kappa), 1e-6)
        w *= w0_sum / w.sum()
        stats[f"kappa_mean_it{t + 1}"] = float(kappa.mean()); stats[f"frac_neg_it{t + 1}"] = float((kappa < 0).mean())
    return sgm.all_pairs_shortest_paths(mk(w), n_jobs), stats
