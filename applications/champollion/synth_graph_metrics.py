"""
Distances pour GW : euclidienne, géodésique kNN, ORC-flow (kNN repondéré par
W1 entre mesures paresseuses). Réplique la spécification de production
(graph_metrics.py, telle que décrite dans la note du 30/09 et le rapport §4.2.2) :

  * graphe kNN symétrisé (arête {i,j} si j in kNN(i) ou i in kNN(j)),
    poids = distance euclidienne ;
  * si déconnecté : k += 1 jusqu'à connexité ou plafond 2k+40, puis pontage
    des composantes par un arbre couvrant minimum sur leurs centroïdes,
    chaque arête de l'arbre réalisée par la paire de points réels la plus proche ;
  * ORC-flow : w~_ij = max(W1(m_i, m_j), 1e-6) avec m_i = 1/2 delta_i +
    1/(2k) sum_{N_k(i)} delta, N_k(i) = k plus proches voisins dans le nuage
    entier (k final après augmentation), coût de transport euclidien ;
    puis plus courts chemins.

W1 exact : m_i et m_j sont deux mesures uniformes à 2k atomes (on duplique
k fois le centre, de masse 1/2 = k * 1/(2k)) ; par le théorème de Birkhoff,
W1 = (1/2k) * min_{sigma} sum c(a_s, b_sigma(s)), résolu exactement par
scipy.optimize.linear_sum_assignment. Option 'pot' (ot.emd2) pour contrôle.
"""
from __future__ import annotations

import math
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def n_jobs_default() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("N_JOBS")
    return int(v) if v else (os.cpu_count() or 1)


# --------------------------------------------------------------------------
def euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def parse_metric(name: str) -> tuple[str, int | None]:
    if name == "euclidean":
        return "euclidean", None
    kind, ks = name.rsplit("_k", 1)
    if kind not in ("geodesic", "orc"):
        raise ValueError(name)
    return kind, int(ks)


# --------------------------------------------------------------------------
def _sym_knn_graph(X, nbr_idx, nbr_dist, k):
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = nbr_idx[:, :k].reshape(-1)
    w = np.maximum(nbr_dist[:, :k].reshape(-1), 1e-12)
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)  # union symétrique (même poids des deux côtés)
    return G.tocsr()


def connected_knn_graph(X: np.ndarray, k: int):
    """Retourne (G csr symétrique, k_final, n_comp_initial, n_bridges, nbr_idx)."""
    n = X.shape[0]
    k_cap = min(2 * k + 40, n - 1)
    nn = NearestNeighbors(n_neighbors=min(k_cap + 1, n)).fit(X)
    dist, idx = nn.kneighbors(X)
    # retirer le point lui-même (robuste aux doublons)
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
        raise RuntimeError("graphe non connexe après pontage (ne devrait pas arriver)")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------
def _w1_edges_chunk(X, nbr_idx, edges, solver):
    k = nbr_idx.shape[1]
    out = np.empty(len(edges))
    sel = np.concatenate([np.zeros(k, dtype=np.int64), np.arange(1, k + 1)])
    if solver == "pot":
        import ot
        a = np.full(k + 1, 1.0 / (2 * k)); a[0] = 0.5
    for e, (i, j) in enumerate(edges):
        Ai = np.concatenate([[i], nbr_idx[i]])
        Bj = np.concatenate([[j], nbr_idx[j]])
        C = cdist(X[Ai], X[Bj])
        if solver == "lsa":
            Cf = C[np.ix_(sel, sel)]
            r, c = linear_sum_assignment(Cf)
            out[e] = Cf[r, c].sum() / (2 * k)
        else:
            out[e] = ot.emd2(a, a, C)
    return out


def orc_flow_weights(X, G, nbr_idx, n_jobs, solver="lsa"):
    U = G.tocoo()
    mask = U.row < U.col
    ei, ej, w = U.row[mask], U.col[mask], U.data[mask]
    edges = np.stack([ei, ej], 1)
    chunks = np.array_split(np.arange(len(edges)), max(1, min(len(edges), n_jobs * 8)))
    res = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=None)(
        delayed(_w1_edges_chunk)(X, nbr_idx, edges[c], solver) for c in chunks)
    W1 = np.concatenate(res)
    kappa = 1.0 - W1 / np.maximum(w, 1e-12)
    wt = np.maximum(W1, 1e-6)
    n = G.shape[0]
    Gw = coo_matrix((np.concatenate([wt, wt]), (np.concatenate([ei, ej]), np.concatenate([ej, ei]))),
                    shape=(n, n)).tocsr()
    stats = dict(orc_kappa_mean=float(kappa.mean()), orc_kappa_median=float(np.median(kappa)),
                 orc_frac_neg=float((kappa < 0).mean()), n_edges=int(len(edges)))
    return Gw, stats


# --------------------------------------------------------------------------
def compute_distance(X: np.ndarray, metric: str, n_jobs: int | None = None,
                     orc_solver: str = "lsa") -> tuple[np.ndarray, dict]:
    """Matrice de distances brute (float64) + statistiques du graphe."""
    n_jobs = n_jobs or n_jobs_default()
    kind, k = parse_metric(metric)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if kind == "euclidean":
        return euclidean(X), dict(k_final=-1, n_comp0=1, n_bridges=0)
    G, k_fin, nc0, nb, nbr_idx = connected_knn_graph(X, k)
    st = dict(k_final=int(k_fin), n_comp0=int(nc0), n_bridges=int(nb))
    if kind == "orc":
        G, s2 = orc_flow_weights(X, G, nbr_idx, n_jobs, orc_solver)
        st.update(s2)
    D = all_pairs_shortest_paths(G, n_jobs)
    return D, st
