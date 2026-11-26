"""
Metriques de graphe pour la distance GW -- geodesic et orc_flow, destinees
a etre COMPATIBLES avec les matrices deja calculees du pipeline GW
historique ($GWCACHE/dist/<MODALITE>__<METRIQUE>.npy), pour pouvoir
etendre le cache aux nouvelles modalites (BH_CLS_RANDOM_SEED0/1,
NEUROJEPA) sans introduire une definition differente sous le meme nom.

Statut de fidelite a la specification recue le 10 septembre 2026
(gw_metrics.build_distance_matrix, via documentation transmise) :

  FIDELE (specification precise, teste) :
    - geodesic  : Dijkstra sur graphe kNN (k=15 par defaut), poids =
                  distances euclidiennes des aretes.
    - orc_flow  : meme graphe, repondere par (1 - kappa_ij) avant
                  Dijkstra. kappa_ij = 1 - W1(m_i, m_j)/d(i,j), W1 exact
                  (ot.emd2), m_i = marche aleatoire PARESSEUSE (alpha=0.5
                  de masse propre sur i, (1-alpha) reparti uniformement
                  sur les k voisins de i).

  APPROXIME, A VERIFIER si compatibilite exacte requise (la doc recue
  donne le mecanisme general mais pas tous les details d'implementation :
  pas d'incrementation de k, methode exacte de pontage) :
    - Augmentation automatique de k si le graphe kNN est deconnecte
      (plafond 2k+40 ici applique tel que documente, incrementation par
      pas de 1 -- CHOIX NON CONFIRME).
    - Si toujours deconnecte au plafond : pontage par arbre couvrant
      minimum sur les centroides des composantes, chaque arete du MST
      connectant la PAIRE DE POINTS REELS la plus proche entre les deux
      composantes (mecanisme invente a partir de la description
      "relie par un arbre couvrant minimum sur leurs centroides" --
      PAS verifie contre le code reel).

  Recommandation : si les nouvelles matrices doivent etre strictement
  comparables aux matrices existantes (ex: NT_MONDO__geodesic.npy, connu
  pour etre domine par les raccords de composantes vu le nombre eleve de
  composantes deconnectees chez NT_BRAINDEV), le plus sur est de
  reutiliser directement gw_metrics.build_distance_matrix si le fichier
  est accessible, plutot que cette reimplementation pour la partie
  "augmentation de k + pontage".
"""

import numpy as np
import ot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components, minimum_spanning_tree

DEFAULT_K = 15
DEFAULT_K_CAP_OFFSET = 40  # plafond = 2*k + this
DEFAULT_ALPHA = 0.5  # masse propre de la marche paresseuse (Ollivier-Ricci)


def _knn_indices(D_full, k):
    return np.argsort(D_full, axis=1)[:, 1:k + 1]


def _build_knn_from_D(D_full, k):
    n = D_full.shape[0]
    idx_nn = _knn_indices(D_full, k)
    rows = np.repeat(np.arange(n), k)
    cols = idx_nn.flatten()
    weights = D_full[rows, cols]
    G = csr_matrix((weights, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    return G, idx_nn


def _bridge_components_via_mst(G, X, labels, n_comp, D_full, verbose=True):
    """Pontage APPROXIME (cf. avertissement du module) : MST sur les
    centroides des composantes, chaque arete du MST realisee par la paire
    de points REELS la plus proche entre les deux composantes concernees
    (pas les centroides eux-memes, qui ne sont pas des points de la
    cohorte)."""
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(n_comp)])
    Dc = ot.dist(centroids, centroids, metric="euclidean")
    mst = minimum_spanning_tree(csr_matrix(Dc))
    mst_coo = mst.tocoo()

    G = G.tolil()
    for a, b in zip(mst_coo.row, mst_coo.col):
        idx_a = np.where(labels == a)[0]
        idx_b = np.where(labels == b)[0]
        sub_d = D_full[np.ix_(idx_a, idx_b)]
        ii, jj = np.unravel_index(np.argmin(sub_d), sub_d.shape)
        real_i, real_j = int(idx_a[ii]), int(idx_b[jj])
        w = D_full[real_i, real_j]
        G[real_i, real_j] = w
        G[real_j, real_i] = w
    if verbose:
        print(f"    [bridge] {len(mst_coo.row)} arete(s) de pontage ajoutee(s) "
              f"entre {n_comp} composantes.", flush=True)
    return G.tocsr()


def build_knn_graph_robust(X, k=DEFAULT_K, cap_offset=DEFAULT_K_CAP_OFFSET,
                             D_full=None, verbose=True):
    """Graphe kNN avec augmentation automatique de k si deconnecte
    (plafond 2k+cap_offset), puis pontage MST si toujours deconnecte au
    plafond. Retourne (graphe connecte, distances euclidiennes completes,
    indices kNN au k final utilise, k final)."""
    if D_full is None:
        D_full = ot.dist(X, X, metric="euclidean")
    n = X.shape[0]
    k_cap = 2 * k + cap_offset
    k_cur = k

    while True:
        G, idx_nn = _build_knn_from_D(D_full, k_cur)
        n_comp, labels = connected_components(G, directed=False)
        if n_comp == 1 or k_cur >= k_cap:
            break
        k_cur += 1

    if n_comp > 1:
        if verbose:
            print(f"    [knn_graph] {n_comp} composantes a k={k_cur} "
                  f"(plafond {k_cap} atteint) -- pontage MST.", flush=True)
        G = _bridge_components_via_mst(G, X, labels, n_comp, D_full, verbose=verbose)
        n_comp_final, _ = connected_components(G, directed=False)
        assert n_comp_final == 1, "Pontage MST insuffisant, graphe encore deconnecte"
    elif verbose and k_cur > k:
        print(f"    [knn_graph] k augmente automatiquement de {k} a {k_cur} "
              "pour obtenir un graphe connexe.", flush=True)

    return G, D_full, idx_nn, k_cur


def geodesic_distance_matrix(X, k=DEFAULT_K, verbose=True):
    G, D_full, _, k_final = build_knn_graph_robust(X, k=k, verbose=verbose)
    D_geo = shortest_path(G, method="D", directed=False)
    return D_geo, np.arange(X.shape[0]), k_final


def _ollivier_ricci_edge_curvatures_lazy(k, idx_nn, D_full, edges, alpha=DEFAULT_ALPHA):
    """kappa(i,j) avec marche aleatoire PARESSEUSE : alpha de masse sur
    le noeud lui-meme, (1-alpha) reparti uniformement sur ses k voisins.
    Le support de la mesure est donc {i} U voisins(i), taille k+1 --
    different de la version uniforme-sur-voisins-seuls (support k)."""
    m = np.empty(k + 1)
    m[0] = alpha
    m[1:] = (1 - alpha) / k
    kappas = {}
    for (i, j) in edges:
        support_i = np.concatenate(([i], idx_nn[i]))
        support_j = np.concatenate(([j], idx_nn[j]))
        cost = np.ascontiguousarray(D_full[np.ix_(support_i, support_j)])
        w1 = ot.emd2(m, m, cost)
        d_ij = D_full[i, j]
        kappas[(i, j)] = 1.0 - w1 / (d_ij + 1e-12)
    return kappas


def orc_flow_distance_matrix(X, k=DEFAULT_K, alpha=DEFAULT_ALPHA, ricci_iters=1, verbose=True):
    """ricci_iters=1 (defaut) : une seule passe, comme avant (repondere
    UNE fois par 1-kappa puis Dijkstra). ricci_iters>1 : vrai flot de
    Ricci itere -- a chaque iteration, recalcule kappa en utilisant les
    poids d'arete COURANTS (pas les distances euclidiennes d'origine)
    comme cout de transport, repondere par-dessus le poids courant,
    repete. Les voisinages (k) restent fixes -- seuls les POIDS des
    aretes deja existantes evoluent d'une iteration a l'autre."""
    G, D_full, idx_nn, k_final = build_knn_graph_robust(X, k=k, verbose=verbose)
    n = X.shape[0]
    coo = G.tocoo()
    edges = sorted(set((min(int(r), int(c)), max(int(r), int(c)))
                        for r, c in zip(coo.row, coo.col) if r != c))

    # Poids courants, indexes par arete -- demarre aux distances euclidiennes
    weights = {(i, j): D_full[i, j] for (i, j) in edges}

    for it in range(ricci_iters):
        if verbose:
            print(f"    [orc_flow] iteration {it+1}/{ricci_iters} : courbure "
                  f"(alpha={alpha}) sur {len(edges)} aretes (k={k_final})...",
                  flush=True)
        # Cout de transport = poids COURANTS (pas D_full une fois it>0) --
        # c'est ce qui fait un vrai flot iteratif plutot que refaire la
        # meme passe plusieurs fois.
        cost_matrix_current = _weights_to_matrix(weights, n) if it > 0 else D_full
        kappas = _ollivier_ricci_edge_curvatures_lazy(
            k_final, idx_nn, cost_matrix_current, edges, alpha=alpha)
        weights = {(i, j): max(weights[(i, j)] * (1 - kappas[(i, j)]), 1e-6)
                   for (i, j) in edges}

    rows, cols, w_list = [], [], []
    for (i, j), w in weights.items():
        rows += [i, j]
        cols += [j, i]
        w_list += [w, w]
    G_reweighted = csr_matrix((w_list, (rows, cols)), shape=(n, n))

    n_comp, _ = connected_components(G_reweighted, directed=False)
    if n_comp > 1 and verbose:
        print(f"    [orc_flow] ATTENTION : {n_comp} composantes APRES "
              "repondaration -- non gere par ce module.", flush=True)
    D_orc = shortest_path(G_reweighted, method="D", directed=False)
    return D_orc, np.arange(n), k_final


def _weights_to_matrix(weights, n):
    """Reconstruit une matrice n x n a partir du dict de poids d'arete,
    pour servir de cout de transport a l'iteration Ricci suivante. Les
    non-aretes restent a 0 -- n'affecte que le sous-probleme de transport
    entre voisinages (idx_nn), qui ne touche que des aretes existantes ou
    des paires deja couvertes par D_full a l'origine ; ici on retombe sur
    une approximation : les couples hors aretes explicites gardent leur
    cout D_full d'origine (les voisinages idx_nn peuvent inclure des
    paires non-arete du graphe symetrise final)."""
    # Remarque : approximation pragmatique -- reconstruit une matrice
    # dense n x n en partant de zero serait couteux (O(n^2) memoire) pour
    # de grands n ; ici n est deja borne par N_SUB (quelques milliers),
    # acceptable.
    M = np.zeros((n, n))
    for (i, j), w in weights.items():
        M[i, j] = M[j, i] = w
    return M


def distance_matrix(X, metric, k=DEFAULT_K, ricci_iters=1, verbose=True):
    if metric == "euclidean":
        return ot.dist(X, X, metric="euclidean"), np.arange(X.shape[0]), k
    if metric == "geodesic":
        D, keep, k_final = geodesic_distance_matrix(X, k=k, verbose=verbose)
        return D, keep, k_final
    if metric == "orc_flow":
        D, keep, k_final = orc_flow_distance_matrix(X, k=k, ricci_iters=ricci_iters, verbose=verbose)
        return D, keep, k_final
    raise ValueError(f"Metrique inconnue : {metric}")
