#!/usr/bin/env python
"""
178 — r choisit-il mieux que du bruit pur ? Test conditionnel par substituts « de même spectre ».

Idée. r_GW est invariant par permutation des sujets (départs produit / aléatoire) : un test par permutation
de r_GW est vide (p = 1). Le seul nul pertinent pour r_GW est un nul de FORME : « l'espace X est un nuage sans
structure ayant le même spectre (et éventuellement les mêmes excentricités) ». On le réalise exactement par
rotation de Haar dans l'espace des sujets (orthogonal à 1) :
  native     X' = U' S V^T, avec X_c = U S V^T et U' uniforme sur Stiefel(1^perp) : même covariance empirique,
             exactement ; test exact du nul « lignes i.i.d. gaussiennes, covariance quelconque ». Avec des
             distances euclidiennes seules, X est reconstruit exactement par MDS classique de D∘D.
  elliptic   native, puis chaque ligne remise à la norme d'une ligne d'origine tirée au hasard : mêmes normes
             (donc ~ même part d'excentricité alpha), directions sans structure. Nul « bruit elliptique / hubness »
             (approché, pas exact).
  spectral   à partir de D seule, toute métrique : A = J D J = W L W^T, A' = W' L W'^T (W' Haar sur 1^perp),
             D'_ij = G'_ii + G'_jj - 2 G'_ij avec G' = -A'/2. Même spectre de J D J, même moyenne. Attention :
             D' est un carré de distance gaussienne, ses excentricités sont gonflées (alpha change) -> préférer
  spectral_ecc  spectral + excentricités d'origine ré-imposées (permutées au hasard) : alpha conservé.
La validité (niveau exact) ne demande PAS que le solveur trouve l'optimum global : il suffit que la même
procédure, aveugle à l'appariement, soit appliquée à l'observé et aux substituts. L'optimisation ne joue que
sur la puissance.

Décomposition exacte (vérifiée numériquement), D normalisée RMS, e = moyennes de lignes - rho, A = J D J :
    sigma^2 = 2 V + |A|_F^2 / n^2,   alpha = 2V / sigma^2  (part « excentricité » de la variance)
    r(T) = r_ecc + r_diag + r_rest
    r_ecc  = 2 e_x^T T e_y / (sx sy)                 <= sqrt(alpha_x alpha_y) · corr. monotone
    r_diag = sum_ij A^x_ii A^y_jj T_ij^2 / (sx sy)   ~ sqrt(q_x q_y) · n |T|_F^2   (A_ii = -rho - 2 e_i)
    r_rest = le reste de <A_x, T A_y T^T> / (sx sy)
r_off (comme 177) exclut les quadruplets i=k ou j=l.

    # réel, distances euclidiennes (native + elliptic via MDS exacte) — même ordre de sujets partout
    python 178_r_null_test.py --target CIBLE=.../D_cible.npy --models NEW=...,OLD=...,EVO2=... \
        --randoms RAND0=...,RAND1=... --kind dist --metric euclidean --B 19 --rel 0.3,0.1,0.03,0.01 \
        --n-sub 2000 --out $METHODO/r_null
    # distances de graphe sans embeddings : --metric geodesic_k20 --surr spectral_ecc
    # embeddings + métrique de graphe recalculée sur chaque substitut : --kind emb --metric geodesic_k20
    # synthétique (contrôle : niveau, puissance, pièges), puis calibration
    python 178_r_null_test.py --synthetic --n 300 --B 9 --calib 20 --out synth_r_null

Sorties (agrégats uniquement) : runs.csv (observé), surr.csv (chaque substitut), tests.csv (z, p par eps et
max sur eps), pairing.csv (tests d'appariement par permutation, gratuits), summary.txt, calib.csv (synthétique).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
DT = torch.float64
L2 = False   # --l2 : centrer puis normaliser chaque point (‖x‖ = 1) avant les distances — appliqué à TOUT (cible, modèles, substituts)


# ----------------------------------------------------------------------------- distances et descripteurs
def rms(D):
    D = 0.5 * (D + D.T)
    D.fill_diagonal_(0)
    return D / torch.sqrt((D ** 2).mean())


def l2_points(X):
    if torch.is_tensor(X):
        Xc = X - X.mean(0, keepdim=True); return Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X = np.asarray(X, dtype=np.float64); Xc = X - X.mean(0)
    return Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)


def dist_from_emb(X, metric, dev):
    if L2:
        X = l2_points(X)
    if metric == "euclidean":
        X = torch.as_tensor(X, dtype=DT, device=dev) if not torch.is_tensor(X) else X.to(dev, DT)
        return rms(torch.cdist(X, X))
    X = np.asarray(X.cpu().numpy() if torch.is_tensor(X) else X, dtype=np.float64)
    try:                                              # métrique de production (graph_metrics.py du pipeline) si disponible
        kind, k = metric.split("_k")
        if kind != "geodesic":
            raise ImportError
        import graph_metrics as gm
        D = gm.distance_matrix(X, "geodesic", k=int(k), ricci_iters=None, verbose=False)[0]
    except ImportError:
        from synth_graph_metrics import compute_distance
        D = compute_distance(X, metric, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))[0]
    return rms(torch.as_tensor(np.asarray(D), dtype=DT, device=dev))


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


def ceiling(px, py):
    """Plafond sans résolution : r(T) <= sqrt(a a') c_mono + sqrt((1-a)(1-a')) R_spec pour TOUT couplage T
    (réarrangement pour le terme d'excentricité ; von Neumann + Birkhoff pour <A_x, T A_y T^T>)."""
    c_mono = float((px["es"] * py["es"]).sum() / (px["es"].norm() * py["es"].norm()).clamp_min(1e-300))
    R_spec = float((px["sv"] * py["sv"]).sum() / (px["sv"].norm() * py["sv"].norm()))
    ax, ay = px["alpha"], py["alpha"]
    return dict(c_mono=c_mono, R_spec=R_spec, r_ceil=math.sqrt(ax * ay) * c_mono + math.sqrt((1 - ax) * (1 - ay)) * R_spec)


def eps_c(px, py):
    return 4.0 * max(px["lp"] * py["lp"], px["ln"] * py["ln"])     # convention POT


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


# ----------------------------------------------------------------------------- solveur (démarrage à chaud possible)
@torch.no_grad()
def gw_solve(Dx, Dy, eps, T0=None, max_outer=1000, tol=1e-7, max_inner=2000, tol_inner=1e-8):
    """GW entropique, convention POT (E + eps KL), même schéma que synth_gw_solver.entropic_gw,
    avec un plan de départ T0 (continuation en eps). T0=None -> produit (équivariant par permutation)."""
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


def run_path(px, py, rels, path, a):
    """Applique LA procédure (identique pour observé et substituts) sur la grille eps = c · eps_c(pair)."""
    ec = eps_c(px, py)
    out, Tprev = {}, None
    for c in sorted(rels, reverse=True):
        eps = c * ec
        cands = []
        if path in ("cont", "best") and Tprev is not None:
            cands.append(("cont",) + gw_solve(px["D"], py["D"], eps, Tprev, a.max_outer, a.tol, a.max_inner))
        if path in ("cold", "best") or Tprev is None:
            cands.append(("cold",) + gw_solve(px["D"], py["D"], eps, None, a.max_outer, a.tol, a.max_inner))
        tag, T, F, conv, it = min(cands, key=lambda z: z[2])
        st = plan_stats(px, py, T)
        st.update(eps=eps, eps_c=ec, rel=c, F=F, converged=conv, n_outer=it, branch=tag)
        out[c] = (st, T)
        Tprev = T
    return out


# ----------------------------------------------------------------------------- substituts
def _stiefel_perp(n, k, gen, dev):
    W = torch.randn(n, k, generator=gen, dtype=DT).to(dev)
    W = W - W.mean(0, keepdim=True)                              # colonnes dans 1^perp
    Q, R = torch.linalg.qr(W)
    return Q * torch.sign(torch.diagonal(R))[None, :]            # Haar exact


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
    return d[:, None] + d[None, :] - 2 * G                    # même moyenne que D ; non renormalisée ici


def surr_spectral_ecc(D, gen, dev, cache):
    """spectral, puis excentricités remplacées par une permutation aléatoire des excentricités d'origine
    (D'' = D' + d 1^T + 1 d^T, diagonale remise à 0 : spectre de J D J conservé à O(n^-1/2) près, alpha conservé)."""
    Ds = surr_spectral(D, gen, dev, cache)
    rho = float(D.mean())
    e = D.mean(1) - rho
    es = Ds.mean(1) - float(Ds.mean())
    dlt = e[torch.randperm(D.shape[0], generator=gen).to(dev)] - es
    Ds = Ds + dlt[:, None] + dlt[None, :]
    Ds.fill_diagonal_(0)
    return Ds


def mds_embedding(D):
    """Pour D euclidienne (non carrée) : MDS classique de D∘D -> nuage natif centré (à rotation près), exact."""
    B = -0.5 * (lambda M: M - M.mean(0, keepdim=True) - M.mean(1, keepdim=True) + M.mean())(D ** 2)
    w, V = torch.linalg.eigh(B)
    keep = w > w.max() * 1e-10
    neg = float(-w[w < 0].sum() / w.abs().sum())
    return V[:, keep] * torch.sqrt(w[keep])[None, :], neg


def make_surrogate(kind, src, gen, dev, metric, cache):
    if kind == "spectral":
        return rms(surr_spectral(src["D"], gen, dev, cache))
    if kind == "spectral_ecc":
        return rms(surr_spectral_ecc(src["D"], gen, dev, cache))
    if "X" not in src:
        raise SystemExit("substituts native/elliptic : il faut des embeddings (--kind emb) ou des distances euclidiennes (--metric euclidean)")
    Z = surr_native(src["X"], gen, dev) if kind == "native" else surr_elliptic(src["X"], gen, dev)
    return dist_from_emb(Z if metric == "euclidean" else Z.cpu().numpy(), metric, dev)


# ----------------------------------------------------------------------------- tests
def mc_test(obs, sur):
    """obs: dict rel -> valeur ; sur: liste (B) de dicts rel -> valeur.
    z descriptif = (obs - moyenne nulle) / sd nulle (substituts seuls) ;
    p de rang par eps, et p du max sur eps : statistique max_eps z_pool, z_pool standardisé sur les B+1 valeurs
    (fonction symétrique, donc échangeable sous H0 : niveau exact, contrôle familial sur la grille)."""
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
    """Tests d'appariement (procédure équivariante => nul par permutation des colonnes du plan, sans résoudre)."""
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


STATS = ("r", "r_off", "r_ecc", "r_diag", "r_rest")


def analyse_model(name, kind, src, pt, a, dev, rels, surrs, seed):
    px = describe(src["D"])
    absmap = {e / eps_c(px, pt): e for e in getattr(a, "abs_list", [])}   # eps absolus -> profondeur relative du modèle
    rels = sorted(set(rels) | set(absmap))
    obs = run_path(px, pt, rels, a.path, a)
    cl = ceiling(px, pt)
    rows = [dict(model=name, kind=kind, abs_eps=absmap.get(c, np.nan), **cl, alpha_m=px["alpha"], alpha_t=pt["alpha"],
                 q_m=px["q"], q_t=pt["q"], rho_m=px["rho"], **{k: v for k, v in st.items()}) for c, (st, _) in obs.items()]
    gen = torch.Generator().manual_seed(seed)
    pair = [dict(model=name, kind=kind, rel=c, abs_eps=absmap.get(c, np.nan), **pairing_tests(px, pt, T, a.n_perm, gen))
            for c, (_, T) in obs.items()]
    srows, trows = [], []
    for sk in surrs:
        cache, draws = {}, []
        for b in range(a.B):
            Ds = make_surrogate(sk, src, gen, dev, a.metric, cache)
            ps = describe(Ds)
            res = run_path(ps, pt, rels, a.path, a)
            draws.append({c: st for c, (st, _) in res.items()})
            for c, (st, _) in res.items():
                srows.append(dict(model=name, kind=kind, surr=sk, b=b, alpha_s=ps["alpha"], rho_s=ps["rho"], **st))
            del Ds, ps
        for stat in STATS:
            rl, z, p, mu, sd, zm, pm, cstar = mc_test({c: obs[c][0][stat] for c in rels}, [{c: d[c][stat] for c in rels} for d in draws])
            for i, c in enumerate(rl):
                trows.append(dict(model=name, kind=kind, surr=sk, stat=stat, rel=c, abs_eps=absmap.get(c, np.nan), obs=obs[c][0][stat], null_mean=mu[i],
                                  null_sd=sd[i], excess=obs[c][0][stat] - mu[i], z=z[i], p=p[i], zmax=zm, p_max=pm, rel_star=cstar))
    return rows, srows, trows, pair


# ----------------------------------------------------------------------------- scénarios synthétiques
def synth_scenarios(n, seed):
    g = np.random.default_rng(seed)

    def roll(m):
        t = 1.5 * np.pi * (1 + 2 * g.random(m)); h = 20 * g.random(m)
        return t, h, np.c_[t * np.cos(t), h, t * np.sin(t)]

    t, h, S = roll(n)
    Lt = g.standard_normal((3, 30)) / 10
    Y = S @ Lt + 0.05 * g.standard_normal((n, 30))
    W1 = g.standard_normal((2, 40)) * 0.5

    def enc(t, h):
        return np.tanh(np.c_[t, h / 4] @ W1) + 0.05 * g.standard_normal((len(t), 40))

    Xs = enc(t, h)
    t2, h2, _ = roll(n)
    lam = np.arange(1, 51) ** -1.0
    Ga = g.standard_normal((n, 50)) * np.sqrt(lam)
    tdist = g.standard_normal((n, 50)) * np.sqrt(lam) / np.sqrt(g.chisquare(3, n) / 3)[:, None]
    gen = torch.Generator().manual_seed(seed + 1000)
    twin = surr_native(Xs, gen, "cpu").numpy()
    scale = np.exp(0.6 * g.standard_normal(n))[:, None]                     # cible à excentricités hétérogènes
    Yecc = (Y - Y.mean(0)) * scale
    models = {"partage": Xs, "forme_non_appariee": enc(t2, h2), "bruit_jumeau": twin, "bruit_aniso": Ga, "bruit_lourd": tdist}
    return {"roll": Y, "roll_ecc": Yecc}, models


# ----------------------------------------------------------------------------- main
def parse(s):
    return [x.split("=", 1) for x in s.split(",") if x]


def load_src(path, kind, metric, idx, dev):
    A = np.load(path, mmap_mode="r")
    if kind == "dist":
        D = rms(torch.tensor(np.asarray(A[idx])[:, idx], dtype=DT, device=dev))
        if metric != "euclidean":
            return dict(D=D)
        X, neg = mds_embedding(D)                            # nuage natif reconstruit : native/elliptic possibles
        err = float((torch.cdist(X, X) - D).abs().max())
        if neg > 1e-4 or err > 1e-3:
            print(f"  attention : {path} pas exactement euclidienne (masse négative {neg:.2e}, erreur MDS {err:.1e})", flush=True)
        return dict(D=D, X=X)
    X = np.asarray(A[idx], dtype=np.float64)
    return dict(X=X, D=dist_from_emb(X, metric, dev))


def summarize(R, Tt, P):
    L = []
    for X in (R, Tt, P):
        if len(X) and "abs_eps" not in X:
            X["abs_eps"] = np.nan
    ceil = R.groupby(["target", "model"]).r_ceil.first() if len(R) and "r_ceil" in R else None
    for (tg, sk), S in Tt.groupby(["target", "surr"]):
        L.append(f"\n=== cible {tg} — substitut {sk} : r_off observé / moyenne nulle « même spectre » (z, p) ===")
        S2 = S[S.stat == "r_off"]
        for (m, kd), U in S2.groupby(["model", "kind"], sort=False):
            lab = lambda c, ae: f"{c:.3g}" + (f"[eps={ae:g}]" if ae == ae else "")
            parts = "  ".join(f"{lab(c, ae)}:{o:.3f}/{mu:.3f}(z{z:+.1f},p{p:.2f})" for c, ae, o, mu, z, p in
                              U.sort_values("rel", ascending=False)[["rel", "abs_eps", "obs", "null_mean", "z", "p"]].itertuples(index=False))
            cl = f" plafond {ceil[(tg, m)]:.3f}" if ceil is not None and (tg, m) in ceil.index else ""
            L.append(f"{m:20s} [{kd:9s}] {parts} | max sur eps : p={U.p_max.iloc[0]:.3f} (eps*={U.rel_star.iloc[0]:g}){cl}")
        rel_only = S2[S2.abs_eps.isna()]
        groups = [(f"eps/eps_c={c:g}", rel_only[rel_only.rel == c]) for c in sorted(rel_only.rel.unique(), reverse=True)]
        groups += [(f"eps={e:g} (absolu)", S2[S2.abs_eps == e]) for e in sorted(S2.abs_eps.dropna().unique(), reverse=True)]
        for lab, V in groups:
            L.append(f"  {lab:18s} classement r_off brut : {' > '.join(V.sort_values('obs', ascending=False).model)}")
            L.append(f"  {'':18s} classement par z      : {' > '.join(V.sort_values('z', ascending=False).model)}")
        E = S[S.stat.isin(["r_ecc", "r_diag", "r_rest"])].pivot_table(index=["model", "rel"], columns="stat", values="excess", sort=False)
        L.append("excès sur le nul par terme de la décomposition (obs - moyenne nulle) :\n" + E.round(4).to_string())
    if P is not None and len(P):
        L.append("\n=== appariement (nul par permutation, sans re-résoudre) : n·tr(T) (hasard = 1) et Mantel hors diagonale ===")
        L.append(P.round(4).to_string(index=False))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--randoms", default="")
    ap.add_argument("--kind", default="dist", choices=["dist", "emb"])
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--surr", default="native,elliptic",
                    help="native,elliptic (embeddings, ou distances euclidiennes via MDS exacte) ; spectral, spectral_ecc (toute métrique, D seule)")
    ap.add_argument("--B", type=int, default=19)
    ap.add_argument("--rel", default="0.3,0.1,0.03,0.01")
    ap.add_argument("--abs-eps", default="", help="eps absolus (convention POT, D RMS) ajoutés à la grille, ex. 0.01,0.0008 ;"
                    " chaque substitut est résolu à la même profondeur eps/eps_c que le modèle")
    ap.add_argument("--path", default="cont", choices=["cont", "cold", "best"])
    ap.add_argument("--n-sub", type=int, default=2000)
    ap.add_argument("--subsample", default="perm", choices=["perm", "prefix"],
                    help="prefix : les n-sub premières lignes (ordre de cohorte emboîté) ; perm : préfixe d'une permutation (graine --seed)")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--max-outer", type=int, default=1000)
    ap.add_argument("--max-inner", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", action="store_true", help="centrer + normaliser L2 chaque point avant les distances (cible, modèles ET substituts ; "
                    "--kind emb seulement). Rend κ ≈ 0 : ε_c devient le seuil exact de l'estimateur (cf. 191-193)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--synth-targets", default="roll,roll_ecc")
    ap.add_argument("--synth-models", default="partage,forme_non_appariee,bruit_jumeau,bruit_aniso,bruit_lourd")
    ap.add_argument("--calib", type=int, default=0, help="synthétique : répétitions sous H0 (bruit_aniso) pour la calibration")
    ap.add_argument("--out", default="r_null")
    ap.add_argument("--merge", default="", help="dossier racine : fusionne <racine>/*/{runs,tests,pairing}.csv (sorties d'array)")
    a = ap.parse_args()
    global L2
    L2 = a.l2
    if L2 and a.kind != "emb" and not a.synthetic:
        raise SystemExit("--l2 demande des embeddings (--kind emb)")
    if a.merge:
        root = Path(a.merge)
        cat = lambda nm: pd.concat([pd.read_csv(f) for f in sorted(root.glob(f"*/{nm}.csv"))], ignore_index=True)
        R, Tt, P = cat("runs"), cat("tests"), cat("pairing")
        for nm, X in (("runs", R), ("tests", Tt), ("pairing", P)):
            X.to_csv(root / f"{nm}_all.csv", index=False)
        txt = summarize(R, Tt, P); print(txt); open(root / "summary_all.txt", "w").write(txt)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rels = [float(x) for x in a.rel.split(",") if x]
    a.abs_list = [float(x) for x in a.abs_eps.split(",") if x]
    surrs = a.surr.split(",")
    R, S, Tt, P = [], [], [], []

    def flush():
        for nm, L in (("runs", R), ("surr", S), ("tests", Tt), ("pairing", P)):
            if L:
                pd.DataFrame(L).to_csv(out / f"{nm}.csv", index=False)

    if a.synthetic:
        a.kind = "emb"
        targets, models = synth_scenarios(a.n, a.seed)
        keep = a.synth_models.split(",")
        models = {k: v for k, v in models.items() if k in keep}
        for tg, Y in [(k, v) for k, v in targets.items() if k in a.synth_targets.split(",")]:
            pt = describe(dist_from_emb(Y, a.metric, dev))
            for i, (nm, X) in enumerate(models.items()):
                t0 = time.time()
                src = dict(X=X, D=dist_from_emb(X, a.metric, dev))
                r, s, t, p = analyse_model(nm, "synth", src, pt, a, dev, rels, surrs, a.seed + 100 * i)
                for L, z in ((R, r), (S, s), (Tt, t), (P, p)):
                    L += [dict(target=tg, **row) for row in z]
                print(f"[{tg}] {nm} fini en {time.time() - t0:.0f} s", flush=True); flush()
        if a.calib:
            C = []
            for k in range(a.calib):
                g = np.random.default_rng(10_000 + k)
                lam = np.arange(1, 51) ** -1.0
                X = g.standard_normal((a.n, 50)) * np.sqrt(lam)
                tg, Y = "roll", targets["roll"]
                pt = describe(dist_from_emb(Y, a.metric, dev))
                _, _, t, _ = analyse_model(f"H0_{k}", "calib", dict(X=X, D=dist_from_emb(X, a.metric, dev)), pt, a, dev, rels, ["native"], 20_000 + k)
                C += [row for row in t if row["stat"] == "r_off"]
                pd.DataFrame(C).to_csv(out / "calib.csv", index=False)
                print(f"calibration {k + 1}/{a.calib}", flush=True)
            Cd = pd.DataFrame(C)
            Pm = Cd.groupby("model").p_max.first()
            print(f"\ncalibration sous H0 : P(p_max <= 0.1) = {(Pm <= 0.1).mean():.2f} (attendu ~0.1, B={a.B}) ; "
                  f"P(p <= 0.1) par eps : {Cd.assign(s=Cd.p <= 0.1).groupby('rel').s.mean().round(2).to_dict()}")
    else:
        (tname, tpath), = parse(a.target)
        N = np.load(tpath, mmap_mode="r").shape[0]
        # sous-échantillon emboîté (préfixe d'une permutation unique, comme la cohorte v5)
        if a.n_sub <= 0 or a.n_sub >= N:
            idx = np.arange(N)
        elif a.subsample == "prefix":                  # fichiers déjà dans l'ordre de la cohorte (181 --cohort *_order*.txt)
            idx = np.arange(a.n_sub)
        else:
            idx = np.sort(np.random.default_rng(a.seed).permutation(N)[:a.n_sub])
        pt = describe(load_src(tpath, a.kind, a.metric, idx, dev)["D"])
        todo = [(n_, p_, "entraîné") for n_, p_ in parse(a.models)] + [(n_, p_, "randomisé") for n_, p_ in parse(a.randoms)]
        for i, (nm, path, kd) in enumerate(todo):
            t0 = time.time()
            src = load_src(path, a.kind, a.metric, idx, dev)
            r, s, t, p = analyse_model(nm, kd, src, pt, a, dev, rels, surrs, a.seed + 100 * i)
            for L, z in ((R, r), (S, s), (Tt, t), (P, p)):
                L += [dict(target=tname, **row) for row in z]
            print(f"{nm} fini en {time.time() - t0:.0f} s", flush=True); flush()
    flush()
    txt = summarize(pd.DataFrame(R), pd.DataFrame(Tt), pd.DataFrame(P))
    print(txt); open(out / "summary.txt", "w").write(txt)
    print(f"\nagrégats : {out}/runs.csv, surr.csv, tests.csv, pairing.csv, summary.txt")


if __name__ == "__main__":
    main()
