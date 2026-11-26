#!/usr/bin/env python
"""
215 -- Carte GW de Champollion : toutes les paires de régions, à quelques ε, contre des substituts de même spectre.

Pour chaque paire (X, Y) de régions (non ordonnée ; X = première dans l'ordre de la liste) :
  - GW entropique L2/euclidien par continuation descendante en ε/ε_c (ε_c propre à la paire, comme 200) ;
  - alignement rA = ⟨A_x, T A_y Tᵀ⟩ / (σσ') et sa part portée par la 1re paire de modes c11 ;
  - même calcul pour B substituts elliptiques de X (même spectre, aucune correspondance entre sujets) :
    contraste = rA(X) − moyenne rA(substituts), z_ens ;
  - appariement (les deux régions viennent des MÊMES sujets) : n·tr(T) (1 = appariement parfait, 1/n = produit) et
    top1 = part des sujets i dont la masse de T est maximale sur j = i (substituts : ≈ 1/n) ;
  - CKA linéaire débiaisée entre X et Y (appariée, même échantillon de sujets) pour la comparaison GW / CKA.
Décomposition (178.plan_stats) : r = r_ecc + rA (excentricités / forme centrée), rA = r_diag + r_rest, r_off (distances
hors diagonale) ; contraste et z pour chacune (colonnes c_<comp>, z_<comp>). --brut : sans L2 (normes conservées).
Descripteurs par région (une fois) : λ1, écart spectral λ1/λ2, PR du spectre de A.

Une tâche traite une tranche des paires (--shard i --n-shards S), écrit map_shard_<i>.csv (agrégats par paire).
    python 215_champ_map.py --regions all --shard 0 --n-shards 40 --out $METHODO/champ_map_215
    (--regions all = toutes les régions présentes dans --aligned ; sinon liste séparée par des virgules)
GPU. Lanceur : 215_launch_map.sh (run_215_gpu.slurm) ; analyse : 216_champ_map_analyse.py.
"""
import argparse, importlib.util, itertools, os, re, time, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
_l = lambda name, f: (lambda s: (s, s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[2])(importlib.util.spec_from_file_location(name, HERE / f))
r178 = _l("r178", "178_r_null_test.py")
b199 = _l("b199", "199_bifurcation_real.py")
DT = torch.float64
COMPS = ["r", "r_ecc", "rA", "r_diag", "r_rest", "r_off"]   # r = r_ecc + rA ; rA = r_diag + r_rest ; r_off : distances hors diagonale


METRIC = "euclidean"


def distances(X, metric, dev):
    """orc_k<k>_it<T>[_eta<η>] : flot de Ricci itéré (ricci_flow.py) ; sinon métriques de 178 (euclidean, geodesic_k<k>, orc_k<k>)"""
    m = re.fullmatch(r"orc_k(\d+)_it(\d+)(?:_eta([\d.]+))?", metric)
    if not m:
        return r178.dist_from_emb(X, metric, dev)
    import ricci_flow
    Xp = r178.l2_points(X) if r178.L2 else X
    Xp = np.asarray(Xp.cpu().numpy() if torch.is_tensor(Xp) else Xp, dtype=np.float64)
    D, _ = ricci_flow.orc_flow_distance(Xp, int(m[1]), iters=int(m[2]), eta=float(m[3] or 1.0),
                                        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    return r178.rms(torch.as_tensor(D, dtype=DT, device=dev))


def prep_region(X, dev, metric=None):
    p = r178.describe(distances(X, metric or METRIC, dev))
    w, U = torch.linalg.eigh(p["A"]); o = torch.argsort(w.abs(), descending=True); w, U = w[o], U[:, o]
    lam = (w / p["n"]).cpu().numpy(); a = np.abs(lam)
    p.update(w1=w[0], u1=U[:, 0], lam1=float(lam[0]), gap=float(a[0] / a[1]), pr=float(a.sum() ** 2 / (a ** 2).sum()))
    return p


def surr_pcperm(X, gen, dev):
    """substitut « marges des composantes » : scores ACP permutés indépendamment colonne par colonne, puis ramenés
    dans l'espace d'origine avec la moyenne. Garde la moyenne (décalage, cône vu de l'origine), le spectre et la
    distribution de chaque composante (asymétrie, bord le long d'un axe) ; détruit les dépendances entre composantes."""
    X = torch.as_tensor(np.asarray(X), dtype=DT, device=dev); mu = X.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X - mu, full_matrices=False); Z = U * S
    n = Z.shape[0]
    Zp = torch.stack([Z[torch.randperm(n, generator=gen).to(dev), k] for k in range(Z.shape[1])], 1)
    return (Zp @ Vt + mu).cpu().numpy()


def cka_u(X, Y, dev):
    def gram0(Z):
        Z = torch.as_tensor(Z, dtype=DT, device=dev); Z = Z - Z.mean(0); G = Z @ Z.T; G.fill_diagonal_(0); return G
    def hs(K, L):
        n = K.shape[0]; k1, l1 = K.sum(0), L.sum(0)
        return float(((K * L).sum() + k1.sum() * l1.sum() / ((n - 1) * (n - 2)) - 2 * (k1 @ l1) / (n - 2)) / (n * (n - 3)))
    K, L = gram0(X), gram0(Y); kl, kk, ll = hs(K, L), hs(K, K), hs(L, L)
    return kl / np.sqrt(kk * ll) if kk > 0 and ll > 0 else np.nan


def save_plan(T, fn, topk):
    """plan n × n ; topk > 0 : pour chaque sujet i, les topk sujets j de plus forte masse (indices, n²·T) ; 0 : plan complet (n²·T, float32)"""
    n = T.shape[0]; S = (T * n ** 2).float()
    if topk and topk < n:
        v, j = torch.topk(S, topk, dim=1)
        np.savez_compressed(fn, idx=j.cpu().numpy().astype(np.int16), val=v.cpu().numpy(), diag=torch.diagonal(S).cpu().numpy(),
                            row_mass_kept=(v.sum(1) / S.sum(1)).cpu().numpy(), n=n)
    else:
        np.savez_compressed(fn, plan=S.cpu().numpy(), n=n)


def path(px, py, grid, a, dev, plan_dir=None):
    """continuation descendante ; renvoie une ligne par ε : rA, c11, n·tr T, top1, itérations ; plan_dir : sauvegarde des plans"""
    n = px["n"]; s = px["sig"] * py["sig"]; ec = r178.eps_c(px, py)
    T0 = torch.full((n, n), 1.0 / n ** 2, dtype=DT, device=dev)
    P = torch.outer(px["u1"], py["u1"]); T = (T0 * (1 + a.delta * P / P.abs().max())).clamp_min(1e-300); T = T / T.sum()
    out = []
    for rel in grid:
        T, it, conv, rate = b199.solve(px["D"], py["D"], rel * ec, T, a.max_outer, a.tol)
        if plan_dir is not None and rel < 1:
            plan_dir.mkdir(parents=True, exist_ok=True); save_plan(T, plan_dir / f"rel{rel:g}.npz", a.plan_topk)
        rA = float((px["A"] * (T @ py["A"] @ T.T)).sum() / s)
        c11 = float(px["w1"] * py["w1"] * (px["u1"] @ T @ py["u1"]) ** 2 / s)
        st = r178.plan_stats(px, py, T)
        out.append(dict(rel=rel, eps_c=ec, rA=rA, c11=c11, ntrT=float(n * torch.trace(T)),
                        **{k: st[k] for k in COMPS if k != "rA"},
                        top1=float((T.argmax(1) == torch.arange(n, device=dev)).double().mean()), it=it, conv=conv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default=os.path.join(os.environ.get("METHODO", "."), "aligned_champ"))
    ap.add_argument("--regions", default="all")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--grid", default="1.5,0.7,0.3,0.1,0.03", help="ε/ε_c, décroissant (le premier point > 1 sert de contrôle ≈ 0)")
    ap.add_argument("--metric", default="euclidean", help="euclidean, geodesic_k<k> (graphe kNN), orc_k<k> (courbure d'Ollivier-Ricci), comme 178")
    ap.add_argument("--surr-B", type=int, default=4)
    ap.add_argument("--surr-kind", default="elliptic", choices=["elliptic", "pcperm"],
                    help="elliptic : gaussien de même covariance (normes réattribuées) ; pcperm : composantes ACP permutées indépendamment")
    ap.add_argument("--n-sub", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--max-outer", type=int, default=3000); ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--brut", action="store_true", help="sans normalisation L2 des points (garde normes et excentricités ; κ ~ 1, seuil ε_c décalé)")
    ap.add_argument("--subset", default="all", help="all | homol+<N> : les homologues + N paires non homologues tirées une fois pour toutes (graine 12345) | fichier de paires X:Y")
    ap.add_argument("--save-plans", default="none", help="none | homologues | subset | X:Y,... : paires dont on garde les plans de transport (réel + 1er substitut)")
    ap.add_argument("--plan-topk", type=int, default=50, help="par sujet, les k plus fortes masses (0 : plan complet, ≈ 16 Mo par plan)")
    ap.add_argument("--count", action="store_true", help="afficher le nombre de paires du sous-ensemble et quitter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r178.L2 = not a.brut
    global METRIC; METRIC = a.metric
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); A = Path(a.aligned)
    regs = sorted(f.stem for f in A.glob("*.npy") if f.stem.endswith(("_left", "_right"))) if a.regions == "all" else [r for r in a.regions.split(",") if r]
    pairs = list(itertools.combinations(regs, 2))
    base = lambda r: re.sub(r"_(left|right)$", "", r)
    homol = [(x, y) for x, y in pairs if base(x) == base(y)]
    if a.subset.startswith("homol+"):
        others = [p_ for p_ in pairs if base(p_[0]) != base(p_[1])]
        nn = min(int(a.subset.split("+")[1]), len(others))
        pick = np.random.default_rng(12345).choice(len(others), nn, replace=False)
        pairs = homol + [others[i] for i in sorted(pick)]
    elif a.subset != "all":
        pairs = [tuple(l.strip().split(":")) for l in open(a.subset) if ":" in l]
    if a.count:
        print(len(pairs)); return
    keep_plans = (set(homol) if a.save_plans == "homologues" else set(pairs) if a.save_plans == "subset" else
                  set() if a.save_plans == "none" else {tuple(p_.split(":")) for p_ in a.save_plans.split(",")})
    mine = pairs[a.shard::a.n_shards]
    print(f"{len(regs)} régions, {len(pairs)} paires ; tranche {a.shard}/{a.n_shards} : {len(mine)} paires ; {dev}", flush=True)
    N = np.load(A / f"{regs[0]}.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(a.seed).permutation(N)[: a.n_sub]) if 0 < a.n_sub < N else np.arange(N)
    load = lambda nm: np.asarray(np.load(A / f"{nm}.npy", mmap_mode="r")[idx], dtype=np.float64)
    grid = sorted({float(x) for x in a.grid.split(",")}, reverse=True)
    if a.shard == 0:
        np.save(out / "sujets_positions.npy", idx)                # positions (dans aligned) des sujets tirés : relie les plans entre runs                                               # descripteurs par région (une seule tranche les écrit)
        rows = []
        for r in regs:
            p = prep_region(load(r), dev); rows.append(dict(region=r, lam1=p["lam1"], gap=p["gap"], pr=p["pr"]))
        pd.DataFrame(rows).to_csv(out / "regions.csv", index=False)
    cache, rows = {}, []
    fn = out / f"map_shard_{a.shard:03d}.csv"
    done = set()
    if fn.exists():                                                # reprise après interruption
        old = pd.read_csv(fn); rows = old.to_dict("records"); done = set(zip(old.src, old.tgt))
    def region(nm):
        if nm not in cache:
            X = load(nm); gen = torch.Generator().manual_seed(zlib.crc32(nm.encode()) % 2 ** 31)
            cache[nm] = dict(X=X, p=prep_region(X, dev),
                             s=[prep_region(r178.surr_elliptic(X, gen, dev) if a.surr_kind == "elliptic" else surr_pcperm(X, gen, dev), dev)
                                for _ in range(a.surr_B)])
            if len(cache) > 12:                                    # mémoire GPU : on garde les régions récentes
                cache.pop(next(iter(cache)))
        return cache[nm]
    for k, (x, y) in enumerate(mine):
        if (x, y) in done:
            continue
        t0 = time.time(); RX, RY = region(x), region(y)
        pdir = out / "plans" / f"{x}__{y}" if (x, y) in keep_plans else None
        obs = path(RX["p"], RY["p"], grid, a, dev, pdir)
        sur = [path(ps, RY["p"], grid, a, dev, (pdir / "substitut0") if (pdir is not None and b_ == 0) else None) for b_, ps in enumerate(RX["s"])]
        ck = cka_u(RX["X"], RY["X"], dev) if a.brut else cka_u(r178.l2_points(RX["X"]), r178.l2_points(RY["X"]), dev)
        for g_i, o in enumerate(obs):
            R = np.array([s_[g_i]["rA"] for s_ in sur]); c = o["rA"] - R.mean()
            sd = R.std(ddof=1) if len(R) > 1 else np.nan
            z = c / (sd * np.sqrt(1 + 1 / len(R))) if sd and sd > 1e-9 * max(abs(o["rA"]), 1e-12) else np.nan
            dec = {}
            for k_ in COMPS:                                       # décomposition : contraste et z par composante
                Rk = np.array([s_[g_i][k_] for s_ in sur]); ck_ = o[k_] - Rk.mean(); sk = Rk.std(ddof=1) if len(Rk) > 1 else np.nan
                dec[f"{k_}_surr"] = float(Rk.mean()); dec[f"c_{k_}"] = ck_
                dec[f"z_{k_}"] = ck_ / (sk * np.sqrt(1 + 1 / len(Rk))) if sk and sk > 1e-9 * max(abs(o[k_]), 1e-12) else np.nan
            rows.append(dict(src=x, tgt=y, **o, **{k_: v for k_, v in dec.items() if k_ != "rA_surr"}, rA_surr=float(R.mean()), contrast=c, z=z,
                             c11_surr=float(np.mean([s_[g_i]["c11"] for s_ in sur])),
                             ntrT_surr=float(np.mean([s_[g_i]["ntrT"] for s_ in sur])),
                             top1_surr=float(np.mean([s_[g_i]["top1"] for s_ in sur])), cka=ck))
        pd.DataFrame(rows).to_csv(fn, index=False)
        best = max(obs, key=lambda o: o["rA"])
        print(f"  [{k + 1}/{len(mine)}] {x} -> {y} : contraste à ε/ε_c={grid[-1]:g} {rows[-1]['contrast']:+.4f} (z {rows[-1]['z']:.1f}) ;"
              f" top1 {obs[-1]['top1']:.3f} ; CKA {ck:.3f} ; {time.time() - t0:.0f} s", flush=True)
    print(f"écrit : {fn}")


if __name__ == "__main__":
    main()
