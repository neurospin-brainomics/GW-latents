import numpy as np
import pytest
import torch

import latent_gw as lg


def cloud(n=150, d=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)) * np.linspace(2, 0.3, d)


def test_r_is_zero_for_product_and_one_for_identity():
    D = lg.dist_from_emb(cloud(), "euclidean")
    p = lg.describe(D); n = D.shape[0]
    assert abs(lg.plan_stats(p, p, torch.full((n, n), 1 / n**2, dtype=torch.float64))["r"]) < 1e-10
    assert np.isclose(lg.plan_stats(p, p, torch.eye(n, dtype=torch.float64) / n)["r"], 1.0)


def test_r_is_scaled_gw_cost_gain():
    rng = np.random.default_rng(1)
    DX, DY = lg.dist_from_emb(cloud(40), "euclidean"), lg.dist_from_emb(cloud(40, seed=2), "euclidean")
    T = rng.random((40, 40))
    for _ in range(300):
        T /= T.sum(1, keepdims=True) * 40; T /= T.sum(0, keepdims=True) * 40
    T = torch.tensor(T); Dx, Dy = DX.double(), DY.double()
    cost = lambda P: (Dx**2).mean() + (Dy**2).mean() - 2 * (Dx * (P @ Dy @ P.T)).sum()
    U = torch.full((40, 40), 1 / 1600, dtype=torch.float64)
    px, py = lg.describe(DX), lg.describe(DY)
    gain = 0.5 * float(cost(U) - cost(T)) / (px["sig"] * py["sig"])
    assert np.isclose(lg.plan_stats(px, py, T)["r"], gain, atol=1e-8)


@pytest.mark.parametrize("metric", ["euclidean", "geodesic_k10", "orc_k10", "orc_k10_it2"])
def test_distances(metric):
    D = lg.dist_from_emb(cloud(80), metric, l2=True)
    assert torch.allclose(D, D.T) and float((D**2).mean()) == pytest.approx(1.0) and torch.isfinite(D).all()


@pytest.mark.parametrize("kind", ["native", "pcperm"])
def test_surrogates_keep_spectrum(kind):
    X = cloud(500)
    Z = {"native": lg.surr_native, "pcperm": lg.surr_pcperm}[kind](X, torch.Generator().manual_seed(0), "cpu")
    Z = Z.cpu().numpy() if torch.is_tensor(Z) else Z
    ev = lambda A: np.sort(np.linalg.eigvalsh(np.cov(A.T)))
    assert np.allclose(ev(Z), ev(X), rtol=0.15)


def test_compare_separates_shape_from_spectrum():
    rng = np.random.default_rng(3)
    X = cloud(150); Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    kw = dict(grid=(1.5, 0.1), B=3, dev="cpu", surrogate="pcperm")
    rel = lg.compare(X, X @ Q + 0.05 * rng.normal(size=X.shape), **kw).set_index("rel")
    unrel = lg.compare(X, cloud(150, seed=9), **kw).set_index("rel")
    assert abs(rel.loc[1.5, "rA"]) < 0.05                       # above eps_c: nothing resolved
    assert rel.loc[0.1, "contrast"] > 0.05 and rel.loc[0.1, "ntrT"] > 10
    assert abs(unrel.loc[0.1, "contrast"]) < 0.05 and unrel.loc[0.1, "ntrT"] < 3


def test_pair_test():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(20, 20)); A, B = np.arange(10), np.arange(10, 20)
    M[A, B] += 3
    t = lg.pair_test(M, A, B, cand_a=A, cand_b=B, n_null=500)
    assert t["p"] < 0.01 and t["n_pos"] >= 9


def test_cka_matrix_matches_pairwise_and_finds_homologues(tmp_path):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(120, 4))
    arrays = {}
    for b in range(4):
        shared = Z @ rng.normal(size=(4, 6))
        arrays[f"r{b}_left"] = shared + 0.5 * rng.normal(size=(120, 6)) + 2 * rng.normal(size=(120, 1)) * b
        arrays[f"r{b}_right"] = shared @ np.linalg.qr(rng.normal(size=(6, 6)))[0] + 0.5 * rng.normal(size=(120, 6))
    M = lg.cka_matrix(arrays, progress=False)
    assert np.isclose(M.loc["r0_left", "r2_right"], lg.cka_u(lg.l2_points(arrays["r0_left"]), lg.l2_points(arrays["r2_right"])))
    assert np.allclose(np.diag(M), 1)
    for f, X in arrays.items():
        np.save(tmp_path / f"{f}.npy", X)
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "latent_gw.cka_map", "--inputs", str(tmp_path), "--out", str(tmp_path / "o"),
                          "--n-sub", "0", "--homologues", "--n-null", "200"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "o" / "cka_pairs.csv").exists() and "homologues" in out.stdout


def test_plots_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib; matplotlib.use("Agg")
    from latent_gw import plots
    X = cloud(60); Y = X @ np.linalg.qr(np.random.default_rng(1).normal(size=(6, 6)))[0]
    res = {"euclidean": lg.compare(X, Y, grid=(1.5, 0.1), B=2, dev="cpu", progress=False)}
    T = plots.transport_plans(X, Y, "euclidean", grid=(1.5, 0.1), dev="cpu")
    assert T[0.1].shape == (60, 60) and np.isclose(T[0.1].sum(), 1)
    B = plots.pool_blocks(T[0.1], plots.spectral_order(Y, k=8), n_blocks=6)
    assert B.shape == (6, 6) and np.isclose(B.mean(), 1, rtol=0.05)
    for fig in (plots.scores_vs_eps(res), plots.retrieval_vs_eps(res), plots.transport_plans_figure({"e": T[0.1]}, n_blocks=6),
                plots.spectra({"X": X, "Y": Y}), plots.score_matrix(lg.cka_matrix({"a": X, "b": Y}, progress=False))):
        assert fig is not None
