# latent-gw — comparing latent spaces with entropic Gromov–Wasserstein

<p align="center">
  <img src="assets/gw_clusters.gif" width="900" alt="Entropic GW matching two clouds of clusters as ε decreases">
</p>

Do two latent spaces share a **shape**? The spaces can be embeddings of the same samples by two models, two brain regions or two modalities. Their axes need not match, and the comparison does not even need to know which sample is which.

`latent_gw` couples two point clouds with **entropic Gromov–Wasserstein (GW)**, which only looks at distances *within* each cloud. Around it, the package provides:
- distances that follow the data manifold: geodesic and Ollivier–Ricci, with discrete Ricci flow;
- the critical regularisation $\varepsilon_c$, computed from the spectra of the distance matrices, and a resolution parameter $\varepsilon/\varepsilon_c$;
- a normalised score $r$ and its decomposition;
- surrogates that keep the variance spectrum and destroy everything else, with an exact Monte-Carlo test;
- a test for expected pairs among many spaces.

The functions are taken unchanged from the research scripts that produced the results; `compare` chains them.

The repository has two parts:
- **`src/latent_gw/`** — the method, independent of any application, tested and documented.
- **`applications/champollion/`** — its application to the Champollion sulcal-region encoders (brain MRI).

> Research code from an internship at NeuroSpin (CEA), Brainomics team, 2026. No data is shipped.

---

## Why not just CKA, or a Euclidean comparison?

- **CKA, CCA and Procrustes need the pairing.** They compare sample *i* in space A with sample *i* in space B. GW does not, so it tests whether the geometry *alone* is shared, and whether it is enough to recover the pairing.
- **A raw GW score is not evidence of shared shape.** Two unrelated clouds with the same anisotropy already score high (see the quickstart). Comparing with **surrogates that keep the spectrum** shows what remains beyond it.
- **Latent spaces are rarely flat.** Distances along the manifold can reveal shared structure that straight-line distances blur.

---

## Install

```bash
pip install -e .                 # numpy, scipy, pandas, scikit-learn, joblib, torch (CPU or GPU)
pip install -e ".[figures,test]" # + matplotlib, imageio, POT (for the animation), pytest
pytest -q                        # 10 tests, about 30 s on CPU
```

## Quickstart

```python
import latent_gw as lg

res = lg.compare(X, Y, metric="geodesic_k20", surrogate="pcperm", B=4)
```

`compare` runs, for one pair, exactly the computation of the Champollion maps. `res` has one row per $\varepsilon/\varepsilon_c$ (default grid 1.5, 0.7, 0.3, 0.1, 0.03), with the following columns:
- `rA`, plus `r`, `r_ecc`, `r_diag`, `r_rest`, `r_off`: the GW score and its parts;
- `rA_surr`: the mean over surrogates of `X` that keep the spectrum;
- `contrast` $=$ `rA` $-$ `rA_surr`;
- `z` $=$ contrast $/(\mathrm{sd}\sqrt{1+1/B})$, and `c_<part>` / `z_<part>` for each part;
- `ntrT` $= n\cdot\mathrm{tr}(T)$, the mass kept on the same sample (×1 = chance), and `top1`;
- `c11`: the part of `rA` carried by the first pair of modes;
- `cka`: debiased linear CKA.

Metrics are named `euclidean`, `geodesic_k20`, `orc_k20` (one Ollivier–Ricci re-weighting) and `orc_k20_it3` (three steps of Ricci flow). Surrogates are `elliptic` (default in the maps script) or `pcperm`.

`python examples/quickstart.py` compares a cloud with a rotated, noisy copy of itself, and with an **unrelated cloud of the same spectrum** (8 pcperm surrogates, a few minutes on CPU):

```
related                                         unrelated, same spectrum
 rel    rA  rA_surr  contrast      z    ntrT     rel    rA  rA_surr  contrast     z  ntrT
1.50 0.000    0.000     0.000  1.551   1.003    1.50 0.000    0.000     0.000 1.548 1.000
0.30 0.541    0.502     0.039  3.724  16.016    0.30 0.490    0.489     0.001 0.068 1.026
0.10 0.944    0.770     0.174 52.543 147.882    0.10 0.768    0.764     0.004 0.529 1.174
0.03 0.991    0.837     0.154 68.407 197.335    0.03 0.835    0.835     0.000 0.050 1.832
```

How to read it:
- **Above $\varepsilon_c$, nothing is resolved** ($r_A = 0$).
- **Below $\varepsilon_c$, the unrelated cloud still reaches $r_A = 0.84$**, from its spectrum alone. Its surrogates do just as well, so its contrast and $z$ stay near 0, and it recovers no sample.
- **The related cloud stands far above its surrogates** ($z = 53$ at $0.1\,\varepsilon_c$) and recovers its samples up to ×197 over chance, without being told which is which.

---

## The method in brief

| Step | Module (origin) | |
|---|---|---|
| Preprocess | `preprocessing` (178) | Optional L2 normalisation of the points (centre, unit norm); distances rescaled so that $\operatorname{mean}(D^2)=1$. The same pipeline is applied to real clouds and to surrogates. |
| Graphs | `graphs` (synth_graph_metrics) | Symmetric kNN graph with Euclidean edge weights; if disconnected, $k$ is increased up to $2k+40$, then components are bridged by a minimum spanning tree on their centroids. All-pairs shortest paths. |
| Curvature | `curvature` (ricci_flow) | Ollivier–Ricci curvature with lazy measures $m_i=\tfrac12\delta_i+\tfrac1{2k}\sum_{N_k(i)}\delta$, exact $W_1$. Edge weights $w\leftarrow d\,(1-\eta\kappa)$, renormalised; iterating gives a discrete Ricci flow. |
| Distances | `distances` (178) | `dist_from_emb(X, metric, l2=...)` chains the three steps above. |
| Solver | `solver` (178, 199) | `describe` computes $\rho$, eccentricities $e$, $A=JDJ$ and its extreme eigenvalues. $\varepsilon_c = 4\max(\lambda^+_X\lambda^+_Y,\ \lambda^-_X\lambda^-_Y)$: above it, the plan is the product coupling. `gw_solve` is entropic GW (POT convention) with a log-domain Sinkhorn inner loop on CPU or GPU. `solve` (199) is the same solver, also returning the contraction rate. `run_path` solves at $\varepsilon=c\,\varepsilon_c$, largest first, each plan warm-starting the next. |
| Scores | `scores` (178) | $r = r_\text{ecc} + r_A$: the GW-cost gain of $T$ over the product coupling, divided by $\sigma_X\sigma_Y$. $r_A=\langle A_X, TA_YT^\top\rangle/(\sigma_X\sigma_Y)$ and $r_\text{ecc}=2e_X^\top Te_Y/(\sigma_X\sigma_Y)$. Also $n\cdot\mathrm{tr}(T)$, top-1 retrieval, and `ceiling`: the largest $r$ any coupling can reach, given the spectra alone. |
| Surrogates | `surrogates` (178) | `native`: random Haar rotation in sample space, exactly the same covariance. `elliptic`: native with the original norms. `spectral` / `spectral_ecc`: from $D$ alone, for any metric. `pcperm` (215): PCA scores permuted independently per component. |
| Tests | `inference` (178, 224) | `mc_test`: $z$, rank $p$ per $\varepsilon$, and $p$ of the max over $\varepsilon$ (exact under exchangeability). `pairing_tests`: $n\cdot\mathrm{tr}(T)$ and Mantel against permutations. `pair_test`: expected pairs among many spaces (baseline, residuals, re-pairings, sign test). |
| Per pair | `compare` (215) | `prep_region` (distances, descriptors, leading eigenpair), `path` (continuation from the slightly tilted product coupling), and `compare`: contrast and $z$ against surrogates. |
| Baseline | `cka` (215) | Debiased linear CKA (`cka_u`), which uses the pairing; plain `linear_cka`. |

## Layout

```
src/latent_gw/          the method (no application code)
tests/                  unit tests (r = scaled GW-cost gain, metrics, surrogates, shape vs spectrum, pair test)
examples/               quickstart.py, gw_clusters_gif.py (the animation above)
applications/champollion/   application to Champollion sulcal regions, see its README
assets/                 README figures
```

## Practical notes

- **float64 throughout.** The solver works in float64 (CPU or GPU); float32 can produce NaNs at small $\varepsilon$.
- **Cost.** Dense GW is $O(n^2)$ in memory and roughly $O(n^3)$ per iteration. A few thousand points per cloud is the practical range on CPU; beyond that, subsample.
- **Check kNN-graph connectivity** at the sample size you actually use.
- **Negative Ollivier–Ricci curvature is generic in high dimension;** it is not a finding by itself.
- **Restricted data:** when inputs are subject-level, distance matrices and transport plans are subject-level too. Keep them on the authorised infrastructure, export only aggregates, and show plans pooled in blocks of samples.

---

Thomas Lambelin — NeuroSpin (CEA), Brainomics team, GAIA lab — supervised by Vincent Frouin and Jean-Christophe Aude. Built on PyTorch, SciPy and scikit-learn; the animation uses [POT](https://pythonot.github.io/).
