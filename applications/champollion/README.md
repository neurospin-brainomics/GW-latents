# Application: Champollion sulcal regions

Champollion encodes the cortical folding of each sulcal region into a 32-dimensional latent space (one self-supervised encoder per region). We ask whether these latent spaces share a shape, using 15 left/right homologous regions as a **positive control**: the two sides of a sulcus should resemble each other.

> These are the scripts exactly as they were run. The `latent_gw` package reuses their core functions (from `synth_graph_metrics.py`, `ricci_flow.py`, `178_r_null_test.py`, `199_bifurcation_real.py`, `215_champ_map.py` and `224_gw_story.py`), so `latent_gw` and these scripts compute the same quantities: `lg.compare` is the per-pair loop of `215_champ_map.py`. The scripts are kept unchanged so the results below can be reproduced as they were obtained.

## Data

The inputs are aligned embeddings, one array per region and side: `$ALIGNED/<region>_<side>.npy`, shape `(n_subjects, 32)`. All arrays list **the same subjects in the same order**. The data come from UK Biobank and are not distributed: embeddings, distance matrices and transport plans stay on the authorised infrastructure, and scripts export aggregates only.

## Pipeline

**1. Score maps (expensive, GPU recommended).** For every pair of the 30 regions and every $\varepsilon/\varepsilon_c$, compute $r_A$ and its parts, the contrast against same-spectrum surrogates of the source region (`--surr-kind elliptic` by default, or `pcperm`), $n\cdot\mathrm{tr}(T)$, top-1 and debiased CKA. Pairs are split into shards (one GPU task each):
```bash
python 215_champ_map.py --metric geodesic_k20 --surr-kind pcperm --regions all --shard 0 --n-shards 40 --out $OUT/champ_map_geo20
```

**2. Figures and tables (fast, CPU).** These scripts only read the maps:
```bash
export EUC=... GEO=... ORC=...      # map folders produced by step 1
python 231_final_figs.py  --aligned $ALIGNED --out $OUT/F  --run euc=$EUC --run geo20=$GEO --run orc20=$ORC --rel 0.1
python 226_key_figures.py --aligned $ALIGNED --out $OUT/K  --run euc=$EUC --run geo20=$GEO --run orc20=$ORC --all-eps
python 224_gw_story.py    --aligned $ALIGNED --out $OUT/S  --run euc=$EUC --run geo20=$GEO --run orc20=$ORC
python 225_spectra.py     --aligned $ALIGNED --out $OUT/SP --run euc=$EUC --run geo20=$GEO --run orc20=$ORC
python 230_cka_figs.py    --run geo20=$GEO --out $OUT/C
python 232_overlap_table.py --run euc=$EUC --run geo20=$GEO --run orc20=$ORC --rels 0.1,0.03
```
The scripts load each other by file name, so keep them in the same folder. The reference test is `BASELINE = "local"` in `224_gw_story.py`.

| File | Role |
|---|---|
| `graph_metrics.py`, `synth_graph_metrics.py`, `ricci_flow.py` | kNN graphs, geodesic and Ollivier–Ricci distances |
| `178_r_null_test.py`, `199_bifurcation_real.py` | entropic GW, r score, surrogates, behaviour across ε |
| `215_champ_map.py` | per-pair score maps |
| `222_champ_figs_jz.py` | transport-plan helpers (block pooling, spectral ordering) |
| `224_gw_story.py` | per-metric analysis and the reference test |
| `225_spectra.py` | variance spectra and the spectrum control |
| `226_key_figures.py`, `230_cka_figs.py`, `231_final_figs.py` | figures |
| `232_overlap_table.py` | per-pair table: CKA, contrast, residual, n·tr(T) |
| `champ_style.py` | figure style |

## Results ($\varepsilon = 0.1\,\varepsilon_c$, about 2000 subjects per region)

- **Raw score:** every metric places most homologues above baseline: Euclidean 13/15, geodesic 12/15, ORC 12/15.
- **The Euclidean score follows the variance spectrum.** Homologues have closer spectra (p < 0.001), and the Euclidean score correlates with spectral similarity. The geodesic and ORC scores do not.
- **Beyond the spectrum** (contrast against same-spectrum surrogates): **geodesic 13/15, ORC 12/15, both p < 0.001**. Euclidean drops to 7/15: its mean stays above the null (p = 0.003), but the effect is carried by a few pairs. The geodesic excess holds at every ε tested, and across $k$ and a new sample of subjects.
- **CKA follows anatomical overlap** and ranks all 15 homologues above baseline, but it uses the subject pairing.
- **Same-subject retrieval** is strong only for a few overlapping regions (up to ≈ ×180 over chance, geodesic); for homologues it is ≈ ×2. The shared shape is a population-level geometry, not individual identity.
