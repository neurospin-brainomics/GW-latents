# Embedding pipelines

Scripts that produce the latent spaces compared with `latent_gw`: DNA sequences and their embeddings by genomic language models, and brain-MRI embeddings by BrainHarmonix. Each script is standalone and configured through command-line arguments; no data is included.

> These scripts were written for a GPU cluster and are shared as they were used. Paths in the usage examples are placeholders.

```bash
pip install -r pipelines/requirements.txt
```

The pinned versions (`torch`, `transformers`, `huggingface_hub`, `timm`…) are those the pipelines ran with. The NTv3 code is loaded from the model repository (`trust_remote_code`) and is sensitive to the `transformers` version. BrainHarmonix also needs its own repository and its requirements.

```
pipelines/
  genomic/
    dna_sequence_generation.py     plausible DNA sequences (pyensembl genes + gnomAD common variants)
    embed_ntv3.py                  NTv3 embeddings, one per transcript per sample
    embed_ntv3_randinit.py         same with RANDOM weights (architecture-only control)
    embed_generator.py             GENERator embeddings, same pipeline
  imaging/
    preprocess/
      fmri_pipeline.py             fMRI -> 450-ROI time series (Schaefer-400 + Tian subcortex)
      Qc_fmri.py                   quality control of the ROI time series
      preprocess_t1_harmonix.py    T1 crop/pad to 160x192x160 + z-score (BrainHarmonix input)
      Qc_check.py                  quality control of preprocessed T1 volumes
    inference_brainharmonix/
      run_brainharmony_encoders_only.py   fMRI and T1 encoder tokens (no fusion)
      run_brainharmony_fusion.py          final BrainHarmonix output (fusion through the Harmonizer)
      run_brainharmony_fusion_randinit.py same with RANDOM weights (architecture-only control)
```

---

## Genomic

### 1. `dna_sequence_generation.py`: sequences

Builds "plausible" biological DNA sequences. Genes are taken from Ensembl (GRCh38, release 109) through `pyensembl`, and common variants are taken from gnomAD through its GraphQL API.

Requirements:
- `pyensembl`;
- the Ensembl GRCh38 primary-assembly FASTA, downloaded separately into `~/.cache/pyensembl/GRCh38/ensembl109/`:
```bash
  wget https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```
  The file is read line by line with `gzip`; no BGZF or `pyfaidx` is needed.

### 2. `embed_ntv3.py` / `embed_generator.py`: embeddings

One embedding per transcript per sample, computed by mean-pooling over the last hidden layer (weighted by the attention mask). The input is one multi-sequence FASTA per sample, one sequence per transcript, with headers such as

```
>TRANSCRIPT_ID LOC=...|STRAND=...|LEN=...|GENE=...|TRANSCRIPT=...|SOURCE=...|STATE=VCF|VCF_COUNT=N
```

(`STATE=REFERENCE` for the reference genome).

```bash
python embed_ntv3.py      --fasta_dir FASTA/ --embeddings_dir EMB/ --model_cache_dir CACHE_NTV3/
python embed_generator.py --fasta_dir FASTA/ --embeddings_dir EMB/ --model_cache_dir CACHE_GENERATOR/
```

- **Reference shortcut:** a transcript with `VCF_COUNT=0` (identical to the reference) reuses the embedding already computed on the reference genome. That cache is computed once and stored in `<embeddings_dir>/_reference_cache.npz`.
- **Automatic resume:** a sample is processed only if its output `.npz` is missing (or with `--force_recompute`); an interrupted sample never writes a partial output.
- **NTv3** (`AutoModelForMaskedLM`) is a U-Net with 7 down-sampling stages, so token lengths are padded to a multiple of 128. `--max_length N` truncates.
- **GENERator** (`GenerTeam/GENERator-eukaryote-3b-base`, `AutoModelForCausalLM`) is decoder-only and has no length constraint (`--pad_to_multiple_of`, default 1).

---

## Imaging (BrainHarmonix)

### 1. fMRI: `fmri_pipeline.py`, then `Qc_fmri.py`

The input is fMRI already registered to MNI152.
- **Atlas:** 450 ROIs, i.e. [Schaefer-400](https://github.com/ThomasYeoLab/CBIG) (7 networks, 2 mm) plus [Tian subcortex Scale III](https://github.com/yetianmed/subcortex). The atlas is downloaded and built automatically.
- **Per subject:** mean BOLD signal per ROI, `RobustScaler` per ROI, and a centred crop to `--n_timesteps` time points (160 by default).

```bash
python fmri_pipeline.py --fmri_dir FMRI/ --work_dir WORK/ --n_timesteps 160
# -> WORK/atlas/atlas_450rois.nii.gz, WORK/roi_timeseries/<subject>_roi_ts.npy  (450, T)
python Qc_fmri.py --ts_dir WORK/roi_timeseries --out WORK/qc
```

`Qc_fmri.py` flags ROIs with near-zero variance and draws the ROI correlation matrix and an ROI × time heatmap.

### 2. T1: `preprocess_t1_harmonix.py`, then `Qc_check.py`

The input is skull-stripped T1. Each volume is cropped/padded to (160, 192, 160), the input size of the BrainHarmonix T1 encoder, and z-scored over the whole volume (mean 0, standard deviation 1). The inference scripts refuse any other shape.

```bash
python Qc_check.py T1_PREPROCESSED/ --out QC_PNG/
```

`Qc_check.py` saves three cuts per volume and warns when non-zero voxels touch the edges, which can mean that part of the brain was cut off.

### 3. Inference

Requirements: the official Brain-Harmony repository (`--repo_dir`) and its checkpoints (`harmonix-f`, `harmonix-s`, and `harmonizer` for the fusion).

| Script | Output per subject |
|---|---|
| `run_brainharmony_encoders_only.py` | fMRI tokens (7200 × 768) and T1 tokens (1200 × 768), from the frozen encoders only. This is cheaper: there is no Harmonizer and no 8,529-token sequence. |
| `run_brainharmony_fusion.py` | The final BrainHarmonix output: fMRI and T1 tokens are concatenated and passed through the Harmonizer (OneTokRegViT), giving `latent_tokens` (129 × 768: 1 CLS + 128 latent tokens). Saved as `latent_tokens`, `cls_token` and `latents_only`. |

```bash
python run_brainharmony_fusion.py --repo_dir Brain-Harmony/ --checkpoints_dir CKPT/ \
    --roi_ts_dir WORK/roi_timeseries --t1_dir T1_PREPROCESSED/ --out_dir EMB/
```

For downstream analyses, the mean over the 128 latent tokens (`latents_only`, averaged) summarises a subject much better than the CLS token.

---

## Random-weight controls

Do the geometric results come from what the models learned, or from their architecture alone? To answer, the same pipelines run with **randomly initialised weights**: no checkpoint is loaded. Both scripts keep everything else identical to the trained pipelines.

```bash
python genomic/embed_ntv3_randinit.py --fasta_dir FASTA/ --out_dir EMB_RAND/ --model_cache_dir CACHE_NTV3/ --seeds 0 1 2
python imaging/inference_brainharmonix/run_brainharmony_fusion_randinit.py --repo_dir Brain-Harmony/ \
    --roi_ts_dir WORK/roi_timeseries --t1_dir T1_PREPROCESSED/ --out_dir EMB_RAND/ --seeds 0 1 2
```

- **`--init_scheme native` (default):** the weights are those drawn by each model's own initialisation code (e.g. `trunc_normal`, `xavier_uniform` and zero biases for ViT/MAE models; `post_init()` for NTv3), with no reset afterwards. This is the faithful "untrained architecture".
- **`--init_scheme pytorch_default`:** a sensitivity variant only. It resets every layer to PyTorch's generic initialisation, which overwrites the authors' scheme on Linear / Conv / Embedding layers but not on bare parameters (CLS and latent tokens). Its outputs go to `seed_<N>_pytorch_default/` and must not be mixed with the native run.
- **A diagnostic** prints the std and max |bias| of the first linear layer of each model, so you can check which scheme is actually in force.
- **Several seeds:** one random model does not characterise the variability of the initialisation, so each seed writes a complete set of embeddings to its own sub-folder.
- **BrainHarmonix:** the anatomical positional embedding is fixed information, not a learned weight, so it is kept by default. `--randomize_pos_embed` tests that separate hypothesis.
- **NTv3:** the script runs offline (no internet on compute nodes) and resolves the model code explicitly from `--model_cache_dir`. The reference-sequence cache is computed per seed.
