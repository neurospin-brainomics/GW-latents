"""
Variant of run_brainharmony_fusion.py for the "random weights" control.

Goal: isolate the contribution of the ARCHITECTURE alone (without any
training) to the measured GW geometry, distinguishing:
  - "learned signal" hypothesis        -> GW(rand, rand) close to noise / to the ceiling
  - "architecture artefact" hypothesis -> GW(rand, rand) already high, comparable
                                          to the true positive pairs

Differences with the original script:
  - The 3 `load_state_dict(...)` calls (fmri_encoder, t1_encoder, harmonizer)
    are SKIPPED. No checkpoint is needed nor downloaded in this mode.
  - --init_scheme {native, pytorch_default} (corrected 2026-09):
      * native (default, recommended): NO reset after instantiation. A freshly
        instantiated nn.Module already holds weights drawn by the init scheme
        defined in its own __init__ -- for these ViT/MAE-style models, a
        custom scheme applied through self.apply(_init_weights) (e.g.
        trunc_normal(std=0.02), xavier_uniform, zero biases). This is the most
        faithful "untrained architecture" control: we keep the init scheme
        actually chosen by the authors, we do not reinvent it.
      * pytorch_default: the earlier behaviour, kept ONLY as an explicit
        sensitivity variant. It calls reset_parameters() on every sub-module
        (generic PyTorch init: kaiming_uniform for Linear, etc.), which
        silently OVERWRITES the authors' custom init on Linear / Conv /
        Embedding layers while leaving bare nn.Parameter tokens (cls_token,
        latent tokens) at the authors' init -- an incoherent mix. Outputs go
        to a distinct sub-folder (seed_<N>_pytorch_default/) and must never be
        mixed with the native run in the same analysis.
  - Diagnostic _log_weight_std (shape, std and max |bias| of the first
    nn.Linear of each module) to check which scheme was actually applied.
  - `pos_embed` (anatomical geometric harmonics, built from the gradient /
    eigenmode CSV files) is NOT randomised by default: it is not a learned
    weight but fixed anatomical information. The flag `--randomize_pos_embed`
    tests this variant separately (distinct hypothesis: does the anatomical
    position itself contribute to the anisotropy?); not to be mixed with the
    main run.
  - Several independent seeds are mandatory (`--seeds 0 1 2 ...`): a single
    random instance does not characterise the variance of the init itself.
    Each seed produces a full set of embeddings in a separate sub-folder.

Usage:
    python run_brainharmony_fusion_randinit.py \\
        --repo_dir ~/brainharmony/Brain-Harmony \\
        --roi_ts_dir ~/brainharmony/fmri_work/roi_timeseries \\
        --t1_dir ~/brainharmony/t1_work/preprocessed \\
        --out_dir /path/to/random_embeddings \\
        --seeds 0 1 2 \\
        --device cuda

Output (per seed):
    <out_dir>/seed_<N>[_pytorch_default]/<subject>_harmonizer_final_randinit_seed<N>.npz
    with the same keys as the original: latent_tokens, cls_token, latents_only

Inputs: same as run_brainharmony_fusion.py -- fMRI parcellated with
fmri_pipeline.py (*_roi_ts.npy, 450 x T, raw) and T1 prepared with
preprocess_t1_harmonix.py ((160, 192, 160), z-score).

IMPORTANT -- memory: same footprint as the original (8529 tokens per
subject), nothing changes on that point.
"""
import time
import argparse
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO_URL = "https://github.com/hzlab/Brain-Harmony.git"

N_CORTICAL = 400
PATCH_SIZE = 48
NUM_PATCHES = 18
EMBED_DIM = 768
NUM_LATENT_TOKENS = 128


def ensure_repo(repo_dir: Path) -> None:
    if (repo_dir / "libs" / "model.py").exists():
        print(f"BrainHarmonix repository already present: {repo_dir}")
    else:
        print(f"Cloning {REPO_URL} into {repo_dir}...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)

    _patch_known_repo_bugs(repo_dir)


def _patch_known_repo_bugs(repo_dir: Path) -> None:
    """Same patches as in the original script (see that file for the details)."""
    ft_path = repo_dir / "libs" / "flex_transformer.py"
    content = ft_path.read_text()
    fixed = content.replace(
        "from attn_utils import expand_mask",
        "from attn_utils.expand_mask import _expand_mask as expand_mask",
    )
    fixed = "\n".join(line for line in fixed.split("\n") if line.strip() != "breakpoint()")
    fixed = fixed.replace('attn_mode="flash_attention_2"', 'attn_mode="eager"')
    if fixed != content:
        ft_path.write_text(fixed)
        print(f"  patches applied to {ft_path}")

    t1_path = repo_dir / "modules" / "harmonizer" / "util" / "t1_encoder.py"
    content = t1_path.read_text()
    already_patched = "except ImportError:\n    flash_attn_qkvpacked_func = None" in content
    fixed = content
    if not already_patched:
        fixed = content.replace(
            "from flash_attn import flash_attn_qkvpacked_func",
            "try:\n    from flash_attn import flash_attn_qkvpacked_func\n"
            "except ImportError:\n    flash_attn_qkvpacked_func = None",
        )
    if fixed != content:
        t1_path.write_text(fixed)
        print(f"  patch applied to {t1_path}")
    elif already_patched:
        print(f"  {t1_path} already patched, nothing to do")


def robust_scale_per_roi(ts: np.ndarray) -> np.ndarray:
    median = np.median(ts, axis=1, keepdims=True)
    q75 = np.percentile(ts, 75, axis=1, keepdims=True)
    q25 = np.percentile(ts, 25, axis=1, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (ts - median) / iqr


def build_fmri_input(roi_ts_path: Path) -> tuple:
    ts_full = np.load(roi_ts_path)
    if ts_full.shape[0] < N_CORTICAL:
        raise ValueError(f"{roi_ts_path}: {ts_full.shape[0]} ROIs < 400 expected")

    ts_400 = ts_full[:N_CORTICAL, :]
    ts_400_scaled = robust_scale_per_roi(ts_400.astype(np.float32))

    n_rois, t = ts_400_scaled.shape
    target_len = PATCH_SIZE * NUM_PATCHES
    n_valid_patches = int(np.ceil(t / PATCH_SIZE))
    if n_valid_patches > NUM_PATCHES:
        raise ValueError(f"fMRI signal too long ({t} TR) for NUM_PATCHES={NUM_PATCHES}")

    padded = np.zeros((n_rois, target_len), dtype=np.float32)
    padded[:, :t] = ts_400_scaled[:, :t]
    x_fmri = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(torch.float32)

    patch_valid = torch.arange(NUM_PATCHES) < n_valid_patches
    mask2d = patch_valid.unsqueeze(0).repeat(n_rois, 1)
    attn_mask_fmri = mask2d.reshape(-1).unsqueeze(0).int()

    return x_fmri, attn_mask_fmri


def load_t1_volume(t1_path: Path) -> torch.Tensor:
    img = nib.load(str(t1_path))
    volume = img.get_fdata().astype(np.float32)
    if volume.shape != (160, 192, 160):
        raise ValueError(
            f"{t1_path}: shape {volume.shape} != (160,192,160). "
            "Use preprocess_t1_harmonix.py to reformat this T1."
        )
    return torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)


def _reinit_module_pytorch_default(module: torch.nn.Module) -> int:
    """
    Generic PyTorch reset (kaiming_uniform for Linear, etc.) via
    reset_parameters() on every sub-module that exposes it (Linear,
    LayerNorm, Conv*, Embedding, ...). This is NOT the authors' scheme when
    the model applies a custom init in __init__ (self.apply(_init_weights)):
    used only by --init_scheme pytorch_default (sensitivity variant) and by
    --randomize_pos_embed.

    Returns the number of sub-modules actually reset, for checking (see the
    warning in load_models() if this number is 0).
    """
    n_reset = 0
    for m in module.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()
            n_reset += 1
    return n_reset


def _log_weight_std(module: torch.nn.Module, label: str, seed: int):
    """
    Diagnostic: std and max |bias| of the first nn.Linear found. A non-zero
    bias is a strong sign that the generic PyTorch init is in force
    (reset_parameters, or no custom init) rather than a xavier/trunc_normal
    scheme with zero bias (typical of ViT/MAE).
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            std = m.weight.detach().float().std().item()
            bias_max = m.bias.detach().float().abs().max().item() if m.bias is not None else float("nan")
            print(f"  [seed={seed}] {label}: first nn.Linear shape={tuple(m.weight.shape)} "
                  f"std={std:.5f} bias_abs_max={bias_max:.5f} "
                  f"(indicative values -- kaiming_uniform(a=sqrt(5)) ~0.021, xavier_uniform ~0.026-0.036, "
                  f"trunc_normal(0.02) ~0.02; bias ~0 => custom init likely, non-zero bias => "
                  f"generic PyTorch init likely)")
            return
    print(f"  [seed={seed}] {label}: no nn.Linear found for the diagnostic")


def load_models(repo_dir: Path, device: str, seed: int, randomize_pos_embed: bool, init_scheme: str):
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "libs"))

    import libs.model as model_lib
    import libs.position_embedding as pos_embeds
    import modules.harmonizer.util.t1_encoder as t1_encoder_model
    from modules.harmonizer.stage1_pretrain.models import onetokreg_vit_base_patch16

    # seed set right before instantiation: __init__ draws the initial
    # weights, so the seed must be active at that precise moment for the
    # weights to be reproducible per seed.
    torch.manual_seed(seed)

    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    pos_embed_args = Args(
        grid_size=(N_CORTICAL, NUM_PATCHES),
        embed_dim=EMBED_DIM,
        predictor_embed_dim=384,
        cls_token=False,
        grad_dim=30,
        gradient=str(repo_dir / "brainharmony_pos_embed" / "gradient_mapping_400.csv"),
        geoh_dim=200,
        geo_harm=str(repo_dir / "brainharmony_pos_embed" / "schaefer400_roi_eigenmodes.csv"),
        use_pos_embed_decoder=False,
    )
    pos_embed = pos_embeds.BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed(device, pos_embed_args)
    pos_embed = pos_embed.to(device)

    if randomize_pos_embed:
        n_reset = _reinit_module_pytorch_default(pos_embed)
        print(f"  [seed={seed}] pos_embed: {n_reset} sub-module(s) randomised "
              f"(--randomize_pos_embed active -- non-standard variant, not to be "
              f"mixed with the main run)")
    else:
        print(f"  [seed={seed}] pos_embed: left INTACT (fixed anatomical information, not a learned weight)")

    fmri_encoder = model_lib.vit_base_flex(
        pos_embed=pos_embed,
        cls_token=None,
        img_size=(N_CORTICAL, PATCH_SIZE * NUM_PATCHES),
        patch_size=PATCH_SIZE,
        gradient_checkpointing=False,
        attn_mode="eager",
    ).to(device)

    t1_encoder = t1_encoder_model.__dict__["mae_vit_base_patch16"](img_size=(160, 192, 160)).to(device)
    for blk in t1_encoder.blocks:
        blk.attn.attn_mode = "normal"

    harmonizer = onetokreg_vit_base_patch16(num_latent_tokens=NUM_LATENT_TOKENS).to(device)

    if init_scheme == "pytorch_default":
        n_reset = {name: _reinit_module_pytorch_default(m)
                   for name, m in (("fmri_encoder", fmri_encoder), ("t1_encoder", t1_encoder), ("harmonizer", harmonizer))}
        print(f"  [seed={seed}] init_scheme=pytorch_default: sub-modules reset to the generic PyTorch init -- "
              + ", ".join(f"{k}: {v}" for k, v in n_reset.items())
              + " (sensitivity variant, NOT the authors' scheme)")
        if 0 in n_reset.values():
            print("  WARNING: a count is 0 -- those weights stay as drawn at construction "
                  "(already random), so not a blocking bug, but check that no expected module "
                  "was silently missed.")
    else:
        print(f"  [seed={seed}] init_scheme=native: no reset after construction -- "
              f"weights as drawn by each model's own __init__ (authors' scheme).")

    for name, m in (("fmri_encoder", fmri_encoder), ("t1_encoder", t1_encoder), ("harmonizer", harmonizer)):
        m.eval()
        _log_weight_std(m, name, seed)

    # NO checkpoint loaded here -- that is the whole point of the control.
    return fmri_encoder, t1_encoder, harmonizer


def process_batch(subject_names, roi_paths, t1_paths, fmri_encoder, t1_encoder, harmonizer, device, out_dir, seed):
    print(f"\n[seed={seed}] [batch of {len(subject_names)}] {subject_names}")

    x_fmri_list, mask_list, x_t1_list = [], [], []
    for roi_path, t1_path in zip(roi_paths, t1_paths):
        xf, mf = build_fmri_input(roi_path)
        x_fmri_list.append(xf.squeeze(0))
        mask_list.append(mf.squeeze(0))
        x_t1_list.append(load_t1_volume(t1_path).squeeze(0))

    x_fmri = torch.stack(x_fmri_list).to(device)
    attn_mask_fmri = torch.stack(mask_list).to(device)
    x_t1 = torch.stack(x_t1_list).to(device)

    with torch.no_grad():
        fmri_tokens = fmri_encoder(x_fmri, PATCH_SIZE, attention_mask=attn_mask_fmri)
        t1_tokens = t1_encoder(x_t1)
        x = torch.cat([fmri_tokens, t1_tokens], dim=1)
        latent_tokens, _ = harmonizer.forward_encoder(x, attn_mask_fmri)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, subject_name in enumerate(subject_names):
        latent_np = latent_tokens[i].cpu().numpy()
        cls_np = latent_np[0]
        latents_only_np = latent_np[1:]
        out_path = out_dir / f"{subject_name}_harmonizer_final_randinit_seed{seed}.npz"
        np.savez(out_path, latent_tokens=latent_np, cls_token=cls_np, latents_only=latents_only_np)
        print(f"  [{subject_name}] latent_tokens shape={latent_np.shape}")
        print(f"  -> saved: {out_path}")


def extract_subject_id(name: str):
    m = re.search(r"sub-?\d+", name)
    return m.group(0) if m else None


def run_one_seed(args, repo_dir, roi_ts_dir, t1_dir, out_dir_seed, seed, device):
    fmri_encoder, t1_encoder, harmonizer = load_models(
        repo_dir, device, seed=seed, randomize_pos_embed=args.randomize_pos_embed, init_scheme=args.init_scheme
    )

    ts_files = {}
    for f in roi_ts_dir.glob("*_roi_ts.npy"):
        sid = extract_subject_id(f.name)
        if sid:
            ts_files[sid] = f
        else:
            print(f"  warning: could not extract a sub-XXX id from {f.name}, skipped")

    t1_files = {}
    for f in list(t1_dir.glob("*.nii.gz")) + list(t1_dir.glob("*.nii")):
        sid = extract_subject_id(f.name)
        if sid:
            t1_files[sid] = f
        else:
            print(f"  warning: could not extract a sub-XXX id from {f.name}, skipped")

    common_subjects = sorted(set(ts_files) & set(t1_files))

    already_done = {
        f.stem.replace(f"_harmonizer_final_randinit_seed{seed}", "")
        for f in out_dir_seed.glob(f"*_harmonizer_final_randinit_seed{seed}.npz")
    }
    n_before = len(common_subjects)
    common_subjects = [s for s in common_subjects if s not in already_done]
    print(f"[seed={seed}] Resume: {n_before - len(common_subjects)} subject(s) already done, "
          f"{len(common_subjects)} remaining.")

    common_subjects = common_subjects[args.start_idx:args.end_idx]
    print(f"[seed={seed}] Slice [{args.start_idx}:{args.end_idx}]: {len(common_subjects)} subject(s) assigned.")

    if args.max_subjects is not None:
        common_subjects = common_subjects[:args.max_subjects]
        print(f"[seed={seed}] Limited to {len(common_subjects)} subject(s) (--max_subjects={args.max_subjects}).")

    missing_t1 = sorted(set(ts_files) - set(t1_files))
    missing_fmri = sorted(set(t1_files) - set(ts_files))
    if missing_t1:
        print(f"[seed={seed}] Subjects with fMRI but no matching T1 (skipped): {missing_t1}")
    if missing_fmri:
        print(f"[seed={seed}] Subjects with T1 but no matching fMRI (skipped): {missing_fmri}")

    if not common_subjects:
        print(f"[seed={seed}] Nothing to process (already done, or no paired subject). Skip.")
        return

    print(f"[seed={seed}] {len(common_subjects)} subject(s) to process...")
    t_loop_start = time.time()
    for i in range(0, len(common_subjects), args.batch_size):
        batch_names = common_subjects[i:i + args.batch_size]
        batch_roi_paths = [ts_files[n] for n in batch_names]
        batch_t1_paths = [t1_files[n] for n in batch_names]
        process_batch(
            batch_names, batch_roi_paths, batch_t1_paths,
            fmri_encoder, t1_encoder, harmonizer, device, out_dir_seed, seed,
        )
        n_done = min(i + args.batch_size, len(common_subjects))
        elapsed = time.time() - t_loop_start
        rate = elapsed / n_done if n_done > 0 else 0
        eta_min = rate * (len(common_subjects) - n_done) / 60
        pct = 100 * n_done / len(common_subjects)
        nbars = int(40 * n_done / len(common_subjects))
        bar = "#" * nbars + " " * (40 - nbars)
        print(f">>> [seed={seed}][PROGRESS] [{bar}] {n_done}/{len(common_subjects)} ({pct:.0f}%) - ETA: {eta_min:.1f} min")

    # free the GPU before the next seed
    del fmri_encoder, t1_encoder, harmonizer
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="BrainHarmonix extraction with RANDOM weights (architecture-only control)")
    parser.add_argument("--repo_dir", type=str, required=True)
    parser.add_argument("--roi_ts_dir", type=str, required=True)
    parser.add_argument("--t1_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output root; one sub-folder seed_<N>/ is created per seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Independent seeds to generate (>= 2 recommended to characterise "
                             "the variance of the init itself)")
    parser.add_argument("--init_scheme", choices=["native", "pytorch_default"], default="native",
                        help="native (default): authors' init, no reset; pytorch_default: generic PyTorch reset, "
                             "sensitivity variant only (written to seed_<N>_pytorch_default/)")
    parser.add_argument("--randomize_pos_embed", action="store_true",
                        help="Also randomise pos_embed (non-standard variant, distinct hypothesis "
                             "-- see docstring). Default: pos_embed stays intact.")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Limit the number of subjects processed (useful for testing)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    roi_ts_dir = Path(args.roi_ts_dir)
    t1_dir = Path(args.t1_dir)
    out_dir = Path(args.out_dir)

    ensure_repo(repo_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f">>> [RANDINIT] seeds={args.seeds} | init_scheme={args.init_scheme} | "
          f"randomize_pos_embed={args.randomize_pos_embed} | device={device}")
    if args.device == "cuda" and device == "cpu":
        print("CUDA not available, falling back to CPU (may be slow).")

    suffix = "" if args.init_scheme == "native" else f"_{args.init_scheme}"
    for seed in args.seeds:
        out_dir_seed = out_dir / f"seed_{seed}{suffix}"
        run_one_seed(args, repo_dir, roi_ts_dir, t1_dir, out_dir_seed, seed, device)

    print(f"\nDone. Random-weight embeddings in: {out_dir} (one sub-folder per seed)")


if __name__ == "__main__":
    main()
