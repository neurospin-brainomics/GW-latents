import time
"""
Extract the intermediate BrainHarmonix embeddings (encoders only, without
going through the Harmonizer / fusion stage):

  - fMRI (Schaefer-400, robust scaling, 48-TR patches) -> harmonix-f -> (7200, 768)
  - T1 (160x192x160 crop, z-score)                     -> harmonix-s -> (1200, 768)

Unlike run_brainharmony_fusion.py, this script does not load the
harmonizer/model.pth checkpoint and does not run the fusion forward_encoder:
this saves compute time and GPU memory (no need to handle the 8529-token
concatenated sequence).

Requirements (same as run_brainharmony_fusion.py):
  - fMRI already parcellated (fmri_pipeline.py): *_roi_ts.npy (450, T) RAW
  - T1 already preprocessed with preprocess_t1_harmonix.py: (160,192,160), z-score
  - 2 checkpoints downloaded: harmonix-f, harmonix-s (harmonizer not required)

Usage:
    python run_brainharmony_encoders_only.py \\
        --repo_dir ~/brainharmony/Brain-Harmony \\
        --checkpoints_dir ~/brainharmony/checkpoints \\
        --roi_ts_dir ~/brainharmony/fmri_work/roi_timeseries \\
        --t1_dir ~/brainharmony/t1_work/preprocessed \\
        --out_dir ~/brainharmony/intermediate_embeddings \\
        --device cuda
"""

import argparse
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

CHECKPOINTS_GDRIVE_FOLDER_ID = "12MkUAOcegU60YVlK8u8_Owmgk4eQVheB"


def ensure_checkpoints(checkpoints_dir: Path) -> dict:
    # Only harmonix-f and harmonix-s are required here (not the harmonizer)
    paths = {
        "fmri": checkpoints_dir / "harmonix-f" / "model.pth",
        "t1": checkpoints_dir / "harmonix-s" / "model.pth",
    }
    if all(p.exists() for p in paths.values()):
        return paths

    print("One or more checkpoints missing, attempting to download the full folder...")
    try:
        import gdown

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=CHECKPOINTS_GDRIVE_FOLDER_ID, output=str(checkpoints_dir), quiet=False, use_cookies=False
        )
    except Exception as e:
        print(f"Automatic download failed: {e}")
        print(
            f"Download manually from https://drive.google.com/drive/folders/{CHECKPOINTS_GDRIVE_FOLDER_ID} "
            f"and place at least harmonix-f and harmonix-s in {checkpoints_dir}"
        )

    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoints still missing: {missing}")

    return paths


def ensure_repo(repo_dir: Path) -> None:
    if (repo_dir / "libs" / "model.py").exists():
        print(f"BrainHarmonix repo already present: {repo_dir}")
    else:
        print(f"Cloning {REPO_URL} into {repo_dir}...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)

    _patch_known_repo_bugs(repo_dir)


def _patch_known_repo_bugs(repo_dir: Path) -> None:
    """Same patches as run_brainharmony_fusion.py (bugs observed as of 2026-07-22)."""
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


def load_encoders(repo_dir: Path, ckpt_paths: dict, device: str):
    """Loads only fmri_encoder (harmonix-f) and t1_encoder (harmonix-s).
    The Harmonizer is NOT loaded here."""
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "libs"))

    import libs.model as model_lib
    import libs.position_embedding as pos_embeds
    import modules.harmonizer.util.t1_encoder as t1_encoder_model

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

    fmri_encoder = model_lib.vit_base_flex(
        pos_embed=pos_embed,
        cls_token=None,
        img_size=(N_CORTICAL, PATCH_SIZE * NUM_PATCHES),
        patch_size=PATCH_SIZE,
        gradient_checkpointing=False,
        attn_mode="eager",
    ).to(device)

    ckpt = torch.load(str(ckpt_paths["fmri"]), map_location="cpu", weights_only=True)
    prefix = "encoder_ema."
    state_dict = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    model_sd = fmri_encoder.state_dict()
    for key in list(state_dict.keys()):
        if key in model_sd and state_dict[key].size() != model_sd[key].size():
            del state_dict[key]
    msg = fmri_encoder.load_state_dict(state_dict, strict=False)
    print(f"  fmri_encoder (harmonix-f) load_state_dict: {msg}")
    fmri_encoder.eval()

    t1_encoder = t1_encoder_model.__dict__["mae_vit_base_patch16"](img_size=(160, 192, 160)).to(device)
    for blk in t1_encoder.blocks:
        blk.attn.attn_mode = "normal"

    t1_ckpt = torch.load(str(ckpt_paths["t1"]), map_location="cpu", weights_only=False)
    t1_state_dict = t1_ckpt["model"] if "model" in t1_ckpt else t1_ckpt
    msg = t1_encoder.load_state_dict(t1_state_dict, strict=False)
    print(f"  t1_encoder (harmonix-s) load_state_dict: {msg}")
    t1_encoder.eval()

    return fmri_encoder, t1_encoder


def process_batch(subject_names, roi_paths, t1_paths, fmri_encoder, t1_encoder, device, fmri_out_dir, t1_out_dir):
    print(f"\n[batch of {len(subject_names)}] {subject_names}")

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
        # NO fusion / harmonizer.forward_encoder here

    fmri_out_dir.mkdir(parents=True, exist_ok=True)
    t1_out_dir.mkdir(parents=True, exist_ok=True)
    for i, subject_name in enumerate(subject_names):
        fmri_np = fmri_tokens[i].cpu().numpy()
        t1_np = t1_tokens[i].cpu().numpy()

        fmri_path = fmri_out_dir / f"{subject_name}_harmonizer_fmri.npy"
        t1_path_out = t1_out_dir / f"{subject_name}_harmonizer_T1.npy"
        np.save(fmri_path, fmri_np)
        np.save(t1_path_out, t1_np)

        print(f"  [{subject_name}] fmri_tokens shape={fmri_np.shape} -> {fmri_path}")
        print(f"  [{subject_name}] t1_tokens shape={t1_np.shape} -> {t1_path_out}")


def main():
    parser = argparse.ArgumentParser(description="Extract intermediate embeddings (harmonix-f / harmonix-s), no fusion")
    parser.add_argument("--repo_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--roi_ts_dir", type=str, required=True)
    parser.add_argument("--t1_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Root output folder; outputs go into <out_dir>/harmonizer_fmri/ and <out_dir>/harmonizer_T1/")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_subjects", type=int, default=None,
                         help="Limit the number of subjects processed (useful for testing)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    roi_ts_dir = Path(args.roi_ts_dir)
    t1_dir = Path(args.t1_dir)
    out_dir = Path(args.out_dir)
    fmri_out_dir = out_dir / "harmonizer_fmri"
    t1_out_dir = out_dir / "harmonizer_T1"

    ckpt_paths = ensure_checkpoints(checkpoints_dir)
    ensure_repo(repo_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f">>> [DEBUG ENCODERS-ONLY] args.device={args.device} | torch.cuda.is_available()={torch.cuda.is_available()} | final device={device}")
    if args.device == "cuda" and device == "cpu":
        print("CUDA not available, falling back to CPU (risk of slowness/OOM).")

    fmri_encoder, t1_encoder = load_encoders(repo_dir, ckpt_paths, device)

    import re

    def extract_subject_id(name: str):
        m = re.search(r"sub-?\d+", name)
        return m.group(0) if m else None

    ts_files = {}
    for f in roi_ts_dir.glob("*_roi_ts.npy"):
        sid = extract_subject_id(f.name)
        if sid:
            ts_files[sid] = f

    t1_files = {}
    for f in list(t1_dir.glob("*.nii.gz")) + list(t1_dir.glob("*.nii")):
        sid = extract_subject_id(f.name)
        if sid:
            t1_files[sid] = f

    common_subjects = sorted(set(ts_files) & set(t1_files))

    # Resume: skip subjects already processed (present in BOTH sub-folders)
    fmri_done = {f.stem.replace("_harmonizer_fmri", "") for f in fmri_out_dir.glob("*_harmonizer_fmri.npy")} if fmri_out_dir.exists() else set()
    t1_done = {f.stem.replace("_harmonizer_T1", "") for f in t1_out_dir.glob("*_harmonizer_T1.npy")} if t1_out_dir.exists() else set()
    already_done = fmri_done & t1_done
    n_before = len(common_subjects)
    common_subjects = [s for s in common_subjects if s not in already_done]
    print(f"Resume: {n_before - len(common_subjects)} subject(s) already done, {len(common_subjects)} remaining.")

    common_subjects = common_subjects[args.start_idx:args.end_idx]
    if args.max_subjects is not None:
        common_subjects = common_subjects[:args.max_subjects]
        print(f"Limited to {len(common_subjects)} subject(s) (--max_subjects={args.max_subjects}).")

    if not common_subjects:
        raise RuntimeError("No subject with both fMRI AND T1 matched (or all already processed). Check the filenames.")

    print(f"\n{len(common_subjects)} subject(s) to process (fMRI + T1 matched)...")
    t_loop_start = time.time()
    for i in range(0, len(common_subjects), args.batch_size):
        batch_names = common_subjects[i:i + args.batch_size]
        batch_roi_paths = [ts_files[n] for n in batch_names]
        batch_t1_paths = [t1_files[n] for n in batch_names]
        process_batch(batch_names, batch_roi_paths, batch_t1_paths, fmri_encoder, t1_encoder, device, fmri_out_dir, t1_out_dir)
        n_done = min(i + args.batch_size, len(common_subjects))
        elapsed = time.time() - t_loop_start
        rate = elapsed / n_done if n_done > 0 else 0
        eta_min = rate * (len(common_subjects) - n_done) / 60
        pct = 100 * n_done / len(common_subjects)
        nbars = int(40 * n_done / len(common_subjects))
        bar = "#" * nbars + " " * (40 - nbars)
        print(f">>> [PROGRESS] [{bar}] {n_done}/{len(common_subjects)} ({pct:.0f}%) - ETA: {eta_min:.1f} min")

    print(f"\nDone.")
    print(f"  harmonix-f (fMRI) embeddings in: {fmri_out_dir}")
    print(f"  harmonix-s (T1) embeddings in:   {t1_out_dir}")


if __name__ == "__main__":
    main()
