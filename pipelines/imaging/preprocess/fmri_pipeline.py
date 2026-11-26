"""
Preprocessing pipeline to use Brain Harmony:

  0) Downloads and builds the 450 Schaeffer ROI atlas (400: cortical, 50: subcortical)
  1) Dissects each fMRI (already into MNI152) by aggregating through a mean BOLD signal to each ROI.
  2) RobustScaler from scikit-learn on each ROI.
  3) Cropping centered on 160 timesteps.

- Schaefer-400 (7 networks, 2mm, espace FSL MNI152) :
    https://github.com/ThomasYeoLab/CBIG
- Tian Subcortex Scale III (3T, 2mm, espace MNI152NLin6Asym) :
    https://github.com/yetianmed/subcortex

Requirements:
    pip install nilearn scikit-learn nibabel joblib requests --break-system-packages

Use :
    python fmri_pipeline.py \\
        --fmri_dir /chemin/vers/fmri_preprocessed \\
        --work_dir /chemin/vers/dossier_travail \\
        --n_timesteps 160

Eventually, gives:
    work_dir/atlas/atlas_450rois.nii.gz     <- atlas
    work_dir/roi_timeseries/*_roi_ts.npy    <- (450, T) for each subject
    work_dir/final/robust_scaler.joblib     <- scaler of the cohort
    work_dir/final/*_final.npy              <- (450, 160) ready to use
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import requests
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from sklearn.preprocessing import RobustScaler

SCHAEFER_400_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"
    "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)
TIAN_S3_URL = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/"
    "3T/Subcortex-Only/Tian_Subcortex_S3_3T.nii"
)


# --------------------------------------------------------------------------
# Step 0 : Atlas
# --------------------------------------------------------------------------


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  déjà présent : {dest}")
        return dest
    print(f"  téléchargement : {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  -> saved : {dest} ({len(resp.content) / 1e6:.1f} Mo)")
    return dest


def build_combined_atlas(atlas_dir: Path) -> Path:
    """
    Downloads Schaefer-400 (cortical) and Tian-III (subcortical) and fuses them.
    """
    atlas_dir.mkdir(parents=True, exist_ok=True)
    combined_path = atlas_dir / "atlas_450rois.nii.gz"

    if combined_path.exists():
        print(f"Atlas already built : {combined_path}")
        return combined_path

    print("Building the atlas (450 ROIs)...")
    schaefer_path = download_file(SCHAEFER_400_URL, atlas_dir / "schaefer400.nii.gz")
    tian_path = download_file(TIAN_S3_URL, atlas_dir / "tian_s3.nii")

    schaefer_img = nib.load(str(schaefer_path))
    tian_img = nib.load(str(tian_path))

    # Align Tian on Schaefer if the two do not exactly fit
    if schaefer_img.shape != tian_img.shape or not np.allclose(
        schaefer_img.affine, tian_img.affine, atol=1e-3
    ):
        print("  Schaefer/Tian grids are different -> resampling nearest-neighbor Tian onto Schaefer")
        tian_img = resample_to_img(tian_img, schaefer_img, interpolation="nearest", force_resample=True)

    schaefer_data = schaefer_img.get_fdata().astype(np.int32)
    tian_data = tian_img.get_fdata().astype(np.int32)

    tian_shifted = np.where(tian_data > 0, tian_data + 400, 0)
    combined_data = np.where(schaefer_data > 0, schaefer_data, tian_shifted).astype(np.int32)

    n_rois_found = len(np.unique(combined_data)) - 1  # -1 pour exclure le label 0 (fond)
    print(f"  nombre de ROIs dans l'atlas combiné : {n_rois_found} (attendu: 450)")

    combined_img = nib.Nifti1Image(combined_data, affine=schaefer_img.affine, header=schaefer_img.header)
    nib.save(combined_img, combined_path)
    print(f"Atlas saved : {combined_path}")
    return combined_path


# --------------------------------------------------------------------------
# 1 - Dissecting the ROIs
# --------------------------------------------------------------------------


def build_masker(atlas_path: Path) -> NiftiLabelsMasker:
    return NiftiLabelsMasker(
        labels_img=str(atlas_path),
        standardize=False,  # scaling later
        detrend=False,      # already done in UKB datasets
        resampling_target="data",
        strategy="mean",
    )


def extract_all_timeseries(fmri_dir: Path, atlas_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    masker = build_masker(atlas_path)

    files = sorted(fmri_dir.glob("*.nii.gz")) + sorted(fmri_dir.glob("*.nii"))
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz found {fmri_dir}")

    print(f"\n{len(files)} subject(s)")
    out_paths = []
    for f in files:
        out_path = out_dir / f"{f.stem.replace('.nii', '')}_roi_ts.npy"
        if out_path.exists():
            print(f"  [{f.name}] already extracted -> {out_path.name}")
            out_paths.append(out_path)
            continue
        print(f"  [{f.name}] extracting...")
        ts = masker.fit_transform(str(f)).T  # (T, 450) -> (450, T)
        np.save(out_path, ts)
        print(f"    shape={ts.shape}")
        out_paths.append(out_path)

    return out_paths


# --------------------------------------------------------------------------
# 2 - Scaling
# --------------------------------------------------------------------------


def fit_robust_scaler(ts_paths: list[Path]) -> RobustScaler:
    print("\nRobust scaler by ROIs")
    stacked = np.concatenate([np.load(p).T for p in ts_paths], axis=0)  # (sum(T_i), 450)
    scaler = RobustScaler(with_centering=True, with_scaling=True)
    scaler.fit(stacked)
    return scaler


def crop_or_pad_time(arr: np.ndarray, n_timesteps: int) -> np.ndarray:
    n_rois, t = arr.shape
    if t == n_timesteps:
        return arr
    if t > n_timesteps:
        start = (t - n_timesteps) // 2
        return arr[:, start : start + n_timesteps]
    pad_total = n_timesteps - t
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)


def scale_and_pad_all(ts_paths: list[Path], out_dir: Path, n_timesteps: int) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = fit_robust_scaler(ts_paths)
    joblib.dump(scaler, out_dir / "robust_scaler.joblib")
    print(f"Scaler sauvegardé : {out_dir / 'robust_scaler.joblib'}")

    print("\n Scaling application + crop/pad...")
    for p in ts_paths:
        arr = np.load(p)  # (450, T)
        scaled = scaler.transform(arr.T).T
        final = crop_or_pad_time(scaled, n_timesteps)

        out_name = p.name.replace("_roi_ts.npy", "_final.npy")
        out_path = out_dir / out_name
        np.save(out_path, final)
        print(f"  [{p.name}] T_original={arr.shape[1]} -> final shape={final.shape}")


# --------------------------------------------------------------------------
# Executing
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Complete fMRI preprocessing pipeline for Brain Harmony : atlas + parcellisation + robust scaling + crop/pad"
    )
    parser.add_argument("--fmri_dir", type=str, required=True, help="fMRI file (.nii.gz)")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory (atlas/, roi_timeseries/, final/)")
    parser.add_argument("--n_timesteps", type=int, default=160, help="Timesteps (by default: 160)")
    args = parser.parse_args()

    fmri_dir = Path(args.fmri_dir)
    work_dir = Path(args.work_dir)

    atlas_path = build_combined_atlas(work_dir / "atlas")
    ts_paths = extract_all_timeseries(fmri_dir, atlas_path, work_dir / "roi_timeseries")
    scale_and_pad_all(ts_paths, work_dir / "final", args.n_timesteps)

    print(f"\n Finished. End results (450, {args.n_timesteps}) in : {work_dir / 'final'}")


if __name__ == "__main__":
    main()