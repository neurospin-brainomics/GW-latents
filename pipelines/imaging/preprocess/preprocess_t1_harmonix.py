"""
Preprocessing of already skull-stripped T1 for BrainHarmonix:
  1) Centred crop / pad to the required TARGET_SHAPE (160, 192, 160), whatever
     the raw input shape (crop if too large, pad if too small, axis by axis,
     independently).
  2) Intensity normalisation over the volume (z-score, mean 0 / std 1).

The output is what the BrainHarmonix T1 encoder (harmonix-s) expects: the
inference scripts refuse any other shape.

Tools: nibabel, numpy
Requirements:
    pip install nibabel numpy

Usage:
    python preprocess_t1_harmonix.py T1_SKULL_STRIPPED/ T1_PREPROCESSED/ [--suffix _harmonix_preprocessed]
"""
from pathlib import Path

import nibabel as nib
import numpy as np

TARGET_SHAPE = (160, 192, 160)  # final shape expected by harmonix-s


def crop_or_pad_center(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Centres `data` on `target_shape`, axis by axis:
      - if the input axis is longer than the target  -> centred crop
      - if the input axis is shorter than the target -> centred pad (zeros)
      - if equal -> unchanged
    Works whatever the raw input shape.
    """
    out = data
    for axis, target_len in enumerate(target_shape):
        in_len = out.shape[axis]
        if in_len == target_len:
            continue
        elif in_len > target_len:
            start = (in_len - target_len) // 2
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(start, start + target_len)
            out = out[tuple(slicer)]
        else:
            pad_total = target_len - in_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_left, pad_right)
            out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """(img - mean) / (std + eps), exactly as T1_BaseDataset.normalize() in BrainHarmonix
    (statistics over the whole volume, background included)."""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + 1e-9)


def preprocess_volume(input_path: str, output_path: str) -> None:
    img_nii = nib.load(input_path)
    volume = img_nii.get_fdata().astype(np.float32)

    original_shape = volume.shape
    cropped = crop_or_pad_center(volume, TARGET_SHAPE)  # -> (160, 192, 160), whatever original_shape
    normalized = zscore_normalize(cropped)

    out_img = nib.Nifti1Image(normalized, affine=img_nii.affine)
    nib.save(out_img, output_path)

    print(
        f"  -> raw shape={original_shape} -> final shape={normalized.shape}, "
        f"mean={normalized.mean():.4f}, std={normalized.std():.4f}"
    )


def find_nifti_files(input_dir: Path) -> list:
    return sorted(list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii")))


def process_directory(input_dir: str, output_dir: str, suffix: str = "_harmonix_preprocessed") -> None:
    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    files = find_nifti_files(input_dir_p)
    if not files:
        print(f"No .nii/.nii.gz file found in {input_dir_p}")
        return

    print(f"{len(files)} file(s) found in {input_dir_p}\n")

    n_ok, n_fail = 0, 0
    for f in files:
        base_name = f.name[:-7] if f.name.endswith(".nii.gz") else f.stem
        out_path = output_dir_p / f"{base_name}{suffix}.nii.gz"
        print(f"[{f.name}]")
        try:
            preprocess_volume(str(f), str(out_path))
            n_ok += 1
        except Exception as e:
            print(f"  !! ERROR: {e}")
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess T1 for BrainHarmonix: generic centred crop/pad to 160x192x160 + z-score"
    )
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--suffix", type=str, default="_harmonix_preprocessed")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.suffix)
