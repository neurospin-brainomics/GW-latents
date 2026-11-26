"""
This file checks whether or not a preprocess pipeline has damaged the T1 MRI file.
Generates :
  - an image with 3 different brain cuts
  - a warning if non-null information is close to the edges (might be that a part of the brain got cut)

Use:
    python qc_check.py /chemin/vers/brain_preprocessed.nii.gz
    python qc_check.py /chemin/vers/dossier_preprocessed --out /chemin/vers/qc_png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def check_edge_touch(volume: np.ndarray, threshold: float = 0.02) -> dict:
    """
    Verifies whether there is non null data on the edges.
    """
    results = {}
    axis_names = ["X (sagittal)", "Y (coronal)", "Z (axial)"]
    for axis, name in enumerate(axis_names):
        first_slice = np.take(volume, 0, axis=axis)
        last_slice = np.take(volume, volume.shape[axis] - 1, axis=axis)
        frac_first = (first_slice > threshold).mean()
        frac_last = (last_slice > threshold).mean()
        results[name] = {"beginning": round(float(frac_first), 4), "end": round(float(frac_last), 4)}
    return results


def make_mosaic(volume: np.ndarray, out_path: Path, title: str) -> None:
    """3 cuts PNG."""
    x_mid, y_mid, z_mid = (s // 2 for s in volume.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.rot90(volume[x_mid, :, :]), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Sagittal (X mid)")
    axes[1].imshow(np.rot90(volume[:, y_mid, :]), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Coronal (Y mid)")
    axes[2].imshow(np.rot90(volume[:, :, z_mid]), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Axial (Z mid)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def qc_single_file(nifti_path: Path, out_dir: Path, threshold: float = 0.02) -> None:
    img = nib.load(str(nifti_path))
    volume = img.get_fdata()

    print(f"\n[{nifti_path.name}]")
    print(f"  shape        : {volume.shape}")
    print(f"  intensities   : min={volume.min():.4f}, max={volume.max():.4f}")

    edge_report = check_edge_touch(volume, threshold=threshold)
    warn = False
    for axis_name, vals in edge_report.items():
        print(f"  bord {axis_name:15s} : beginning={vals['debut']*100:.1f}%  end={vals['fin']*100:.1f}% (voxels > seuil)")
        if vals["debut"] > 0.01 or vals["fin"] > 0.01:
            warn = True

    if warn:
        print("  Warning: non null signal close to the edges")
        print("      -> cropping might have cut parts of the brain")
    else:
        print("  OK ")

    out_path = out_dir / f"{nifti_path.stem.replace('.nii', '')}_qc.png"
    make_mosaic(volume, out_path, title=nifti_path.name)
    print(f"  -> PNG saved : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual check for MRI data")
    parser.add_argument("input_path", type=str, help=".nii.gz or directory containing .nii.gz")
    parser.add_argument("--out", type=str, default=None, help="End directory")
    parser.add_argument("--threshold", type=float, default=0.02, help="Edge detection threshold value")
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_dir():
        files = sorted(input_path.glob("*.nii.gz")) + sorted(input_path.glob("*.nii"))
        out_dir = Path(args.out) if args.out else input_path
    else:
        files = [input_path]
        out_dir = Path(args.out) if args.out else input_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        print("No .nii/.nii.gz found.")
        return

    for f in files:
        qc_single_file(f, out_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()