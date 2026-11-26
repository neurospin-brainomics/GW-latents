"""
This file checks whether a processed fMRI data is plausible or not.
Examines: 
- Close to null variances between ROIs
- Heatmap of correlation matrix between ROIs
- Heatmap of ROI x time to detect anomalies

How to use:
    python qc_fmri.py --ts_dir ~/Source/fmri_work/roi_timeseries --out ~/Source/fmri_work/qc
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def check_variance(ts: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Checks which ROIs have close to null variance."""
    variances = ts.var(axis=1)  # (450,)
    dead_rois = np.where(variances < threshold)[0]
    return dead_rois, variances


def make_carpet_plot(ts: np.ndarray, out_path: Path, title: str) -> None:
    """
    Carpet plot : ROIs x time
    """
    ts_display = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ts_display, aspect="auto", cmap="gray", vmin=-2, vmax=2)
    ax.set_xlabel("Temps (TR)")
    ax.set_ylabel("ROI (1-450)")
    ax.set_title(f"Carpet plot — {title}")
    fig.colorbar(im, ax=ax, label="z-score (affichage uniquement)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_fc_matrix(ts: np.ndarray, out_path: Path, title: str) -> np.ndarray:
    """Functional connectivity heatmap : Pearson correlation between ROIs."""
    fc = np.corrcoef(ts)  # (450, 450)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    ax.set_title(f"Functional connectivity — {title}")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return fc


def qc_single_subject(ts_path: Path, out_dir: Path) -> None:
    ts = np.load(ts_path)  # (450, T)
    name = ts_path.stem.replace("_roi_ts", "")

    print(f"\n[{name}]")
    print(f"  shape : {ts.shape}")

    dead_rois, variances = check_variance(ts)
    print(f"  mean variance : {variances.mean():.4f}  (min={variances.min():.6f}, max={variances.max():.4f})")

    if len(dead_rois) > 0:
        print(f"  Warning : {len(dead_rois)} ROI(s) with close to null variance : {dead_rois.tolist()}")
    else:
        print("  OK")

    carpet_path = out_dir / f"{name}_carpet.png"
    make_carpet_plot(ts, carpet_path, name)
    print(f"  -> carpet plot : {carpet_path}")

    fc_path = out_dir / f"{name}_fc_matrix.png"
    fc = make_fc_matrix(ts, fc_path, name)
    print(f"  -> matrice FC : {fc_path}")

    # petit résumé numérique de la structure de connectivité
    off_diag = fc[~np.eye(fc.shape[0], dtype=bool)]
    print(f"  off diagonal correlations : mean={off_diag.mean():.3f}, std={off_diag.std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Quick check fMRI (450 ROIs)")
    parser.add_argument("--ts_dir", type=str, required=True, help="File *_roi_ts.npy (before scaling)")
    parser.add_argument("--out", type=str, required=True, help="PNG files")
    args = parser.parse_args()

    ts_dir = Path(args.ts_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ts_dir.glob("*_roi_ts.npy"))
    if not files:
        print(f"No ROIs found in {ts_dir}")
        return

    for f in files:
        qc_single_subject(f, out_dir)


if __name__ == "__main__":
    main()