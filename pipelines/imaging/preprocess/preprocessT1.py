"""
Pré-processing IRM T1 skull-strippée pour foundation model :
  1) Crop / pad centré à la taille spatiale cible (167, 212, 160)
  2) Normalisation d'intensité par voxel entre 0 et 1 (min-max scaling)

Outil : MONAI (monai.transforms), standard de facto pour ce type de
pipeline dans les foundation models d'imagerie cérébrale 3D.

Installation :
    pip install monai nibabel --break-system-packages
"""

from pathlib import Path

import nibabel as nib
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, ResizeWithPadOrCrop, ScaleIntensity

TARGET_SHAPE = (167, 212, 160)


def build_transforms() -> Compose:
    """Construit le pipeline MONAI : chargement -> canal -> crop/pad -> normalisation."""
    return Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=False),
            EnsureChannelFirst(),  # ajoute la dimension canal -> (1, X, Y, Z)
            ResizeWithPadOrCrop(spatial_size=TARGET_SHAPE, mode="constant", constant_values=0),
            ScaleIntensity(minv=0.0, maxv=1.0),  # min-max par volume, voxel par voxel
        ]
    )


def preprocess_volume(input_path: str, output_path: str, transforms: Compose) -> None:
    """Applique le pipeline à un fichier NIfTI et sauvegarde le résultat."""
    img = transforms(input_path)  # tenseur MetaTensor, shape (1, 167, 212, 160)

    # On retire la dimension canal pour ré-écrire un NIfTI 3D classique
    volume = img[0].numpy()

    # On récupère l'affine d'origine pour garder les métadonnées spatiales cohérentes
    original_affine = nib.load(input_path).affine

    out_img = nib.Nifti1Image(volume, affine=original_affine)
    nib.save(out_img, output_path)

    print(f"  -> shape={volume.shape}, min={volume.min():.4f}, max={volume.max():.4f}")


def find_nifti_files(input_dir: Path) -> list[Path]:
    """Liste tous les fichiers .nii et .nii.gz d'un dossier (non récursif)."""
    files = sorted(list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii")))
    return files


def process_directory(input_dir: str, output_dir: str, suffix: str = "_preprocessed") -> None:
    """Boucle sur tous les fichiers NIfTI d'un dossier et applique le pipeline à chacun."""
    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    files = find_nifti_files(input_dir_p)
    if not files:
        print(f"Aucun fichier .nii/.nii.gz trouvé dans {input_dir_p}")
        return

    print(f"{len(files)} fichier(s) trouvé(s) dans {input_dir_p}\n")

    transforms = build_transforms()  # construit une seule fois, réutilisé pour chaque fichier

    n_ok, n_fail = 0, 0
    for f in files:
        # gère proprement le double suffixe .nii.gz
        base_name = f.name[:-7] if f.name.endswith(".nii.gz") else f.stem
        out_path = output_dir_p / f"{base_name}{suffix}.nii.gz"

        print(f"[{f.name}]")
        try:
            preprocess_volume(str(f), str(out_path), transforms)
            n_ok += 1
        except Exception as e:
            print(f"  !! ERREUR : {e}")
            n_fail += 1

    print(f"\nTerminé : {n_ok} réussi(s), {n_fail} échec(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess T1 MRI (dossier entier) : crop/pad 167x212x160 + normalisation [0,1]"
    )
    parser.add_argument("input_dir", type=str, help="Dossier contenant les fichiers NIfTI d'entrée (.nii/.nii.gz)")
    parser.add_argument("output_dir", type=str, help="Dossier de sortie pour les fichiers prétraités")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_preprocessed",
        help="Suffixe ajouté au nom de fichier de sortie (défaut: _preprocessed)",
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.suffix)