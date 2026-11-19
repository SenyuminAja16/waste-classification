"""Jalankan pipeline PCD ke semua foto, simpan hasilnya, dan buat contact sheet untuk pemeriksaan."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import CLASSES, DATA_DIR, METADATA_CSV, PROCESSED_DIR, REPORTS_DIR
from core.image import load_image, resize_43
from pcd.pipeline import run_pcd
from pcd.processing import plausible_mask

QC_DIR = REPORTS_DIR / "pcd"
SHEET_COLUMNS = 10


def output_path(raw_path: str, kind: str) -> Path:
    """Lokasi hasil, misalnya processed/masks/plastik/PL-F-001/PL-F-001-atas.png."""
    return PROCESSED_DIR / kind / Path(raw_path).relative_to("raw").with_suffix(".png")


def outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tepi mask berwarna hijau di atas citra, dalam RGB untuk ditampilkan."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.cvtColor(cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2), cv2.COLOR_BGR2RGB)


def save_contact_sheet(qc: pd.DataFrame, previews: dict, label: str) -> None:
    """Semua foto satu kelas dalam satu lembar; judul merah menandai mask yang luas atau letaknya tidak wajar."""
    subset = qc[qc["label"] == label]
    rows = -(-len(subset) // SHEET_COLUMNS)
    fig, axes = plt.subplots(rows, SHEET_COLUMNS, figsize=(1.8 * SHEET_COLUMNS, 2.2 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, photo in zip(axes.ravel(), subset.itertuples()):
        ax.imshow(previews[photo.raw_path])
        ax.set_title(f"{photo.object_id} {photo.view}", fontsize=7, color="black" if photo.plausible else "red")
    fig.tight_layout()
    fig.savefig(QC_DIR / f"pcd_qc_{label}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    photos = pd.read_csv(METADATA_CSV)
    photos = photos[photos["split"].notna()].reset_index(drop=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    areas, plausible, previews = [], [], {}
    for number, raw_path in enumerate(photos["raw_path"], 1):
        result = run_pcd(resize_43(load_image(DATA_DIR / raw_path)))
        for kind, image in (("pcd", result["segmented"]), ("masks", result["mask"])):
            target = output_path(raw_path, kind)
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), image)
        areas.append((result["mask"] > 0).mean())
        plausible.append(plausible_mask(result["mask"]))
        previews[raw_path] = outline(result["resized"], result["mask"])
        print(f"PCD {number}/{len(photos)}", end="\r")
    print()

    qc = photos.assign(mask_area=areas, plausible=plausible)
    for label in CLASSES:
        save_contact_sheet(qc, previews, label)

    flagged = qc[~qc["plausible"]]
    print(f"Hasil PCD: {PROCESSED_DIR / 'pcd'}")
    print(f"Mask:      {PROCESSED_DIR / 'masks'}")
    print(f"Contact sheet: {QC_DIR}")
    print(f"Luas mask (median): {qc['mask_area'].median():.0%}")
    print(f"Mask tidak wajar: {len(flagged)} dari {len(qc)} foto")
    if not flagged.empty:
        print(flagged[["object_id", "view", "split", "mask_area"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
