"""Bandingkan metode segmentasi dengan mask acuan memakai IoU dan Dice."""

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from core.config import DATA_DIR, REPORTS_DIR
from core.image import load_image, resize_43
from pcd import processing
from pcd.evaluate import dice, iou
from pcd.pipeline import run_pcd
from pcd.scripts.annotate_gt import mask_path
from pcd.scripts.select_gt_photos import SELECTION_CSV

OUTPUT_DIR = REPORTS_DIR / "pcd"
FAILED_IOU = 0.5
REFERENCE = "pipeline"

METHODS = {
    "otsu_gray": lambda img: processing.clean_mask(processing.otsu_mask(processing.blur(processing.clahe(img)), "gray")),
    "otsu_saturation": lambda img: processing.clean_mask(
        processing.otsu_mask(processing.blur(processing.clahe(img)), "saturation")
    ),
    "grabcut_rect": lambda img: processing.clean_mask(processing.grabcut_mask(processing.clahe(img))),
    "pipeline": lambda img: run_pcd(img)["mask"],
}


def score_photos(selection: pd.DataFrame) -> pd.DataFrame:
    """Hitung IoU dan Dice setiap foto untuk setiap metode."""
    rows = []
    for number, photo in enumerate(selection.itertuples(), 1):
        image = resize_43(load_image(DATA_DIR / photo.raw_path))
        truth = cv2.imread(str(mask_path(photo.object_id, photo.view)), cv2.IMREAD_GRAYSCALE)
        for method, segment in METHODS.items():
            pred = segment(image)
            rows.append({
                "object_id": photo.object_id, "view": photo.view, "label": photo.label, "method": method,
                "iou": iou(pred, truth), "dice": dice(pred, truth),
            })
        print(f"Evaluasi {number}/{len(selection)}", end="\r")
    print()
    return pd.DataFrame(rows)


def confidence_interval(values: pd.Series) -> float:
    """Hitung nilai ± untuk selang kepercayaan 95% dari rata-rata."""
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def summarize(scores: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Hitung skor rata-rata tiap metode dan bandingkan per foto dengan metode acuan."""
    baseline = scores[scores["method"] == reference].set_index(["object_id", "view"])["iou"]
    rows = []
    for method, group in scores.groupby("method", sort=False):
        difference = group.set_index(["object_id", "view"])["iou"] - baseline
        rows.append({
            "metode": method,
            "IoU rata-rata": group["iou"].mean(),
            "IoU ±95%": confidence_interval(group["iou"]),
            "IoU median": group["iou"].median(),
            "Dice rata-rata": group["dice"].mean(),
            f"gagal (IoU<{FAILED_IOU})": int((group["iou"] < FAILED_IOU).sum()),
            f"lebih baik dari {reference}": int((difference > 0.01).sum()),
            f"lebih buruk dari {reference}": int((difference < -0.01).sum()),
        })
    return pd.DataFrame(rows).set_index("metode")


def main() -> None:
    selection = pd.read_csv(SELECTION_CSV)
    missing = [f"{p.object_id} {p.view}" for p in selection.itertuples() if not mask_path(p.object_id, p.view).exists()]
    if missing:
        raise SystemExit(f"Mask acuan belum lengkap: {', '.join(missing)}")

    scores = score_photos(selection)
    summary = summarize(scores)
    per_class = scores.pivot_table(index="method", columns="label", values="iou", sort=False)

    scores.round(4).to_csv(OUTPUT_DIR / "segmentation_scores.csv", index=False)
    summary.round(3).to_csv(OUTPUT_DIR / "segmentation_summary.csv")
    print(summary.round(3).to_string(), "\n")
    print("IoU rata-rata per kelas")
    print(per_class.round(3).to_string(), "\n")
    print(f"Foto dengan IoU pipeline < {FAILED_IOU}")
    failed = scores[(scores["method"] == REFERENCE) & (scores["iou"] < FAILED_IOU)]
    print(failed[["object_id", "view", "label", "iou"]].round(3).to_string(index=False) if not failed.empty else "tidak ada")


if __name__ == "__main__":
    main()
