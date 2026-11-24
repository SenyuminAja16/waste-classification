"""Cari setelan segmentasi terbaik berdasarkan mask acuan."""

from itertools import product

import cv2
import pandas as pd

from core.config import DATA_DIR
from core.image import load_image, resize_43
from pcd import processing
from pcd.evaluate import dice, iou
from pcd.scripts.annotate_gt import mask_path
from pcd.scripts.evaluate_segmentation import OUTPUT_DIR, otsu_grabcut, summarize
from pcd.scripts.select_gt_photos import SELECTION_CSV

INPUTS = {
    "clahe": processing.clahe,
    "asli": lambda img: img,
    "clahe_blur": lambda img: processing.blur(processing.clahe(img)),
}
MARGINS = (0.03, 0.05, 0.10)
ITERATIONS = (5, 10)
FILL_HOLES = (True, False)
BASELINE = "grabcut clahe m0.05 it5 isi"
OTSU_CENTER = "otsu_center + grabcut"


def config_name(source: str, margin: float, iterations: int, fill_holes: bool) -> str:
    return f"grabcut {source} m{margin:.2f} it{iterations} {'isi' if fill_holes else 'tanpa_isi'}"


def predictions(img):
    """Jalankan semua setelan pada satu foto."""
    for source, margin, iterations in product(INPUTS, MARGINS, ITERATIONS):
        raw = processing.grabcut_mask(INPUTS[source](img), margin, iterations)
        for fill_holes in FILL_HOLES:
            yield config_name(source, margin, iterations, fill_holes), processing.clean_mask(raw, fill_holes=fill_holes)
    yield OTSU_CENTER, otsu_grabcut(img, polarity="center")


def main() -> None:
    selection = pd.read_csv(SELECTION_CSV)
    rows = []
    for number, photo in enumerate(selection.itertuples(), 1):
        image = resize_43(load_image(DATA_DIR / photo.raw_path))
        truth = cv2.imread(str(mask_path(photo.object_id, photo.view)), cv2.IMREAD_GRAYSCALE)
        for method, pred in predictions(image):
            rows.append({
                "object_id": photo.object_id, "view": photo.view, "label": photo.label, "method": method,
                "iou": iou(pred, truth), "dice": dice(pred, truth),
            })
        print(f"Tuning {number}/{len(selection)}", end="\r")
    print()

    scores = pd.DataFrame(rows)
    summary = summarize(scores, BASELINE).sort_values("IoU rata-rata", ascending=False)
    summary.round(3).to_csv(OUTPUT_DIR / "segmentation_tuning.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
