"""Pipeline PCD untuk ML, DL, dan demo.

run_pcd menerima citra hasil core.image.resize_43 dengan output:
    "resized"   : citra input (BGR)
    "enhanced"  : hasil CLAHE (BGR)
    "mask"      : mask biner uint8 (objek = 255)
    "segmented" : citra enhanced dengan background hitam (BGR)
    "roi"       : crop objek berbentuk persegi (BGR)

Segmentasi dilakukan pada citra asli menggunakan GrabCut, sedangkan CLAHE
diterapkan pada citra hasil. Parameter GrabCut dituning menggunakan 30 citra
train yang sudah dianotasi dengan hasil IoU: 0,885.
"""

import numpy as np

from pcd import processing

GRABCUT_MARGIN = 0.10
GRABCUT_RETRY_MARGIN = 0.05
GRABCUT_ITERATIONS = 10
ROI_PADDING = 10


def run_pcd(img_bgr: np.ndarray) -> dict:
    mask = segment(img_bgr)
    enhanced = processing.clahe(img_bgr)
    segmented = processing.apply_mask(enhanced, mask)
    return {
        "resized": img_bgr,
        "enhanced": enhanced,
        "mask": mask,
        "segmented": segmented,
        "roi": processing.pad_square(processing.crop_roi(segmented, mask, ROI_PADDING)),
    }


def segment(img: np.ndarray) -> np.ndarray:
    """Segmentasikan objek menggunakan GrabCut dan operasi morfologi.

    Jika mask kosong, GrabCut diulang dengan margin lebih kecil.
    Jika masih kosong, seluruh citra digunakan sebagai foreground.
    """
    for margin in (GRABCUT_MARGIN, GRABCUT_RETRY_MARGIN):
        mask = processing.clean_mask(processing.grabcut_mask(img, margin, GRABCUT_ITERATIONS))
        if mask.any():
            return mask
    return np.full(img.shape[:2], 255, np.uint8)
