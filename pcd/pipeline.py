"""Pipeline PCD yang dipakai ML, DL, dan demo.

Kontrak run_pcd(img_bgr), dengan input yang sudah melalui core.image.resize_43:
    "resized"   : citra input apa adanya (BGR)
    "enhanced"  : citra setelah perbaikan kontras (BGR)
    "mask"      : uint8 0/255, objek = 255, ukurannya sama dengan citra
    "segmented" : citra enhanced dengan latar dihitamkan (BGR)

Isi pipeline masih sementara untuk Progress 2 dan ditetapkan di Progress 3 setelah dievaluasi dengan IoU/Dice.
"""

import numpy as np

from pcd import processing


def run_pcd(img_bgr: np.ndarray) -> dict:
    """CLAHE, segmentasi gabungan Otsu saturation dan GrabCut, morfologi, lalu masking."""
    enhanced = processing.clahe(img_bgr)
    mask = processing.clean_mask(segment(enhanced))
    return {
        "resized": img_bgr,
        "enhanced": enhanced,
        "mask": mask,
        "segmented": processing.apply_mask(enhanced, mask),
    }


def segment(img: np.ndarray) -> np.ndarray:
    """Otsu saturation sebagai tebakan awal GrabCut.

    Kalau tebakan atau hasil GrabCut-nya tidak wajar, GrabCut diulang dari kotak di tengah foto.
    """
    seed = processing.clean_mask(processing.otsu_mask(processing.blur(img), "saturation"))
    if processing.plausible_mask(seed):
        mask = processing.grabcut_from_mask(img, seed)
        if processing.plausible_mask(processing.clean_mask(mask)):
            return mask
    return processing.grabcut_mask(img)
