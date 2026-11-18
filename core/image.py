"""Pembacaan dan resize citra yang dipakai semua skenario."""
import cv2
import numpy as np
from PIL import Image, ImageOps

from core.config import SIZE_LANDSCAPE


def load_image(file) -> np.ndarray:
    """Baca path atau file-like menjadi array BGR dengan orientasi EXIF diterapkan."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_43(img: np.ndarray) -> np.ndarray:
    """Resize ke 320x240, atau 240x320 untuk citra potret."""
    h, w = img.shape[:2]
    size = SIZE_LANDSCAPE if w >= h else (SIZE_LANDSCAPE[1], SIZE_LANDSCAPE[0])
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
