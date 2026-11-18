"""Utilitas dasar (sudah terisi, tidak perlu diubah)."""
import cv2
import numpy as np
from PIL import Image, ImageOps

from core.config import SIZE_LANDSCAPE


def load_image(file) -> np.ndarray:
    """Baca file (path / file-like) jadi array BGR, dengan koreksi orientasi EXIF."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_43(img: np.ndarray) -> np.ndarray:
    """Resize ke ukuran seragam rasio 4:3 (landscape 320x240, portrait 240x320)."""
    h, w = img.shape[:2]
    size = SIZE_LANDSCAPE if w >= h else (SIZE_LANDSCAPE[1], SIZE_LANDSCAPE[0])
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
