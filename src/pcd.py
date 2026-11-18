"""PCD (Pengolahan Citra Digital).   >>> TITIK GANTI untuk PCD (Mafaza) <<<

Kontrak yang harus dipenuhi kalau isinya diganti:

    resize_43(img_bgr)  -> img_bgr  (ukuran seragam)
    run_pcd(img_bgr)    -> dict dengan kunci:
        "resized"   : citra BGR hasil resize
        "enhanced"  : citra BGR setelah perbaikan kontras / filtering
        "mask"      : uint8 (0 atau 255), objek = 255, ukuran sama dengan citra
        "segmented" : citra BGR dengan latar dihitamkan (dipakai ML skenario 1 dan DL skenario 3)

Default sekarang: resize 4:3 → CLAHE (kanal L pada LAB) → Gaussian blur → Otsu + morfologi.
Setelah mengubah file ini, latih ulang model PCD (python scripts/train_ml.py --refresh).
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

from config import SIZE_LANDSCAPE


def load_image(file) -> np.ndarray:
    """Baca file (path / file-like) jadi array BGR, dengan koreksi orientasi EXIF."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_43(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    size = SIZE_LANDSCAPE if w >= h else (SIZE_LANDSCAPE[1], SIZE_LANDSCAPE[0])
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def enhance(img: np.ndarray) -> np.ndarray:
    """CLAHE pada kanal L (LAB), lalu Gaussian blur untuk mengurangi derau."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(out, (5, 5), 0)


def segment(enhanced: np.ndarray) -> np.ndarray:
    """Otsu → polaritas otomatis → morfologi → kontur terbesar.
    Mengembalikan mask uint8 (0/255). Bila segmentasi gagal, mask penuh (seluruh citra)."""
    h, w = enhanced.shape[:2]
    full = np.full((h, w), 255, np.uint8)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Latar biasanya menempati tepi citra: bila tepi dominan putih, balik mask.
    m = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([th[:m].ravel(), th[-m:].ravel(), th[:, :m].ravel(), th[:, -m:].ravel()])
    if (border > 0).mean() > 0.5:
        th = cv2.bitwise_not(th)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full
    c = max(cnts, key=cv2.contourArea)
    ratio = cv2.contourArea(c) / float(h * w)
    if ratio < 0.05 or ratio > 0.95:
        return full
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


def run_pcd(img_bgr: np.ndarray) -> dict:
    """Jalankan seluruh PCD. Input: citra yang sudah di-resize (BGR)."""
    enhanced = enhance(img_bgr)
    mask = segment(enhanced)
    segmented = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    return {"resized": img_bgr, "enhanced": enhanced, "mask": mask, "segmented": segmented}
