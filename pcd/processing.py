"""Operasi dasar PCD: enhancement, segmentasi, morfologi, dan masking."""

import cv2
import numpy as np

from core.config import SEED

CHANNELS = ("gray", "saturation")
POLARITIES = ("border", "center")


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Histogram equalization global pada kanal L, sebagai pembanding CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((cv2.equalizeHist(lightness), a, b)), cv2.COLOR_LAB2BGR)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Tingkatkan kontras citra menggunakan CLAHE pada kanal L."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Terapkan koreksi gamma menggunakan lookup table."""
    table = ((np.arange(256) / 255.0) ** gamma * 255).round().astype(np.uint8)
    return cv2.LUT(img, table)


def blur(img: np.ndarray, size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (size, size), 0)


def otsu_mask(img: np.ndarray, channel: str = "gray", polarity: str = "border") -> np.ndarray:
    """Buat mask menggunakan threshold Otsu pada kanal grayscale atau saturation.

    Polarity menentukan orientasi mask berdasarkan area tepi atau tengah citra.
    """
    if channel == "gray":
        values = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif channel == "saturation":
        values = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    else:
        raise ValueError(f"channel harus salah satu dari {CHANNELS}")
    _, mask = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if polarity == "border":
        return background_at_border(mask)
    if polarity == "center":
        return object_at_center(mask)
    raise ValueError(f"polarity harus salah satu dari {POLARITIES}")


def background_at_border(mask: np.ndarray) -> np.ndarray:
    """Sesuaikan mask berdasarkan piksel pada tepi citra."""
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return cv2.bitwise_not(mask) if border.mean() > 127 else mask


def object_at_center(mask: np.ndarray) -> np.ndarray:
    """Sesuaikan mask berdasarkan dominasi piksel foreground di tengah citra."""
    height, width = mask.shape
    center = mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return cv2.bitwise_not(mask) if center.mean() < mask.mean() else mask


def grabcut_mask(img: np.ndarray, margin: float = 0.05, iterations: int = 5) -> np.ndarray:
    """Segmentasikan objek menggunakan GrabCut dengan bounding box awal."""
    height, width = img.shape[:2]
    dx, dy = int(width * margin), int(height * margin)
    rect = (dx, dy, width - 2 * dx, height - 2 * dy)
    labels = np.zeros((height, width), np.uint8)
    return run_grabcut(img, labels, rect, cv2.GC_INIT_WITH_RECT, iterations)


def grabcut_from_mask(img: np.ndarray, seed: np.ndarray, border: float = 0.03, iterations: int = 5) -> np.ndarray:
    """Perbaiki mask awal menggunakan GrabCut.

    Area di sepanjang tepi citra ditetapkan sebagai background.
    """
    height, width = img.shape[:2]
    labels = np.where(seed > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    band = max(int(min(height, width) * border), 1)
    labels[:band] = labels[-band:] = cv2.GC_BGD
    labels[:, :band] = labels[:, -band:] = cv2.GC_BGD
    return run_grabcut(img, labels, None, cv2.GC_INIT_WITH_MASK, iterations)


def run_grabcut(img: np.ndarray, labels: np.ndarray, rect, mode: int, iterations: int) -> np.ndarray:
    cv2.setRNGSeed(SEED)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, labels, rect, background_model, foreground_model, iterations, mode)
    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    return np.where(foreground, 255, 0).astype(np.uint8)


def plausible_mask(mask: np.ndarray, min_area: float = 0.03, max_area: float = 0.8, max_border: float = 0.3) -> bool:
    """Validasi mask berdasarkan luas area dan proporsi piksel di tepi citra."""
    area = (mask > 0).mean()
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return min_area <= area <= max_area and (border > 0).mean() <= max_border


def clean_mask(mask: np.ndarray, size: int = 5, min_share: float = 0.1, fill_holes: bool = True) -> np.ndarray:
    """Bersihkan mask melalui operasi morfologi dan seleksi contour.

    Contour dipertahankan berdasarkan luas relatif terhadap contour terbesar.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    if contours:
        largest = max(cv2.contourArea(contour) for contour in contours)
        kept = [contour for contour in contours if cv2.contourArea(contour) >= min_share * largest]
        cv2.drawContours(cleaned, kept, -1, 255, thickness=cv2.FILLED)
    return cleaned if fill_holes else cv2.bitwise_and(cleaned, mask)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask)


def crop_roi(img: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop area objek berdasarkan mask dengan tambahan padding.

    Jika mask kosong, citra asli dikembalikan.
    """
    points = cv2.findNonZero(mask)
    if points is None:
        return img
    x, y, width, height = cv2.boundingRect(points)
    top, left = max(y - pad, 0), max(x - pad, 0)
    return img[top : y + height + pad, left : x + width + pad]


def pad_square(img: np.ndarray) -> np.ndarray:
    """Tambahkan padding untuk mempertahankan rasio aspek citra."""
    height, width = img.shape[:2]
    size = max(height, width)
    top, left = (size - height) // 2, (size - width) // 2
    return cv2.copyMakeBorder(img, top, size - height - top, left, size - width - left, cv2.BORDER_CONSTANT, value=0)
