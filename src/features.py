"""Ekstraksi fitur handcrafted.   >>> TITIK GANTI untuk fitur (Farhan) <<<

Kontrak:
    extract_features(img_bgr, mask=None) -> np.ndarray 1D float32, panjang SELALU sama
        mask=None berarti seluruh citra dianggap objek (skenario Raw).

Default: warna HSV (32) + tekstur GLCM (5) + LBP (10) + bentuk (4) + Hu Moments (7) = 58 dimensi.
Setelah mengubah file ini, latih ulang model ML (python scripts/train_ml.py --refresh).
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Peta posisi fitur di dalam vektor (dipakai scripts/analyze_ml.py untuk ablation & plot).
# Kalau extract_features diganti, perbarui dua konstanta ini (atau hapus, ablation akan dilewati).
FEATURE_GROUPS = {"warna": slice(0, 32), "tekstur": slice(32, 47), "bentuk": slice(47, 58)}
COLOR_BINS = (16, 8, 8)  # panjang histogram H, S, V di dalam grup "warna"

_GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def _norm(h: np.ndarray) -> np.ndarray:
    h = h.flatten().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def extract_features(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if mask is None:
        mask = np.full((h, w), 255, np.uint8)

    # Warna: histogram H(16) + S(8) + V(8) di dalam mask
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        _norm(cv2.calcHist([hsv], [0], mask, [16], [0, 180])),
        _norm(cv2.calcHist([hsv], [1], mask, [8], [0, 256])),
        _norm(cv2.calcHist([hsv], [2], mask, [8], [0, 256])),
    ])

    # Tekstur: pada crop bounding box objek
    x, y, bw, bh = cv2.boundingRect(mask)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[y:y + bh, x:x + bw] if bw >= 16 and bh >= 16 else gray

    g32 = (crop // 8).astype(np.uint8)
    glcm = graycomatrix(g32, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    glcm_feat = np.array([graycoprops(glcm, p).mean() for p in _GLCM_PROPS], np.float32)

    lbp = local_binary_pattern(crop, P=8, R=1, method="uniform")
    lbp_hist = _norm(np.histogram(lbp, bins=10, range=(0, 10))[0])

    # Bentuk: kontur terbesar
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        rx, ry, rw, rh = cv2.boundingRect(c)
        shape = np.array([area / (h * w), area / max(rw * rh, 1), rw / max(rh, 1), area / hull_area], np.float32)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    else:
        shape = np.zeros(4, np.float32)
        hu = np.zeros(7)

    return np.concatenate([hist, glcm_feat, lbp_hist, shape, hu.astype(np.float32)]).astype(np.float32)
