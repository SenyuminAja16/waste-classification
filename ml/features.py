"""Ekstraksi fitur handcrafted.

>>> BAGIAN YANG DIISI SENDIRI (Fitur) <<<

Kontrak:

    extract_features(img_bgr, mask=None) -> np.ndarray 1D (float), panjangnya SELALU sama
        img_bgr : citra BGR (skenario PCD: citra `segmented`; skenario Raw: citra hasil resize)
        mask    : uint8 0/255 dari PCD, atau None (skenario Raw: seluruh citra dianggap objek)

Opsional, untuk ablation dan plot di ml/scripts/analyze_ml.py (kalau tidak didefinisikan, bagian itu dilewati):

    FEATURE_GROUPS = {"warna": slice(0, 32), "tekstur": slice(32, 47), "bentuk": slice(47, 61)}
    COLOR_BINS = (16, 8, 8)     # panjang histogram H, S, V di dalam grup "warna"

Rencana di proposal: warna (histogram HSV), tekstur (GLCM + LBP), bentuk (kontur + Hu Moments, dll.).
Contoh implementasi yang jalan ada di klasifikasi_sampah.ipynb (bagian 4).

Setelah isi file ini diubah, latih ulang: python -m ml.scripts.train_ml --refresh
"""
import numpy as np
import pandas as pd

from core.config import FEATURES_CACHE_DIR, ROOT
from core.image import load_image, resize_43
from pcd.pipeline import run_pcd


def extract_features(img_bgr, mask=None):
    raise NotImplementedError("Isi extract_features() di ml/features.py")


def build_features(df: pd.DataFrame, tag: str, refresh: bool = False) -> np.ndarray:
    """Matriks fitur (n_foto x n_fitur). tag 'pcd': resize → PCD → fitur. tag 'raw': resize → fitur.
    Di-cache di cache/features/. Pakai refresh=True kalau PCD / fitur diubah."""
    cache = FEATURES_CACHE_DIR / f"features_{tag}.npz"
    paths = df.path.to_numpy()
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(z["paths"], paths):
            print(f"[{tag}] fitur dari cache: {cache.name}")
            return z["X"]
    feats = []
    for i, p in enumerate(paths, 1):
        img = resize_43(load_image(ROOT / p))
        if tag == "pcd":
            s = run_pcd(img)
            feats.append(extract_features(s["segmented"], s["mask"]))
        else:
            feats.append(extract_features(img))
        if i % 50 == 0 or i == len(paths):
            print(f"[{tag}] ekstraksi fitur {i}/{len(paths)}", end="\r")
    print()
    X = np.stack(feats)
    FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, paths=paths)
    return X
