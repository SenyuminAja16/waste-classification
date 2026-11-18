"""Pemuatan fitur dengan cache (dipakai train_ml.py dan analyze_ml.py)."""
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, ROOT
from src.features import extract_features
from src.pcd import load_image, resize_43, run_pcd


def build_features(df: pd.DataFrame, tag: str, refresh: bool = False) -> np.ndarray:
    """Matriks fitur (n_foto x n_fitur). tag: 'pcd' (resize→PCD→fitur) atau 'raw' (resize→fitur).
    Hasil di-cache di data/processed/. Pakai refresh=True kalau PCD / fitur diubah."""
    cache = PROCESSED_DIR / f"features_{tag}.npz"
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, paths=paths)
    return X
