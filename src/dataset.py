"""Membaca data dari Google Sheet + Drive, dan ekstraksi fitur dengan cache (sudah terisi)."""
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

from config import FOTO_DIR, PROCESSED_DIR, ROOT, CLASSES

# Kolom sheet dibaca berdasarkan POSISI (ada nama kolom ganda: Sub-tipe x3, Masukkan Foto x2).
KOLOM = ["timestamp", "id_atas", "id_miring", "anggota", "sesi", "tanggal", "kelas", "sub1", "sub2", "sub3",
         "lokasi", "latar", "cahaya", "kondisi", "catatan", "perspektif", "url_atas", "url_miring"]


def baca_log(src) -> pd.DataFrame:
    d = pd.read_csv(src, dtype=str).fillna("")
    if d.shape[1] < len(KOLOM):
        sys.exit(f"Sheet punya {d.shape[1]} kolom, harusnya minimal {len(KOLOM)}. "
                 "Cek apakah ini sheet 'Form Responses 1'.")
    d = d.iloc[:, :len(KOLOM)].copy()
    d.columns = KOLOM
    return d


def file_id(url: str):
    m = re.search(r"[?&]id=([\w-]+)|/d/([\w-]+)", url)
    return (m.group(1) or m.group(2)) if m else None


def tabel_foto(sumber) -> pd.DataFrame:
    """Satu baris sheet (satu objek) -> dua baris (foto atas dan miring)."""
    log = pd.concat([baca_log(s) for s in sumber], ignore_index=True)
    rows = []
    for _, r in log.iterrows():
        obj = re.sub(r"-(atas|miring)$", "", r.id_atas.strip(), flags=re.I)
        for posisi, url in (("atas", r.url_atas), ("miring", r.url_miring)):
            fid = file_id(url)
            if fid:
                rows.append(dict(file_id=fid, id_objek=obj, posisi=posisi, kelas=r.kelas.strip().lower(),
                                 subtipe=next((x for x in (r.sub1, r.sub2, r.sub3) if x.strip()), ""),
                                 anggota=r.anggota.strip(), sesi=r.sesi.strip(), lokasi=r.lokasi.strip(),
                                 latar=r.latar.strip(), cahaya=r.cahaya.split(" (")[0].strip(),
                                 kondisi=r.kondisi.strip()))
    df = pd.DataFrame(rows).drop_duplicates("file_id").reset_index(drop=True)
    bad = ~df.kelas.isin(CLASSES)
    if bad.any():
        print(f"PERINGATAN: {bad.sum()} foto dibuang, kelas tidak dikenal: {sorted(set(df[bad].kelas))}")
        df = df[~bad].reset_index(drop=True)
    cnt = df.groupby("id_objek").size()
    if (cnt != 2).any():
        print("PERINGATAN: objek dengan jumlah foto tidak 2 ->", cnt[cnt != 2].to_dict())
    return df


def unduh(fid: str):
    """Unduh satu foto dari Drive ke cache/foto/<id>.jpg. None bila gagal."""
    p = FOTO_DIR / f"{fid}.jpg"
    if not p.exists():
        try:
            import gdown
            FOTO_DIR.mkdir(parents=True, exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={fid}", str(p), quiet=True)
        except Exception:
            return None
    try:
        Image.open(p).verify()
        return p
    except Exception:
        p.unlink(missing_ok=True)  # biasanya halaman "akses ditolak", bukan gambar
        return None


def muat_tabel_foto(sumber) -> pd.DataFrame:
    """Tabel foto lengkap dengan kolom `path` (relatif terhadap ROOT). Foto yang gagal diunduh dibuang."""
    df = tabel_foto(sumber)
    print(f"Mengunduh/memeriksa {len(df)} foto...")
    paths = [unduh(f) for f in df.file_id]
    gagal = sum(p is None for p in paths)
    if gagal:
        print(f"PERINGATAN: {gagal} foto gagal diunduh (cek sharing folder Drive 'Anyone with the link').")
    df["path"] = [p.relative_to(ROOT).as_posix() if p else None for p in paths]
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    if df.empty:
        sys.exit("Tidak ada foto yang bisa dibaca. Lihat petunjuk sharing di README.")
    return df


def build_features(df: pd.DataFrame, tag: str, refresh: bool = False) -> np.ndarray:
    """Matriks fitur (n_foto x n_fitur). tag 'pcd': resize → PCD → fitur. tag 'raw': resize → fitur.
    Di-cache di data/processed/. Pakai refresh=True kalau PCD / fitur diubah."""
    from src.features import extract_features
    from src.pcd import run_pcd
    from src.utils import load_image, resize_43

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
