"""Baca sheet log_akuisisi, unduh foto dari Drive, dan buat data/splits/split.csv (dibagi PER OBJEK).

Pemakaian:
    python -m dataset.scripts.prepare_dataset                       # baca Google Sheet (lihat config.SUMBER_LOG)
    python -m dataset.scripts.prepare_dataset --sumber log.csv      # atau dari CSV hasil download sheet
    python -m dataset.scripts.prepare_dataset --sumber a.csv b.csv  # gabungan beberapa sumber

Pembagian: train / val / test = 70 / 15 / 15 per objek, seimbang per kelas.
Kalau ada kelas dengan objek < config.MIN_OBJEK_SPLIT, semua data jadi 'train'
(train_ml.py lalu hanya memakai cross-validation).
"""
import argparse
import math
import random
import sys
from pathlib import Path

from core.config import CLASSES, MIN_OBJEK_SPLIT, ROOT, SEED, SPLIT_CSV, SUMBER_LOG, TEST_FRAC, VAL_FRAC
from dataset.acquisition import muat_tabel_foto


def bagi(ids, rng):
    ids = sorted(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val, n_test = math.floor(VAL_FRAC * n), math.ceil(TEST_FRAC * n)
    return {o: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, o in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="*", default=None, help="URL atau path CSV sheet (default: config.SUMBER_LOG)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=SPLIT_CSV)
    a = ap.parse_args()

    df = muat_tabel_foto(a.sumber or SUMBER_LOG)
    n_obj = df.groupby("kelas").id_objek.nunique().reindex(CLASSES).fillna(0).astype(int)
    if n_obj.min() >= MIN_OBJEK_SPLIT:
        rng = random.Random(a.seed)
        sub = {}
        for k in CLASSES:
            sub.update(bagi(df.loc[df.kelas == k, "id_objek"].unique().tolist(), rng))
        df["subset"] = df["id_objek"].map(sub)
    else:
        df["subset"] = "train"
        print(f"\nCATATAN: objek per kelas baru {n_obj.to_dict()} (minimal {MIN_OBJEK_SPLIT} per kelas untuk split). "
              "Semua data dijadikan 'train'; training hanya memakai cross-validation.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nDisimpan: {a.out.relative_to(ROOT)}  ({len(df)} foto, {df.id_objek.nunique()} objek)")
    print("\nObjek per kelas dan subset:")
    print(df.groupby(["kelas", "subset"]).id_objek.nunique().unstack(fill_value=0).reindex(CLASSES).to_string())
    print("\nObjek per anggota x kelas:")
    print(df.groupby(["anggota", "kelas"]).id_objek.nunique().unstack(fill_value=0).reindex(columns=CLASSES).to_string())
    for kol in ("latar", "lokasi", "cahaya", "kondisi"):
        print(f"\n{kol}:", df.drop_duplicates("id_objek")[kol].value_counts().to_dict())


if __name__ == "__main__":
    main()
