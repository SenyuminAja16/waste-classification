"""Baca data/raw/, validasi, dan buat data/splits/split.csv (train/val/test dibagi PER OBJEK).

Struktur yang diharapkan (sesuai panduan pengambilan data):
    data/raw/plastik/PL-F-001/IMG_xxx.jpg   (tepat 2 foto: atas dulu, lalu miring)
    data/raw/kertas/KT-M-003/...
    data/raw/kardus/KD-J-010/...
data/raw/hardset/ diabaikan (tidak dipakai untuk training).

Urutan foto ditentukan dari waktu EXIF (fallback: nama file): yang paling awal = atas, berikutnya = miring.

Pemakaian:  python scripts/prepare_dataset.py
"""
import argparse
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from config import CLASS_CODE, CLASSES, RAW_DIR, ROOT, SEED, SPLIT_CSV, TEST_FRAC, VAL_FRAC

IMG_EXT = {".jpg", ".jpeg", ".png"}
ID_RE = re.compile(r"^(PL|KT|KD)-(M|F|J)-(\d{3})$")


def exif_time(p: Path) -> str:
    try:
        ex = Image.open(p).getexif()
        sub = ex.get_ifd(0x8769)  # Exif IFD
        t = sub.get(36867) or ex.get(306)  # DateTimeOriginal / DateTime
        ss = sub.get(37521) or ""           # SubSecTimeOriginal
        if t:
            return f"{t}.{ss}"
    except Exception:
        pass
    return ""


def split_ids(ids: list[str], rng: random.Random) -> dict[str, str]:
    ids = sorted(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val, n_test = math.floor(VAL_FRAC * n), math.ceil(TEST_FRAC * n)
    out = {}
    for i, k in enumerate(ids):
        out[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=SPLIT_CSV)
    args = ap.parse_args()

    rows, problems = [], []
    for kelas in CLASSES:
        for d in sorted((RAW_DIR / kelas).glob("*")):
            if not d.is_dir():
                continue
            m = ID_RE.match(d.name)
            if not m or m.group(1) != CLASS_CODE[kelas]:
                problems.append(f"{d.relative_to(ROOT)}: nama folder bukan ID objek yang valid untuk kelas {kelas}")
                continue
            imgs = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
            if len(imgs) != 2:
                problems.append(f"{d.relative_to(ROOT)}: berisi {len(imgs)} foto (harus tepat 2)")
                continue
            imgs.sort(key=lambda p: (exif_time(p), p.name))
            for pos, p in zip(["atas", "miring"], imgs):
                rows.append({"path": p.relative_to(ROOT).as_posix(), "kelas": kelas, "id_objek": d.name,
                             "anggota": m.group(2), "posisi": pos,
                             "nama_standar": f"{d.name}_{pos}{p.suffix.lower()}"})

    if problems:
        print("PERHATIAN (folder berikut dilewati):")
        for p in problems:
            print("  -", p)
    if not rows:
        sys.exit("Belum ada data valid di data/raw/. Isi dulu folder plastik/kertas/kardus.")

    df = pd.DataFrame(rows)
    rng = random.Random(args.seed)
    subset = {}
    for kelas in CLASSES:
        subset.update(split_ids(df.loc[df.kelas == kelas, "id_objek"].unique().tolist(), rng))
    df["subset"] = df["id_objek"].map(subset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nDisimpan: {args.out.relative_to(ROOT)}  ({len(df)} foto, {df.id_objek.nunique()} objek)")
    tab = df.groupby(["kelas", "subset"]).id_objek.nunique().unstack(fill_value=0)
    tab = tab.reindex(CLASSES)[["train", "val", "test"]] if set(["train", "val", "test"]) <= set(tab.columns) else tab
    print("\nJumlah OBJEK per kelas dan subset (foto = 2 x objek):")
    print(tab.to_string())
    per_member = df.groupby(["anggota", "kelas"]).id_objek.nunique().unstack(fill_value=0)
    print("\nObjek per anggota x kelas:")
    print(per_member.reindex(columns=CLASSES).to_string())


if __name__ == "__main__":
    main()
