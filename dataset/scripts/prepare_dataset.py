"""Unduh foto, validasi log akuisisi, dan bagi dataset per objek menjadi metadata siap pakai."""

import argparse

import pandas as pd

from core.config import DATA_DIR, LOG_SOURCES, METADATA_CSV, OBJECT_SPLITS_CSV, ROOT, VALIDATION_CSV
from dataset.acquisition import download_photos, read_log, to_photo_table
from dataset.metadata import validate
from dataset.split import SPLIT_COLUMNS, assign_splits

METADATA_COLUMNS = [
    "object_id", "view", "label", "subtype", "member", "taken_on", "location", "lighting",
    "condition", "note", "split", "raw_path", "file_id", "download_status", "submitted_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=LOG_SOURCES, help="URL Sheet atau path CSV log akuisisi")
    parser.add_argument("--skip-download", action="store_true", help="pakai foto yang sudah ada tanpa mengunduh")
    parser.add_argument("--workers", type=int, default=8, help="jumlah unduhan paralel")
    return parser.parse_args()


def read_existing_splits() -> pd.DataFrame:
    if OBJECT_SPLITS_CSV.exists():
        return pd.read_csv(OBJECT_SPLITS_CSV, dtype=str)
    return pd.DataFrame(columns=SPLIT_COLUMNS)


def main() -> None:
    args = parse_args()
    if not args.source:
        raise SystemExit("WASTE_SHEET_ID belum diatur (lihat .env.example)")

    log = pd.concat([read_log(source) for source in args.source], ignore_index=True)
    photos = to_photo_table(log)
    if args.skip_download:
        exists = photos["raw_path"].map(lambda path: (DATA_DIR / path).exists())
        photos = photos.assign(download_status=exists.map({True: "exists", False: "missing"}))
    else:
        photos = download_photos(photos, workers=args.workers)

    issues = validate(log, photos)
    problem_ids = set(issues["object_id"])
    objects = photos.loc[~photos["object_id"].isin(problem_ids), ["object_id", "label"]].drop_duplicates()

    # Objek yang pernah dibagi tetap disimpan walaupun sedang bermasalah, supaya split-nya tidak berubah.
    existing = read_existing_splits()
    splits = assign_splits(objects, existing)
    history = existing[~existing["object_id"].isin(splits["object_id"])]
    metadata = photos.merge(splits[["object_id", "split"]], on="object_id", how="left")[METADATA_COLUMNS]

    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([splits, history]).sort_values("object_id").to_csv(OBJECT_SPLITS_CSV, index=False)
    metadata.to_csv(METADATA_CSV, index=False)
    issues.to_csv(VALIDATION_CSV, index=False)

    print(f"isian: {len(log)} | objek: {photos['object_id'].nunique()} | foto: {len(photos)}")
    print("unduhan:", photos["download_status"].value_counts().to_dict())
    print(f"validasi: {len(issues)} masalah, lihat {VALIDATION_CSV.relative_to(ROOT).as_posix()}")
    if problem_ids:
        print("objek tidak dimasukkan ke split:", sorted(problem_ids))
    print("\nsplit per kelas (objek):")
    print(pd.crosstab(splits["label"], splits["split"], margins=True, margins_name="total"))
    print(f"\ndisimpan: {METADATA_CSV.relative_to(ROOT).as_posix()}, {OBJECT_SPLITS_CSV.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
