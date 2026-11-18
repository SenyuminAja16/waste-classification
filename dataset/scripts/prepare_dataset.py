"""Unduh foto, validasi log akuisisi, dan bagi dataset per objek menjadi metadata siap pakai."""

import argparse

import pandas as pd

from core.config import DATA_DIR, DATASET_SUMMARY, LOG_SOURCES, METADATA_CSV, OBJECT_SPLITS_CSV, ROOT
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
    parser.add_argument("--workers", type=int, default=4, help="jumlah unduhan paralel")
    return parser.parse_args()


def read_existing_splits() -> pd.DataFrame:
    if OBJECT_SPLITS_CSV.exists():
        return pd.read_csv(OBJECT_SPLITS_CSV, dtype=str)
    return pd.DataFrame(columns=SPLIT_COLUMNS)


def split_table(splits: pd.DataFrame) -> pd.DataFrame:
    """Jumlah objek per kelas dan subset, dengan baris dan kolom total."""
    table = pd.crosstab(splits["label"], splits["split"], margins=True, margins_name="total")
    return table.reindex(columns=["train", "val", "test", "total"], fill_value=0).reset_index()


def markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def build_summary(log: pd.DataFrame, photos: pd.DataFrame, issues: pd.DataFrame, splits: pd.DataFrame) -> str:
    """Laporan ringkas dataset dalam Markdown."""
    downloads = "\n".join(f"- {status}: {count}" for status, count in photos["download_status"].value_counts().items())
    if issues.empty:
        validation = "Tidak ada masalah."
    else:
        validation = (
            f"{len(issues)} masalah. Objek berikut tidak dimasukkan ke split sampai diperbaiki.\n\n"
            + markdown_table(issues)
        )
    return f"""# Ringkasan Dataset

Isian Form terakhir: {log["submitted_at"].max():%Y-%m-%d %H:%M}

## Jumlah

- Isian Form: {len(log)}
- Objek: {photos["object_id"].nunique()}
- Foto: {len(photos)}

## Unduhan

{downloads}

## Validasi

{validation}

## Split per kelas (objek)

{markdown_table(split_table(splits))}
"""


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

    DATASET_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([splits, history]).sort_values("object_id").to_csv(OBJECT_SPLITS_CSV, index=False)
    metadata.to_csv(METADATA_CSV, index=False)
    DATASET_SUMMARY.write_text(build_summary(log, photos, issues, splits), encoding="utf-8")

    print(f"isian: {len(log)} | objek: {photos['object_id'].nunique()} | foto: {len(photos)}")
    print("unduhan:", photos["download_status"].value_counts().to_dict())
    print(f"validasi: {len(issues)} masalah")
    if problem_ids:
        print("objek tidak dimasukkan ke split:", sorted(problem_ids))
    print("\nsplit per kelas (objek):")
    print(split_table(splits).to_string(index=False))
    saved = [METADATA_CSV, OBJECT_SPLITS_CSV, DATASET_SUMMARY]
    print("\ndisimpan:", ", ".join(path.relative_to(ROOT).as_posix() for path in saved))


if __name__ == "__main__":
    main()
