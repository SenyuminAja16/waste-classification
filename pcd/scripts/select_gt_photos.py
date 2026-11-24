"""Pilih foto train yang akan diberi mask acuan untuk evaluasi segmentasi."""

import matplotlib.pyplot as plt
import pandas as pd

from core.config import CLASSES, DATA_DIR, METADATA_CSV, REPORTS_DIR, ROOT, SEED
from core.image import load_image, resize_43

GT_DIR = ROOT / "pcd" / "ground_truth"
SELECTION_CSV = GT_DIR / "selection.csv"
PER_CLASS = 10
EXPLORATION_SAMPLE = [
    "PL-J-003", "PL-F-004", "PL-M-003",
    "KT-J-002", "KT-M-003", "KT-F-002",
    "KD-F-007", "KD-J-003", "KD-M-004",
]
INCLUDE = {"PL-F-010": "miring", "KT-F-006": "miring", "KD-F-012": "miring"}


def pick_objects(train: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ambil objek secara acak dari satu kelas. Objek di INCLUDE dimasukkan menggantikan pilihan terakhir."""
    objects = train[train["label"] == label].drop_duplicates("object_id").sample(frac=1, random_state=SEED)
    forced = objects[objects["object_id"].isin(INCLUDE)].assign(view=lambda df: df["object_id"].map(INCLUDE))
    others = objects[~objects["object_id"].isin(INCLUDE)]
    others = others.head(PER_CLASS - len(forced)).assign(
        view=lambda df: ["atas" if number % 2 == 0 else "miring" for number in range(len(df))]
    )
    return pd.concat([others, forced])[["object_id", "view"]]


def save_preview(selection: pd.DataFrame) -> None:
    """Simpan gambar semua foto terpilih, tiap kelas satu baris."""
    fig, axes = plt.subplots(len(CLASSES), PER_CLASS, figsize=(2 * PER_CLASS, 2 * len(CLASSES)))
    for row, label in zip(axes, CLASSES):
        for ax, photo in zip(row, selection[selection["label"] == label].itertuples()):
            ax.imshow(resize_43(load_image(DATA_DIR / photo.raw_path))[..., ::-1])
            ax.set_title(f"{photo.object_id} {photo.view}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pcd" / "gt_selection.png", dpi=80)
    plt.close(fig)


def main() -> None:
    if SELECTION_CSV.exists():
        raise SystemExit(
            f"{SELECTION_CSV} sudah ada dan mask acuan dibuat dari daftar ini. "
            "Hapus file itu dulu kalau memang mau memilih ulang."
        )
    photos = pd.read_csv(METADATA_CSV)
    train = photos[(photos["split"] == "train") & ~photos["object_id"].isin(EXPLORATION_SAMPLE)]
    picked = pd.concat([pick_objects(train, label) for label in CLASSES])
    selection = picked.merge(photos, on=["object_id", "view"])[
        ["object_id", "view", "label", "subtype", "lighting", "condition", "raw_path"]
    ]

    GT_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(SELECTION_CSV, index=False)
    save_preview(selection)

    print(f"{len(selection)} foto tersimpan di {SELECTION_CSV}")
    for column in ("label", "view", "lighting", "condition"):
        print(selection[column].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
