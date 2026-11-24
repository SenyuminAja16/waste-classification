"""Gabungkan semua mask acuan dalam satu gambar untuk dicek."""

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from core.config import DATA_DIR, REPORTS_DIR
from core.image import load_image, resize_43
from pcd.scripts.annotate_gt import mask_path
from pcd.scripts.run_pcd import outline
from pcd.scripts.select_gt_photos import SELECTION_CSV

COLUMNS = 6
OUTPUT = REPORTS_DIR / "pcd" / "gt_review.png"


def main() -> None:
    selection = pd.read_csv(SELECTION_CSV)
    done = [photo for photo in selection.itertuples() if mask_path(photo.object_id, photo.view).exists()]
    if not done:
        print("Belum ada mask acuan.")
        return

    rows = -(-len(done) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(3 * COLUMNS, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, photo in zip(axes.ravel(), done):
        image = resize_43(load_image(DATA_DIR / photo.raw_path))
        mask = cv2.imread(str(mask_path(photo.object_id, photo.view)), cv2.IMREAD_GRAYSCALE)
        ax.imshow(outline(image, mask))
        ax.set_title(f"{photo.object_id} {photo.view}, {(mask > 0).mean():.0%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=90)
    plt.close(fig)
    print(f"{len(done)} dari {len(selection)} mask, lembar tersimpan di {OUTPUT}")


if __name__ == "__main__":
    main()
