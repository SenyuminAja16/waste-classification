"""Metadata, dataset foto, augmentasi, dan DataLoader untuk Raw dan PCD."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from core.config import CLASSES, DATA_DIR
from core.image import load_image, resize_43
from dl.transforms import to_rgb224, to_tensor
from pcd.pipeline import run_pcd

PHOTO_COLUMNS = ["object_id", "raw_path", "label", "split"]
CLASS_INDEX = {label: i for i, label in enumerate(CLASSES)}


def read_photos(path: Path) -> pd.DataFrame:
    photos = pd.read_csv(path, dtype=str)
    missing = set(PHOTO_COLUMNS) - set(photos.columns)
    if missing:
        raise ValueError(f"Kolom metadata belum lengkap: {', '.join(sorted(missing))}.")
    photos = photos.loc[photos["split"].notna(), PHOTO_COLUMNS].reset_index(drop=True)
    if photos.isna().any().any() or not photos["label"].isin(CLASSES).all():
        raise ValueError("Metadata berisi ID, path atau kelas yang tidak valid.")
    if not photos["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Kolom split harus berisi train, val, atau test.")
    if photos.groupby("object_id")["split"].nunique().gt(1).any():
        raise ValueError("Foto dari objek yang sama harus berada dalam satu split.")
    return photos


class PhotoDataset(Dataset):
    def __init__(self, photos: pd.DataFrame, scenario: str, augment: bool):
        self.labels = photos["label"].map(CLASS_INDEX).tolist()
        self.images = []
        self.transform = None
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        for raw_path in photos["raw_path"]:
            img = resize_43(load_image(DATA_DIR / raw_path))
            if scenario == "pcd":
                img = run_pcd(img)["segmented"]
            self.images.append(to_rgb224(img))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb = self.images[index]
        if self.transform is not None:
            rgb = np.asarray(self.transform(Image.fromarray(rgb))).copy()
        return to_tensor(rgb), self.labels[index]


def make_loader(photos, split, scenario, batch_size):
    subset = photos[photos["split"] == split]
    if subset.empty:
        raise ValueError(f"Split {split} kosong di metadata.")
    print(f"[{scenario}] Menyiapkan {len(subset)} foto {split}...", flush=True)
    training = split == "train"
    return DataLoader(PhotoDataset(subset, scenario, augment=training), batch_size=batch_size, shuffle=training)
