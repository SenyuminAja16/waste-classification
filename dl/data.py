"""Metadata, dataset foto, augmentasi, dan DataLoader untuk Raw dan PCD."""

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from core.config import CLASSES, DATA_DIR, DL_INPUT
from core.image import load_image, resize_43
from dl.augmentation import FixedVariantSampler, build_transform, materialize_variants, transform_rgb
from dl.transforms import to_rgb, to_tensor
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


def prepare_image(raw_path: str, scenario: str, image_size: int):
    img = resize_43(load_image(DATA_DIR / raw_path))
    if scenario == "pcd":
        img = run_pcd(img)["segmented"]
    return to_rgb(img, image_size)


class PhotoDataset(Dataset):
    def __init__(
        self, photos: pd.DataFrame, scenario: str, image_size: int, augmentation: dict | None, *, prepared=None,
    ):
        self.labels = photos["label"].map(CLASS_INDEX).tolist()
        self.transform = build_transform(augmentation) if augmentation and augmentation["enabled"] else None
        self.images = [
            prepare_image(path, scenario, image_size) if prepared is None else prepared[path]
            for path in photos["raw_path"]
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb = self.images[index]
        if self.transform is not None:
            rgb = transform_rgb(rgb, self.transform)
        return to_tensor(rgb), self.labels[index]


def make_loader(
    photos, split, scenario, batch_size, *, image_size=DL_INPUT, augmentation=None, prepared=None, augmentation_seed=0,
):
    subset = photos[photos["split"] == split]
    if subset.empty:
        raise ValueError(f"Split {split} kosong di metadata.")
    print(f"[{scenario}] Menyiapkan {len(subset)} foto {split}...", flush=True)
    training = split == "train"
    offline = training and augmentation and augmentation["enabled"] and augmentation.get("mode") == "offline"
    online_augmentation = augmentation if training and not offline else None
    dataset = PhotoDataset(subset, scenario, image_size, online_augmentation, prepared=prepared)
    if offline:
        variants = augmentation.get("variants_per_photo", 2)
        sources = len(dataset)
        dataset.images = materialize_variants(dataset.images, augmentation, variants, augmentation_seed)
        dataset.labels *= variants + 1
        sampler = FixedVariantSampler(sources, variants, augmentation_seed)
        print(f"[{scenario}] Augmentasi offline: {sources} sumber, {len(dataset)} gambar.", flush=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training)
