"""Transformasi foto dan penjadwalan varian offline dengan anggaran langkah tetap."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision import transforms as T


def build_transform(augmentation):
    return T.Compose([
        T.RandomHorizontalFlip(p=augmentation["horizontal_flip"]),
        T.RandomRotation(augmentation["rotation"]),
        T.ColorJitter(brightness=augmentation["brightness"], contrast=augmentation["contrast"]),
    ])


def transform_rgb(rgb, transform):
    return np.array(transform(Image.fromarray(rgb)))


def materialize_variants(images, augmentation, variants_per_photo, seed):
    """Urutan hasil: seluruh citra asli, varian pertama, lalu varian berikutnya."""
    transform = build_transform(augmentation)
    expanded = list(images)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for _ in range(variants_per_photo):
            expanded.extend(transform_rgb(rgb, transform) for rgb in images)
    return expanded


class FixedVariantSampler(Sampler[int]):
    """Ambil satu versi tiap foto sumber per epoch, dengan slot awal yang berbeda antar-foto."""

    def __init__(self, n_sources: int, variants_per_photo: int, seed: int):
        self.n_sources = n_sources
        self.n_versions = variants_per_photo + 1
        self.generator = torch.Generator().manual_seed(seed)
        self.offsets = torch.randint(self.n_versions, (n_sources,), generator=self.generator)
        self.epoch = 0

    def __iter__(self):
        order = torch.randperm(self.n_sources, generator=self.generator)
        versions = (self.offsets[order] + self.epoch) % self.n_versions
        self.epoch += 1
        return iter((versions * self.n_sources + order).tolist())

    def __len__(self):
        return self.n_sources
