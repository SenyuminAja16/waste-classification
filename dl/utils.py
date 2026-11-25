"""Seed, pemilihan perangkat, dan pemuatan checkpoint DL."""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(setting: str) -> str:
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA tidak tersedia. Gunakan device: auto atau cpu.")
    return setting


def load_weights(model, checkpoint: Path, device: str) -> None:
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
