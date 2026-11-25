"""Seed, pemilihan perangkat, dan pemuatan checkpoint DL."""

import random
from pathlib import Path

import numpy as np
import torch

from core.config import CLASSES, DL_INPUT
from dl.model import DEFAULT_DROPOUT, DEFAULT_MODEL, build_model


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


def save_checkpoint(path: Path, model, config: dict) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "model_name": config["model"]["name"],
        "dropout": config["model"]["dropout"],
        "image_size": config["data"]["image_size"],
        "classes": CLASSES,
    }, path)


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in checkpoint:
        # Checkpoint awal hanya berisi state_dict MobileNetV2 dengan input 224.
        checkpoint = {"model_state": checkpoint}
    if checkpoint.get("classes", CLASSES) != CLASSES:
        raise ValueError("Urutan kelas pada checkpoint berbeda dari konfigurasi proyek.")
    return checkpoint


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, int]:
    checkpoint = read_checkpoint(path)
    model = build_model(
        checkpoint.get("model_name", DEFAULT_MODEL),
        pretrained=False,
        dropout=checkpoint.get("dropout", DEFAULT_DROPOUT),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint.get("image_size", DL_INPUT)
