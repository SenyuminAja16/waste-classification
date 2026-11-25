"""Pemuatan YAML, validasi pengaturan, dan lokasi hasil eksperimen."""

import math
from datetime import datetime
from pathlib import Path

import yaml

from core.config import ROOT

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML tidak valid di {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path} harus berisi konfigurasi YAML.")
    return config


def validate_config(config: dict) -> None:
    if config["data"]["scenario"] not in {"raw", "pcd", "both"}:
        raise ValueError("data.scenario harus raw, pcd, atau both.")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device harus auto, cpu, atau cuda.")
    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed harus bilangan bulat dari 0 sampai 2**32 - 1.")
    batch_size = config["data"]["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("data.batch_size harus bilangan bulat positif.")
    if not isinstance(config["model"]["pretrained"], bool):
        raise ValueError("model.pretrained harus true atau false.")

    train = config["train"]
    epochs = [train["epochs_head"], train["epochs_finetune"]]
    if any(type(n) is not int or n < 0 for n in epochs) or sum(epochs) == 0:
        raise ValueError("Jumlah epoch tiap fase harus ≥ 0 dan minimal satu fase harus dijalankan.")
    for key in ("lr_head", "lr_finetune"):
        lr = float(train[key])
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"train.{key} harus positif dan berhingga.")
        train[key] = lr


def resolve_run_dir(config: dict, scenario: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / config["output"]["dir"] / f"{scenario}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_config(run_dir: Path, config: dict, scenario: str) -> None:
    snapshot = {**config, "data": {**config["data"], "scenario": scenario}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
