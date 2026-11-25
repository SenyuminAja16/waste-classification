"""Pemuatan YAML, validasi pengaturan, dan lokasi hasil eksperimen."""

import math
from datetime import datetime
from pathlib import Path

import yaml

from core.config import DL_INPUT, ROOT
from dl.model import DEFAULT_DROPOUT, DEFAULT_MODEL, MODEL_NAMES

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
    data, model, train = (config[key] for key in ("data", "model", "train"))
    augmentation = data.setdefault("augmentation", {})
    optimizer = config.setdefault("optimizer", {})

    if data["scenario"] not in {"raw", "pcd", "both"}:
        raise ValueError("data.scenario harus raw, pcd, atau both.")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device harus auto, cpu, atau cuda.")
    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed harus bilangan bulat dari 0 sampai 2**32 - 1.")
    batch_size = data["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("data.batch_size harus bilangan bulat positif.")
    image_size = data.setdefault("image_size", DL_INPUT)
    if type(image_size) is not int or image_size < 32:
        raise ValueError("data.image_size harus bilangan bulat minimal 32.")
    if not isinstance(augmentation.setdefault("enabled", True), bool):
        raise ValueError("data.augmentation.enabled harus true atau false.")
    if augmentation.get("mode", "online") not in ("online", "offline"):
        raise ValueError("data.augmentation.mode harus online atau offline.")
    variants = augmentation.get("variants_per_photo", 2)
    if type(variants) is not int or variants < 1:
        raise ValueError("data.augmentation.variants_per_photo harus bilangan bulat positif.")
    for key, default, maximum in (
        ("horizontal_flip", 0.5, 1.0), ("rotation", 10, 180), ("brightness", 0.2, 1.0), ("contrast", 0.2, 1.0),
    ):
        augmentation[key] = validate_number(augmentation.get(key, default), f"data.augmentation.{key}", maximum)

    if model.setdefault("name", DEFAULT_MODEL) not in MODEL_NAMES:
        raise ValueError(f"model.name harus salah satu dari {', '.join(MODEL_NAMES)}.")
    if not isinstance(model["pretrained"], bool):
        raise ValueError("model.pretrained harus true atau false.")
    model["dropout"] = validate_number(model.get("dropout", DEFAULT_DROPOUT), "model.dropout", 1.0)
    if model["dropout"] == 1:
        raise ValueError("model.dropout harus lebih kecil dari 1.")

    if optimizer.setdefault("name", "adam") not in {"adam", "adamw"}:
        raise ValueError("optimizer.name harus adam atau adamw.")
    optimizer["weight_decay"] = validate_number(optimizer.get("weight_decay", 0.0), "optimizer.weight_decay")

    epochs = [train["epochs_head"], train["epochs_finetune"]]
    if any(type(n) is not int or n < 0 for n in epochs) or sum(epochs) == 0:
        raise ValueError("Jumlah epoch tiap fase harus ≥ 0 dan minimal satu fase harus dijalankan.")
    finetune_blocks = train.setdefault("finetune_blocks", 4)
    if type(finetune_blocks) is not int or finetune_blocks < 0 or (train["epochs_finetune"] and not finetune_blocks):
        raise ValueError("train.finetune_blocks harus bilangan bulat ≥ 0 dan minimal 1 untuk fase fine-tuning.")
    patience = train.setdefault("early_stopping_patience", 0)
    if type(patience) is not int or patience < 0:
        raise ValueError("train.early_stopping_patience harus bilangan bulat ≥ 0.")
    for key in ("lr_head", "lr_finetune"):
        lr = validate_number(train[key], f"train.{key}")
        if not lr:
            raise ValueError(f"train.{key} harus positif.")
        train[key] = lr


def validate_number(value, key: str, maximum: float = math.inf) -> float:
    try:
        number = float(value) if not isinstance(value, bool) else math.nan
    except (ValueError, TypeError):
        number = math.nan
    if not math.isfinite(number) or not 0 <= number <= maximum:
        limits = f"0–{maximum:g}" if math.isfinite(maximum) else "≥ 0"
        raise ValueError(f"{key} harus berupa angka berhingga {limits}.")
    return number


def resolve_run_dir(config: dict, scenario: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / config["output"]["dir"] / f"{config['model']['name']}_{scenario}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_config(run_dir: Path, config: dict, scenario: str) -> None:
    snapshot = {**config, "data": {**config["data"], "scenario": scenario}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
