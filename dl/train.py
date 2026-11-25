"""Latih skenario Raw dan PCD berdasarkan dl/config.yaml."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from core.config import ROOT
from core.evaluate import print_metrics, save_report
from dl.config import DEFAULT_CONFIG, load_config, resolve_run_dir, save_run_config, validate_config
from dl.data import make_loader, read_photos
from dl.engine import score_model, train_one_epoch
from dl.logger import RunLogger, save_learning_curves
from dl.model import build_model, set_trainable
from dl.utils import read_checkpoint, resolve_device, save_checkpoint, set_seed


def fit(model, train_loader, val_loader, config, device, checkpoint, logger):
    train = config["train"]
    optimizer_config = config["optimizer"]
    optimizer_factory = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_config["name"]]
    criterion = nn.CrossEntropyLoss()
    stages = [
        ("head", train["epochs_head"], train["lr_head"], 0),
        ("finetune", train["epochs_finetune"], train["lr_finetune"], train["finetune_blocks"]),
    ]
    epoch = 0
    best_score = (-math.inf, -math.inf)
    patience = train["early_stopping_patience"]
    for stage, epochs, lr, blocks in stages:
        if not epochs:
            continue
        if stage == "finetune" and epoch > 0:
            model.load_state_dict(read_checkpoint(checkpoint)["model_state"])
        set_trainable(model, config["model"]["name"], blocks)
        optimizer = optimizer_factory(
            (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=optimizer_config["weight_decay"],
        )
        stage_best_score = (-math.inf, -math.inf)
        stale_epochs = 0

        for _ in range(epochs):
            epoch += 1
            description = f"[{logger.scenario}] {stage} ep{epoch:02d}"
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, description=description)
            metrics, val_loss = score_model(model, val_loader, criterion, device)
            f1 = metrics["f1_macro"]
            score = (f1, -val_loss)
            if score > best_score:
                best_score = score
                best_epoch, best_metrics = epoch, metrics
                save_checkpoint(checkpoint, model, config)
            logger.log_epoch(epoch, stage, lr, train_loss, val_loss, f1)
            if score > stage_best_score:
                stage_best_score = score
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience and stale_epochs >= patience:
                    print(f"[{stage}] Early stopping setelah {stale_epochs} epoch tanpa perbaikan.")
                    break
    return best_epoch, best_metrics


def train_scenario(config, photos, scenario, device):
    set_seed(config["seed"])
    data = config["data"]
    loader_config = {"batch_size": data["batch_size"], "image_size": data["image_size"]}
    train_loader = make_loader(
        photos, "train", scenario, **loader_config, augmentation=data["augmentation"], augmentation_seed=config["seed"],
    )
    val_loader = make_loader(photos, "val", scenario, **loader_config)
    model = build_model(**config["model"]).to(device)

    run_dir = resolve_run_dir(config, scenario)
    save_run_config(run_dir, config, scenario)
    photos.to_csv(run_dir / "metadata.csv", index=False)
    checkpoint = run_dir / "best.pt"
    print(f"[{scenario}] Device: {device} | Output: {run_dir}", flush=True)

    logger = RunLogger(run_dir, scenario)
    best_epoch, metrics = fit(model, train_loader, val_loader, config, device, checkpoint, logger)
    info = {"scenario": scenario, "model": config["model"]["name"], "best_epoch": best_epoch}
    save_report("val", metrics, info, output_dir=run_dir)
    save_learning_curves(logger.history, best_epoch, run_dir / "learning_curves.png")
    print_metrics(f"DL {scenario} VAL (epoch {best_epoch})", metrics)
    print(f"Checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="konfigurasi YAML eksperimen")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        validate_config(config)
        device = resolve_device(config["device"])
        metadata = ROOT / config["data"]["metadata"]
        if not metadata.is_file():
            raise ValueError(f"{metadata} belum ada. Jalankan dulu python -m dataset.scripts.prepare_dataset.")
        photos = read_photos(metadata)
        scenario = config["data"]["scenario"]
        for tag in ("raw", "pcd") if scenario == "both" else (scenario,):
            train_scenario(config, photos, tag, device)
    except KeyError as exc:
        parser.exit(2, f"Konfigurasi belum lengkap: {exc}\n")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
