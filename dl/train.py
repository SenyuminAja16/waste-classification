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
from dl.logger import RunLogger
from dl.model import build_model, set_trainable
from dl.utils import load_weights, resolve_device, set_seed


def fit(model, train_loader, val_loader, train_config, device, checkpoint, logger):
    criterion = nn.CrossEntropyLoss()
    stages = [
        ("head", train_config["epochs_head"], train_config["lr_head"]),
        ("finetune", train_config["epochs_finetune"], train_config["lr_finetune"]),
    ]
    epoch = 0
    best_score = (-math.inf, -math.inf)
    for stage, epochs, lr in stages:
        if not epochs:
            continue
        if stage == "finetune" and epoch > 0:
            load_weights(model, checkpoint, device)
        set_trainable(model, stage)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

        for _ in range(epochs):
            epoch += 1
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            metrics, val_loss = score_model(model, val_loader, criterion, device)
            f1 = metrics["f1_macro"]
            if (f1, -val_loss) > best_score:
                best_score = (f1, -val_loss)
                best_epoch, best_metrics = epoch, metrics
                torch.save(model.state_dict(), checkpoint)
            logger.log_epoch(epoch, stage, lr, train_loss, val_loss, f1)
    return best_epoch, best_metrics


def train_scenario(config, photos, scenario, device):
    set_seed(config["seed"])
    batch_size = config["data"]["batch_size"]
    train_loader = make_loader(photos, "train", scenario, batch_size)
    val_loader = make_loader(photos, "val", scenario, batch_size)
    model = build_model(pretrained=config["model"]["pretrained"]).to(device)

    run_dir = resolve_run_dir(config, scenario)
    save_run_config(run_dir, config, scenario)
    photos.to_csv(run_dir / "metadata.csv", index=False)
    checkpoint = run_dir / "best.pt"
    print(f"[{scenario}] Device: {device} | Output: {run_dir}", flush=True)

    logger = RunLogger(run_dir, scenario)
    best_epoch, metrics = fit(model, train_loader, val_loader, config["train"], device, checkpoint, logger)
    save_report("val", metrics, {"scenario": scenario, "best_epoch": best_epoch}, output_dir=run_dir)
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
