"""Ukur kestabilan DL dengan validasi silang per objek pada data train."""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from tqdm.auto import tqdm

from core.config import CLASSES, ROOT
from core.evaluate import save_report
from dl.config import DEFAULT_CONFIG, load_config, validate_config
from dl.data import make_loader, prepare_image, read_photos
from dl.engine import score_model
from dl.logger import RunLogger
from dl.model import MODEL_NAMES, build_model
from dl.train import fit
from dl.utils import load_checkpoint, resolve_device, set_seed
from pcd.scripts.select_gt_photos import EXPLORATION_SAMPLE, SELECTION_CSV


def assign_folds(train, reserved_ids, n_splits, seed):
    """Tetapkan satu fold evaluasi per objek; 0 berarti tetap di train setiap fold."""
    objects = train[["object_id", "label"]].drop_duplicates().set_index("object_id")
    if objects.index.has_duplicates:
        raise ValueError("Setiap object_id harus memiliki tepat satu kelas.")
    eligible = train[~train["object_id"].isin(reserved_ids)]
    counts = eligible.groupby("label")["object_id"].nunique().reindex(CLASSES, fill_value=0)
    if counts.min() < n_splits:
        raise ValueError(f"Butuh minimal {n_splits} objek yang bisa dievaluasi per kelas: {counts.to_dict()}")

    objects["fold"] = 0
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for number, (_, held_indices) in enumerate(splitter.split(eligible, eligible["label"], eligible["object_id"]), 1):
        held = eligible.iloc[held_indices]
        if set(held["label"]) != set(CLASSES):
            raise ValueError(f"Fold {number} tidak memuat semua kelas. Kurangi n_splits atau ganti seed.")
        objects.loc[held["object_id"].unique(), "fold"] = number
    return objects.reset_index()


def calibration_ids(options, photos):
    calibration = options.get("pcd_calibration", str(SELECTION_CSV))
    annotated = set() if calibration is None else set(pd.read_csv(ROOT / calibration, dtype=str)["object_id"].dropna())
    reserved = (annotated | set(EXPLORATION_SAMPLE)) & set(photos["object_id"])
    if reserved & set(photos.loc[photos["split"] != "train", "object_id"]):
        raise ValueError("Objek yang pernah dipakai untuk pengembangan PCD harus berada pada split train.")
    return reserved


def evaluate_fold(config, device, train_loader, val_loader, held_loader, fold_dir):
    name, scenario = config["model"]["name"], config["data"]["scenario"]
    fold_dir.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(fold_dir, f"{name}/{scenario}/{fold_dir.name}")
    set_seed(config["seed"])
    model = build_model(**config["model"]).to(device)
    with TemporaryDirectory(prefix="waste-dl-fold-") as scratch:
        checkpoint = Path(scratch) / "best.pt"
        epoch, val_metrics = fit(model, train_loader, val_loader, config, device, checkpoint, logger)
        del model
        best, _ = load_checkpoint(checkpoint, device)
        metrics, loss = score_model(best, held_loader, nn.CrossEntropyLoss(), device)

    selection = {
        "best_epoch": epoch,
        "best_stage": logger.history[epoch - 1]["stage"],
        "checkpoint_val_f1": val_metrics["f1_macro"],
    }
    save_report("fold", metrics, {"model": name, "scenario": scenario, "loss": loss, **selection}, output_dir=fold_dir)
    return metrics, selection


def save_summary(run_dir, scores, n_splits, reserved_count, validation_count):
    measures = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = scores.groupby(["model", "scenario"], sort=False)[measures].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(run_dir / "summary.csv", float_format="%.6f")

    markdown = [
        "# Validasi silang Deep Learning",
        "",
        f"StratifiedGroupKFold {n_splits} fold pada split train dengan pengelompokan berdasarkan `object_id`.",
        f"Sebanyak {reserved_count} objek pengembangan PCD selalu berada di train setiap fold.",
        f"Checkpoint dipilih dari validation asli ({validation_count} objek) dan dinilai pada fold yang ditahan.",
        "Skor dihitung per foto. Angka berikut adalah rata-rata ± simpangan baku sampel antar-fold.",
        "Split test asli tetap terpisah dari validasi silang.",
        "",
        "| Model | Skenario | Accuracy | Precision macro | Recall macro | F1 macro |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (name, scenario), row in summary.iterrows():
        values = [f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}" for m in measures]
        markdown.append(f"| {name} | {scenario.upper()} | {' | '.join(values)} |")

    pivot = scores.pivot(index=["model", "fold"], columns="scenario", values="f1_macro")
    if {"raw", "pcd"} <= set(pivot.columns):
        paired = (pivot["pcd"] - pivot["raw"]).groupby("model").agg(["mean", "std"])
        paired.to_csv(run_dir / "paired_delta_f1.csv", float_format="%.6f")
        markdown.extend(["", "Selisih F1 macro berpasangan (PCD − Raw) per fold.", ""])
        markdown.extend(f"- {name}: {row['mean']:+.3f} ± {row['std']:.3f}" for name, row in paired.iterrows())

    (run_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown), flush=True)


def run_cross_validation(config, device):
    options = config.get("kfold", {})
    n_splits = options.get("n_splits", 3)
    models = options.get("models", ["mobilenet_v2", "efficientnet_b0"])
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("kfold.n_splits harus bilangan bulat minimal 2.")
    if not isinstance(models, list) or not models or any(name not in MODEL_NAMES for name in models):
        raise ValueError(f"kfold.models harus berupa daftar dari {', '.join(MODEL_NAMES)}.")
    if len(models) != len(set(models)):
        raise ValueError("kfold.models berisi model yang sama lebih dari sekali.")

    data = config["data"]
    photos = read_photos(ROOT / data["metadata"])
    reserved = calibration_ids(options, photos)
    development = photos[photos["split"].isin(["train", "val"])]
    train = development[development["split"] == "train"]
    checkpoint_val = development[development["split"] == "val"]
    if checkpoint_val.empty:
        raise ValueError("Split val asli diperlukan untuk memilih checkpoint di setiap fold.")
    assignments = assign_folds(train, reserved, n_splits, config["seed"])

    scenarios = ("raw", "pcd") if data["scenario"] == "both" else (data["scenario"],)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / config["output"]["dir"] / f"kfold_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assignments.to_csv(run_dir / "fold_assignments.csv", index=False)
    development.to_csv(run_dir / "development.csv", index=False)
    print(f"K-fold: {len(models) * len(scenarios) * n_splits} run | Device: {device} | Output: {run_dir}", flush=True)

    scores = []
    for scenario in scenarios:
        paths = development["raw_path"].unique()
        print(f"[{scenario}] Menyiapkan {len(paths)} foto train/val untuk semua fold...", flush=True)
        prepared = {
            path: prepare_image(path, scenario, data["image_size"])
            for path in tqdm(paths, desc=f"[{scenario}] Persiapan foto", unit="foto", disable=None)
        }
        loader_options = {"batch_size": data["batch_size"], "image_size": data["image_size"], "prepared": prepared}
        val_loader = make_loader(checkpoint_val, "val", scenario, **loader_options)
        for number in range(1, n_splits + 1):
            held_ids = set(assignments.loc[assignments["fold"] == number, "object_id"])
            held = train[train["object_id"].isin(held_ids)].assign(split="val")
            remaining = train[~train["object_id"].isin(held_ids)]
            held_loader = make_loader(held, "val", scenario, **loader_options)
            for name in models:
                train_loader = make_loader(
                    remaining, "train", scenario, **loader_options,
                    augmentation=data["augmentation"], augmentation_seed=config["seed"] + number,
                )
                fold_dir = run_dir / f"{name}_{scenario}" / f"fold_{number}"
                fold_config = {
                    **config, "model": {**config["model"], "name": name}, "data": {**data, "scenario": scenario},
                }
                metrics, selection = evaluate_fold(fold_config, device, train_loader, val_loader, held_loader, fold_dir)
                scores.append({
                    "model": name, "scenario": scenario, "fold": number,
                    "train_objects": remaining["object_id"].nunique(), "held_objects": len(held_ids),
                    "train_source_photos": len(remaining), "train_pool_photos": len(train_loader.dataset),
                    "train_steps_per_epoch": len(train_loader),
                    "held_photos": len(held), **selection,
                    **{key: metrics[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                })
                pd.DataFrame(scores).to_csv(run_dir / "fold_metrics.csv", index=False)
                print(f"[{name}/{scenario}/fold_{number}] F1 macro held-out={metrics['f1_macro']:.3f}", flush=True)

    save_summary(run_dir, pd.DataFrame(scores), n_splits, len(reserved), checkpoint_val["object_id"].nunique())
    print(f"Ringkasan: {run_dir / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="konfigurasi YAML eksperimen")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        validate_config(config)
        run_cross_validation(config, resolve_device(config["device"]))
    except KeyError as exc:
        parser.exit(2, f"Konfigurasi atau kolom CSV belum lengkap: {exc}\n")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
