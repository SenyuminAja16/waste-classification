"""Evaluasi checkpoint dari satu run pada split test yang tersimpan."""

import argparse
from pathlib import Path

from torch import nn

from core.evaluate import print_metrics, save_report
from dl.config import load_config
from dl.data import make_loader, read_photos
from dl.engine import score_model
from dl.model import build_model
from dl.utils import load_weights, resolve_device


def evaluate_run(run_dir: Path):
    config = load_config(run_dir / "config.yaml")
    scenario = config["data"]["scenario"]
    device = resolve_device(config["device"])
    model = build_model(pretrained=False).to(device)
    load_weights(model, run_dir / "best.pt", device)

    photos = read_photos(run_dir / "metadata.csv")
    test_loader = make_loader(photos, "test", scenario, config["data"]["batch_size"])
    metrics, loss = score_model(model, test_loader, nn.CrossEntropyLoss(), device)
    save_report("test", metrics, {"scenario": scenario, "loss": loss}, output_dir=run_dir)
    print_metrics(f"DL {scenario} TEST", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="folder eksperimen yang berisi best.pt")
    args = parser.parse_args()
    try:
        evaluate_run(args.run_dir)
    except KeyError as exc:
        parser.exit(2, f"Konfigurasi belum lengkap: {exc}\n")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
