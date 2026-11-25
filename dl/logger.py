"""Pencatatan metrik tiap epoch ke terminal dan history.json."""

import json
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path, scenario: str):
        self.path = run_dir / "history.json"
        self.scenario = scenario
        self.history = []

    def log_epoch(self, epoch, stage, lr, train_loss, val_loss, val_f1):
        self.history.append({
            "epoch": epoch,
            "stage": stage,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        })
        self.path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"[{self.scenario}] {stage} ep{epoch:02d} train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} val_f1={val_f1:.3f}", flush=True)
