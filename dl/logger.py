"""Pencatatan metrik tiap epoch dan grafik kurva pelatihan."""

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


def save_learning_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    f1_scores = [row["val_f1"] for row in history]
    best_f1 = f1_scores[epochs.index(best_epoch)]

    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    loss_ax.plot(epochs, [row["train_loss"] for row in history], "o-", label="Train")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], "s-", label="Validation")
    loss_ax.set(ylabel="Cross-entropy loss", title="Train dan validation loss")
    loss_ax.legend(frameon=False)

    f1_ax.plot(epochs, f1_scores, "o-", color="tab:green")
    f1_ax.scatter(best_epoch, best_f1, color="tab:red", zorder=3, label=f"Terbaik: {best_f1:.3f}")
    f1_ax.set(ylabel="F1 macro", title="F1 macro validation", ylim=(0, 1))
    f1_ax.legend(frameon=False)

    for ax in (loss_ax, f1_ax):
        ax.axvline(best_epoch, color="tab:red", linestyle="--", alpha=0.55)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(path, dpi=160)
    plt.close(fig)
