"""Metrik evaluasi dan confusion matrix yang dipakai ML dan DL."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from core.config import CLASSES, REPORTS_DIR


def evaluate(y_true, y_pred) -> dict:
    labels = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "per_class": {c: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def save_report(name: str, metrics: dict, extra: dict | None = None, *, output_dir=REPORTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = dict(metrics, **(extra or {}))
    (output_dir / f"{name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title(name)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_cm.png", dpi=150)
    plt.close(fig)


def print_metrics(name: str, m: dict):
    print(f"[{name}] acc={m['accuracy']:.3f}  P={m['precision_macro']:.3f}  "
          f"R={m['recall_macro']:.3f}  F1={m['f1_macro']:.3f}  (n={m['n_test']})")
