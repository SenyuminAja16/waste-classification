"""Metrik segmentasi untuk membandingkan mask hasil PCD dengan mask acuan."""

import numpy as np


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred > 0, truth > 0
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 1.0


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred > 0, truth > 0
    total = pred.sum() + truth.sum()
    return float(2 * np.logical_and(pred, truth).sum() / total) if total else 1.0
