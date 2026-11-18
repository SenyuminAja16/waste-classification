"""Pembagian train/val/test per objek yang tidak berubah saat data bertambah."""

import math

import numpy as np
import pandas as pd

from core.config import SEED, TEST_FRAC, VAL_FRAC

SPLIT_COLUMNS = ["object_id", "label", "split"]


def target_counts(n_objects: int) -> dict:
    """Jumlah objek per subset untuk satu kelas: test dibulatkan ke atas, val ke bawah."""
    n_test = math.ceil(TEST_FRAC * n_objects)
    n_val = math.floor(VAL_FRAC * n_objects)
    return {"train": n_objects - n_val - n_test, "val": n_val, "test": n_test}


def assign_splits(objects: pd.DataFrame, existing: pd.DataFrame | None = None, seed: int = SEED) -> pd.DataFrame:
    """Tetapkan split per objek; objek di `existing` tidak dipindah, objek baru mengisi kekurangan tiap kelas."""
    if existing is None:
        existing = pd.DataFrame(columns=SPLIT_COLUMNS)
    known = existing[existing["object_id"].isin(objects["object_id"])][SPLIT_COLUMNS]
    new = objects[~objects["object_id"].isin(known["object_id"])]

    rng = np.random.default_rng(seed)
    assigned = []
    for label, group in new.groupby("label"):
        target = target_counts((objects["label"] == label).sum())
        have = known.loc[known["label"] == label, "split"].value_counts()
        missing = {split: max(count - have.get(split, 0), 0) for split, count in target.items()}

        object_ids = rng.permutation(np.sort(group["object_id"].to_numpy()))
        queue = ["test"] * missing["test"] + ["val"] * missing["val"]
        splits = queue[: len(object_ids)] + ["train"] * max(len(object_ids) - len(queue), 0)
        assigned.append(pd.DataFrame({"object_id": object_ids, "label": label, "split": splits}))

    result = pd.concat([known, *assigned], ignore_index=True)
    return result.sort_values("object_id").reset_index(drop=True)
