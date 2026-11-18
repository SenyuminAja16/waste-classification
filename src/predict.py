"""Inference untuk 4 skenario (dipakai app.py)."""
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import CLASSES, MODELS_DIR, SCENARIOS
from src import dl
from src.features import extract_features
from src.pcd import resize_43, run_pcd


class ModelNotFound(Exception):
    def __init__(self, scenario, path):
        super().__init__(f"Model untuk {scenario.name} belum ada: {path}")
        self.scenario, self.path = scenario, path


@dataclass
class Result:
    scenario_id: int
    label: str
    probs: dict
    is_demo: bool = False
    pcd_steps: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.probs[self.label]


def model_path(scenario_id: int) -> Path:
    return MODELS_DIR / SCENARIOS[scenario_id].file


def model_available(scenario_id: int) -> bool:
    return model_path(scenario_id).exists()


@lru_cache(maxsize=4)
def _load_ml(path: str):
    import joblib
    return joblib.load(path)


def _predict_ml(path: Path, img, mask) -> np.ndarray:
    model = _load_ml(str(path))
    x = extract_features(img, mask).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
    else:
        p = np.zeros(len(model.classes_))
        p[list(model.classes_).index(model.predict(x)[0])] = 1.0
    out = np.zeros(len(CLASSES))
    for cls, val in zip(model.classes_, p):
        idx = CLASSES.index(cls) if isinstance(cls, str) else int(cls)
        out[idx] = val
    return out


@lru_cache(maxsize=2)
def _load_dl(path: str):
    import torch
    m = dl.build_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def _predict_dl(path: Path, img) -> np.ndarray:
    import torch
    model = _load_dl(str(path))
    x = dl.to_tensor(dl.to_rgb224(img)).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()


def _demo_probs(img) -> np.ndarray:
    seed = int(hashlib.md5(img.tobytes()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).dirichlet([1.0] * len(CLASSES))


def predict(scenario_id: int, img_bgr: np.ndarray, allow_demo: bool = False) -> Result:
    sc = SCENARIOS[scenario_id]
    base = resize_43(img_bgr)
    steps, mask, model_input = {}, None, base
    if sc.use_pcd:
        steps = run_pcd(base)
        model_input, mask = steps["segmented"], steps["mask"]

    path, is_demo = model_path(scenario_id), False
    if not path.exists():
        if not allow_demo:
            raise ModelNotFound(sc, path)
        probs, is_demo = _demo_probs(img_bgr), True
    elif sc.kind == "ml":
        probs = _predict_ml(path, model_input, mask)
    else:
        probs = _predict_dl(path, model_input)

    probs_d = {c: float(p) for c, p in zip(CLASSES, probs)}
    return Result(scenario_id, max(probs_d, key=probs_d.get), probs_d, is_demo, steps)
