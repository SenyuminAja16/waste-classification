"""Latih Deep Learning (skenario 3 = PCD+DL, skenario 4 = Raw+DL) dengan MobileNetV2 transfer learning.

Tahap 1: hanya classifier dilatih (backbone beku).  Tahap 2: fine-tune beberapa blok terakhir (lr kecil).
Epoch terbaik dipilih berdasarkan F1 macro di data VAL; data TEST dipakai sekali di akhir.
Augmentasi (flip, rotasi kecil, brightness/contrast) hanya untuk data train.

Butuh torch + torchvision.
Pemakaian:
    python -m dl.scripts.train_dl --scenario pcd
    python -m dl.scripts.train_dl --scenario raw --epochs-head 5 --epochs-ft 15 --batch 16
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from core.config import CLASSES, MODELS_DIR, REPORTS_DIR, ROOT, SCENARIOS, SEED, SPLIT_CSV
from core.evaluate import evaluate, print_metrics, save_report
from core.image import load_image, resize_43
from dl import models as dl
from pcd.pipeline import run_pcd

DL_SCENARIO = {"pcd": 3, "raw": 4}


def load_arrays(df: pd.DataFrame, tag: str) -> list:
    out = []
    for i, p in enumerate(df.path, 1):
        img = resize_43(load_image(ROOT / p))
        if tag == "pcd":
            img = run_pcd(img)["segmented"]
        out.append(dl.to_rgb224(img))
        if i % 50 == 0 or i == len(df):
            print(f"[{tag}] memuat citra {i}/{len(df)}", end="\r")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["pcd", "raw"], required=True)
    ap.add_argument("--split", type=Path, default=SPLIT_CSV)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if not a.split.exists():
        sys.exit(f"{a.split} belum ada. Jalankan dulu: python -m dataset.scripts.prepare_dataset")
    df = pd.read_csv(a.split)
    if not {"train", "val", "test"} <= set(df.subset):
        sys.exit("Deep Learning butuh split train/val/test. Data belum cukup (lihat catatan prepare_dataset.py).")
    y_all = df.kelas.map(CLASSES.index).to_numpy()
    arrays = load_arrays(df, a.scenario)

    class DS(Dataset):
        def __init__(self, idx, train):
            self.idx, self.train = idx, train
            self.aug = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10),
                                  T.ColorJitter(brightness=0.2, contrast=0.2)])

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            im = Image.fromarray(arrays[j])
            if self.train:
                im = self.aug(im)
            return dl.to_tensor(np.array(im)), int(y_all[j])

    def loader(sub, train):
        idx = np.where(df.subset.to_numpy() == sub)[0]
        if len(idx) == 0:
            sys.exit(f"Subset {sub} kosong.")
        return DataLoader(DS(idx, train), batch_size=a.batch, shuffle=train, num_workers=0)

    tr_dl, va_dl, te_dl = loader("train", True), loader("val", False), loader("test", False)
    model = dl.build_model(pretrained=True).to(device)
    crit = nn.CrossEntropyLoss()

    def predict_all(dloader):
        model.eval()
        ys, ps, loss = [], [], 0.0
        with torch.no_grad():
            for x, y in dloader:
                out = model(x.to(device))
                loss += crit(out, y.to(device)).item() * len(y)
                ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
        return np.array(ys), np.array(ps), loss / max(len(ys), 1)

    best = {"f1": -1.0, "loss": 1e9, "state": None, "epoch": 0}
    history = []

    def run_stage(mode, epochs, lr):
        dl.set_trainable(model, mode)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for ep in range(1, epochs + 1):
            model.train()
            if mode == "head":
                model.features.eval()  # backbone beku: jangan update statistik BatchNorm
            tot = 0.0
            for x, y in tr_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
            yv, pv, vloss = predict_all(va_dl)
            f1 = f1_score(yv, pv, average="macro", labels=list(range(len(CLASSES))), zero_division=0)
            tloss = tot / len(tr_dl.dataset)
            history.append({"stage": mode, "epoch": ep, "train_loss": tloss, "val_loss": vloss, "val_f1": f1})
            print(f"[{a.scenario}] {mode:8s} ep{ep:02d}  train_loss={tloss:.3f}  val_loss={vloss:.3f}  val_f1={f1:.3f}")
            if f1 > best["f1"] or (f1 == best["f1"] and vloss < best["loss"]):
                best.update(f1=f1, loss=vloss, epoch=len(history),
                            state=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))

    run_stage("head", a.epochs_head, a.lr_head)
    run_stage("finetune", a.epochs_ft, a.lr_ft)

    model.load_state_dict(best["state"])
    yt, pt, _ = predict_all(te_dl)
    m = evaluate(yt, pt)
    print(f"[{a.scenario}] epoch terbaik (urutan global) = {best['epoch']}, val_f1={best['f1']:.3f}")
    print_metrics(f"DL {a.scenario} TEST", m)

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / SCENARIOS[DL_SCENARIO[a.scenario]].file
    torch.save(best["state"], out)
    print(f"[{a.scenario}] model disimpan: {out}")
    save_report(f"dl_{a.scenario}", m, {"best_epoch": best["epoch"], "val_f1": best["f1"], "args": vars(a) | {"split": str(a.split)}})
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"dl_{a.scenario}_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        sys.exit(f"\nBagian ini belum diisi: {e}")
