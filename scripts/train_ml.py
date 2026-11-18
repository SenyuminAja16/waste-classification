"""Latih Machine Learning (skenario 1 = PCD+ML, skenario 2 = Raw+ML).

Alur: ekstraksi fitur (di-cache) → grid search + validasi silang per objek (GroupKFold) pada data train
→ pilih model terbaik berdasarkan F1 macro di data VAL → evaluasi → simpan model.

Pemakaian:
    python scripts/train_ml.py                       # pcd dan raw
    python scripts/train_ml.py --scenario pcd
    python scripts/train_ml.py --refresh             # hitung ulang fitur (wajib bila PCD / fitur diubah)
    python scripts/train_ml.py --eval-on val         # selama masih iterasi: JANGAN lihat skor test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from config import CLASSES, MODELS_DIR, SCENARIOS, SEED, SPLIT_CSV
from src.dataset import build_features
from src.evaluate import evaluate, print_metrics, save_report
from src.ml import candidates

ML_SCENARIO = {"pcd": 1, "raw": 2}


def run(tag: str, df: pd.DataFrame, refresh: bool, eval_on: str):
    X = build_features(df, tag, refresh)
    y = df.kelas.map(CLASSES.index).to_numpy()
    g = df.id_objek.to_numpy()
    tr, va, te = [(df.subset == s).to_numpy() for s in ("train", "val", "test")]
    if not (tr.any() and va.any() and te.any()):
        sys.exit("Subset train/val/test ada yang kosong. Tambah data atau jalankan prepare_dataset.py lagi.")

    n_splits = min(5, len(set(g[tr])))
    best, cands = None, {}
    for name, (est, grid) in candidates(SEED).items():
        gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=n_splits), scoring="f1_macro", n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=g[tr])
        f1_val = f1_score(y[va], gs.predict(X[va]), average="macro", labels=list(range(len(CLASSES))))
        cands[name] = {"params": gs.best_params_, "cv_f1": float(gs.best_score_), "val_f1": float(f1_val)}
        print(f"[{tag}] {name:14s} cv_f1={gs.best_score_:.3f}  val_f1={f1_val:.3f}  {gs.best_params_}")
        if best is None or f1_val > best[0]:
            best = (f1_val, name, gs.best_estimator_, gs.best_params_)

    f1_val, name, model, params = best
    ev = va if eval_on == "val" else te
    m = evaluate(y[ev], model.predict(X[ev]))
    print(f"[{tag}] model terpilih: {name} (val_f1={f1_val:.3f})")
    print_metrics(f"ML {tag} {eval_on.upper()}", m)

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / SCENARIOS[ML_SCENARIO[tag]].file
    joblib.dump(model, out)
    print(f"[{tag}] model disimpan: {out}")
    save_report(f"ml_{tag}", m, {"model": name, "params": params, "val_f1": f1_val,
                                 "eval_subset": eval_on, "candidates": cands})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["pcd", "raw", "both"], default="both")
    ap.add_argument("--split", type=Path, default=SPLIT_CSV)
    ap.add_argument("--refresh", action="store_true", help="hitung ulang fitur (abaikan cache)")
    ap.add_argument("--eval-on", choices=["val", "test"], default="test",
                    help="subset untuk laporan metrik. Pakai 'val' selama tuning agar test tetap bersih.")
    a = ap.parse_args()
    if not a.split.exists():
        sys.exit(f"{a.split} belum ada. Jalankan dulu: python scripts/prepare_dataset.py")
    df = pd.read_csv(a.split)
    for tag in (["pcd", "raw"] if a.scenario == "both" else [a.scenario]):
        run(tag, df, a.refresh, a.eval_on)


if __name__ == "__main__":
    main()
