"""Latih Machine Learning (skenario 1 = PCD+ML, skenario 2 = Raw+ML).

Alur: ekstraksi fitur (di-cache) → grid search + cross-validation per objek pada data train
→ pilih model terbaik → evaluasi → simpan model.
  - Data cukup (ada val dan test): model dipilih berdasarkan F1 macro di VAL, metrik dilaporkan di VAL/TEST.
  - Data masih sedikit (tanpa val/test): model dipilih berdasarkan skor CV, metrik = prediksi out-of-fold.

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
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from config import CLASSES, MODELS_DIR, SCENARIOS, SEED, SPLIT_CSV
from src.dataset import build_features
from src.evaluate import evaluate, print_metrics, save_report
from src.ml import candidates

ML_SCENARIO = {"pcd": 1, "raw": 2}


def run(tag, df, refresh, eval_on):
    X = build_features(df, tag, refresh)
    y = df.kelas.map(CLASSES.index).to_numpy()
    g = df.id_objek.to_numpy()
    tr, va, te = [(df.subset == s).to_numpy() for s in ("train", "val", "test")]
    kecil = not (va.any() and te.any())
    if kecil:
        print(f"[{tag}] CATATAN: belum ada split val/test, hanya cross-validation yang dilaporkan.")
        tr = np.ones(len(df), bool)

    min_obj = min(len(set(g[tr & (y == c)])) for c in range(len(CLASSES)))
    if min_obj < 2:
        sys.exit("Tiap kelas butuh minimal 2 objek untuk cross-validation. Data belum cukup.")
    n_splits = min(5, min_obj)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    best, cands = None, {}
    for name, (est, grid) in candidates(SEED).items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=g[tr])
        f1_val = None if kecil else float(f1_score(y[va], gs.predict(X[va]), average="macro",
                                                   labels=list(range(len(CLASSES)))))
        score = gs.best_score_ if kecil else f1_val
        cands[name] = {"params": gs.best_params_, "cv_f1": float(gs.best_score_), "val_f1": f1_val}
        print(f"[{tag}] {name:14s} cv_f1={gs.best_score_:.3f}" + ("" if kecil else f"  val_f1={f1_val:.3f}") + f"  {gs.best_params_}")
        if best is None or score > best[0]:
            best = (score, name, gs.best_estimator_, gs.best_params_)

    score, name, model, params = best
    print(f"[{tag}] model terpilih: {name}")
    if kecil:
        oof = cross_val_predict(model, X, y, groups=g, cv=cv)
        m, subset_name = evaluate(y, oof), "cv"
    else:
        ev = va if eval_on == "val" else te
        m, subset_name = evaluate(y[ev], model.predict(X[ev])), eval_on
    print_metrics(f"ML {tag} {subset_name.upper()}", m)

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / SCENARIOS[ML_SCENARIO[tag]].file
    joblib.dump(model, out)
    print(f"[{tag}] model disimpan: {out}")
    save_report(f"ml_{tag}", m, {"model": name, "params": params, "val_f1": None if kecil else score,
                                 "eval_subset": subset_name, "candidates": cands})


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
    try:
        for tag in (["pcd", "raw"] if a.scenario == "both" else [a.scenario]):
            run(tag, df, a.refresh, a.eval_on)
    except NotImplementedError as e:
        sys.exit(f"\nBagian ini belum diisi: {e}")


if __name__ == "__main__":
    main()
