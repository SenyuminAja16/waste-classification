"""Analisis lengkap bagian ML untuk laporan / PPT progress.

Jalankan SETELAH scripts/train_ml.py. Menghasilkan di results/analysis/:
  summary.md              tabel siap tempel: CV mean±std, metrik, ΔF1, ablation, pasangan error, efisiensi
  cv_models.csv           F1 macro (mean±std) tiap model, tiap skenario
  ablation_fitur.csv      kontribusi grup fitur (warna / tekstur / bentuk)
  errors_<tag>.csv        daftar foto yang salah prediksi (default: val; --include-test untuk menambah test)
  errors_<tag>.png        grid foto yang salah prediksi
  timing.csv              waktu PCD, ekstraksi fitur, inferensi (ms/foto) + ukuran model
  feature_hsv_<tag>.png   histogram HSV rata-rata per kelas
  feature_pca_<tag>.png   sebaran fitur (PCA 2D) per kelas

CV memakai train+val (TIDAK menyentuh test), dibagi per objek, diulang beberapa seed.

Pemakaian:
    python scripts/analyze_ml.py
    python scripts/analyze_ml.py --scenario pcd --seeds 5
"""
import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

from config import CLASSES, MODELS_DIR, RESULTS_DIR, ROOT, SCENARIOS, SEED, SPLIT_CSV
from src import features as F
from src.dataset import build_features
from src.ml import candidates
from src.pcd import run_pcd
from src.utils import load_image, resize_43

ML_SCENARIO = {"pcd": 1, "raw": 2}
LABEL = {"pcd": "PCD+ML", "raw": "Raw+ML"}
OUT = RESULTS_DIR / "analysis"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def repeated_cv(est, X, y, g, seeds, n_splits):
    scores = []
    for s in range(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
        scores += list(cross_val_score(clone(est), X, y, groups=g, cv=cv, scoring="f1_macro", n_jobs=-1))
    return float(np.mean(scores)), float(np.std(scores))


def tuned(name, params):
    est, _ = candidates(SEED)[name]
    return est.set_params(**params)


def group_combos():
    names = list(F.FEATURE_GROUPS)
    for r in range(1, len(names) + 1):
        for c in itertools.combinations(names, r):
            yield c


def cols_for(combo):
    n = max(s.stop for s in F.FEATURE_GROUPS.values())
    idx = np.arange(n)
    return np.concatenate([idx[F.FEATURE_GROUPS[c]] for c in combo])


def plot_hsv(tag, X, y):
    sl = F.FEATURE_GROUPS["warna"]
    col = X[:, sl]
    edges = np.cumsum([0, *F.COLOR_BINS])
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, title, a, b in zip(axes, ["Hue", "Saturation", "Value"], edges[:-1], edges[1:]):
        for ci, c in enumerate(CLASSES):
            ax.plot(col[y == ci][:, a:b].mean(axis=0), label=c, marker="o", ms=3)
        ax.set_title(title); ax.set_xlabel("bin")
    axes[0].set_ylabel("proporsi piksel rata-rata"); axes[0].legend()
    fig.suptitle(f"Histogram HSV rata-rata per kelas ({LABEL[tag]})"); fig.tight_layout()
    fig.savefig(OUT / f"feature_hsv_{tag}.png", dpi=150); plt.close(fig)


def plot_pca(tag, X, y):
    Z = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for ci, c in enumerate(CLASSES):
        ax.scatter(Z[y == ci, 0], Z[y == ci, 1], s=14, label=c, alpha=0.75)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title(f"Sebaran fitur ({LABEL[tag]})")
    fig.tight_layout(); fig.savefig(OUT / f"feature_pca_{tag}.png", dpi=150); plt.close(fig)


def error_analysis(tag, df, X, y, g, model, subsets, n_splits):
    ev = df.subset.isin(subsets).to_numpy()
    if ev.any():
        sub = df[ev].reset_index(drop=True)
        proba = model.predict_proba(X[ev])
        cls = list(model.classes_)
        true = y[ev]
        asal = "+".join(subsets)
    else:  # data kecil (belum ada split): pakai prediksi out-of-fold atas seluruh data
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        proba = cross_val_predict(clone(model), X, y, groups=g, cv=cv, method="predict_proba")
        sub, cls, true, asal = df.reset_index(drop=True), list(range(len(CLASSES))), y, "CV out-of-fold"
    pred = np.array(cls)[proba.argmax(1)]
    wrong = pred != true
    rows = [{"path": sub.path[i], "id_objek": sub.id_objek[i], "posisi": sub.posisi[i], "subset": sub.subset[i],
             "benar": CLASSES[true[i]], "prediksi": CLASSES[int(pred[i])],
             "prob_prediksi": round(float(proba[i].max()), 3),
             "prob_benar": round(float(proba[i][cls.index(true[i])]), 3)} for i in np.where(wrong)[0]]
    pd.DataFrame(rows).to_csv(OUT / f"errors_{tag}.csv", index=False)

    pairs = Counter((r["benar"], r["prediksi"]) for r in rows)
    if rows:
        n = min(12, len(rows))
        cols = min(4, n)
        nr = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nr, cols, figsize=(3 * cols, 2.6 * nr), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), rows[:n]):
            img = cv2.cvtColor(resize_43(load_image(ROOT / r["path"])), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"{r['benar']} → {r['prediksi']} ({r['prob_prediksi']:.2f})", fontsize=9, color="firebrick")
        fig.suptitle(f"Foto salah prediksi di {asal} ({LABEL[tag]}): benar → prediksi (prob.)")
        fig.tight_layout(); fig.savefig(OUT / f"errors_{tag}.png", dpi=130); plt.close(fig)
    return int(wrong.sum()), int(len(true)), pairs


def timing(tag, df, model, n=30):
    sub = df[df.subset.isin(["val", "test"])].head(n)
    imgs = [resize_43(load_image(ROOT / p)) for p in sub.path]
    t_pcd = t_feat = t_inf = 0.0
    for img in imgs:
        t0 = time.perf_counter()
        if tag == "pcd":
            s = run_pcd(img); inp, mask = s["segmented"], s["mask"]
        else:
            inp, mask = img, None
        t1 = time.perf_counter()
        f = F.extract_features(inp, mask)
        t2 = time.perf_counter()
        model.predict_proba(f.reshape(1, -1))
        t3 = time.perf_counter()
        t_pcd += t1 - t0; t_feat += t2 - t1; t_inf += t3 - t2
    k = max(len(imgs), 1)
    path = MODELS_DIR / SCENARIOS[ML_SCENARIO[tag]].file
    return {"skenario": LABEL[tag], "pcd_ms": 1000 * t_pcd / k, "fitur_ms": 1000 * t_feat / k,
            "inferensi_ms": 1000 * t_inf / k, "total_ms": 1000 * (t_pcd + t_feat + t_inf) / k,
            "model_kb": path.stat().st_size / 1024, "n_foto": len(imgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["pcd", "raw", "both"], default="both")
    ap.add_argument("--split", type=Path, default=SPLIT_CSV)
    ap.add_argument("--seeds", type=int, default=5, help="jumlah seed untuk CV berulang")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--include-test", action="store_true",
                    help="ikutkan data test di analisis error (default: hanya val, supaya test tetap bersih)")
    a = ap.parse_args()
    err_subsets = ["val", "test"] if a.include_test else ["val"]
    if not a.split.exists():
        sys.exit(f"{a.split} belum ada. Jalankan dulu: python scripts/prepare_dataset.py")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.split)
    y = df.kelas.map(CLASSES.index).to_numpy()
    g = df.id_objek.to_numpy()
    dev = df.subset.isin(["train", "val"]).to_numpy()
    min_obj = min(len(set(g[dev & (y == c)])) for c in range(len(CLASSES)))
    n_splits = max(2, min(5, min_obj))
    N_SPLIT_ERR = n_splits
    has_groups = bool(getattr(F, "FEATURE_GROUPS", None)) and hasattr(F, "COLOR_BINS")

    tags = ["pcd", "raw"] if a.scenario == "both" else [a.scenario]
    metas, cv_rows, abl_rows, best_cv, timings, err_info = {}, [], [], {}, [], {}
    for tag in tags:
        mp = RESULTS_DIR / f"ml_{tag}.json"
        model_file = MODELS_DIR / SCENARIOS[ML_SCENARIO[tag]].file
        if not mp.exists() or not model_file.exists():
            print(f"[{tag}] dilewati: jalankan dulu scripts/train_ml.py"); continue
        meta = json.loads(mp.read_text())
        if "candidates" not in meta:
            print(f"[{tag}] dilewati: hasil train_ml lama, jalankan ulang scripts/train_ml.py"); continue
        metas[tag] = meta
        X = build_features(df, tag, a.refresh)
        model = joblib.load(model_file)

        print(f"[{tag}] CV berulang ({a.seeds} seed x {n_splits} fold) per model...")
        for name, c in meta["candidates"].items():
            m, s = repeated_cv(tuned(name, c["params"]), X[dev], y[dev], g[dev], a.seeds, n_splits)
            cv_rows.append({"skenario": LABEL[tag], "model": name, "f1_mean": m, "f1_std": s})
            if name == meta["model"]:
                best_cv[tag] = (m, s)

        if has_groups:
            print(f"[{tag}] ablation fitur...")
            est = tuned(meta["model"], meta["candidates"][meta["model"]]["params"])
            for combo in group_combos():
                cc = cols_for(combo)
                m, s = repeated_cv(est, X[dev][:, cc], y[dev], g[dev], a.seeds, n_splits)
                abl_rows.append({"skenario": LABEL[tag], "fitur": " + ".join(combo), "n_fitur": len(cc),
                                 "f1_mean": m, "f1_std": s})
            plot_hsv(tag, X, y)
        else:
            print(f"[{tag}] FEATURE_GROUPS/COLOR_BINS tidak ada di src/features.py → ablation & plot HSV dilewati")
        plot_pca(tag, X, y)

        err_info[tag] = error_analysis(tag, df, X, y, g, model, err_subsets, N_SPLIT_ERR)
        timings.append(timing(tag, df, model))

    if not metas:
        sys.exit("Tidak ada skenario yang bisa dianalisis.")

    # ---------- simpan CSV ----------
    pd.DataFrame(cv_rows).to_csv(OUT / "cv_models.csv", index=False)
    if abl_rows:
        pd.DataFrame(abl_rows).to_csv(OUT / "ablation_fitur.csv", index=False)
    pd.DataFrame(timings).to_csv(OUT / "timing.csv", index=False)

    # ---------- summary.md ----------
    md = ["# Ringkasan analisis ML", "",
          f"CV: train+val, dibagi per objek (StratifiedGroupKFold, {n_splits} fold × {a.seeds} seed). Test tidak dipakai.", ""]

    md += ["## 1. F1 macro hasil CV (mean ± std)", ""]
    names = list(dict.fromkeys(r["model"] for r in cv_rows))
    rows = []
    for n in names:
        row = [n]
        for t in tags:
            r = next((x for x in cv_rows if x["model"] == n and x["skenario"] == LABEL[t]), None)
            row.append(fmt(r["f1_mean"], r["f1_std"]) if r else "-")
        rows.append(row)
    md += [md_table(["Model"] + [LABEL[t] for t in tags if t in metas], rows), ""]

    md += ["## 2. Metrik model terpilih", ""]
    rows = []
    for t, m in metas.items():
        rows.append([LABEL[t], m["model"], m.get("eval_subset", "test"), f"{m['accuracy']:.3f}",
                     f"{m['precision_macro']:.3f}", f"{m['recall_macro']:.3f}", f"{m['f1_macro']:.3f}", m["n_test"]])
    md += [md_table(["Skenario", "Model", "Subset", "Accuracy", "Precision", "Recall", "F1", "n foto"], rows), ""]

    if "pcd" in best_cv and "raw" in best_cv:
        d_cv = best_cv["pcd"][0] - best_cv["raw"][0]
        d_ev = metas["pcd"]["f1_macro"] - metas["raw"]["f1_macro"]
        md += ["## 3. Pengaruh PCD (ΔF1 = F1 PCD+ML − F1 Raw+ML)", "",
               md_table(["Sumber", "ΔF1"], [["CV (train+val)", f"{d_cv:+.3f}"],
                                            [f"Evaluasi ({metas['pcd'].get('eval_subset', 'test')})", f"{d_ev:+.3f}"]]),
               "", "Positif = PCD membantu, negatif = PCD menurunkan performa.", ""]

    if abl_rows:
        md += ["## 4. Ablation fitur (F1 macro CV, mean ± std)", ""]
        combos = list(dict.fromkeys(r["fitur"] for r in abl_rows))
        rows = []
        for cmb in combos:
            row = [cmb]
            for t in tags:
                r = next((x for x in abl_rows if x["fitur"] == cmb and x["skenario"] == LABEL[t]), None)
                row.append(fmt(r["f1_mean"], r["f1_std"]) if r else "-")
            rows.append(row)
        md += [md_table(["Fitur"] + [LABEL[t] for t in tags if t in metas], rows), ""]

    md += [f"## 5. Analisis error ({' + '.join(err_subsets)}, model terpilih)", ""]
    for t, (nw, n, pairs) in err_info.items():
        md.append(f"**{LABEL[t]}**: {nw} salah dari {n} foto.")
        if pairs:
            md += ["", md_table(["Benar", "Diprediksi", "Jumlah"],
                                [[b, p, c] for (b, p), c in pairs.most_common(6)])]
        md.append("")
    md.append("Foto yang salah: `errors_<tag>.png` dan `errors_<tag>.csv`.\n")

    md += ["## 6. Efisiensi (rata-rata per foto)", ""]
    rows = [[t["skenario"], f"{t['pcd_ms']:.1f}", f"{t['fitur_ms']:.1f}", f"{t['inferensi_ms']:.2f}",
             f"{t['total_ms']:.1f}", f"{t['model_kb']:.0f}"] for t in timings]
    md += [md_table(["Skenario", "PCD (ms)", "Fitur (ms)", "Inferensi (ms)", "Total (ms)", "Ukuran model (KB)"], rows),
           "", f"Waktu diukur pada {timings[0]['n_foto']} foto, di mesin ini (CPU)."]

    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nSelesai. Lihat {OUT.relative_to(ROOT)}/summary.md")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        sys.exit(f"\nBagian ini belum diisi: {e}")
