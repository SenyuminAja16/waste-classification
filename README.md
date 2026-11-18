# Klasifikasi Sampah Anorganik (Plastik, Kertas, Kardus)

Final Project PCD & Visi Komputer. Kelompok: Mafaza, Farhan, Jefri.

Empat skenario yang dibandingkan:

| # | Skenario | File model |
|---|---|---|
| 1 | PCD + Machine Learning (SVM / Random Forest) | `models/ml_pcd.joblib` |
| 2 | Raw + Machine Learning | `models/ml_raw.joblib` |
| 3 | PCD + Deep Learning (MobileNetV2) | `models/dl_pcd.pt` |
| 4 | Raw + Deep Learning (MobileNetV2) | `models/dl_raw.pt` |

## Langsung coba (tanpa data)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Selama `models/` kosong, app berjalan dalam **Mode demo**: tampilan bisa dicoba, tapi hasilnya acak dan ada peringatan di layar. Jangan dipresentasikan sebagai hasil asli.

## Alur kerja lengkap

```bash
# 1. Taruh foto di data/raw/<kelas>/<ID-objek>/  (lihat data/README.md)
python scripts/prepare_dataset.py      # validasi + split train/val/test per objek
python scripts/train_ml.py --eval-on val   # skenario 1 dan 2 → models/*.joblib (selama iterasi, test tetap bersih)
python scripts/analyze_ml.py           # bahan slide ML: CV, ablation fitur, error, efisiensi → results/analysis/
pip install torch torchvision          # hanya untuk DL
python scripts/train_dl.py --scenario pcd   # skenario 3 → models/dl_pcd.pt
python scripts/train_dl.py --scenario raw   # skenario 4 → models/dl_raw.pt
streamlit run app.py                   # demo dengan model sungguhan
```

Hasil evaluasi (Accuracy, Precision, Recall, F1 macro, confusion matrix) tersimpan di `results/` sebagai `.json` dan `.png`.
`scripts/analyze_ml.py` menambah di `results/analysis/`: `summary.md` (tabel siap tempel), ablation fitur, grid foto salah prediksi, plot fitur, dan waktu inferensi. Jalankan setelah `train_ml.py`. Skor test baru dilihat di akhir: `python scripts/train_ml.py` (tanpa `--eval-on val`).

## Struktur dan titik ganti

```
app.py                  UI Streamlit
config.py               kelas, ukuran input, definisi skenario, split
src/
  pcd.py                ← GANTI: Pengolahan Citra (Mafaza)
  features.py           ← GANTI: fitur handcrafted (Farhan)
  ml.py                 ← GANTI: kandidat model ML + grid (Farhan)
  dl.py                 ← GANTI: arsitektur & preprocessing DL (Jefri)
  predict.py            inference 4 skenario (biasanya tidak perlu diubah)
  evaluate.py           metrik + confusion matrix
  dataset.py            ekstraksi fitur + cache
scripts/
  prepare_dataset.py    validasi data & split per objek
  train_ml.py           latih skenario 1 dan 2
  analyze_ml.py         analisis ML untuk laporan / PPT
  train_dl.py           latih skenario 3 dan 4
data/  models/  results/
docs/panduan-pengambilan-data.md
report/                 laporan LaTeX
```

Setiap file bertanda **GANTI** punya kontrak di docstring-nya (nama fungsi, input, output). Selama kontrak itu dipenuhi, isi file bebas diganti dan seluruh app serta script training tetap jalan.

**Aturan penting:**
- Setelah `pcd.py` atau `features.py` diubah, latih ulang: `python scripts/train_ml.py --refresh` (dan `train_dl.py` untuk DL). Model lama tidak cocok dengan PCD baru.
- Urutan kelas ada di `config.CLASSES`. Jangan diubah setelah ada model terlatih.
- Test set hanya dipakai sekali di akhir training. Pemilihan model memakai validation.
- Data dibagi **per objek**, jadi kedua foto satu objek selalu ada di subset yang sama (mencegah kebocoran data).
