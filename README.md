# Klasifikasi Sampah: Plastik, Kertas, Kardus

Final Project PCD & Visi Komputer. Empat skenario dibandingkan:

| # | Skenario | File model |
|---|---|---|
| 1 | PCD + Machine Learning | `models/ml_pcd.joblib` |
| 2 | Raw + Machine Learning | `models/ml_raw.joblib` |
| 3 | PCD + Deep Learning | `models/dl_pcd.pt` |
| 4 | Raw + Deep Learning | `models/dl_raw.pt` |

## Struktur proyek

| Folder | Isi |
|---|---|
| `core/` | Konfigurasi kelas dan skenario, pembacaan gambar, serta metrik evaluasi. |
| `dataset/` | Akuisisi, validasi, pembagian per objek, dan notebook eksplorasi data. |
| `pcd/` | Pipeline pengolahan citra, anotasi mask, dan evaluasi segmentasi. |
| `ml/` | Ekstraksi fitur, kandidat model, serta script training dan analisis ML. |
| `dl/` | Konfigurasi YAML, training MobileNetV2, EfficientNet-B0, atau ConvNeXt V2 Femto, dan evaluasi checkpoint. |
| `demo/` | Aplikasi Streamlit dan inferensi. |
| `reports/` | Hasil evaluasi dataset, PCD, dan eksperimen model. |
| `report/` | Naskah laporan LaTeX. |

## Menyiapkan dataset

Jalankan dari root proyek. Salin `.env.example` ke `.env` jika belum ada, lalu isi `WASTE_SHEET_ID` dengan ID Google Sheet `log_akuisisi` dari grup.

```bash
uv sync
uv run python -m dataset.scripts.prepare_dataset
```

Script menyimpan foto di `data/raw/` dan metadata di `data/metadata.csv`. Foto atas dan miring dari setiap objek masuk ke subset yang sama. Panduan sumber data, pembagian, dan alternatif CSV lokal ada di [panduan data](docs/panduan-data.md).

## Eksperimen Deep Learning

Edit [`dl/config.yaml`](dl/config.yaml) untuk memilih model, skenario `raw`, `pcd`, atau `both`, dan hyperparameter. Konfigurasi awal melatih head MobileNetV2 pada Raw. Evaluasi checkpoint terpilih pada test menggunakan folder run yang dicetak saat training.

```bash
uv run --group dl python -m dl.train
uv run --group dl python -m dl.predict reports/dl/NAMA_FOLDER_RUN
```

Training memilih `best.pt` berdasarkan F1 macro validation. Folder run juga berisi riwayat epoch, laporan metrik, dan confusion matrix. Lihat [panduan eksperimen DL](docs/panduan-data.md#eksperimen-deep-learning) untuk opsi konfigurasi dan pemakaian di Colab.

## Machine Learning

`ml/features.py` dan `ml/models.py` berisi kontrak ekstraksi fitur dan kandidat model yang perlu diimplementasikan. Script di `ml/scripts/` masih membaca format split sebelumnya. Pemetaan ke `data/metadata.csv` tersedia di [panduan migrasi](docs/panduan-data.md#pindah-dari-format-lama).

## Demo

Untuk memakai hasil eksperimen DL, salin `best.pt` dari run yang dipilih ke `models/dl_raw.pt` atau `models/dl_pcd.pt` sesuai skenarionya.

```bash
uv run --group dl streamlit run demo/app.py
```

Demo memakai urutan kelas dari `core/config.py` dan pipeline PCD yang sama dengan training. Saat checkpoint belum tersedia, mode demo menampilkan hasil simulasi.
