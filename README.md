# Klasifikasi Sampah: Plastik, Kertas, Kardus

Final Project PCD & Visi Komputer. Empat skenario dibandingkan:

| # | Skenario | File model |
|---|---|---|
| 1 | PCD + Machine Learning | `models/ml_pcd.joblib` |
| 2 | Raw + Machine Learning | `models/ml_raw.joblib` |
| 3 | PCD + Deep Learning | `models/dl_pcd.pt` |
| 4 | Raw + Deep Learning | `models/dl_raw.pt` |

Project ini berupa **kerangka**. Bagian PCD, fitur, ML, dan DL sengaja dikosongkan untuk diisi sendiri. Sisanya (baca data dari Sheet, training, evaluasi, analisis, app demo) sudah terisi dan siap dipakai.

---

## 1. Struktur folder

```
klasifikasi-sampah/
├── app.py                     UI demo (Streamlit): upload foto → pilih skenario → hasil
├── config.py                  kelas, ukuran input, sumber data, definisi 4 skenario
├── requirements.txt
├── README.md
│
├── src/                       ← MODUL: dipanggil (di-import) oleh file lain
│   ├── pcd.py                 ★ ISI SENDIRI: Pengolahan Citra Digital
│   ├── features.py            ★ ISI SENDIRI: ekstraksi fitur (warna, tekstur, bentuk)
│   ├── ml.py                  ★ ISI SENDIRI: kandidat model Machine Learning
│   ├── dl.py                  ★ ISI SENDIRI: model Deep Learning
│   ├── utils.py               sudah terisi: baca foto, resize 4:3
│   ├── dataset.py             sudah terisi: baca Sheet + unduh foto Drive + cache fitur
│   ├── evaluate.py            sudah terisi: metrik + confusion matrix
│   └── predict.py             sudah terisi: prediksi 4 skenario (dipakai app.py)
│
├── scripts/                   ← PROGRAM: dijalankan lewat terminal
│   ├── prepare_dataset.py     baca Sheet, unduh foto, split train/val/test per objek
│   ├── train_ml.py            latih skenario 1 dan 2
│   ├── train_dl.py            latih skenario 3 dan 4
│   └── analyze_ml.py          analisis ML: CV, ablation, foto salah, efisiensi
│
├── data/
│   ├── splits/split.csv       (dibuat otomatis) daftar foto + label + subset
│   └── processed/             (dibuat otomatis) cache fitur
├── cache/foto/                (dibuat otomatis) foto hasil unduhan dari Drive
├── models/                    hasil training (.joblib dan .pt)
└── results/                   metrik, confusion matrix, hasil analisis
```

### `src/` atau `scripts/`: file mana ditaruh di mana?

- **`src/`** = potongan yang **dipanggil** file lain. Isi PCD, fitur, ML, dan DL **ditaruh di `src/`** (`pcd.py`, `features.py`, `ml.py`, `dl.py`).
- **`scripts/`** = file yang **dijalankan** (`python scripts/...`). Ia hanya merangkai modul-modul `src/`, jadi biasanya tidak perlu diubah.

---

## 2. Empat file yang diisi sendiri

Isinya sekarang hanya kerangka: setiap fungsi memunculkan `NotImplementedError`. Selama **nama fungsi, input, dan outputnya sesuai kontrak** di bawah, isi di dalamnya bebas, dan seluruh app serta script akan tetap jalan.

| File | Fungsi yang harus ada | Input → output |
|---|---|---|
| `src/pcd.py` | `run_pcd(img_bgr)` | citra BGR (sudah di-resize 4:3) → `dict` berisi `"resized"`, `"enhanced"`, `"mask"` (uint8 0/255, objek=255), `"segmented"` (latar dihitamkan) |
| `src/features.py` | `extract_features(img_bgr, mask=None)` | citra BGR + mask atau `None` → vektor 1D `np.ndarray` dengan panjang selalu sama |
| `src/ml.py` | `candidates(seed)` | → `dict`: `{"nama": (estimator_sklearn, param_grid)}`; estimator harus punya `predict_proba` (disarankan Pipeline dengan StandardScaler) |
| `src/dl.py` | `build_model(pretrained)`, `set_trainable(model, mode)`, `to_rgb224(img_bgr)`, `to_tensor(rgb)` | model dengan `len(CLASSES)` keluaran; `mode` = `"head"` atau `"finetune"`; gambar → tensor ternormalisasi |

Penjelasan lengkap kontrak ada di docstring tiap file.

Opsional di `src/features.py`: konstanta `FEATURE_GROUPS` dan `COLOR_BINS` (format di docstring) supaya `analyze_ml.py` bisa membuat ablation per kelompok fitur dan plot HSV. Kalau tidak didefinisikan, bagian itu dilewati.

**Contoh implementasi yang jalan** untuk keempat bagian ini ada di `klasifikasi_sampah.ipynb` (bagian 3, 4, dan 6), kalau mau dijadikan acuan.

---

## 3. Alur kerja

```bash
pip install -r requirements.txt

python scripts/prepare_dataset.py            # 1. baca Sheet, unduh foto, split per objek
python scripts/train_ml.py --eval-on val     # 2. latih skenario 1 dan 2 (test tetap bersih)
python scripts/analyze_ml.py                 # 3. bahan analisis ML → results/analysis/summary.md
pip install torch torchvision                #    (hanya untuk DL)
python scripts/train_dl.py --scenario pcd    # 4. skenario 3
python scripts/train_dl.py --scenario raw    #    skenario 4
python scripts/train_ml.py                   # 5. di AKHIR saja: laporkan skor test
streamlit run app.py                         # 6. demo
```

Urutan pengisian yang disarankan: `pcd.py` → `features.py` → `ml.py` (lalu langkah 2 dan 3) → `dl.py` (lalu langkah 4).

### Sebelum bagian diisi
- `prepare_dataset.py` sudah bisa dijalankan.
- `train_ml.py`, `analyze_ml.py`, `train_dl.py` akan berhenti dengan pesan **"Bagian ini belum diisi: ..."** yang menyebut file mana yang masih kosong.
- `app.py` tetap terbuka dalam **Mode demo** (hasil acak dan diberi peringatan, bukan prediksi sungguhan).

---

## 4. Data

Data dibaca dari sheet **log_akuisisi** (jawaban Google Form). Satu baris = satu objek dengan dua foto (atas dan miring). Foto diunduh dari Drive ke `cache/foto/`.

- Sheet dan folder foto harus di-share **"Anyone with the link: Viewer"**.
- Kalau tidak mau mengubah sharing: download sheet sebagai CSV, lalu jalankan `python scripts/prepare_dataset.py --sumber log_akuisisi.csv`. Foto tetap harus bisa diunduh.
- Beberapa sumber (misal sheet teman) bisa digabung: `--sumber a.csv b.csv`, atau ubah `SUMBER_LOG` di `config.py`.
- Sheet harus punya **minimal 18 kolom** dengan urutan yang sama seperti Form sekarang (kolom dibaca berdasarkan posisi).

**Pembagian data:** 70/15/15 (train/val/test) **per objek**, seimbang per kelas. Foto atas dan miring dari satu objek selalu masuk subset yang sama, supaya tidak terjadi *data leakage*.

**Data masih sedikit:** kalau ada kelas dengan objek < 10 (`MIN_OBJEK_SPLIT` di `config.py`), semua data jadi `train` dan training hanya memakai cross-validation. Metriknya berasal dari prediksi out-of-fold. Deep Learning butuh split lengkap, jadi `train_dl.py` baru bisa jalan setelah data cukup.

---

## 5. Aturan penting

- **Setelah `pcd.py` atau `features.py` diubah**, hitung ulang fitur: `python scripts/train_ml.py --refresh` (dan latih ulang DL). Model lama tidak cocok dengan PCD atau fitur yang baru.
- **Urutan kelas** ada di `config.CLASSES`. Jangan diubah setelah ada model terlatih.
- **Skor test dilihat sekali, di akhir.** Selama tuning pakai `--eval-on val`.
- **PCD yang dipakai saat training harus sama persis dengan yang dipakai saat demo.** Karena itu `run_pcd` dan `extract_features` dipanggil dari satu tempat (`src/`) oleh training maupun app.
- **Tipe model:** model ML disimpan dengan `joblib`, model DL sebagai `state_dict` (`torch.save(model.state_dict(), ...)`).

---

## 6. Masalah umum

| Gejala | Penyebab dan solusi |
|---|---|
| `Bagian ini belum diisi: ...` | Fungsi di file yang disebut masih kosong. Isi sesuai kontrak (bagian 2) |
| `foto gagal diunduh` | Sharing folder Drive belum "Anyone with the link". Atur sharing, atau pakai CSV lokal |
| `Sheet punya N kolom, harusnya minimal 18` | Urutan atau jumlah kolom Form berubah |
| `Tiap kelas butuh minimal 2 objek` | Data belum cukup untuk cross-validation |
| `Deep Learning butuh split train/val/test` | Objek per kelas masih < 10. Tambah data lalu jalankan ulang `prepare_dataset.py` |
| Hasil lama tidak berubah setelah PCD/fitur diganti | Cache fitur lama dipakai. Jalankan `train_ml.py --refresh` |
| App menampilkan "Mode demo" | Model belum ada di `models/`. Latih dulu (langkah 2 dan 4) |
