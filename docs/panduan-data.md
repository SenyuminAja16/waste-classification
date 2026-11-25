# Panduan Data

Dokumen ini menjelaskan data apa saja yang tersedia untuk bagian ML dan DL, di mana letaknya, dan bagaimana cara memakainya. Bagian PCD akan dilengkapi setelah pipeline PCD selesai.

## Mulai cepat

Dari root repo:

```bash
git pull
uv sync
```

Salin `.env.example` menjadi `.env`, lalu isi `WASTE_SHEET_ID` dengan ID Sheet dari grup:

```bash
cp .env.example .env
```

Di PowerShell, pakai `Copy-Item .env.example .env`.

Buat data. Proses pertama mengunduh sekitar 290 MB foto:

```bash
uv run python -m dataset.scripts.prepare_dataset
```

Setelah selesai, `data/metadata.csv` siap dipakai. Penjelasan setiap langkah ada di bagian berikutnya.

## Setup

Di laptop, cukup sekali setelah clone:

1. Jalankan `uv sync`. Perintah ini memasang Python 3.12 dan semua library sesuai `uv.lock`. Kalau mengerjakan DL secara lokal, pakai `uv sync --group dl` supaya torch ikut terpasang.
2. Salin `.env.example` menjadi `.env`, lalu isi `WASTE_SHEET_ID` dengan ID Google Sheet log akuisisi. ID-nya ada di grup, tidak disimpan di repo.
3. Jalankan script dari root repo dengan `uv run python -m <modul>`.

Di Colab, clone repo lalu jalankan `pip install -e . pyyaml 'timm>=1.0.15,<2'` dari root repo. Simpan ID Sheet di Colab Secrets dengan nama `WASTE_SHEET_ID`, kemudian set ke environment sebelum meng-import `core`:

```python
import os
from google.colab import userdata

os.environ["WASTE_SHEET_ID"] = userdata.get("WASTE_SHEET_ID")
```

Torch tidak perlu dipasang ulang di Colab karena sudah tersedia.

## Sumber data

Semua data berasal dari Google Form akuisisi. Setiap isian Form adalah satu objek dengan dua foto, tampak atas dan miring. Jawaban Form masuk ke Sheet `log_akuisisi`, dan fotonya tersimpan di Drive. Script membaca Sheet itu langsung, jadi data yang dipakai selalu yang terbaru tanpa perlu mengekspor CSV.

Aturan pengambilan foto ada di `docs/panduan-pengambilan-data.md`.

## Membuat data

```bash
uv run python -m dataset.scripts.prepare_dataset
```

Perintah ini membaca Sheet, mengunduh foto yang belum ada, memeriksa isian Form, lalu membagi objek ke train, val, dan test. Aman dijalankan berulang kali: foto yang sudah ada tidak diunduh lagi dan objek yang sudah dibagi tidak berpindah subset. Pakai `--skip-download` kalau hanya ingin memperbarui tabel tanpa mengunduh, atau `--help` untuk melihat semua opsi.

Hasilnya:

```
data/
  raw/<label>/<object_id>/<object_id>-atas.jpg
  raw/<label>/<object_id>/<object_id>-miring.jpg
  metadata.csv
dataset/object_splits.csv
reports/dataset/summary.md
```

Folder `data/` tidak masuk git. `object_splits.csv` dan `summary.md` masuk git supaya semua anggota memakai split yang sama.

`summary.md` berisi ringkasan dataset: jumlah isian, objek, dan foto, status unduhan, hasil pemeriksaan, serta jumlah objek per split. Kalau ada masalah pada isian Form atau unduhan, daftarnya tercantum di bagian Validasi dan objek tersebut tidak dimasukkan ke split sampai diperbaiki. Foto yang gagal diunduh dicoba ulang otomatis sampai tiga kali, dan perintah yang sama bisa dijalankan lagi untuk mencoba sisanya.

## metadata.csv

Satu baris per foto. Ini tabel yang dipakai untuk EDA, PCD, ML, dan DL.

| Kolom | Isi |
|---|---|
| `object_id` | ID objek, misalnya `PL-F-001` |
| `view` | `atas` atau `miring` |
| `label` | `plastik`, `kertas`, atau `kardus` |
| `subtype` | Sub-tipe dari Form, misalnya `botol` atau `kotak utuh` |
| `member` | Anggota yang memotret |
| `taken_on` | Tanggal pemotretan |
| `location`, `lighting`, `condition`, `note` | Isian Form lainnya |
| `split` | `train`, `val`, atau `test` |
| `raw_path` | Lokasi foto, relatif terhadap folder `data/` |
| `file_id`, `download_status`, `submitted_at` | Informasi unduhan dan waktu isian Form |

`raw_path` relatif terhadap `data/`, bukan root repo. Contoh membaca foto train:

```python
import pandas as pd

from core.config import DATA_DIR, METADATA_CSV
from core.image import load_image

metadata = pd.read_csv(METADATA_CSV)
train = metadata[metadata["split"] == "train"]
image = load_image(DATA_DIR / train["raw_path"].iloc[0])
```

Foto dari objek yang gagal pemeriksaan tetap tercatat, tetapi kolom `split`-nya kosong. Baris seperti ini dilewati saja.

## Split

Pembagian dilakukan per objek, sehingga foto atas dan miring dari objek yang sama selalu berada di subset yang sama. Setiap kelas dibagi 70/15/15: test 15% dibulatkan ke atas, val 15% dibulatkan ke bawah, sisanya train.

Subset setiap objek disimpan di `dataset/object_splits.csv` dan tidak pernah berubah. Objek baru hanya mengisi subset yang masih kurang. Dengan begitu, data yang sudah dipakai untuk memilih metode atau model tidak akan pindah ke test. Perubahan file ini hanya di-commit oleh Farhan sebagai pemegang data. Anggota lain cukup `git pull` dan menjalankan `prepare_dataset`, dan hasilnya akan sama.

Selama pengembangan, evaluasi memakai val. Test hanya dipakai sekali untuk hasil akhir.

## Pindah dari format lama

Kerangka awal memakai `data/splits/split.csv`. File itu tidak dibuat lagi. Script yang masih membacanya perlu disesuaikan:

| Format lama | `metadata.csv` |
|---|---|
| `kelas` | `label` |
| `id_objek` | `object_id` |
| `subset` | `split` |
| `posisi` | `view` |
| `path`, dibaca dengan `ROOT / path` | `raw_path`, dibaca dengan `DATA_DIR / raw_path` |
| `config.SPLIT_CSV` | `config.METADATA_CSV` |

Script yang perlu disesuaikan:

- `ml/scripts/train_ml.py`, `ml/scripts/analyze_ml.py`, dan `build_features` di `ml/features.py` (Jefri)

Training DL sudah membaca `data/metadata.csv`. Setelah script ML disesuaikan, keduanya bisa memakai metadata yang sama.

`SPLIT_CSV` masih ada di `core/config.py` supaya import lama tidak error, dan akan dihapus setelah semua script pindah.

## Raw dan PCD

Citra Raw adalah foto yang diputar sesuai EXIF dengan `core.image.load_image`, lalu di-resize dengan `core.image.resize_43`. Citra PCD adalah citra Raw yang sudah melewati `pcd.pipeline.run_pcd`. Pipeline PCD yang sama dipakai untuk semua foto, baik train, val, test, maupun foto baru di demo.

## Eksperimen Deep Learning

Setelah `data/metadata.csv` dan foto tersedia, edit [`dl/config.yaml`](../dl/config.yaml) lalu jalankan dari root repo:

```bash
uv run --group dl python -m dl.train
```

Atur `model.name` ke `mobilenet_v2`, `efficientnet_b0`, atau `convnextv2_femto`. Gunakan `data.scenario: raw` untuk benchmark Raw, `pcd` untuk PCD saja, atau `both` untuk membandingkan kedua skenario dengan pengaturan yang sama. Ketiga model mendukung bobot pretrained. ConvNeXt V2 Femto dimuat melalui `timm`.

Konfigurasi awal memakai input 224, batch 16, dan 10 epoch untuk head. Ubah `model.dropout`, `data.augmentation`, dan `optimizer` untuk eksperimen regularisasi. `train.epochs_finetune: 0` hanya melatih head. Untuk fine-tuning, naikkan `epochs_finetune` dan tentukan `finetune_blocks` untuk membuka unit terakhir backbone sesuai urutan model. `train.early_stopping_patience` berlaku per fase, dengan nilai `0` untuk menjalankan semua epoch.

`data.augmentation.mode: online` membuat variasi saat foto dimuat. Mode `offline` menyiapkan `variants_per_photo` variasi tetap per foto train sebelum pelatihan dan menyimpannya di memori. Setiap epoch mengambil satu versi per foto sumber dan mengunjungi semua versi dalam satu siklus. Dengan batch size dan jumlah epoch yang sama, jumlah langkah optimizer setara dengan mode online. Foto validation dan test memakai versi asli tanpa augmentasi.

Untuk menyimpan pengaturan beberapa eksperimen, buat salinan YAML dan jalankan `uv run --group dl python -m dl.train --config path/eksperimen.yaml`. Di Colab, gunakan `python -m dl.train`.

| Modul di `dl/` | Tanggung jawab |
|---|---|
| `config.py` | Baca dan validasi YAML, tentukan folder run. |
| `data.py`, `transforms.py` | Baca metadata, siapkan Raw dan PCD, augmentasi train, dan preprocessing untuk inferensi. |
| `model.py` | Bangun arsitektur yang dipilih dan atur blok yang dilatih di tiap fase. |
| `engine.py` | Jalankan training atau evaluasi satu epoch. |
| `logger.py`, `utils.py` | Catat riwayat epoch, atur seed dan device, serta muat bobot. |
| `train.py` | Jalankan kedua fase training dan simpan checkpoint terbaik. |
| `predict.py` | Evaluasi checkpoint terpilih pada split test. |
| `kfold.py` | Ukur kestabilan skor dengan validasi silang per objek pada split train. |

Terminal menampilkan progress per batch dan ringkasan per epoch. Setiap skenario menyimpan checkpoint `best.pt` berdasarkan F1 macro validation, `history.json`, `learning_curves.png`, metrik dan confusion matrix validation, serta salinan konfigurasi dan metadata di folder baru di bawah `reports/dl/`.

Setelah memilih run final, jalankan perintah berikut untuk mengevaluasi `best.pt` pada foto test yang tercatat di metadata run itu. Ganti `RUN_DIR` dengan folder run yang ditampilkan oleh training.

```bash
uv run --group dl python -m dl.predict RUN_DIR
```

Hasil evaluasi disimpan sebagai `test.json` dan `test_cm.png` di folder run. Untuk demo, salin `best.pt` dari run Raw sebagai `models/dl_raw.pt` dan dari run PCD sebagai `models/dl_pcd.pt`. Checkpoint menyimpan arsitektur dan ukuran input, sehingga evaluasi dan demo dapat memuatnya tanpa melihat konfigurasi eksperimen yang sedang aktif.

### Validasi silang DL

Jalankan `uv run --group dl python -m dl.kfold`. Bagian `kfold` di YAML menentukan jumlah fold dan daftar model. Dengan `data.scenario: both`, ketiga fold dipakai bersama oleh MobileNetV2 dan EfficientNet-B0 pada Raw dan PCD. Gunakan hyperparameter yang sudah dipilih untuk eksperimen utama.

Fold dibentuk dari split train berdasarkan `object_id`. Objek yang dipakai untuk anotasi acuan dan eksplorasi PCD selalu masuk train pada setiap fold. Validation asli memilih checkpoint, objek yang ditahan di tiap fold dipakai untuk menghitung skor, dan split test asli tetap terpisah. Varian offline dibuat setelah fold dipisah, hanya dari foto train pada fold tersebut. Rata-rata dan simpangan baku antar-fold ada di `reports/dl/kfold_*/summary.md` dan `summary.csv`. `fold_metrics.csv` memuat skor individual, jumlah gambar train, dan langkah optimizer per epoch. `paired_delta_f1.csv` merangkum selisih PCD − Raw pada fold yang sama.
