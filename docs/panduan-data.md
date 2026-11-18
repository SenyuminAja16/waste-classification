# Panduan Data

Dokumen ini menjelaskan data apa saja yang tersedia untuk bagian ML dan DL, di mana letaknya, dan bagaimana cara memakainya. Bagian PCD akan dilengkapi setelah pipeline PCD selesai.

## Setup

Di laptop, cukup sekali setelah clone:

1. Jalankan `uv sync`. Perintah ini memasang Python 3.12 dan semua library sesuai `uv.lock`. Kalau mengerjakan DL secara lokal, pakai `uv sync --group dl` supaya torch ikut terpasang.
2. Salin `.env.example` menjadi `.env`, lalu isi `WASTE_SHEET_ID` dengan ID Google Sheet log akuisisi. ID-nya ada di grup, tidak disimpan di repo.
3. Jalankan script dari root repo dengan `uv run python -m <modul>`.

Di Colab, clone repo lalu jalankan `pip install -e .` dari root repo. Simpan ID Sheet di Colab Secrets dengan nama `WASTE_SHEET_ID`, kemudian set ke environment sebelum meng-import `core`:

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
  object_splits.csv
reports/dataset/validation.csv
```

`validation.csv` berisi daftar masalah pada isian Form atau unduhan. Kalau tabelnya kosong, semua data lolos pemeriksaan.

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

Subset setiap objek disimpan di `object_splits.csv` dan tidak pernah berubah. Objek baru hanya mengisi subset yang masih kurang. Dengan begitu, data yang sudah dipakai untuk memilih metode atau model tidak akan pindah ke test.

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

`SPLIT_CSV` masih ada di `core/config.py` supaya import lama tidak error, dan akan dihapus setelah semua script pindah.

## Raw dan PCD

Citra Raw adalah foto yang diputar sesuai EXIF dengan `core.image.load_image`, lalu di-resize dengan `core.image.resize_43`. Citra PCD adalah citra Raw yang sudah melewati `pcd.pipeline.run_pcd`. Pipeline PCD yang sama dipakai untuk semua foto, baik train, val, test, maupun foto baru di demo.
