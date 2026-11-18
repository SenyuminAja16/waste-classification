# Kontrak Data

Dokumen ini menjelaskan data apa saja yang tersedia untuk bagian ML dan DL, di mana letaknya, dan bagaimana cara memakainya. Isinya masih draf dan akan dilengkapi di PR `dataset/prepare` dan `pcd/preprocessing`. Kalau ada yang belum jelas atau kurang, tanyakan ke Farhan.

## Setup

Di laptop, cukup sekali setelah clone:

1. Jalankan `uv sync`. Perintah ini memasang Python 3.12 dan semua library sesuai `uv.lock`. Kalau mengerjakan DL secara lokal, pakai `uv sync --group dl` supaya torch ikut terpasang.
2. Salin `.env.example` menjadi `.env`, lalu isi `WASTE_SHEET_ID` dengan ID Google Sheet log akuisisi. ID-nya ada di grup, tidak disimpan di repo.
3. Jalankan script dari root repo dengan `uv run python -m <modul>`, misalnya `uv run python -m dataset.scripts.prepare_dataset`.

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

## Kondisi saat ini

Untuk sementara, `dataset/scripts/prepare_dataset.py` masih memakai kerangka awal. Script ini mengunduh foto ke `cache/photos/<file_id>.jpg` dan menulis `data/splits/split.csv` dengan kolom berikut:

`file_id, id_objek, posisi, kelas, subtipe, anggota, sesi, lokasi, latar, cahaya, kondisi, path, subset`

Satu baris adalah satu foto. Kolom `subset` berisi `train`, `val`, atau `test`, dan pembagiannya dilakukan per objek sehingga foto atas dan miring dari objek yang sama selalu berada di subset yang sama. Format ini akan berubah di PR `dataset/prepare`. Perubahannya akan disepakati dulu dengan pemakai, terutama `ml/scripts/train_ml.py`.

## Rencana berikutnya

Bagian di bawah ini diisi setelah PR terkait selesai.

### Struktur `data/`

Foto mentah akan disimpan per objek di `data/raw/<kelas>/<id_objek>/`, dengan nama `<id_objek>-atas.jpg` dan `<id_objek>-miring.jpg`, menggantikan `cache/photos/`.

### `metadata.csv`

Satu baris per foto, berisi informasi dari Form dan hasil pemeriksaan otomatis seperti resolusi, skor blur, dan kompleksitas latar.

### Split

Subset setiap objek ditetapkan sekali dan tidak berubah walaupun data bertambah. Objek baru hanya ditambahkan ke subset yang proporsinya masih kurang. Dengan begitu, data yang sudah dipakai untuk memilih metode atau model tidak akan pindah ke test. Selama pengembangan, evaluasi memakai val. Test hanya dipakai sekali untuk hasil akhir.

### Raw dan PCD

Citra Raw adalah foto yang hanya diputar sesuai EXIF lalu di-resize dengan `core.image.load_image` dan `resize_43`. Citra PCD adalah citra Raw yang sudah melewati `pcd.pipeline.run_pcd`. Pipeline PCD yang sama dipakai untuk semua foto, baik train, val, test, maupun foto baru di demo.
