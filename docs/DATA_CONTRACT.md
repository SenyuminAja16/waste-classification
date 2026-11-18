# Folder data

Isi `data/raw/` dengan foto hasil akuisisi. Aturan lengkap ada di `docs/panduan-pengambilan-data.md`.

```
data/
  raw/
    plastik/   PL-F-001/  (tepat 2 foto: atas dulu, lalu miring)
    kertas/    KT-M-003/
    kardus/    KD-J-010/
    hardset/   (opsional, tidak dipakai training)
  splits/      split.csv   (dibuat oleh scripts/prepare_dataset.py)
  processed/   cache fitur (dibuat otomatis)
```

- Satu folder per objek, nama folder = ID objek `KELAS-ANGGOTA-NOMOR`.
- **Nama file asli jangan diubah** dan jangan diedit. Urutan atas/miring dibaca dari waktu EXIF.
- Foto asli tidak ikut di-commit ke git (lihat `.gitignore`). Simpan di Drive.
- Setelah data masuk: `python scripts/prepare_dataset.py`. Script ini memeriksa folder (tepat 2 foto, nama ID valid) dan membagi data train/val/test **per objek**.
