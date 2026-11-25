# Perbandingan augmentasi offline, online, dan tanpa augmentasi

[Tanpa augmentasi](../kfold_20261125-221536-804150/summary.md) · [Augmentasi online](../kfold_20261125-220047-935497/summary.md) · [Augmentasi offline](summary.md)

Tiga kondisi memakai objek train/validation, seed, pembagian fold, dan hyperparameter yang sama. Varian offline dibuat tetap di memori sebelum model belajar dari masing-masing fold, setelah preprocessing skenario Raw atau PCD. Setiap foto train punya dua varian tambahan, dengan jenis transformasi yang sama dengan augmentasi online. Pool berisi 510–516 gambar dari 170–172 foto asli per fold. Setiap epoch mengambil satu versi per foto sumber, sehingga tiga kondisi sama-sama menjalankan 11 langkah optimizer per epoch. Validation dan fold yang ditahan memakai foto asli tanpa augmentasi. Split test final tetap terpisah.

| Model | Input | Tanpa augmentasi | Online | Offline | Δ offline − online (poin %) |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | RAW | 0.827 ± 0.041 | 0.848 ± 0.038 | 0.856 ± 0.030 | +0.78 ± 1.58 |
| efficientnet_b0 | RAW | 0.881 ± 0.033 | 0.851 ± 0.038 | 0.852 ± 0.023 | +0.13 ± 4.78 |
| mobilenet_v2 | PCD | 0.804 ± 0.018 | 0.820 ± 0.049 | 0.812 ± 0.013 | -0.83 ± 5.51 |
| efficientnet_b0 | PCD | 0.799 ± 0.035 | 0.806 ± 0.021 | 0.753 ± 0.073 | -5.25 ± 9.10 |

Tabel melaporkan rata-rata ± simpangan baku sampel F1 macro antar-fold. Selisih dihitung berpasangan pada fold yang sama.
