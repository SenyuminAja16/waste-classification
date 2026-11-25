# Ablation augmentasi Deep Learning

Run dengan augmentasi: [`kfold_20261125-220047-935497`](../kfold_20261125-220047-935497/summary.md).
Run tanpa augmentasi: [`kfold_20261125-221536-804150`](summary.md).

Kedua run memakai 3 fold per objek, data, seed 42, dan hyperparameter yang sama. Perbedaannya hanya penggunaan flip horizontal, rotasi, serta perubahan kecerahan dan kontras pada data train. Checkpoint dipilih dari validation asli. Skor dihitung pada objek yang ditahan di tiap fold.

| Model | Input | F1 macro tanpa aug. | F1 macro dengan aug. | Δ dengan − tanpa (poin %) | Fold membaik |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | RAW | 0.827 ± 0.041 | 0.848 ± 0.038 | +2.15 ± 3.62 | 2/3 |
| efficientnet_b0 | RAW | 0.881 ± 0.033 | 0.851 ± 0.038 | -3.04 ± 4.76 | 1/3 |
| mobilenet_v2 | PCD | 0.804 ± 0.018 | 0.820 ± 0.049 | +1.62 ± 4.30 | 2/3 |
| efficientnet_b0 | PCD | 0.799 ± 0.035 | 0.806 ± 0.021 | +0.73 ± 5.16 | 1/3 |

Angka ± adalah simpangan baku sampel antar-fold. Selisih dihitung berpasangan pada fold yang sama. Eksperimen ini menggunakan satu seed dan tiga fold; split test final tetap terpisah.
