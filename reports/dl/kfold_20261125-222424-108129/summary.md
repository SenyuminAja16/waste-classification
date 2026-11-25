# Validasi silang Deep Learning

StratifiedGroupKFold 3 fold pada split train dengan pengelompokan berdasarkan `object_id`.
Sebanyak 39 objek pengembangan PCD selalu berada di train setiap fold.
Checkpoint dipilih dari validation asli (21 objek) dan dinilai pada fold yang ditahan.
Skor dihitung per foto. Angka berikut adalah rata-rata ± simpangan baku sampel antar-fold.
Split test asli tetap terpisah dari validasi silang.

| Model | Skenario | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | RAW | 0.857 ± 0.030 | 0.873 ± 0.035 | 0.854 ± 0.029 | 0.856 ± 0.030 |
| efficientnet_b0 | RAW | 0.850 ± 0.022 | 0.865 ± 0.029 | 0.849 ± 0.023 | 0.852 ± 0.023 |
| mobilenet_v2 | PCD | 0.814 ± 0.011 | 0.820 ± 0.020 | 0.810 ± 0.011 | 0.812 ± 0.013 |
| efficientnet_b0 | PCD | 0.756 ± 0.073 | 0.787 ± 0.080 | 0.757 ± 0.069 | 0.753 ± 0.073 |

Selisih F1 macro berpasangan (PCD − Raw) per fold.

- efficientnet_b0: -0.098 ± 0.054
- mobilenet_v2: -0.044 ± 0.021
