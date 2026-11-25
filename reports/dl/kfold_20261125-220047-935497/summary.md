# Validasi silang Deep Learning

StratifiedGroupKFold 3 fold pada split train dengan pengelompokan berdasarkan `object_id`.
Sebanyak 39 objek pengembangan PCD selalu berada di train setiap fold.
Checkpoint dipilih dari validation asli (21 objek) dan dinilai pada fold yang ditahan.
Skor dihitung per foto. Angka berikut adalah rata-rata ± simpangan baku sampel antar-fold.
Split test asli tetap terpisah dari validasi silang.

| Model | Skenario | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | RAW | 0.850 ± 0.036 | 0.867 ± 0.038 | 0.846 ± 0.036 | 0.848 ± 0.038 |
| efficientnet_b0 | RAW | 0.851 ± 0.033 | 0.877 ± 0.027 | 0.846 ± 0.040 | 0.851 ± 0.038 |
| mobilenet_v2 | PCD | 0.822 ± 0.049 | 0.832 ± 0.058 | 0.818 ± 0.046 | 0.820 ± 0.049 |
| efficientnet_b0 | PCD | 0.807 ± 0.022 | 0.832 ± 0.001 | 0.805 ± 0.022 | 0.806 ± 0.021 |

Selisih F1 macro berpasangan (PCD − Raw) per fold.

- efficientnet_b0: -0.045 ± 0.043
- mobilenet_v2: -0.028 ± 0.067
