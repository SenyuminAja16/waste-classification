# Validasi silang Deep Learning

StratifiedGroupKFold 3 fold pada split train dengan pengelompokan berdasarkan `object_id`.
Sebanyak 39 objek pengembangan PCD selalu berada di train setiap fold.
Checkpoint dipilih dari validation asli (21 objek) dan dinilai pada fold yang ditahan.
Skor dihitung per foto. Angka berikut adalah rata-rata ± simpangan baku sampel antar-fold.
Split test asli tetap terpisah dari validasi silang.

| Model | Skenario | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | RAW | 0.828 ± 0.040 | 0.861 ± 0.021 | 0.824 ± 0.041 | 0.827 ± 0.041 |
| efficientnet_b0 | RAW | 0.879 ± 0.033 | 0.903 ± 0.027 | 0.876 ± 0.033 | 0.881 ± 0.033 |
| mobilenet_v2 | PCD | 0.807 ± 0.017 | 0.820 ± 0.015 | 0.803 ± 0.018 | 0.804 ± 0.018 |
| efficientnet_b0 | PCD | 0.800 ± 0.036 | 0.824 ± 0.053 | 0.798 ± 0.034 | 0.799 ± 0.035 |

Selisih F1 macro berpasangan (PCD − Raw) per fold.

- efficientnet_b0: -0.082 ± 0.063
- mobilenet_v2: -0.023 ± 0.046
