"""Kandidat model Machine Learning.

>>> BAGIAN YANG DIISI SENDIRI (ML) <<<

Kontrak:

    candidates(seed) -> dict[nama_model] = (estimator, param_grid)

    estimator  : objek sklearn (disarankan Pipeline dengan StandardScaler) yang punya predict_proba
    param_grid : dict untuk GridSearchCV, memakai nama parameter Pipeline, mis. {"clf__C": [1, 10]}

ml/scripts/train_ml.py mencoba semua kandidat dengan cross-validation per objek, memilih yang terbaik,
lalu menyimpannya ke models/ml_pcd.joblib dan models/ml_raw.joblib.

Rencana di proposal: SVM (kernel RBF dan linear) dan Random Forest.
Contoh implementasi yang jalan ada di klasifikasi_sampah.ipynb (bagian 6).
"""


def candidates(seed: int = 42) -> dict:
    raise NotImplementedError("Isi candidates() di ml/models.py")
