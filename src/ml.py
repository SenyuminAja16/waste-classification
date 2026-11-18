"""Kandidat model Machine Learning.   >>> TITIK GANTI untuk ML (Farhan) <<<

Kontrak:
    candidates(seed) -> dict[nama] = (estimator, param_grid)

- estimator: objek sklearn (disarankan Pipeline) yang punya predict_proba.
- param_grid: dict untuk GridSearchCV (nama parameter Pipeline, mis. "clf__C").
scripts/train_ml.py mencoba semua kandidat dengan validasi silang per objek, memilih yang terbaik
berdasarkan F1 macro di data validation, lalu mengevaluasinya sekali di data test.
"""
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def candidates(seed: int = 42) -> dict:
    return {
        "svm_rbf": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=seed))]),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
        ),
        "svm_linear": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="linear", probability=True, random_state=seed))]),
            {"clf__C": [0.1, 1, 10]},
        ),
        "random_forest": (
            Pipeline([("clf", RandomForestClassifier(random_state=seed))]),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10, 20]},
        ),
    }
