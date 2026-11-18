"""Konfigurasi bersama: dipakai oleh app, script training, dan inference.

PENTING: urutan CLASSES dan ukuran input harus SAMA saat training dan saat demo.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_CSV = DATA_DIR / "splits" / "split.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = Path(os.environ.get("SAMPAH_MODELS_DIR", ROOT / "models"))

# Urutan label (indeks DL = urutan ini). Nama folder di data/raw/ juga memakai nama ini.
CLASSES = ["plastik", "kertas", "kardus"]
CLASS_CODE = {"plastik": "PL", "kertas": "KT", "kardus": "KD"}  # untuk ID objek PL-F-001

CLASS_INFO = {
    "plastik": {"title": "Plastik", "emoji": "🧴",
                "contoh": "botol, gelas cup, wadah/tutup plastik, kresek"},
    "kertas": {"title": "Kertas", "emoji": "📄",
               "contoh": "HVS, koran, struk, kertas cokelat tipis, amplop, tisu"},
    "kardus": {"title": "Kardus", "emoji": "📦",
               "contoh": "karton bergelombang: kardus paket, kardus mi/air mineral"},
}

SIZE_LANDSCAPE = (320, 240)  # (lebar, tinggi) rasio 4:3; foto portrait dibalik jadi (240, 320)
DL_INPUT = 224               # input MobileNetV2

# Pembagian data per objek (bukan per foto). Val = floor, test = ceil, sisanya train.
VAL_FRAC, TEST_FRAC = 0.15, 0.15
SEED = 42


@dataclass(frozen=True)
class Scenario:
    id: int
    name: str
    short: str
    tag: str       # "pcd" atau "raw" (dipakai untuk nama file & argumen script)
    use_pcd: bool
    kind: str      # "ml" atau "dl"
    file: str      # nama file model di folder models/
    desc: str


SCENARIOS = {
    1: Scenario(1, "Skenario 1: PCD + Machine Learning", "PCD + ML", "pcd", True, "ml", "ml_pcd.joblib",
                "PCD → fitur warna/tekstur/bentuk → SVM / Random Forest"),
    2: Scenario(2, "Skenario 2: Raw + Machine Learning", "Raw + ML", "raw", False, "ml", "ml_raw.joblib",
                "Foto mentah → fitur warna/tekstur/bentuk → SVM / Random Forest"),
    3: Scenario(3, "Skenario 3: PCD + Deep Learning", "PCD + DL", "pcd", True, "dl", "dl_pcd.pt",
                "PCD → MobileNetV2 (transfer learning)"),
    4: Scenario(4, "Skenario 4: Raw + Deep Learning", "Raw + DL", "raw", False, "dl", "dl_raw.pt",
                "Foto mentah → MobileNetV2 (transfer learning)"),
}
