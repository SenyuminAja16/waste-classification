"""Pengaturan bersama untuk dataset, pcd, ml, dl, dan demo."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
SPLIT_CSV = DATA_DIR / "splits" / "split.csv"
PHOTO_CACHE_DIR = ROOT / "cache" / "photos"
FEATURES_CACHE_DIR = ROOT / "cache" / "features"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = Path(os.environ.get("WASTE_MODELS_DIR", ROOT / "models"))

SHEET_ID = os.environ.get("WASTE_SHEET_ID", "")
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet=Form%20Responses%201"
LOG_SOURCES = [SHEET_CSV_URL.format(SHEET_ID)] if SHEET_ID else []

CLASSES = ["plastik", "kertas", "kardus"]

CLASS_INFO = {
    "plastik": {"title": "Plastik", "emoji": "🧴", "contoh": "botol, gelas cup, wadah/tutup plastik, kresek"},
    "kertas": {"title": "Kertas", "emoji": "📄", "contoh": "HVS, koran, struk, kertas cokelat tipis, amplop, tisu"},
    "kardus": {"title": "Kardus", "emoji": "📦", "contoh": "karton bergelombang: kardus paket, kardus mi/air mineral"},
}

SIZE_LANDSCAPE = (320, 240)
DL_INPUT = 224

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_OBJECTS_FOR_SPLIT = 10


@dataclass(frozen=True)
class Scenario:
    id: int
    name: str
    short: str
    tag: str
    use_pcd: bool
    kind: str
    file: str
    desc: str


SCENARIOS = {
    1: Scenario(id=1, name="Skenario 1: PCD + Machine Learning", short="PCD + ML", tag="pcd",
                use_pcd=True, kind="ml", file="ml_pcd.joblib",
                desc="PCD, fitur warna/tekstur/bentuk, SVM / Random Forest"),
    2: Scenario(id=2, name="Skenario 2: Raw + Machine Learning", short="Raw + ML", tag="raw",
                use_pcd=False, kind="ml", file="ml_raw.joblib",
                desc="Foto mentah, fitur warna/tekstur/bentuk, SVM / Random Forest"),
    3: Scenario(id=3, name="Skenario 3: PCD + Deep Learning", short="PCD + DL", tag="pcd",
                use_pcd=True, kind="dl", file="dl_pcd.pt",
                desc="PCD, MobileNetV2 (transfer learning)"),
    4: Scenario(id=4, name="Skenario 4: Raw + Deep Learning", short="Raw + DL", tag="raw",
                use_pcd=False, kind="dl", file="dl_raw.pt",
                desc="Foto mentah, MobileNetV2 (transfer learning)"),
}
