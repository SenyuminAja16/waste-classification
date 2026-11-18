"""Membaca log akuisisi dari Google Sheet dan mengunduh foto ke data/raw."""

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from core.config import DATA_DIR, RAW_DIR

LOG_COLUMNS = {
    "Timestamp": "submitted_at",
    "ID foto ATAS": "top_photo_id",
    "ID foto MIRING": "tilted_photo_id",
    "Anggota": "member",
    "Tanggal": "taken_on",
    "Kelas": "label",
    "Lokasi": "location",
    "Cahaya": "lighting",
    "Kondisi objek": "condition",
    "Catatan": "note",
}
SUBTYPE_PREFIX = "Sub-tipe"
UPLOAD_PREFIX = "Masukkan Foto"
VIEWS = ("atas", "miring")
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={}"
DOWNLOAD_RETRIES = 3


def read_log(source) -> pd.DataFrame:
    """Baca log akuisisi (URL Sheet atau path CSV) menjadi satu baris per isian Form."""
    raw = pd.read_csv(source, dtype=str).fillna("")
    missing = [column for column in LOG_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di log: {missing}")

    subtypes = raw.filter(regex=f"^{SUBTYPE_PREFIX}")
    uploads = raw.filter(regex=f"^{UPLOAD_PREFIX}")
    if uploads.shape[1] != len(VIEWS):
        raise ValueError(f"Kolom upload foto harus {len(VIEWS)}, ditemukan {uploads.shape[1]}")

    log = raw[list(LOG_COLUMNS)].rename(columns=LOG_COLUMNS).apply(lambda column: column.str.strip())
    log["object_id"] = log["top_photo_id"].str.replace(r"-atas$", "", regex=True, flags=re.IGNORECASE)
    log["label"] = log["label"].str.lower()
    log["subtype"] = subtypes.apply(lambda row: next((v.strip() for v in row if v.strip()), ""), axis=1).str.lower()
    log["lighting"] = log["lighting"].str.split(" (", regex=False).str[0]
    log["submitted_at"] = pd.to_datetime(log["submitted_at"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    log["taken_on"] = pd.to_datetime(log["taken_on"], format="%m/%d/%Y", errors="coerce").dt.date
    for view, column in zip(VIEWS, uploads.columns):
        log[f"url_{view}"] = uploads[column].str.strip()
    return log


def to_photo_table(log: pd.DataFrame) -> pd.DataFrame:
    """Pecah setiap isian menjadi dua baris foto; kalau ID objek dobel, pakai isian terbaru."""
    latest = log.sort_values("submitted_at").drop_duplicates("object_id", keep="last")
    url_columns = [f"url_{view}" for view in VIEWS]
    photos = pd.concat(
        [latest.assign(view=view, url=latest[f"url_{view}"]) for view in VIEWS],
        ignore_index=True,
    ).drop(columns=url_columns)
    photos["file_id"] = photos["url"].map(drive_file_id)
    photos["raw_path"] = [
        raw_path(label, object_id, view)
        for label, object_id, view in zip(photos["label"], photos["object_id"], photos["view"])
    ]
    return photos.sort_values(["object_id", "view"]).reset_index(drop=True)


def drive_file_id(url: str) -> str:
    """Ambil ID file dari link Drive (format open?id=... atau /d/...)."""
    match = re.search(r"[?&]id=([\w-]+)|/d/([\w-]+)", url)
    return (match.group(1) or match.group(2)) if match else ""


def raw_path(label: str, object_id: str, view: str) -> str:
    """Lokasi foto mentah relatif terhadap folder data, misalnya raw/plastik/PL-F-001/PL-F-001-atas.jpg."""
    path = RAW_DIR / label / object_id / f"{object_id}-{view}.jpg"
    return path.relative_to(DATA_DIR).as_posix()


def download_photos(photos: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Unduh foto yang belum ada di data/raw dan kembalikan tabel dengan kolom download_status."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(download_photo, photos["file_id"], photos["raw_path"]))
    return photos.assign(download_status=statuses)


def download_photo(file_id: str, relative_path: str, retries: int = DOWNLOAD_RETRIES) -> str:
    """Unduh satu foto dan coba ulang kalau gagal; hasilnya exists, downloaded, no_link, failed, atau not_image."""
    target = DATA_DIR / relative_path
    if target.exists():
        return "exists"
    if not file_id:
        return "no_link"

    target.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    for attempt in range(retries):
        if attempt:
            # Jeda 2 lalu 4 detik memberi waktu saat Drive membatasi unduhan yang terlalu rapat.
            time.sleep(2**attempt)
        status = fetch_photo(file_id, target)
        if status == "downloaded":
            break
    return status


def fetch_photo(file_id: str, target) -> str:
    """Satu kali percobaan unduh; file hanya disimpan kalau isinya gambar yang valid."""
    partial = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(DOWNLOAD_URL.format(file_id), timeout=60) as response:
            partial.write_bytes(response.read())
    except OSError:
        partial.unlink(missing_ok=True)
        return "failed"

    try:
        with Image.open(partial) as image:
            image.verify()
    except (OSError, SyntaxError):
        partial.unlink(missing_ok=True)
        return "not_image"

    partial.replace(target)
    return "downloaded"
