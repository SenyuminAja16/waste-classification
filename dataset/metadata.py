"""Validasi log akuisisi dan pengukuran kondisi setiap foto."""

import re

import pandas as pd

LABEL_CODES = {"PL": "plastik", "KT": "kertas", "KD": "kardus"}
MEMBER_CODES = {"M": "Mafaza", "F": "Farhan", "J": "Jefri"}
OBJECT_ID_PATTERN = re.compile(r"^(PL|KT|KD)-(M|F|J)-\d{3}$")
DOWNLOAD_OK = {"downloaded", "exists"}
ISSUE_COLUMNS = ["object_id", "view", "check", "detail"]


def validate(log: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    """Periksa aturan data; satu baris per masalah, tabel kosong berarti data lolos."""
    issues = [*check_log(log), *check_downloads(photos)]
    table = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    return table.sort_values(["object_id", "check"]).reset_index(drop=True)


def check_log(log: pd.DataFrame):
    """Periksa isian Form: ID dobel, format ID, kecocokan kode, ID foto miring, dan tanggal."""
    counts = log["object_id"].value_counts()
    for object_id, count in counts[counts > 1].items():
        yield issue(object_id, "duplicate_id", f"diisi {count} kali")

    for row in log.itertuples(index=False):
        match = OBJECT_ID_PATTERN.match(row.object_id)
        if match:
            label_code, member_code = match.groups()
            if LABEL_CODES[label_code] != row.label:
                yield issue(row.object_id, "label_mismatch", f"kode {label_code} tetapi Kelas = {row.label}")
            if MEMBER_CODES[member_code] != row.member:
                yield issue(row.object_id, "member_mismatch", f"kode {member_code} tetapi Anggota = {row.member}")
        else:
            yield issue(row.object_id, "invalid_id", "format harus KELAS-ANGGOTA-NNN, misalnya PL-F-001")

        if row.tilted_photo_id.lower() != f"{row.object_id}-miring".lower():
            yield issue(row.object_id, "tilted_id", f"ID foto miring = {row.tilted_photo_id}")
        if pd.isna(row.taken_on) or pd.isna(row.submitted_at):
            yield issue(row.object_id, "invalid_date", "tanggal atau timestamp tidak terbaca")


def check_downloads(photos: pd.DataFrame):
    """Laporkan foto yang tidak berhasil diunduh."""
    failed = photos[~photos["download_status"].isin(DOWNLOAD_OK)]
    for row in failed.itertuples(index=False):
        yield issue(row.object_id, "download_failed", row.download_status, view=row.view)


def issue(object_id: str, check: str, detail: str, view: str = "") -> dict:
    return {"object_id": object_id, "view": view, "check": check, "detail": detail}
