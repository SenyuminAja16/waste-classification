"""PCD (Pengolahan Citra Digital).

>>> BAGIAN YANG DIISI SENDIRI (PCD) <<<

Kontrak:

    run_pcd(img_bgr) -> dict dengan kunci:
        "resized"   : citra BGR (input apa adanya atau setelah resize tambahan)
        "enhanced"  : citra BGR setelah perbaikan kontras / filtering
        "mask"      : uint8 (0 atau 255), objek = 255, ukuran sama dengan citra
        "segmented" : citra BGR dengan latar dihitamkan (dipakai skenario 1 dan 3)

    Input `img_bgr` sudah di-resize seragam oleh src.utils.resize_43 (rasio 4:3).

Rencana di proposal: CLAHE (kanal L pada LAB), filtering (Gaussian / median),
segmentasi Otsu atau adaptive threshold + morfologi + kontur terbesar.
Contoh implementasi yang jalan ada di klasifikasi_sampah.ipynb (bagian 3).

Setelah isi file ini diubah, latih ulang: python scripts/train_ml.py --refresh
"""


def run_pcd(img_bgr):
    raise NotImplementedError("Isi run_pcd() di src/pcd.py")
