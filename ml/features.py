"""Ekstraksi fitur handcrafted.

>>> BAGIAN YANG DIISI SENDIRI (Fitur) <<<

Kontrak:

    extract_features(img_bgr, mask=None) -> np.ndarray 1D (float), panjangnya SELALU sama
        img_bgr : citra BGR (skenario PCD: citra `segmented`; skenario Raw: citra hasil resize)
        mask    : uint8 0/255 dari PCD, atau None (skenario Raw: seluruh citra dianggap objek)

Opsional, untuk ablation dan plot di scripts/analyze_ml.py (kalau tidak didefinisikan, bagian itu dilewati):

    FEATURE_GROUPS = {"warna": slice(0, 32), "tekstur": slice(32, 47), "bentuk": slice(47, 61)}
    COLOR_BINS = (16, 8, 8)     # panjang histogram H, S, V di dalam grup "warna"

Rencana di proposal: warna (histogram HSV), tekstur (GLCM + LBP), bentuk (kontur + Hu Moments, dll.).
Contoh implementasi yang jalan ada di klasifikasi_sampah.ipynb (bagian 4).

Setelah isi file ini diubah, latih ulang: python scripts/train_ml.py --refresh
"""


def extract_features(img_bgr, mask=None):
    raise NotImplementedError("Isi extract_features() di src/features.py")
