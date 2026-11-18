# Template LaTeX Final Project PCD & Visi Komputer

Template ini disusun untuk laporan final project bergaya artikel jurnal dua kolom.

## Aturan inti project
- 1 kelompok = 3 mahasiswa.
- Dataset utama wajib **data primer**, diambil sendiri **hanya di dalam lingkungan Universitas Brawijaya**.
- Akuisisi dilakukan pada kondisi yang wajar dan berisiko rendah; hindari identitas personal/sensitif sebagai target utama.
- Project wajib menunjukkan proses PCD sebelum/bersama Computer Vision.
- Wajib membandingkan minimal:
  1. PCD + handcrafted feature + Machine Learning,
  2. Raw Image + Deep Learning,
  3. PCD + Deep Learning.
- Wajib ada evaluasi kuantitatif dan analisis error.
- Final harus memiliki demo/prototipe yang berjalan.

## Pembagian peran
1. Mahasiswa 1: Data & PCD.
2. Mahasiswa 2: Feature Extraction & Machine Learning.
3. Mahasiswa 3: Deep Learning/Computer Vision & Integrasi.

Akuisisi data adalah tanggung jawab bersama. Semua anggota wajib memahami seluruh pipeline.

## Timeline
- Progress 1: 4–5 November 2026 — proposal, desain data, pipeline, contoh data primer awal.
- Progress 2: 18–19 November 2026 — dataset mayoritas selesai, PCD, fitur, baseline ML.
- Progress 3: 25–26 November 2026 — dataset final, DL, perbandingan eksperimen, prototype awal.
- Final: 2–3 Desember 2026 — hasil final, evaluasi, analisis, demo, laporan dan source code.

## Compile
Disarankan menggunakan Overleaf atau TeX Live.

Urutan kompilasi lokal:
1. `pdflatex main.tex`
2. `bibtex main`
3. `pdflatex main.tex`
4. `pdflatex main.tex`

Bagian berwarna abu-abu dan diawali `[Isi: ...]` adalah petunjuk yang harus diganti/dihapus pada laporan final.
