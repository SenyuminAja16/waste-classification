"""Demo klasifikasi sampah: upload foto → pilih skenario → lihat hasil.

Jalankan:  streamlit run demo/app.py
"""
import cv2
import streamlit as st

from core.config import CLASS_INFO, CLASSES, SCENARIOS
from core.image import load_image
from demo.predict import ModelNotFound, model_available, model_path, predict

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️", layout="wide")

st.title("♻️ Klasifikasi Sampah")
st.caption("Final Project PCD & Visi Komputer. Upload foto, pilih skenario, lihat kelas sampahnya.")

# ---------- Sidebar ----------
any_model = any(model_available(i) for i in SCENARIOS)
with st.sidebar:
    st.header("Status model")
    for i, sc in SCENARIOS.items():
        st.write(("✅ " if model_available(i) else "❌ ") + sc.short)
    st.caption("Model dibaca dari folder `models/`.")
    demo_mode = st.checkbox(
        "Mode demo (hasil acak)",
        value=not any_model,
        help="Hanya untuk mencoba tampilan saat model belum dilatih. Hasilnya BUKAN prediksi sungguhan.",
    )

if demo_mode:
    st.warning("Mode demo aktif: hasil di bawah acak dan bukan prediksi model sungguhan.")

# ---------- Input ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("1. Upload foto")
    up = st.file_uploader("Foto sampah (satu objek per foto)", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")
    img = None
    if up is not None:
        img = load_image(up)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Foto yang diupload", use_container_width=True)

with right:
    st.subheader("2. Pilih skenario")
    sid = st.radio("Skenario", list(SCENARIOS), format_func=lambda i: SCENARIOS[i].name,
                   label_visibility="collapsed")
    st.caption(SCENARIOS[sid].desc)
    run = st.button("Klasifikasikan", type="primary", disabled=img is None, use_container_width=True)

    st.subheader("3. Hasil")
    if img is None:
        st.info("Upload foto dulu.")
    elif not run:
        st.info("Pilih skenario lalu klik **Klasifikasikan**.")
    else:
        try:
            with st.spinner("Memproses..."):
                res = predict(sid, img, allow_demo=demo_mode)
        except ModelNotFound:
            st.error(f"Model untuk **{SCENARIOS[sid].short}** belum ada (`models/{model_path(sid).name}`). "
                     "Latih model dulu, atau aktifkan *Mode demo* di sidebar untuk mencoba tampilan.")
        except NotImplementedError as e:
            st.warning(f"Bagian ini belum diisi: {e}")
        except ImportError:
            st.error("Skenario Deep Learning butuh `torch` dan `torchvision`. Install dulu (lihat requirements.txt).")
        else:
            info = CLASS_INFO[res.label]
            st.markdown(f"## {info['emoji']} {info['title']}")
            st.caption(f"Contoh yang termasuk kelas ini: {info['contoh']}")
            st.metric("Keyakinan", f"{res.confidence * 100:.1f}%")
            for c in CLASSES:
                st.write(f"{CLASS_INFO[c]['title']}: {res.probs[c] * 100:.1f}%")
                st.progress(min(max(res.probs[c], 0.0), 1.0))
            if res.is_demo:
                st.caption("⚠️ Hasil demo (acak).")

            if res.pcd_steps:
                with st.expander("Lihat tahap PCD"):
                    s = res.pcd_steps
                    cols = st.columns(4)
                    items = [("Resize 4:3", s["resized"]), ("CLAHE + filtering", s["enhanced"]),
                             ("Mask segmentasi", s["mask"]), ("Hasil segmentasi", s["segmented"])]
                    for col, (cap, im) in zip(cols, items):
                        rgb = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        col.image(rgb, caption=cap, use_container_width=True)
