import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
# import tensorflow as tf  # Nonaktifkan sementara

# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# Inisialisasi session state
for key in ["image", "image_bytes", "filename", "name"]:
    if key not in st.session_state:
        st.session_state[key] = None if "name" not in key else ""

# ======== Kustomisasi Tema ========
st.sidebar.header("🎨 Kustomisasi Tampilan")
theme_choice = st.sidebar.selectbox("Pilih Mode Tema", ["Default", "Terang", "Gelap"])
font_size = st.sidebar.slider("Ukuran Font (px)", 12, 30, 16)

def set_theme_and_font(theme, font_px):
    if theme == "Terang":
        bg_color, text_color = "#ffffff", "#000000"
        button_bg_color, button_text_color = "#929292", "#ffffff"
    elif theme == "Gelap":
        bg_color, text_color = "#000000", "#ffffff"
        button_bg_color, button_text_color = "#424242", "#000000"
    else:
        bg_color, text_color = "#daffb8", "#000000"
        button_bg_color, button_text_color = "#3d8000", "#ffffff"

    st.markdown(f"""
        <style>
            body, .stApp {{
                background-color: {bg_color};
                color: {text_color};
                font-size: {font_px}px;
            }}
            h1, h2, h3, h4, h5, h6, p, label {{
                color: {text_color};
                font-size: {font_px}px;
            }}
            div.stButton > button {{
                background-color: {button_bg_color};
                color: {button_text_color};
                font-size: {font_px}px;
                padding: 10px 20px;
                border-radius: 8px;
            }}
            div.stButton > button:hover {{
                background-color: #45a049;
                color: #ffffff;
            }}
        </style>
    """, unsafe_allow_html=True)

    return text_color

text_color = set_theme_and_font(theme_choice, font_size)

# ======== Navigasi ========
st.title("DRChecker 👁")
st.markdown("Website Pendeteksi Diabetic Retinopathy")

option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# ======== Halaman Beranda ========
if option == "Beranda":
    st.markdown("<h1>Beranda</h1>", unsafe_allow_html=True)
    st.markdown("<p>Selamat datang di situs Pemeriksaan Diabetic Retinopathy</p>", unsafe_allow_html=True)

    name = st.text_input("Masukkan nama Anda", value=st.session_state["name"])
    if name:
        st.session_state["name"] = name
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Halo, {name}!</p>", unsafe_allow_html=True)

    if st.button("Selesai"):
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'</p>", unsafe_allow_html=True)

# ======== Halaman Periksa Retina ========
elif option == "Periksa Retina":
    st.markdown("<h1>Periksa Retina</h1>", unsafe_allow_html=True)
    st.markdown("<p>Unggah Gambar Scan Retina Anda</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        st.session_state["image_bytes"] = bytes_data
        st.session_state["filename"] = uploaded_file.name
        image = Image.open(io.BytesIO(bytes_data))
        st.session_state["image"] = image

        st.success(f"✅ Gambar '{uploaded_file.name}' berhasil diunggah!")
        st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_container_width=True)
    elif st.session_state["image"] is not None:
        st.info(f"Gambar sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=st.session_state["filename"], use_container_width=True)
    else:
        st.info("Silakan unggah gambar berformat PNG, JPG, atau JPEG.")

# ======== Halaman Hasil Pemeriksaan ========
elif option == "Periksa Retina":
    st.markdown("<h1>Periksa Retina</h1>", unsafe_allow_html=True)
    st.markdown("<p>Unggah Gambar Scan Retina Anda</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        st.session_state["image_bytes"] = bytes_data
        st.session_state["filename"] = uploaded_file.name
        image = Image.open(io.BytesIO(bytes_data))
        st.session_state["image"] = image

        st.success(f"✅ Gambar '{uploaded_file.name}' berhasil diunggah!")
        st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_container_width=True)
    elif st.session_state["image"] is not None:
        st.info(f"Gambar sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=st.session_state["filename"], use_container_width=True)
    else:
        st.info("Silakan unggah gambar berformat PNG, JPG, atau JPEG.")

# ======== Halaman Tim Kami ========
elif option == "Tim Kami":
    st.markdown("<h1>Tim Kami</h1>", unsafe_allow_html=True)
    st.markdown("<h2>El STM</h2>", unsafe_allow_html=True)
    st.markdown("""
        <ul>
            <li>Anggota 1</li>
            <li>Anggota 2</li>
            <li>Anggota 3</li>
        </ul>
    """, unsafe_allow_html=True)

# ======== Footer ========
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>drchecker.web@2025</p>", unsafe_allow_html=True)
