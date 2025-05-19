import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow import lite
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
from PIL import Image
import io
import os


# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# Inisialisasi session state
for key in ["image", "image_bytes", "filename", "name", "prediction_result", "confidence", "font_size"]:
    if key not in st.session_state:
        if key == "font_size":
            st.session_state[key] = 16  # Default font size
        else:
            st.session_state[key] = None if key not in ["name", "prediction_result", "confidence"] else ""

# ======== Kustomisasi Tema ========
st.sidebar.header("🎨 Kustomisasi Tampilan")
theme_choice = st.sidebar.selectbox("Pilih Mode Tema", ["Default", "Terang", "Gelap"])

# Font size adjustment with more detailed options
st.sidebar.subheader("Pengaturan Font")
font_size_option = st.sidebar.radio("Pilih Ukuran Font", ["Kecil", "Sedang", "Besar", "Kustom"])

if font_size_option == "Kecil":
    st.session_state.font_size = 14
elif font_size_option == "Sedang":
    st.session_state.font_size = 16
elif font_size_option == "Besar":
    st.session_state.font_size = 20
elif font_size_option == "Kustom":
    st.session_state.font_size = st.sidebar.slider("Ukuran Font Kustom (px)", 10, 30, st.session_state.font_size)

font_size = st.session_state.font_size

# Ukuran heading berdasarkan font dasar
h1_size = font_size * 2
h2_size = font_size * 1.6
h3_size = font_size * 1.3

def set_theme_and_font(theme, font_px):
    if theme == "Terang":
        bg_color, text_color = "#ffffff", "#000000"
        button_bg_color, button_text_color = "#929292", "#ffffff"
    elif theme == "Gelap":
        bg_color, text_color = "#000000", "#ffffff"
        button_bg_color, button_text_color = "#424242", "#000000"
    else:
        bg_color, text_color = "#18421b", "#ffffff"
        button_bg_color, button_text_color = "#017b0a", "#ffffff"

    st.markdown(f"""
        <style>
            body, .stApp {{
                background-color: {bg_color};
                color: {text_color};
            }}
            .stMarkdown, .stText, p, li, div.row-widget.stRadio div {{
                font-size: {font_px}px !important;
            }}
            h1, .stMarkdown h1 {{
                color: {text_color};
                font-size: {h1_size}px !important;
            }}
            h2, .stMarkdown h2 {{
                color: {text_color};
                font-size: {h2_size}px !important;
            }}
            h3, .stMarkdown h3 {{
                color: {text_color};
                font-size: {h3_size}px !important;
            }}
            .stTextInput input, .stTextArea textarea, .stNumberInput input {{
                font-size: {font_px}px !important;
            }}
            .stSelectbox div div div, .stMultiselect div div div {{
                font-size: {font_px}px !important;
            }}
            div.stButton > button {{
                background-color: {button_bg_color};
                color: {button_text_color};
                font-size: {font_px}px !important;
                padding: 10px 20px;
                border-radius: 8px;
            }}
            div.stButton > button:hover {{
                background-color: #45a049;
                color: #ffffff;
            }}
            .stRadio label, .stCheckbox label, .stSlider label {{
                font-size: {font_px}px !important;
            }}
            .stInfo, .stWarning, .stError, .stSuccess {{
                font-size: {font_px}px !important;
            }}
            .stDataFrame, .stTable {{
                font-size: {font_px-2}px !important;
            }}
            .stSidebar .stRadio label, .stSidebar .stCheckbox label {{
                font-size: {font_px-2}px !important;
            }}
            .stPlotBaseGlideElement {{
                font-size: {font_px}px !important;
            }}
            footer {{
                font-size: {font_px-2}px !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    return text_color

text_color = set_theme_and_font(theme_choice, font_size)

# ======== Fungsi untuk menampilkan teks dengan ukuran font yang dapat disesuaikan ========
def custom_text(text, tag="p", style=""):
    st.markdown(f"<{tag} style='font-size:{font_size}px; {style}'>{text}</{tag}>", unsafe_allow_html=True)

# ======== Navigasi ========
st.title("DRChecker 👁")
custom_text("Website Pendeteksi Diabetic Retinopathy")

option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# ======== Fungsi Prediksi ========
def predict_class(image_data):
    """
    Fungsi untuk memprediksi kelas gambar dari data gambar yang sudah dibuka
    
    Args:
        image_data: PIL Image object yang sudah dibuka
    
    Returns:
        tuple (hasil prediksi, persentase keyakinan)
    """
    # Konversi PIL Image ke array numpy
    np_image = np.array(image_data)
    
    # Konversi ke RGB jika gambar berwarna
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        RGBImg = np_image  # Sudah RGB
    elif len(np_image.shape) == 3 and np_image.shape[2] == 4:
        # Gambar dengan Alpha channel
        RGBImg = np_image[:, :, :3]
    else:
        # Gambar grayscale - konversi ke 3 channel
        RGBImg = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    
    # Resize gambar
    RGBImg = cv2.resize(RGBImg, (224, 224))

    # Load arsitektur model
    try:
        with open('64x3-CNN.json', 'r') as json_file:
            json_model = json_file.read()
        new_model = tf.keras.models.model_from_json(json_model)

        # Load bobot ke model yang sama
        new_model.load_weights("64x3-CNN.weights.h5")
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return "Error", 0

    # Tampilkan gambar yang akan diprediksi
    plt.figure(figsize=(3, 3))
    plt.imshow(RGBImg)
    plt.axis('off')
    st.pyplot(plt)

    # Normalisasi dan prediksi
    image = np.array(RGBImg) / 255.0
    predict = new_model.predict(np.array([image]))
    predicted_class = np.argmax(predict, axis=1)[0]
    confidence = predict[0][predicted_class]

    # Tampilkan hasil prediksi
    if predicted_class == 1:
        return "No DR", confidence * 100
    else:
        return "DR", confidence * 100


# ======== Halaman Beranda ========
if option == "Beranda":
    st.markdown(f"<h1 style='font-size:{h1_size}px;'>Beranda</h1>", unsafe_allow_html=True)
    custom_text("Selamat datang di situs Pemeriksaan Diabetic Retinopathy")

    st.markdown("""
    ## Apa itu Diabetic Retinopathy?
    Diabetic Retinopathy (DR) adalah komplikasi diabetes yang memengaruhi mata. Kondisi ini terjadi ketika kadar gula darah tinggi merusak pembuluh darah di retina - jaringan peka cahaya di bagian belakang mata.
    
    ## Tentang Aplikasi Ini
    DRChecker adalah aplikasi yang dirancang untuk membantu deteksi awal Diabetic Retinopathy melalui analisis gambar retina. Aplikasi ini menggunakan kecerdasan buatan untuk mengenali pola-pola pada gambar retina yang mungkin mengindikasikan adanya kondisi DR.
    
    ## Cara Penggunaan
    1. Masukkan nama Anda di bawah ini
    2. Pilih "Periksa Retina" di menu samping
    3. Unggah gambar retina Anda
    4. Lihat hasil analisis di halaman "Hasil Pemeriksaan"
    """)
    
    name = st.text_input("Masukkan nama Anda", value=st.session_state["name"])
    if name:
        st.session_state["name"] = name
        custom_text(f"Halo, {name}!")

    if st.button("Selesai"):
        custom_text("Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'")

# ======== Halaman Periksa Retina ========
elif option == "Periksa Retina":
    st.markdown(f"<h1 style='font-size:{h1_size}px;'>Periksa Retina</h1>", unsafe_allow_html=True)
    custom_text("Unggah Gambar Scan Retina Anda")

    st.info("""
    **Petunjuk:**
    1. Unggah gambar retina yang jelas dan tidak buram
    2. Format gambar yang didukung: PNG, JPG, JPEG
    3. Gambar sebaiknya menampilkan area retina secara lengkap
    4. Pastikan gambar dengan pencahayaan yang cukup
    """)
    
    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        st.session_state["image_bytes"] = bytes_data
        st.session_state["filename"] = uploaded_file.name
        
        try:
            image = Image.open(io.BytesIO(bytes_data))
            st.session_state["image"] = image

            st.success(f"✅ Gambar '{uploaded_file.name}' berhasil diunggah!")
            st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_container_width=True)
        except Exception as e:
            st.error(f"Error saat membuka gambar: {str(e)}")
            st.session_state["image"] = None
            
    elif st.session_state["image"] is not None:
        st.info(f"Gambar sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=st.session_state["filename"], use_container_width=True)
    else:
        st.info("Silakan unggah gambar berformat PNG, JPG, atau JPEG.")

# ======== Halaman Hasil Pemeriksaan ========
elif option == "Hasil Pemeriksaan":
    st.markdown(f"<h1 style='font-size:{h1_size}px;'>Hasil Pemeriksaan</h1>", unsafe_allow_html=True)

    if st.session_state["image"] is None:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        st.image(st.session_state["image"], caption=f"Gambar yang akan diproses: {st.session_state['filename']}", use_container_width=True)

        if st.button("🔍 Prediksi"):
            with st.spinner("Sedang memproses gambar..."):
                try:
                    # Langsung gunakan gambar dari session state
                    result, confidence = predict_class(st.session_state["image"])
                    
                    # Simpan hasil prediksi ke session state
                    st.session_state["prediction_result"] = result
                    st.session_state["confidence"] = confidence
                    
                    # Tampilkan hasil
                    st.markdown(f"<h2 style='font-size:{h2_size}px;'>Hasil Deteksi</h2>", unsafe_allow_html=True)
                    
                    if result == "DR":
                        st.warning(f"⚠️ Terdeteksi indikasi Diabetic Retinopathy dengan tingkat kepercayaan {confidence:.2f}%")
                        st.markdown("""
                        ### Rekomendasi:
                        - Segera konsultasikan dengan dokter mata
                        - Kontrol gula darah secara rutin
                        - Jaga pola makan sehat
                        """)
                    else:
                        st.success(f"✅ Tidak terdeteksi indikasi Diabetic Retinopathy dengan tingkat kepercayaan {confidence:.2f}%")
                        st.markdown("""
                        ### Rekomendasi:
                        - Tetap lakukan pemeriksaan mata rutin setahun sekali
                        - Jaga pola hidup sehat
                        """)
                    
                except Exception as e:
                    st.error(f"Error saat melakukan prediksi: {str(e)}")
            
# ======== Halaman Tim Kami ========
elif option == "Tim Kami":
    st.markdown(f"<h1 style='font-size:{h1_size}px;'>Tim Kami</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:{h2_size}px;'>El STM</h2>", unsafe_allow_html=True)
    
    # Menggunakan ukuran font dari pengaturan
    team_html = f"""
    <ul style='font-size:{font_size}px;'>
        <li>Fayzul Haq Mahardika Basunjaya</li>
        <li>Kevin Surya Prayoga Wibowo</li>
        <li>Rifki Martleo Alfiansyah</li>
    </ul>
    """
    st.markdown(team_html, unsafe_allow_html=True)

# ======== Footer ========
st.markdown("---")
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color}; font-size:{font_size-2}px;'>drchecker.web@2025</p>", unsafe_allow_html=True)

# Display current font settings
with st.sidebar.expander("Info Pengaturan Font Saat Ini"):
    st.write(f"Ukuran Font: {font_size}px")
    st.write(f"Ukuran Heading 1: {h1_size}px")
    st.write(f"Ukuran Heading 2: {h2_size}px")
