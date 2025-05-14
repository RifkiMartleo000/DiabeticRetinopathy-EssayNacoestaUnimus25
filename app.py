import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import tempfile

# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# Model path - default value which will be updated when model is loaded
MODEL_PATH = 'model'

@st.cache_resource
def load_model():
    """Load the saved model"""
    try:
        # Attempt to load the model from the specified path
        with open(os.path.join(MODEL_PATH, "64x3-CNN.json"), "r") as json_file:
            model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights(os.path.join(MODEL_PATH, "64x3-CNN_weights.h5"))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['acc']
            )
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # For demonstration purposes - creating a dummy model
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        dummy_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['acc']
        )
        return dummy_model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Resize image to target size
    image = image.resize(target_size)
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # Handle RGBA
        img_array = img_array[:, :, :3]
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions for batch prediction
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_dr(image):
    """Make prediction using the model"""
    model = load_model()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Get class with highest probability
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][pred_class] * 100
    
    # Map class index to label
    labels = ["Normal", "Diabetic Retinopathy"]
    result = labels[pred_class]
    
    return result, confidence

# Inisialisasi session state
for key in ["image", "image_bytes", "filename", "name", "prediction", "confidence"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["name", "prediction", "confidence"] else ""

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

    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3>Apa itu Diabetic Retinopathy?</h3>
        <p>Diabetic Retinopathy adalah komplikasi diabetes yang mempengaruhi mata. Kondisi ini terjadi ketika tingginya kadar gula darah merusak pembuluh darah di retina — lapisan jaringan sensitif cahaya di bagian belakang mata.</p>
        
        <h3>Mengapa Deteksi Dini Penting?</h3>
        <p>Deteksi dini Diabetic Retinopathy sangat penting untuk mencegah kebutaan. Dengan pemeriksaan rutin dan penanganan yang tepat, risiko kehilangan penglihatan dapat dikurangi hingga 95%.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mulai Pemeriksaan"):
        st.session_state["page"] = "Periksa Retina"
        st.experimental_rerun()

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
    
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3>Panduan Unggah Gambar</h3>
        <ul>
            <li>Pastikan gambar memiliki pencahayaan yang baik</li>
            <li>Gambar retina harus terlihat jelas</li>
            <li>Format yang didukung: JPG, JPEG, dan PNG</li>
            <li>Ukuran maksimum file: 5MB</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ======== Halaman Hasil Pemeriksaan ========
elif option == "Hasil Pemeriksaan":
    st.markdown("<h1>Hasil Pemeriksaan</h1>", unsafe_allow_html=True)

    if st.session_state["image"] is None:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state["image"], caption=f"Gambar: {st.session_state['filename']}", use_container_width=True)
        
        with col2:
            st.markdown("<h3>Informasi Pasien</h3>", unsafe_allow_html=True)
            if st.session_state["name"]:
                st.markdown(f"**Nama:** {st.session_state['name']}")
            else:
                st.markdown("**Nama:** Tidak diketahui")
            
            st.markdown(f"**File Gambar:** {st.session_state['filename']}")
            
            if st.button("🔍 Prediksi"):
                with st.spinner('Menganalisis gambar...'):
                    try:
                        # Make prediction
                        result, confidence = predict_dr(st.session_state["image"])
                        st.session_state["prediction"] = result
                        st.session_state["confidence"] = confidence
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            
            # Display previous prediction if available
            if st.session_state["prediction"]:
                st.markdown("<h3>Hasil Analisis</h3>", unsafe_allow_html=True)
                
                if st.session_state["prediction"] == "Normal":
                    st.success(f"Diagnosis: {st.session_state['prediction']}")
                else:
                    st.error(f"Diagnosis: {st.session_state['prediction']}")
                
                st.markdown(f"Tingkat kepercayaan: {st.session_state['confidence']:.2f}%")
                
                # Provide recommendations based on prediction
                st.markdown("<h3>Rekomendasi</h3>", unsafe_allow_html=True)
                if st.session_state["prediction"] == "Normal":
                    st.markdown("""
                    ✅ Hasil pemeriksaan menunjukkan retina normal.
                    
                    **Rekomendasi:**
                    - Lakukan pemeriksaan rutin setiap tahun
                    - Jaga kadar gula darah tetap normal
                    - Pertahankan gaya hidup sehat
                    """)
                else:
                    st.markdown("""
                    ⚠️ Hasil pemeriksaan menunjukkan indikasi Diabetic Retinopathy.
                    
                    **Rekomendasi:**
                    - Segera konsultasikan dengan dokter mata
                    - Kontrol kadar gula darah dengan ketat
                    - Ikuti pengobatan sesuai anjuran dokter
                    - Lakukan pemeriksaan rutin lebih sering (3-6 bulan sekali)
                    """)

                # Add disclaimer
                st.info("**Disclaimer:** Hasil ini hanya prediksi dan tidak menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter untuk hasil yang akurat.")

# ======== Halaman Tim Kami ========
elif option == "Tim Kami":
    st.markdown("<h1>Tim Kami</h1>", unsafe_allow_html=True)
    st.markdown("<h2>El STM</h2>", unsafe_allow_html=True)
    
    # Display team info in a more organized way
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
            <h3>Anggota 1</h3>
            <p>Machine Learning Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
            <h3>Anggota 2</h3>
            <p>Data Scientist</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
            <h3>Anggota 3</h3>
            <p>Web Developer</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3>Tentang Proyek</h3>", unsafe_allow_html=True)
    st.markdown("""
    DRChecker adalah proyek yang dikembangkan untuk membantu mendeteksi Diabetic Retinopathy secara dini menggunakan teknologi Machine Learning dan Artificial Intelligence. 
    
    Model yang digunakan adalah Convolutional Neural Network (CNN) yang dilatih menggunakan dataset gambar retina. Aplikasi ini bertujuan untuk memudahkan screening awal dan tidak menggantikan diagnosis profesional medis.
    """)

# ======== Footer ========
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>drchecker.web@2025</p>", unsafe_allow_html=True)
