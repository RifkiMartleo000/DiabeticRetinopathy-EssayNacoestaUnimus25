#!/bin/bash

# Memastikan penggunaan Python 3.11
if command -v python3.11 &>/dev/null; then
    PYTHON_CMD=python3.11
    echo "✅ Menggunakan Python 3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version)
    echo "⚠️ Python 3.11 tidak ditemukan. Menggunakan $PYTHON_VERSION"
    echo "⚠️ Beberapa library mungkin tidak kompatibel jika bukan Python 3.11"
    echo "⚠️ Untuk menginstall Python 3.11, kunjungi: https://www.python.org/downloads/"
else
    echo "❌ Python 3 tidak ditemukan. Silakan install Python 3.11 terlebih dahulu"
    echo "Untuk menginstall Python 3.11, kunjungi: https://www.python.org/downloads/"
    exit 1
fi

# Membuat virtual environment dengan Python 3.11 (opsional)
echo "Membuat virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate || source venv/Scripts/activate

# Memastikan pip sudah versi terbaru
echo "Memperbarui pip ke versi terbaru..."
python -m pip install --upgrade pip

# Membuat folder untuk menyimpan model
mkdir -p models
touch models/README.md
echo "Letakkan file model 64x3-CNN.json dan 64x3-CNN.weights.h5 di sini" > models/README.md

# Menginstall library yang diperlukan untuk aplikasi Streamlit DRChecker
echo "Menginstall dependencies..."
pip install -q tensorflow
pip install -q matplotlib
pip install -q opencv-python-headless
pip install -q streamlit
pip install -q pandas
pip install -q numpy
pip install -q pillow
pip install -q gdown

# Membuat file requirements.txt untuk GitHub dan Streamlit Cloud dengan versi yang kompatibel dengan Python 3.11
echo "# Requirements untuk Python 3.11
tensorflow==2.13.0
matplotlib==3.7.2
opencv-python-headless==4.8.0.76
streamlit==1.25.0
pandas==2.0.3
numpy==1.24.3
pillow==9.5.0
gdown==4.7.1
protobuf==4.23.4" > requirements.txt

# Membuat file runtime.txt untuk Streamlit Cloud
echo "python-3.11.6" > runtime.txt

# Membuat file app.py jika belum ada
if [ ! -f app.py ]; then
  echo "Membuat file app.py untuk aplikasi DRChecker"
  cp paste.txt app.py 2>/dev/null || echo "File paste.txt tidak ditemukan, silakan buat file app.py secara manual"
fi

echo "Setup selesai! File requirements.txt dan runtime.txt telah dibuat untuk deployment di GitHub dan Streamlit Cloud."
echo "URL Google Drive di app.py perlu diperbarui dengan URL yang valid."

# Menampilkan instruksi penggunaan
echo ""
echo "=== INSTRUKSI PENGGUNAAN ==="
echo "1. Aktifkan virtual environment setiap kali menjalankan aplikasi:"
echo "   source venv/bin/activate    # Linux/Mac"
echo "   venv\\Scripts\\activate      # Windows"
echo ""
echo "2. Jalankan aplikasi dengan perintah:"
echo "   streamlit run app.py"
echo ""
echo "3. Untuk deployment ke Streamlit Cloud, pastikan file runtime.txt dan requirements.txt sudah di-push ke GitHub"
