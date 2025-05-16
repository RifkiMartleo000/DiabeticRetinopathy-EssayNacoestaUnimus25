#!/bin/bash

# Membuat folder untuk menyimpan model
mkdir -p models
touch models/README.md
echo "Letakkan file model 64x3-CNN.json dan 64x3-CNN.weights.h5 di sini" > models/README.md

# Menginstall library yang diperlukan untuk aplikasi Streamlit DRChecker
pip install -q tensorflow
pip install -q matplotlib
pip install -q opencv-python-headless
pip install -q streamlit
pip install -q pandas
pip install -q numpy
pip install -q pillow
pip install -q gdown

# Membuat file requirements.txt untuk GitHub dan Streamlit Cloud
echo "tensorflow==2.13.0
matplotlib==3.7.2
opencv-python-headless==4.8.0.76
streamlit==1.25.0
pandas==2.0.3
numpy==1.24.3
pillow==9.5.0
gdown==4.7.1" > requirements.txt

# Membuat file app.py jika belum ada
if [ ! -f app.py ]; then
  echo "Membuat file app.py untuk aplikasi DRChecker"
  cp paste.txt app.py 2>/dev/null || echo "File paste.txt tidak ditemukan, silakan buat file app.py secara manual"
fi

echo "Setup selesai! File requirements.txt telah dibuat untuk deployment di GitHub dan Streamlit Cloud."
echo "URL Google Drive di app.py perlu diperbarui dengan URL yang valid."
