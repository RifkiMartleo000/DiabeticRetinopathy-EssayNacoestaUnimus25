#!/bin/bash

# Script untuk troubleshooting aplikasi DRChecker
# Gunakan script ini jika mengalami masalah dengan aplikasi

echo "========== DRChecker Troubleshooter =========="
echo "Menjalankan diagnosa sistem..."

# Cek versi Python
echo -e "\n=== Versi Python ==="
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"

# Cek instalasi dan versi library
echo -e "\n=== Versi Library ==="
python -c "
import sys
libraries = ['tensorflow', 'matplotlib', 'cv2', 'streamlit', 'pandas', 'numpy', 'PIL', 'gdown']
status = 0

for lib in libraries:
    try:
        module = __import__(lib)
        if lib == 'PIL':
            print(f'✅ {lib}: {module.__version__}')
        else:
            print(f'✅ {lib}: {module.__version__}')
    except ImportError:
        print(f'❌ {lib}: Tidak terinstall')
        status = 1
    except AttributeError:
        if lib == 'cv2':
            # OpenCV doesn't expose __version__ in the same way
            print(f'✅ {lib}: {module.__version__}')
        else:
            print(f'✅ {lib}: Terinstall (versi tidak tersedia)')

sys.exit(status)
"

# Periksa status exit dari perintah Python sebelumnya
if [ $? -ne 0 ]; then
    echo -e "\n❌ Beberapa library tidak terinstall. Menjalankan setup ulang..."
    ./setup.sh
else
    echo -e "\n✅ Semua library terinstall dengan baik."
fi

# Cek koneksi internet
echo -e "\n=== Koneksi Internet ==="
if ping -c 1 google.com &> /dev/null; then
    echo "✅ Koneksi internet berfungsi"
else
    echo "❌ Tidak ada koneksi internet. Koneksi diperlukan untuk mengunduh model dari Google Drive."
fi

# Cek keberadaan file model
echo -e "\n=== File Model ==="
model_json="models/64x3-CNN.json"
model_weights="models/64x3-CNN.h5"

if [ -f "$model_json" ]; then
    echo "✅ File model JSON ditemukan: $model_json"
else
    echo "❌ File model JSON tidak ditemukan: $model_json"
fi

if [ -f "$model_weights" ]; then
    echo "✅ File model weights ditemukan: $model_weights"
else
    echo "❌ File model weights tidak ditemukan: $model_weights"
fi

echo -e "\n=== Petunjuk Tambahan ==="
echo "1. Jika ada masalah dengan library, coba aktivasi virtual environment:"
echo "   source venv/bin/activate    # Linux/Mac"
echo "   venv\\Scripts\\activate      # Windows"
echo ""
echo "2. Jika file model tidak ditemukan, periksa URL Google Drive di app.py"
echo "   dan pastikan file diatur sebagai dapat diakses publik."
echo ""
echo "3. Untuk menjalankan aplikasi:"
echo "   streamlit run app.py"
echo ""
echo "4. Jika masalah berlanjut, edit file requirements.txt dan coba versi library yang berbeda."
