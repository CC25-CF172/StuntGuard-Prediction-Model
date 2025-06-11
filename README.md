# 🩺 Aplikasi Prediksi Stunting - WHO Standards

Aplikasi machine learning untuk memprediksi risiko stunting pada anak berdasarkan WHO Child Growth Standards menggunakan Streamlit UI.

## 📋 Deskripsi

Aplikasi ini menggunakan model neural network yang dilatih dengan data stunting dan mengikuti standar WHO untuk:
- Memprediksi risiko stunting pada anak
- Menghitung Z-score Height-for-Age
- Memberikan klasifikasi berdasarkan WHO standards
- Visualisasi hasil prediksi

## 🚀 Cara Clone dan Setup

### 1. Clone Repository
```bash
git clone git@github.com:CC25-CF172/StuntGuard-Prediction-Model.git
cd stunting-prediction-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pastikan File Data Tersedia
Pastikan file `Stunting-1.xlsx` berada di folder utama project.

## 📊 Cara Menjalankan

### 1. Jalankan Notebook untuk Melatih Model
```bash
jupyter notebook notebook.ipynb
```
**Penting**: Pastikan semua cell di notebook dijalankan sampai selesai untuk menghasilkan:
- `stunting_prediction_model.h5`
- `stunting_preprocessor.joblib`

### 2. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Project

```
stunting-prediction-app/
│
├── notebook.ipynb                    # Notebook untuk training model
├── app.py                           # Aplikasi Streamlit UI
├── requirements.txt                 # Dependencies
├── README.md                        # Dokumentasi
├── Stunting-1.xlsx                  # Dataset (perlu disediakan)
│
└── Generated Files (setelah training):
    ├── stunting_prediction_model.h5     # Model terlatih
    ├── stunting_preprocessor.joblib     # Preprocessor
    ├── stunting_model_architecture.json # Arsitektur model
    └── Various PNG files               # Grafik hasil training
```

## 🔧 Fitur Aplikasi

### 1. Prediksi Tunggal
- Input data anak melalui sidebar
- Hasil prediksi real-time
- Visualisasi Z-score
- Interpretasi hasil berdasarkan WHO

### 2. Prediksi Batch
- Upload file CSV dengan multiple data
- Prediksi otomatis untuk semua data
- Download hasil dalam format CSV
- Ringkasan statistik

### 3. Visualisasi
- Gauge chart untuk Z-score
- Interpretasi warna berdasarkan status
- Grafik distribusi (dari notebook)

## 📊 Format Data Input

### Single Prediction
Input melalui form Streamlit dengan field:
- Jenis Kelamin: M/F
- Usia (bulan): 0-60
- Berat Lahir (kg)
- Panjang Lahir (cm)
- Berat Badan Saat Ini (kg)
- Panjang/Tinggi Badan Saat Ini (cm)
- ASI Eksklusif: Yes/No

### Batch Prediction (CSV)
File CSV harus memiliki kolom:
```
Sex,Age,Birth Weight,Birth Length,Body Weight,Body Length,ASI Eksklusif
M,24,3.2,50,12.5,85,Yes
F,36,2.8,48,11.0,82,No
```

## 🏥 WHO Standards

Aplikasi menggunakan WHO Child Growth Standards:
- **Normal**: Z-score ≥ -2
- **Stunted**: Z-score < -2
- **Severely Stunted**: Z-score < -3

## ⚠️ Catatan Penting

1. **Jalankan Notebook Terlebih Dahulu**: Model harus dilatih dulu sebelum menjalankan aplikasi Streamlit
2. **File Data**: Pastikan `Stunting-1.xlsx` tersedia di folder project
3. **Medical Disclaimer**: Hasil prediksi hanya sebagai alat bantu, tidak menggantikan diagnosis medis profesional

## 🛠️ Troubleshooting

### Error: Model tidak ditemukan
```
FileNotFoundError: stunting_prediction_model.h5
```
**Solusi**: Jalankan notebook.ipynb sampai selesai untuk generate model

### Error: Preprocessor tidak ditemukan
```
FileNotFoundError: stunting_preprocessor.joblib
```
**Solusi**: Pastikan notebook dijalankan sampai selesai

### Error: Dataset tidak ditemukan
```
FileNotFoundError: Stunting-1.xlsx
```
**Solusi**: Pastikan file dataset berada di folder yang sama dengan notebook

## 📞 Support

Jika mengalami masalah, pastikan:
1. Semua dependencies terinstall dengan benar
2. Notebook sudah dijalankan sampai selesai
3. File-file yang diperlukan sudah ter-generate
4. Python version compatible (3.8+)

## 🏃‍♂️ Quick Start

```bash
# 1. Clone dan setup
git clone <repo-url>
cd stunting-prediction-app
pip install -r requirements.txt

# 2. Jalankan notebook untuk training
jupyter notebook notebook.ipynb

# 3. Jalankan aplikasi
streamlit run app.py
```

Happy predicting! 🎉
