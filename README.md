# Prediksi Persetujuan Kartu Kredit

Ini adalah aplikasi sederhana untuk memprediksi persetujuan pengajuan kartu kredit berdasarkan beberapa faktor seperti jenis pekerjaan, status pernikahan, pendidikan, dan lainnya.

## Deskripsi Proyek

Proyek ini menggunakan teknik Machine Learning untuk membangun model yang dapat memprediksi apakah suatu pengajuan kartu kredit akan disetujui atau ditolak berdasarkan data input pengguna. Model yang digunakan dalam proyek ini adalah RandomForestClassifier, dan aplikasi dibangun menggunakan Streamlit untuk user interface.

## Fitur

- Input data pengguna melalui antarmuka pengguna (UI).
- Prediksi persetujuan kartu kredit berdasarkan data input.
- Tampilan hasil prediksi dengan menggunakan ikon indikator (✅ untuk disetujui, ❌ untuk ditolak).

## Teknologi yang Digunakan

- Python
- Pandas
- Scikit-learn
- Streamlit

## Cara Menggunakan

1. Pastikan Python dan semua library yang dibutuhkan sudah terinstal.
2. Clone repositori ini ke dalam sistem lokal Anda.
   ```bash
   git clone https://github.com/much66/credit-card-classification.git
   ```
3. Masuk ke direktori proyek.
   ```bash
   cd credit-card-classification
   ```
4. Instal semua dependencies yang diperlukan.
   ```bash
   pip install -r requirements.txt
   ```
5. Jalankan aplikasi menggunakan Streamlit.
   ```bash
   streamlit run app.py
   ```
6. Akses aplikasi melalui browser Anda dengan URL yang diberikan oleh Streamlit.
