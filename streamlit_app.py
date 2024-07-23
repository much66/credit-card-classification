import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('default_predict.joblib')

st.title('Prediksi Default Loan Customer')

Name  = st.text_input('Name', placeholder='Input Your Name..')

# Streamlit input widgets
GENDER = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
AGE = st.slider('Umur (Tahun)', 0, 130, 20)
Type_Occupation = st.selectbox(
    "Jenis Pekerjaan",
    ("High skill tech staff", 'Core staff', 'Sales staff', 'Laborers',
     'Cooking staff', 'Managers', 'Accountants', 'Cleaning staff', 'Drivers',
     'Private service staff', 'Low-skill Laborers', 'IT staff',
     'Waiters/barmen staff', 'Medicine staff', 'Security staff', 'HR staff',
     'Secretaries', 'Realty agents'),
    placeholder="Pilih Pekerjaanmu...",
)
Marital_status = st.selectbox(
    "Status Pernikahan",
    ('Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'),
    placeholder="Pilih Jenis Pendapatanmu...",
)
Family_Members = st.slider('Jumlah Anggota Keluarga', 0, 20, 2)
Type_Income = st.selectbox(
    "Jenis Pendapatan",
    ('Commercial associate', 'Pensioner', 'Working', 'State servant'),
    placeholder="Pilih Jenis Pendapatanmu...",
)
YEAR_EMPLOYED = st.slider('Lama Bekerja (Tahun)', 0, 60, 5)
EDUCATION = st.selectbox(
    "Pendidikan",
    ('Higher education', 'Secondary / secondary special', 'Lower secondary',
     'Incomplete higher', 'Academic degree'),
    placeholder="Pilih Pendidikan Terakhirmu...",
)
Housing_type = st.selectbox(
    "Tipe Rumah",
    ('House / apartment', 'With parents', 'Rented apartment',
     'Municipal apartment', 'Co-op apartment', 'Office apartment'),
    placeholder="Pilih Tipe Rumahmu...",
)

# Mapping dictionaries
Housing_type_map = {
    'House / apartment': 0, 'Rented apartment': 1, 'With parents': 2,
    'Municipal apartment': 3, 'Co-op apartment': 4, 'Office apartment': 5
}
EDUCATION_map = {
    'Higher education': 0, 'Secondary / secondary special': 1, 'Lower secondary': 2,
    'Incomplete higher': 3, 'Academic degree': 4
}
Type_Occupation_map = {
    'Private service staff': 0, 'Laborers': 1, 'Managers': 2, 'Medicine staff': 3,
    'Cooking staff': 4, 'Sales staff': 5, 'Accountants': 6, 'High skill tech staff': 7,
    'Cleaning staff': 8, 'Drivers': 9, 'Low-skill Laborers': 10, 'IT staff': 11,
    'Waiters/barmen staff': 12, 'Core staff': 13, 'Security staff': 14, 'HR staff': 15,
    'Secretaries': 16, 'Realty agents': 17
}
GENDER_map = {'Laki-laki': 1, 'Perempuan': 0}
Marital_status_map = {
    'Married': 0, 'Single / not married': 1, 'Civil marriage': 2, 'Separated': 3, 'Widow': 4
}
Type_Income_map = {
    'Commercial associate': 0, 'Pensioner': 1, 'Working': 2, 'State servant': 3
}

df = pd.DataFrame()

if st.button('Prediksi Loan Customer'):
    Name = Name
    Housing_type_value = Housing_type_map[Housing_type]
    EDUCATION_value = EDUCATION_map[EDUCATION]
    Type_Occupation_value = Type_Occupation_map[Type_Occupation]
    GENDER_value = GENDER_map[GENDER]
    Marital_status_value = Marital_status_map[Marital_status]
    Type_Income_value = Type_Income_map[Type_Income]

    card_credit = [GENDER_value, Type_Occupation_value, Type_Income_value, Marital_status_value, EDUCATION_value, AGE, Housing_type_value, YEAR_EMPLOYED]

    df = pd.DataFrame([card_credit], columns=['GENDER', 'Type_Occupation', 'Type_Income', 'Marital_status', 'EDUCATION', 'AGE', 'Housing_type', 'YEAR_EMPLOYED'])
    if not df.empty:
        c_scaler = scaler.transform(df.values.reshape(1, -1))
        
        loan_prediction = model.predict(c_scaler)
        
        if loan_prediction[0] == 1:
            loan_diagnose = f"Pengajuan Kartu Kredit Atas Nama {Name} Ditolak"
        else:
            loan_diagnose = f"Pengajuan Kartu Kredit Atas Nama {Name} Diterima"
        
        if loan_prediction[0] == 1:
            st.error(loan_diagnose, icon="❌")
        else:
            st.success(loan_diagnose, icon="✅")
    else:
        st.error("Harap isi semua form terlebih dahulu.")
