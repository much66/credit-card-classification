import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Default Kredit",
    page_icon="💳",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .main { background: #f8f9fc; }

    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .result-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .result-accept {
        background: #eafaf1;
        border-left: 6px solid #27ae60;
        color: #1e8449;
    }
    .result-reject {
        background: #fdf2f2;
        border-left: 6px solid #e74c3c;
        color: #c0392b;
    }

    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #ddd;
    }
    .feature-positive { border-left-color: #27ae60; }
    .feature-negative { border-left-color: #e74c3c; }
    .feature-neutral  { border-left-color: #95a5a6; }

    .feature-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .feature-value {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #555;
        margin-top: 0.3rem;
    }

    .prob-bar-bg {
        background: #e8ecf0;
        border-radius: 999px;
        height: 12px;
        width: 100%;
        margin-top: 0.4rem;
    }
    .prob-bar-fill {
        height: 12px;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e8ecf0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('credit_default.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

# ── Mapping dictionaries ──────────────────────────────────────────────────────
Housing_type_map = {
    'House / apartment': 0, 'Rented apartment': 1, 'With parents': 2,
    'Municipal apartment': 3, 'Co-op apartment': 4, 'Office apartment': 5
}
EDUCATION_map = {
    'Higher education': 0, 'Secondary / secondary special': 1,
    'Lower secondary': 2, 'Incomplete higher': 3, 'Academic degree': 4
}
Type_Occupation_map = {
    'Private service staff': 0, 'Laborers': 1, 'Managers': 2,
    'Medicine staff': 3, 'Cooking staff': 4, 'Sales staff': 5,
    'Accountants': 6, 'High skill tech staff': 7, 'Cleaning staff': 8,
    'Drivers': 9, 'Low-skill Laborers': 10, 'IT staff': 11,
    'Waiters/barmen staff': 12, 'Core staff': 13, 'Security staff': 14,
    'HR staff': 15, 'Secretaries': 16, 'Realty agents': 17
}
GENDER_map = {'Laki-laki': 1, 'Perempuan': 0}
Marital_status_map = {
    'Married': 0, 'Single / not married': 1, 'Civil marriage': 2,
    'Separated': 3, 'Widow': 4
}
Type_Income_map = {
    'Commercial associate': 0, 'Pensioner': 1, 'Working': 2, 'State servant': 3
}

# ── Feature analysis helper ───────────────────────────────────────────────────
def analyze_features(gender, occupation, income_type, marital, education,
                     age, housing, years_employed):
    """
    Returns list of dicts:
      {label, value, sentiment (positive/negative/neutral), description}
    Sentiment reflects whether the feature value supports APPROVAL.
    """
    analyses = []

    # --- AGE ---
    if age < 22:
        sent, desc = "negative", (
            "Usia sangat muda (<22 thn) meningkatkan risiko kredit — "
            "riwayat keuangan dan stabilitas pendapatan masih minim."
        )
    elif 22 <= age <= 45:
        sent, desc = "positive", (
            "Usia produktif (22–45 thn) umumnya berkorelasi dengan "
            "pendapatan stabil dan kemampuan bayar yang baik."
        )
    elif 46 <= age <= 60:
        sent, desc = "positive", (
            "Usia matang (46–60 thn) menunjukkan rekam jejak keuangan "
            "panjang, biasanya dinilai positif oleh model kredit."
        )
    else:
        sent, desc = "neutral", (
            "Usia >60 thn bisa berarti mendekati pensiun; "
            "model mempertimbangkan stabilitas pendapatan jangka panjang."
        )
    analyses.append({"label": "Umur", "value": f"{age} tahun",
                     "sentiment": sent, "desc": desc})

    # --- YEARS EMPLOYED ---
    if years_employed == 0:
        sent, desc = "negative", (
            "Belum bekerja atau baru mulai bekerja. "
            "Tidak ada riwayat pendapatan yang bisa dievaluasi."
        )
    elif years_employed < 2:
        sent, desc = "negative", (
            "Masa kerja <2 tahun dianggap kurang stabil — "
            "risiko kehilangan pekerjaan lebih tinggi."
        )
    elif 2 <= years_employed <= 10:
        sent, desc = "positive", (
            "Masa kerja 2–10 tahun menunjukkan stabilitas pekerjaan "
            "yang cukup untuk mendukung pembayaran kredit."
        )
    else:
        sent, desc = "positive", (
            "Masa kerja >10 tahun adalah sinyal kuat stabilitas karier "
            "dan kapasitas pembayaran jangka panjang."
        )
    analyses.append({"label": "Lama Bekerja", "value": f"{years_employed} tahun",
                     "sentiment": sent, "desc": desc})

    # --- EDUCATION ---
    edu_positive = {'Higher education', 'Academic degree', 'Incomplete higher'}
    if education in edu_positive:
        sent, desc = "positive", (
            f"Pendidikan '{education}' umumnya berkorelasi dengan "
            "pendapatan lebih tinggi dan literasi keuangan yang lebih baik."
        )
    else:
        sent, desc = "neutral", (
            f"Pendidikan '{education}' tidak secara otomatis mengurangi "
            "kelayakan, namun pendidikan tinggi sedikit mengungguli."
        )
    analyses.append({"label": "Pendidikan", "value": education,
                     "sentiment": sent, "desc": desc})

    # --- TYPE INCOME ---
    income_positive = {'State servant', 'Commercial associate'}
    income_negative = {'Working'}  # more variable
    if income_type in income_positive:
        sent, desc = "positive", (
            f"Jenis pendapatan '{income_type}' cenderung lebih stabil "
            "dan dapat diprediksi, mengurangi risiko gagal bayar."
        )
    elif income_type == 'Pensioner':
        sent, desc = "neutral", (
            "Pensiunan memiliki pendapatan tetap (pensiun), "
            "namun kapasitas pinjaman jangka panjang lebih terbatas."
        )
    else:
        sent, desc = "neutral", (
            f"'{income_type}' memiliki variabilitas pendapatan lebih tinggi; "
            "model mengevaluasinya bersama fitur lain."
        )
    analyses.append({"label": "Jenis Pendapatan", "value": income_type,
                     "sentiment": sent, "desc": desc})

    # --- OCCUPATION ---
    high_skill = {'High skill tech staff', 'IT staff', 'Managers',
                  'Accountants', 'Medicine staff'}
    low_skill  = {'Low-skill Laborers', 'Cleaning staff',
                  'Cooking staff', 'Waiters/barmen staff'}
    if occupation in high_skill:
        sent, desc = "positive", (
            f"'{occupation}' termasuk kategori pekerjaan high-skill "
            "dengan gaji kompetitif dan permintaan pasar kerja tinggi."
        )
    elif occupation in low_skill:
        sent, desc = "negative", (
            f"'{occupation}' cenderung memiliki gaji lebih rendah dan "
            "fluktuasi pekerjaan lebih tinggi menurut data historis kredit."
        )
    else:
        sent, desc = "neutral", (
            f"'{occupation}' berada di segmen menengah; "
            "model mempertimbangkan kombinasinya dengan faktor lain."
        )
    analyses.append({"label": "Jenis Pekerjaan", "value": occupation,
                     "sentiment": sent, "desc": desc})

    # --- HOUSING ---
    housing_positive = {'House / apartment', 'Co-op apartment'}
    if housing in housing_positive:
        sent, desc = "positive", (
            f"'{housing}' menunjukkan kepemilikan atau aset tetap, "
            "yang menjadi sinyal positif stabilitas finansial."
        )
    elif housing == 'With parents':
        sent, desc = "neutral", (
            "Tinggal bersama orang tua dapat berarti pengeluaran rendah, "
            "namun juga bisa menandakan ketergantungan finansial."
        )
    else:
        sent, desc = "neutral", (
            f"'{housing}' adalah situasi hunian yang umum; "
            "tidak secara kuat memengaruhi keputusan model ke satu arah."
        )
    analyses.append({"label": "Tipe Rumah", "value": housing,
                     "sentiment": sent, "desc": desc})

    # --- MARITAL STATUS ---
    marital_positive = {'Married', 'Civil marriage'}
    marital_negative = {'Separated', 'Widow'}
    if marital in marital_positive:
        sent, desc = "positive", (
            f"Status '{marital}' sering dikaitkan dengan pendapatan ganda "
            "dan tanggung jawab finansial bersama, mengurangi risiko default."
        )
    elif marital in marital_negative:
        sent, desc = "negative", (
            f"Status '{marital}' secara historis berkorelasi sedikit lebih "
            "tinggi dengan tekanan finansial dalam dataset kredit."
        )
    else:
        sent, desc = "neutral", (
            "Status lajang memiliki profil risiko bervariasi tergantung "
            "pendapatan dan faktor lain."
        )
    analyses.append({"label": "Status Pernikahan", "value": marital,
                     "sentiment": sent, "desc": desc})

    # --- GENDER ---
    if gender == 'Perempuan':
        sent, desc = "positive", (
            "Data historis kredit secara global menunjukkan perempuan "
            "memiliki tingkat gagal bayar sedikit lebih rendah."
        )
    else:
        sent, desc = "neutral", (
            "Gender laki-laki memiliki profil risiko yang bervariasi "
            "dan sangat bergantung pada fitur lain."
        )
    analyses.append({"label": "Jenis Kelamin", "value": gender,
                     "sentiment": sent, "desc": desc})

    return analyses


# ── Default values ────────────────────────────────────────────────────────────
DEFAULTS = {
    "input_name":   "",
    "input_gender": "Laki-laki",
    "input_age":    25,
    "input_occupation": "High skill tech staff",
    "input_marital":    "Married",
    "input_family":     2,
    "input_income":     "Working",
    "input_years":      1,
    "input_education":  "Higher education",
    "input_housing":    "House / apartment",
}

# Seed defaults on first load
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Result persists across reruns; None = not yet predicted
if "result" not in st.session_state:
    st.session_state["result"] = None

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("💳 Prediksi Default Loan Customer")
st.caption("Isi formulir di bawah, lalu klik **Analisis** untuk melihat hasil dan penjelasan mendalam.")

col_form, col_result = st.columns([1, 1.1], gap="large")

# ════════════════════════════════════════════════════════════
# LEFT COLUMN — Input form
# ════════════════════════════════════════════════════════════
with col_form:
    st.markdown("### 📋 Data Pemohon")

    st.text_input(
        "Nama Pemohon",
        placeholder="Masukkan nama lengkap...",
        key="input_name",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.radio(
            "Jenis Kelamin",
            ["Laki-laki", "Perempuan"],
            key="input_gender",
        )
    with c2:
        st.number_input(
            "Umur (Tahun)",
            min_value=17, max_value=130,
            key="input_age",
        )

    st.selectbox(
        "Jenis Pekerjaan",
        list(Type_Occupation_map.keys()),
        key="input_occupation",
    )

    c3, c4 = st.columns(2)
    with c3:
        st.selectbox(
            "Status Pernikahan",
            list(Marital_status_map.keys()),
            key="input_marital",
        )
    with c4:
        st.number_input(
            "Jumlah Anggota Keluarga",
            min_value=1, max_value=20,
            key="input_family",
        )

    st.selectbox(
        "Jenis Pendapatan",
        list(Type_Income_map.keys()),
        key="input_income",
    )

    st.number_input(
        "Lama Bekerja (Tahun)",
        min_value=0, max_value=60,
        key="input_years",
    )

    st.selectbox(
        "Pendidikan Terakhir",
        list(EDUCATION_map.keys()),
        key="input_education",
    )

    st.selectbox(
        "Tipe Hunian",
        list(Housing_type_map.keys()),
        key="input_housing",
    )

    st.markdown("")
    analyze_clicked = st.button("🔍 Analisis Kelayakan Kredit")

# ════════════════════════════════════════════════════════════
# Prediction logic (runs when button clicked)
# ════════════════════════════════════════════════════════════
if analyze_clicked:
    # Capture current widget values BEFORE resetting
    name           = st.session_state["input_name"] or "Pemohon"
    gender         = st.session_state["input_gender"]
    age            = st.session_state["input_age"]
    occupation     = st.session_state["input_occupation"]
    marital        = st.session_state["input_marital"]
    income_type    = st.session_state["input_income"]
    years_employed = st.session_state["input_years"]
    education      = st.session_state["input_education"]
    housing        = st.session_state["input_housing"]

    feature_vec = [
        GENDER_map[gender],
        Type_Occupation_map[occupation],
        Type_Income_map[income_type],
        Marital_status_map[marital],
        EDUCATION_map[education],
        age,
        Housing_type_map[housing],
        years_employed,
    ]

    df_input = pd.DataFrame(
        [feature_vec],
        columns=['GENDER','Type_Occupation','Type_Income','Marital_status',
                 'EDUCATION','AGE','Housing_type','YEAR_EMPLOYED']
    )

    scaled = scaler.transform(df_input.values.reshape(1, -1))
    prediction = model.predict(scaled)[0]

    # Try predict_proba; fall back to binary
    try:
        proba = model.predict_proba(scaled)[0]
        prob_approve = float(proba[0]) * 100
        prob_reject  = float(proba[1]) * 100
        has_proba = True
    except Exception:
        prob_approve = 100.0 if prediction == 0 else 0.0
        prob_reject  = 100.0 if prediction == 1 else 0.0
        has_proba = False

    # Feature analyses
    feature_analyses = analyze_features(
        gender, occupation, income_type, marital,
        education, age, housing, years_employed
    )

    # Store result
    st.session_state["result"] = {
        "name": name,
        "prediction": prediction,
        "prob_approve": prob_approve,
        "prob_reject": prob_reject,
        "has_proba": has_proba,
        "features": feature_analyses,
    }

    # ── Reset all widget keys to defaults ──
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

    st.rerun()

# ════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results (persists after re-render)
# ════════════════════════════════════════════════════════════
with col_result:
    st.markdown("### 📊 Hasil Analisis")

    if st.session_state["result"] is None:
        st.info("Hasil analisis akan muncul di sini setelah kamu menekan tombol **Analisis**.", icon="💡")
    else:
        r = st.session_state["result"]
        name       = r["name"]
        prediction = r["prediction"]
        p_approve  = r["prob_approve"]
        p_reject   = r["prob_reject"]
        has_proba  = r["has_proba"]
        features   = r["features"]

        # ── Verdict card ──
        if prediction == 0:
            st.markdown(
                f'<div class="result-card result-accept">'
                f'✅ Pengajuan kredit atas nama <b>{name}</b> — <b>DITERIMA</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card result-reject">'
                f'❌ Pengajuan kredit atas nama <b>{name}</b> — <b>DITOLAK</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability bars ──
        st.markdown('<div class="section-title">🎯 Probabilitas Keputusan</div>', unsafe_allow_html=True)

        approve_color = "#27ae60"
        reject_color  = "#e74c3c"

        st.markdown(
            f"""
            <div style="margin-bottom:0.5rem;">
              <div style="display:flex;justify-content:space-between;font-size:0.9rem;font-weight:600;">
                <span style="color:{approve_color}">✅ Peluang Diterima</span>
                <span style="color:{approve_color}">{p_approve:.1f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p_approve:.1f}%;background:{approve_color};"></div>
              </div>
            </div>
            <div style="margin-bottom:1rem;">
              <div style="display:flex;justify-content:space-between;font-size:0.9rem;font-weight:600;">
                <span style="color:{reject_color}">❌ Peluang Ditolak</span>
                <span style="color:{reject_color}">{p_reject:.1f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p_reject:.1f}%;background:{reject_color};"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not has_proba:
            st.caption("ℹ️ Model tidak mendukung probabilitas; nilai di atas adalah estimasi biner.")

        # ── Feature-level analysis ──
        st.markdown('<div class="section-title">🔍 Analisis Per Faktor</div>', unsafe_allow_html=True)
        st.caption("Setiap faktor dievaluasi berdasarkan pola historis dataset kredit.")

        sentiment_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}

        for feat in features:
            icon  = sentiment_icon[feat["sentiment"]]
            cls   = f"feature-{feat['sentiment']}"
            st.markdown(
                f"""
                <div class="feature-card {cls}">
                  <div class="feature-label">{feat['label']}</div>
                  <div class="feature-value">{icon} {feat['value']}</div>
                  <div class="feature-desc">{feat['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Summary count ──
        pos_count = sum(1 for f in features if f["sentiment"] == "positive")
        neg_count = sum(1 for f in features if f["sentiment"] == "negative")
        neu_count = sum(1 for f in features if f["sentiment"] == "neutral")

        st.markdown(
            f"""
            <div style="background:#f0f4ff;border-radius:12px;padding:1rem 1.25rem;margin-top:0.5rem;font-size:0.9rem;">
              <b>Ringkasan Faktor:</b>&nbsp;&nbsp;
              🟢 <b>{pos_count}</b> positif &nbsp;|&nbsp;
              🔴 <b>{neg_count}</b> negatif &nbsp;|&nbsp;
              🟡 <b>{neu_count}</b> netral
              <br><br>
              <span style="color:#555;">
              Keputusan akhir ditentukan oleh model machine learning yang mempelajari
              pola dari ribuan data historis kredit — bukan semata dari jumlah faktor positif/negatif.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
