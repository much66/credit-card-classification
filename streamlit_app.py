import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f0f0f0; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.08); }
.result-approved { background: linear-gradient(135deg, rgba(6,214,160,0.15), rgba(6,214,160,0.05)); border: 1px solid rgba(6,214,160,0.4); border-radius: 20px; padding: 32px; text-align: center; }
.result-rejected { background: linear-gradient(135deg, rgba(255,92,92,0.15), rgba(255,92,92,0.05)); border: 1px solid rgba(255,92,92,0.4); border-radius: 20px; padding: 32px; text-align: center; }
.metric-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px 20px; text-align: center; }
.metric-label { font-size: 12px; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 22px; font-weight: 700; color: #fff; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge-green { background: rgba(6,214,160,0.2); color: #06D6A0; border: 1px solid rgba(6,214,160,0.4); }
.badge-red   { background: rgba(255,92,92,0.2);  color: #FF5C5C; border: 1px solid rgba(255,92,92,0.4); }
.section-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: rgba(255,255,255,0.4); margin-bottom: 12px; }
.history-row { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.feature-row { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.feature-bar-bg { background: rgba(255,255,255,0.08); border-radius: 99px; height: 8px; margin-top: 6px; }
.feature-bar-fill-green { background: linear-gradient(90deg, #06D6A0, #0aa876); border-radius: 99px; height: 8px; }
.feature-bar-fill-red   { background: linear-gradient(90deg, #FF5C5C, #cc3333); border-radius: 99px; height: 8px; }
.feature-bar-fill-yellow{ background: linear-gradient(90deg, #FFD166, #e6b800); border-radius: 99px; height: 8px; }
div.stButton > button { width: 100%; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 12px; padding: 14px 24px; font-size: 16px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif; }
label { color: rgba(255,255,255,0.75) !important; font-size: 14px !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    model  = joblib.load('credit_default.pkl')
    return scaler, model

scaler, model = load_models()

# ── Session state ─────────────────────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history = []
if "last_result"    not in st.session_state: st.session_state.last_result = None
# Form values — persist across reruns
if "Name"           not in st.session_state: st.session_state.Name = ""
if "GENDER"         not in st.session_state: st.session_state.GENDER = "Laki-laki"
if "AGE"            not in st.session_state: st.session_state.AGE = 30
if "Marital_status" not in st.session_state: st.session_state.Marital_status = "Married"
if "Family_Members" not in st.session_state: st.session_state.Family_Members = 3
if "Housing_type"   not in st.session_state: st.session_state.Housing_type = "House / apartment"
if "Type_Occupation"not in st.session_state: st.session_state.Type_Occupation = "High skill tech staff"
if "Type_Income"    not in st.session_state: st.session_state.Type_Income = "Working"
if "YEAR_EMPLOYED"  not in st.session_state: st.session_state.YEAR_EMPLOYED = 3
if "EDUCATION"      not in st.session_state: st.session_state.EDUCATION = "Higher education"

# ── Mappings ──────────────────────────────────────────────────────────────────
Housing_type_map    = {'House / apartment':0,'Rented apartment':1,'With parents':2,'Municipal apartment':3,'Co-op apartment':4,'Office apartment':5}
EDUCATION_map       = {'Higher education':0,'Secondary / secondary special':1,'Lower secondary':2,'Incomplete higher':3,'Academic degree':4}
Type_Occupation_map = {'Private service staff':0,'Laborers':1,'Managers':2,'Medicine staff':3,'Cooking staff':4,'Sales staff':5,'Accountants':6,'High skill tech staff':7,'Cleaning staff':8,'Drivers':9,'Low-skill Laborers':10,'IT staff':11,'Waiters/barmen staff':12,'Core staff':13,'Security staff':14,'HR staff':15,'Secretaries':16,'Realty agents':17}
GENDER_map          = {'Laki-laki':1,'Perempuan':0}
Marital_status_map  = {'Married':0,'Single / not married':1,'Civil marriage':2,'Separated':3,'Widow':4}
Type_Income_map     = {'Commercial associate':0,'Pensioner':1,'Working':2,'State servant':3}

# ── Feature importance dari Random Forest ─────────────────────────────────────
feature_names = ['GENDER','Type_Occupation','Type_Income','Marital_status','EDUCATION','AGE','Housing_type','YEAR_EMPLOYED']
feature_labels = {
    'GENDER':          'Jenis Kelamin',
    'Type_Occupation': 'Jenis Pekerjaan',
    'Type_Income':     'Jenis Pendapatan',
    'Marital_status':  'Status Pernikahan',
    'EDUCATION':       'Pendidikan',
    'AGE':             'Umur',
    'Housing_type':    'Tipe Hunian',
    'YEAR_EMPLOYED':   'Lama Bekerja',
}

# Konteks penilaian per fitur (untuk penjelasan naratif)
def get_feature_context(feature, value_raw, value_encoded):
    contexts = {
        'AGE': {
            'value_display': f"{value_raw} tahun",
            'analysis': "Usia produktif (25-55 thn) dinilai lebih stabil" if 25 <= value_raw <= 55
                        else ("Usia terlalu muda, riwayat kredit belum terbentuk" if value_raw < 25
                              else "Usia senior, pertimbangkan masa pensiun"),
            'signal': 'positive' if 25 <= value_raw <= 55 else 'negative',
        },
        'YEAR_EMPLOYED': {
            'value_display': f"{value_raw} tahun",
            'analysis': "Pengalaman kerja panjang → stabilitas finansial tinggi" if value_raw >= 5
                        else ("Pengalaman kerja memadai" if value_raw >= 2
                              else "Pengalaman kerja singkat, risiko instabilitas pendapatan"),
            'signal': 'positive' if value_raw >= 5 else ('neutral' if value_raw >= 2 else 'negative'),
        },
        'EDUCATION': {
            'value_display': value_raw,
            'analysis': {
                'Higher education':             "Pendidikan tinggi → kapasitas repayment lebih baik",
                'Academic degree':              "Gelar akademik → potensi pendapatan tinggi",
                'Secondary / secondary special':"Pendidikan menengah, cukup memenuhi syarat",
                'Incomplete higher':            "Pendidikan tidak selesai, poin berkurang",
                'Lower secondary':              "Pendidikan dasar, risiko lebih tinggi",
            }.get(value_raw, ""),
            'signal': 'positive' if value_raw in ['Higher education','Academic degree']
                      else ('neutral' if value_raw == 'Secondary / secondary special' else 'negative'),
        },
        'Type_Income': {
            'value_display': value_raw,
            'analysis': {
                'State servant':        "PNS/pegawai negeri → pendapatan paling stabil",
                'Commercial associate': "Karyawan swasta → pendapatan stabil",
                'Working':              "Karyawan umum, memenuhi syarat",
                'Pensioner':            "Pensiunan → pendapatan tetap tapi terbatas",
            }.get(value_raw, ""),
            'signal': 'positive' if value_raw in ['State servant','Commercial associate']
                      else ('neutral' if value_raw == 'Working' else 'negative'),
        },
        'Type_Occupation': {
            'value_display': value_raw,
            'analysis': {
                'Managers':             "Manajer → pendapatan & stabilitas tinggi",
                'High skill tech staff':"Tenaga ahli teknologi → prospek karir baik",
                'Accountants':          "Akuntan → profesi stabil",
                'IT staff':             "Staf IT → permintaan pasar tinggi",
                'Medicine staff':       "Tenaga medis → profesi prestisius",
                'Laborers':             "Buruh → pendapatan cenderung tidak menentu",
                'Low-skill Laborers':   "Buruh tidak terampil → risiko tinggi",
                'Cleaning staff':       "Staf kebersihan → pendapatan terbatas",
            }.get(value_raw, f"Pekerjaan: {value_raw}"),
            'signal': 'positive' if value_raw in ['Managers','High skill tech staff','Accountants','IT staff','Medicine staff']
                      else ('negative' if value_raw in ['Low-skill Laborers','Cleaning staff','Laborers'] else 'neutral'),
        },
        'Marital_status': {
            'value_display': value_raw,
            'analysis': {
                'Married':              "Menikah → tanggung jawab finansial lebih terjaga",
                'Civil marriage':       "Pernikahan sipil → relatif stabil",
                'Single / not married': "Belum menikah → tanggungan lebih sedikit",
                'Separated':            "Terpisah → potensi beban finansial ganda",
                'Widow':                "Duda/Janda → pertimbangkan sumber pendapatan",
            }.get(value_raw, ""),
            'signal': 'positive' if value_raw in ['Married','Civil marriage']
                      else ('neutral' if value_raw == 'Single / not married' else 'negative'),
        },
        'Housing_type': {
            'value_display': value_raw,
            'analysis': {
                'House / apartment':    "Pemilik rumah/apartemen → aset tetap dimiliki",
                'Co-op apartment':      "Koperasi apartemen → kepemilikan bersama",
                'Municipal apartment':  "Apartemen kota → tinggal mandiri",
                'Rented apartment':     "Sewa apartemen → biaya bulanan rutin",
                'With parents':         "Tinggal dengan orang tua → pengeluaran lebih rendah",
                'Office apartment':     "Fasilitas kantor → tidak punya aset properti",
            }.get(value_raw, ""),
            'signal': 'positive' if value_raw in ['House / apartment','Co-op apartment']
                      else ('neutral' if value_raw in ['Municipal apartment','With parents'] else 'negative'),
        },
        'GENDER': {
            'value_display': value_raw,
            'analysis': "Data gender digunakan sebagai salah satu faktor demografis dalam model",
            'signal': 'neutral',
        },
    }
    return contexts.get(feature, {'value_display': str(value_raw), 'analysis': '', 'signal': 'neutral'})

def signal_to_color(signal):
    return {'positive': '#06D6A0', 'negative': '#FF5C5C', 'neutral': '#FFD166'}.get(signal, '#FFD166')

def signal_to_bar_class(signal):
    return {'positive': 'feature-bar-fill-green', 'negative': 'feature-bar-fill-red', 'neutral': 'feature-bar-fill-yellow'}.get(signal, 'feature-bar-fill-yellow')

def signal_to_icon(signal):
    return {'positive': '✅', 'negative': '⚠️', 'neutral': '➡️'}.get(signal, '➡️')

# ── SIDEBAR — Input Form (pakai session_state agar tidak reset) ───────────────
with st.sidebar:
    st.markdown("## 💳 Credit Analyzer")
    st.markdown("<p style='color:rgba(255,255,255,0.4);font-size:13px;margin-top:-8px;margin-bottom:24px;'>Isi data pemohon di bawah ini</p>", unsafe_allow_html=True)

    Name   = st.text_input("Nama Pemohon", value=st.session_state.Name, placeholder="Masukkan nama lengkap...", key="input_name")
    GENDER = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True,
                      index=["Laki-laki","Perempuan"].index(st.session_state.GENDER), key="input_gender")

    st.divider()
    st.markdown("<div class='section-title'>Data Pribadi</div>", unsafe_allow_html=True)

    AGE            = st.slider("Umur (Tahun)", 17, 80, st.session_state.AGE, key="input_age")
    Marital_status = st.selectbox("Status Pernikahan", list(Marital_status_map.keys()),
                                  index=list(Marital_status_map.keys()).index(st.session_state.Marital_status), key="input_marital")
    Family_Members = st.slider("Jumlah Anggota Keluarga", 1, 10, st.session_state.Family_Members, key="input_family")
    Housing_type   = st.selectbox("Tipe Hunian", list(Housing_type_map.keys()),
                                  index=list(Housing_type_map.keys()).index(st.session_state.Housing_type), key="input_housing")

    st.divider()
    st.markdown("<div class='section-title'>Pekerjaan & Pendidikan</div>", unsafe_allow_html=True)

    Type_Occupation = st.selectbox("Jenis Pekerjaan", list(Type_Occupation_map.keys()),
                                   index=list(Type_Occupation_map.keys()).index(st.session_state.Type_Occupation), key="input_occ")
    Type_Income     = st.selectbox("Jenis Pendapatan", list(Type_Income_map.keys()),
                                   index=list(Type_Income_map.keys()).index(st.session_state.Type_Income), key="input_income")
    YEAR_EMPLOYED   = st.slider("Lama Bekerja (Tahun)", 0, 40, st.session_state.YEAR_EMPLOYED, key="input_year")
    EDUCATION       = st.selectbox("Pendidikan Terakhir", list(EDUCATION_map.keys()),
                                   index=list(EDUCATION_map.keys()).index(st.session_state.EDUCATION), key="input_edu")

    st.divider()
    predict_btn = st.button("🔍 Analisis Sekarang")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("# 💳 Credit Card Approval System")
st.markdown("<p style='color:rgba(255,255,255,0.5);margin-top:-12px;margin-bottom:28px;'>Sistem prediksi kelayakan pengajuan kartu kredit berbasis Machine Learning</p>", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
total    = len(st.session_state.history)
approved = sum(1 for h in st.session_state.history if h["result"] == "Diterima")
rejected = total - approved

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Analisis</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Diterima</div><div class='metric-value' style='color:#06D6A0'>{approved}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Ditolak</div><div class='metric-value' style='color:#FF5C5C'>{rejected}</div></div>", unsafe_allow_html=True)
with col4:
    rate = f"{(approved/total*100):.0f}%" if total > 0 else "—"
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Approval Rate</div><div class='metric-value' style='color:#667eea'>{rate}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not Name.strip():
        st.warning("⚠️ Mohon isi nama pemohon terlebih dahulu.")
    else:
        # Simpan nilai form ke session_state agar tidak reset
        st.session_state.Name            = Name
        st.session_state.GENDER          = GENDER
        st.session_state.AGE             = AGE
        st.session_state.Marital_status  = Marital_status
        st.session_state.Family_Members  = Family_Members
        st.session_state.Housing_type    = Housing_type
        st.session_state.Type_Occupation = Type_Occupation
        st.session_state.Type_Income     = Type_Income
        st.session_state.YEAR_EMPLOYED   = YEAR_EMPLOYED
        st.session_state.EDUCATION       = EDUCATION

        features = [
            GENDER_map[GENDER],
            Type_Occupation_map[Type_Occupation],
            Type_Income_map[Type_Income],
            Marital_status_map[Marital_status],
            EDUCATION_map[EDUCATION],
            AGE,
            Housing_type_map[Housing_type],
            YEAR_EMPLOYED,
        ]
        raw_values = {
            'GENDER': GENDER, 'Type_Occupation': Type_Occupation,
            'Type_Income': Type_Income, 'Marital_status': Marital_status,
            'EDUCATION': EDUCATION, 'AGE': AGE,
            'Housing_type': Housing_type, 'YEAR_EMPLOYED': YEAR_EMPLOYED,
        }

        df      = pd.DataFrame([features], columns=feature_names)
        scaled  = scaler.transform(df)
        pred    = model.predict(scaled)[0]
        prob    = model.predict_proba(scaled)[0]

        # Feature importance dari model
        importances = model.feature_importances_

        # Hitung kontribusi per fitur: importance × nilai fitur yang sudah di-scale (abs)
        scaled_vals     = np.abs(scaled[0])
        contributions   = importances * scaled_vals
        contributions   = contributions / contributions.sum() * 100  # jadi persen

        result_label = "Diterima" if pred == 0 else "Ditolak"
        approve_prob = prob[0] * 100

        # Simpan ke history
        st.session_state.history.insert(0, {
            "name": Name, "result": result_label,
            "prob": f"{approve_prob:.1f}%", "age": AGE,
            "occupation": Type_Occupation, "income": Type_Income,
        })

        # Simpan last result untuk ditampilkan ulang
        st.session_state.last_result = {
            "pred": pred, "prob": prob, "approve_prob": approve_prob,
            "result_label": result_label, "contributions": contributions,
            "raw_values": raw_values, "features": features,
            "Name": Name, "AGE": AGE, "GENDER": GENDER,
            "Marital_status": Marital_status, "Type_Occupation": Type_Occupation,
            "Type_Income": Type_Income, "EDUCATION": EDUCATION, "Housing_type": Housing_type,
        }

# ── Tampilkan hasil (dari session state, tidak hilang saat rerun) ─────────────
if st.session_state.last_result:
    r = st.session_state.last_result

    # ── Row 1: Hasil utama + chart ────────────────────────────────────────────
    col_res, col_chart = st.columns([1, 1], gap="large")

    with col_res:
        if r["pred"] == 0:
            st.markdown(f"""
            <div class='result-approved'>
                <div style='font-size:56px;margin-bottom:8px'>✅</div>
                <div style='font-size:13px;letter-spacing:2px;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:8px'>Hasil Analisis</div>
                <div style='font-size:28px;font-weight:800;color:#06D6A0;margin-bottom:6px'>DISETUJUI</div>
                <div style='font-size:15px;color:rgba(255,255,255,0.7)'>Pengajuan atas nama <strong>{r['Name']}</strong> layak diterima</div>
                <div style='margin-top:16px;font-size:13px;color:rgba(6,214,160,0.8)'>Tingkat keyakinan model: <strong>{r['approve_prob']:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-rejected'>
                <div style='font-size:56px;margin-bottom:8px'>❌</div>
                <div style='font-size:13px;letter-spacing:2px;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:8px'>Hasil Analisis</div>
                <div style='font-size:28px;font-weight:800;color:#FF5C5C;margin-bottom:6px'>DITOLAK</div>
                <div style='font-size:15px;color:rgba(255,255,255,0.7)'>Pengajuan atas nama <strong>{r['Name']}</strong> tidak memenuhi kriteria</div>
                <div style='margin-top:16px;font-size:13px;color:rgba(255,92,92,0.8)'>Tingkat keyakinan model: <strong>{r['approve_prob']:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ringkasan Data Pemohon</div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"**👤 Nama:** {r['Name']}")
            st.markdown(f"**⚧ Jenis Kelamin:** {r['GENDER']}")
            st.markdown(f"**🎂 Umur:** {r['AGE']} tahun")
            st.markdown(f"**💍 Status:** {r['Marital_status']}")
        with r2:
            st.markdown(f"**💼 Pekerjaan:** {r['Type_Occupation']}")
            st.markdown(f"**💰 Pendapatan:** {r['Type_Income']}")
            st.markdown(f"**🎓 Pendidikan:** {r['EDUCATION']}")
            st.markdown(f"**🏠 Hunian:** {r['Housing_type']}")

    with col_chart:
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["approve_prob"],
            title={"text": "Probabilitas Disetujui", "font": {"size": 14, "color": "rgba(255,255,255,0.7)", "family": "Plus Jakarta Sans"}},
            number={"suffix": "%", "font": {"size": 36, "color": "white", "family": "Plus Jakarta Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)", "tickfont": {"color": "rgba(255,255,255,0.4)"}},
                "bar": {"color": "#06D6A0" if r["pred"] == 0 else "#FF5C5C", "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.05)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(255,92,92,0.15)"},
                    {"range": [40, 60], "color": "rgba(255,200,0,0.1)"},
                    {"range": [60, 100],"color": "rgba(6,214,160,0.15)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": 50},
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family": "Plus Jakarta Sans"})
        st.plotly_chart(fig, use_container_width=True)

        # Bar distribusi probabilitas
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Diterima", "Ditolak"],
            y=[r["prob"][0]*100, r["prob"][1]*100],
            marker_color=["#06D6A0", "#FF5C5C"], marker_line_width=0,
            text=[f"{r['prob'][0]*100:.1f}%", f"{r['prob'][1]*100:.1f}%"],
            textposition="outside",
            textfont={"color": "white", "size": 13, "family": "Plus Jakarta Sans"},
        ))
        fig2.update_layout(
            height=220, margin=dict(t=20,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis={"range": [0,115], "showgrid": False, "showticklabels": False, "zeroline": False},
            xaxis={"tickfont": {"color": "rgba(255,255,255,0.6)", "size": 13}},
            bargap=0.4,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Analisis mendalam per fitur ───────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Analisis Mendalam Per Fitur")
    st.markdown("<p style='color:rgba(255,255,255,0.5);margin-top:-8px;margin-bottom:20px;font-size:14px'>Seberapa besar kontribusi setiap faktor terhadap keputusan model, dan mengapa</p>", unsafe_allow_html=True)

    # Penjelasan dari mana persentase berasal
    with st.expander("ℹ️  Bagaimana kontribusi (%) dihitung?"):
        st.markdown("""
Persentase kontribusi dihitung dengan rumus:

```
kontribusi_fitur = feature_importance × |nilai_fitur_setelah_normalisasi|
```

Kemudian dinormalisasi agar total = 100%.

- **Feature importance** didapat dari Random Forest: seberapa sering fitur tersebut digunakan sebagai split di 50 decision tree, dikalikan seberapa besar peningkatan kemurnian (Gini impurity) yang dihasilkan.
- **Nilai fitur setelah normalisasi** (RobustScaler) menunjukkan seberapa jauh nilai kamu dari nilai median data pelatihan.
- Jadi fitur dengan importance tinggi DAN nilainya jauh dari median akan punya kontribusi besar.
        """)

    # Urutkan fitur berdasarkan kontribusi
    sorted_idx = np.argsort(r["contributions"])[::-1]

    col_feat1, col_feat2 = st.columns(2)
    for i, idx in enumerate(sorted_idx):
        fname   = feature_names[idx]
        raw_val = r["raw_values"][fname]
        enc_val = r["features"][idx]
        ctx     = get_feature_context(fname, raw_val, enc_val)
        contrib = r["contributions"][idx]
        color   = signal_to_color(ctx["signal"])
        bar_cls = signal_to_bar_class(ctx["signal"])
        icon    = signal_to_icon(ctx["signal"])
        imp_pct = model.feature_importances_[idx] * 100

        card_html = f"""
        <div class='feature-row'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <span style='font-weight:700;color:white;font-size:14px'>{icon} {feature_labels[fname]}</span>
                    <span style='color:rgba(255,255,255,0.4);font-size:12px;margin-left:8px'>importance model: {imp_pct:.1f}%</span>
                </div>
                <span style='font-weight:700;color:{color};font-size:16px'>{contrib:.1f}%</span>
            </div>
            <div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:2px'>Nilai: <strong style='color:white'>{ctx['value_display']}</strong></div>
            <div class='feature-bar-bg'>
                <div class='{bar_cls}' style='width:{min(contrib*3, 100):.0f}%'></div>
            </div>
            <div style='color:rgba(255,255,255,0.6);font-size:13px;margin-top:8px'>{ctx['analysis']}</div>
        </div>"""

        if i % 2 == 0:
            with col_feat1:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with col_feat2:
                st.markdown(card_html, unsafe_allow_html=True)

    # ── Kesimpulan naratif ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    positives = [feature_labels[feature_names[i]] for i in sorted_idx if get_feature_context(feature_names[i], r["raw_values"][feature_names[i]], r["features"][i])["signal"] == "positive"]
    negatives = [feature_labels[feature_names[i]] for i in sorted_idx if get_feature_context(feature_names[i], r["raw_values"][feature_names[i]], r["features"][i])["signal"] == "negative"]

    verdict_color = "#06D6A0" if r["pred"] == 0 else "#FF5C5C"
    verdict_text  = "mendukung persetujuan" if r["pred"] == 0 else "menunjukkan risiko"

    summary_html = f"""
    <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:20px 24px;'>
        <div style='font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:rgba(255,255,255,0.4);margin-bottom:12px'>Kesimpulan Analisis</div>
        <p style='color:rgba(255,255,255,0.8);font-size:14px;line-height:1.7;margin:0'>
            Secara keseluruhan, profil <strong style='color:white'>{r['Name']}</strong> 
            <strong style='color:{verdict_color}'>{verdict_text}</strong> dengan keyakinan <strong style='color:{verdict_color}'>{r['approve_prob']:.1f}%</strong>.
            {'Faktor pendukung utama: <strong style="color:#06D6A0">' + ', '.join(positives[:3]) + '</strong>.' if positives else ''}
            {'Faktor risiko yang terdeteksi: <strong style="color:#FF5C5C">' + ', '.join(negatives[:3]) + '</strong>.' if negatives else ''}
        </p>
    </div>"""
    st.markdown(summary_html, unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("### 📋 Riwayat Analisis")
    with col_clear:
        if st.button("🗑️ Hapus Semua"):
            st.session_state.history    = []
            st.session_state.last_result = None
            st.rerun()

    for h in st.session_state.history[:10]:
        badge_class = "badge-green" if h["result"] == "Diterima" else "badge-red"
        icon        = "✅" if h["result"] == "Diterima" else "❌"
        st.markdown(f"""
        <div class='history-row'>
            <div>
                <span style='font-weight:600;color:white'>{h['name']}</span>
                <span style='color:rgba(255,255,255,0.4);font-size:13px;margin-left:12px'>{h['occupation']} · {h['income']} · {h['age']} thn</span>
            </div>
            <div>
                <span style='color:rgba(255,255,255,0.4);font-size:13px;margin-right:12px'>{h['prob']}</span>
                <span class='badge {badge_class}'>{icon} {h['result']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:12px'>Credit Card Approval System · Powered by Random Forest · Built with Streamlit</p>", unsafe_allow_html=True)
