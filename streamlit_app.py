import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
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

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}

/* Result cards */
.result-approved {
    background: linear-gradient(135deg, rgba(6,214,160,0.15), rgba(6,214,160,0.05));
    border: 1px solid rgba(6,214,160,0.4);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(255,92,92,0.15), rgba(255,92,92,0.05));
    border: 1px solid rgba(255,92,92,0.4);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.badge-green { background: rgba(6,214,160,0.2); color: #06D6A0; border: 1px solid rgba(6,214,160,0.4); }
.badge-red   { background: rgba(255,92,92,0.2);  color: #FF5C5C; border: 1px solid rgba(255,92,92,0.4); }

/* Section title */
.section-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.4);
    margin-bottom: 12px;
}

/* History table */
.history-row {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 24px;
    font-size: 16px;
    font-weight: 700;
    font-family: 'Plus Jakarta Sans', sans-serif;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.4);
}

/* Input labels */
label { color: rgba(255,255,255,0.75) !important; font-size: 14px !important; }

/* Divider */
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
if "history" not in st.session_state:
    st.session_state.history = []

# ── Mappings ──────────────────────────────────────────────────────────────────
Housing_type_map    = {'House / apartment':0,'Rented apartment':1,'With parents':2,'Municipal apartment':3,'Co-op apartment':4,'Office apartment':5}
EDUCATION_map       = {'Higher education':0,'Secondary / secondary special':1,'Lower secondary':2,'Incomplete higher':3,'Academic degree':4}
Type_Occupation_map = {'Private service staff':0,'Laborers':1,'Managers':2,'Medicine staff':3,'Cooking staff':4,'Sales staff':5,'Accountants':6,'High skill tech staff':7,'Cleaning staff':8,'Drivers':9,'Low-skill Laborers':10,'IT staff':11,'Waiters/barmen staff':12,'Core staff':13,'Security staff':14,'HR staff':15,'Secretaries':16,'Realty agents':17}
GENDER_map          = {'Laki-laki':1,'Perempuan':0}
Marital_status_map  = {'Married':0,'Single / not married':1,'Civil marriage':2,'Separated':3,'Widow':4}
Type_Income_map     = {'Commercial associate':0,'Pensioner':1,'Working':2,'State servant':3}

# ── SIDEBAR — Input Form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Analyzer")
    st.markdown("<p style='color:rgba(255,255,255,0.4);font-size:13px;margin-top:-8px;margin-bottom:24px;'>Isi data pemohon di bawah ini</p>", unsafe_allow_html=True)

    Name   = st.text_input("Nama Pemohon", placeholder="Masukkan nama lengkap...")
    GENDER = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)

    st.divider()
    st.markdown("<div class='section-title'>Data Pribadi</div>", unsafe_allow_html=True)

    AGE            = st.slider("Umur (Tahun)", 17, 80, 30)
    Marital_status = st.selectbox("Status Pernikahan", list(Marital_status_map.keys()))
    Family_Members = st.slider("Jumlah Anggota Keluarga", 1, 10, 3)
    Housing_type   = st.selectbox("Tipe Hunian", list(Housing_type_map.keys()))

    st.divider()
    st.markdown("<div class='section-title'>Pekerjaan & Pendidikan</div>", unsafe_allow_html=True)

    Type_Occupation = st.selectbox("Jenis Pekerjaan", list(Type_Occupation_map.keys()))
    Type_Income     = st.selectbox("Jenis Pendapatan", list(Type_Income_map.keys()))
    YEAR_EMPLOYED   = st.slider("Lama Bekerja (Tahun)", 0, 40, 3)
    EDUCATION       = st.selectbox("Pendidikan Terakhir", list(EDUCATION_map.keys()))

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
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Total Analisis</div>
        <div class='metric-value'>{total}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Diterima</div>
        <div class='metric-value' style='color:#06D6A0'>{approved}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Ditolak</div>
        <div class='metric-value' style='color:#FF5C5C'>{rejected}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    rate = f"{(approved/total*100):.0f}%" if total > 0 else "—"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Approval Rate</div>
        <div class='metric-value' style='color:#667eea'>{rate}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not Name.strip():
        st.warning("⚠️ Mohon isi nama pemohon terlebih dahulu.")
    else:
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
        df = pd.DataFrame([features], columns=['GENDER','Type_Occupation','Type_Income','Marital_status','EDUCATION','AGE','Housing_type','YEAR_EMPLOYED'])
        scaled   = scaler.transform(df)
        pred     = model.predict(scaled)[0]
        prob     = model.predict_proba(scaled)[0]
        prob_pct = prob[0] * 100 if pred == 0 else prob[1] * 100

        result_label = "Diterima" if pred == 0 else "Ditolak"

        # Save to history
        st.session_state.history.insert(0, {
            "name": Name, "result": result_label,
            "prob": f"{prob_pct:.1f}%", "age": AGE,
            "occupation": Type_Occupation, "income": Type_Income,
        })

        # ── Result layout ─────────────────────────────────────────────────────
        col_res, col_chart = st.columns([1, 1], gap="large")

        with col_res:
            if pred == 0:
                st.markdown(f"""
                <div class='result-approved'>
                    <div style='font-size:56px;margin-bottom:8px'>✅</div>
                    <div style='font-size:13px;letter-spacing:2px;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:8px'>Hasil Analisis</div>
                    <div style='font-size:28px;font-weight:800;color:#06D6A0;margin-bottom:6px'>DISETUJUI</div>
                    <div style='font-size:15px;color:rgba(255,255,255,0.7)'>Pengajuan atas nama <strong>{Name}</strong> layak diterima</div>
                    <div style='margin-top:16px;font-size:13px;color:rgba(6,214,160,0.8)'>Tingkat keyakinan model: <strong>{prob_pct:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-rejected'>
                    <div style='font-size:56px;margin-bottom:8px'>❌</div>
                    <div style='font-size:13px;letter-spacing:2px;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:8px'>Hasil Analisis</div>
                    <div style='font-size:28px;font-weight:800;color:#FF5C5C;margin-bottom:6px'>DITOLAK</div>
                    <div style='font-size:15px;color:rgba(255,255,255,0.7)'>Pengajuan atas nama <strong>{Name}</strong> tidak memenuhi kriteria</div>
                    <div style='margin-top:16px;font-size:13px;color:rgba(255,92,92,0.8)'>Tingkat keyakinan model: <strong>{prob_pct:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

            # ── Summary data input ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Ringkasan Data Pemohon</div>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"**👤 Nama:** {Name}")
                st.markdown(f"**⚧ Jenis Kelamin:** {GENDER}")
                st.markdown(f"**🎂 Umur:** {AGE} tahun")
                st.markdown(f"**💍 Status:** {Marital_status}")
            with r2:
                st.markdown(f"**💼 Pekerjaan:** {Type_Occupation}")
                st.markdown(f"**💰 Pendapatan:** {Type_Income}")
                st.markdown(f"**🎓 Pendidikan:** {EDUCATION}")
                st.markdown(f"**🏠 Hunian:** {Housing_type}")

        with col_chart:
            # ── Gauge chart ───────────────────────────────────────────────
            approve_prob = prob[0] * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=approve_prob,
                title={"text": "Probabilitas Disetujui", "font": {"size": 14, "color": "rgba(255,255,255,0.7)", "family": "Plus Jakarta Sans"}},
                number={"suffix": "%", "font": {"size": 36, "color": "white", "family": "Plus Jakarta Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)", "tickfont": {"color": "rgba(255,255,255,0.4)"}},
                    "bar": {"color": "#06D6A0" if pred == 0 else "#FF5C5C", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(255,92,92,0.15)"},
                        {"range": [40, 60], "color": "rgba(255,200,0,0.1)"},
                        {"range": [60, 100],"color": "rgba(6,214,160,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": 50},
                },
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=40, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Plus Jakarta Sans"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Probability bar ───────────────────────────────────────────
            st.markdown("<div class='section-title'>Distribusi Probabilitas</div>", unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Diterima", "Ditolak"],
                y=[prob[0]*100, prob[1]*100],
                marker_color=["#06D6A0", "#FF5C5C"],
                marker_line_width=0,
                text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                textposition="outside",
                textfont={"color": "white", "size": 13, "family": "Plus Jakarta Sans"},
            ))
            fig2.update_layout(
                height=220,
                margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"range": [0, 115], "showgrid": False, "showticklabels": False, "zeroline": False},
                xaxis={"tickfont": {"color": "rgba(255,255,255,0.6)", "size": 13}},
                bargap=0.4,
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("### 📋 Riwayat Analisis")
    with col_clear:
        if st.button("🗑️ Hapus Semua"):
            st.session_state.history = []
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
