import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, .brand-font {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,0.5);
    font-size: 13px;
}

/* ── Section label ── */
.sec-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.25);
    margin: 20px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
.kpi-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #6c63ff);
    border-radius: 14px 14px 0 0;
}
.kpi-label { font-size: 11px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.kpi-val { font-size: 26px; font-weight: 700; font-family: 'Space Grotesk', sans-serif; color: var(--accent, #fff); }

/* ── Result panels ── */
.panel-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(5,150,105,0.04));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px; padding: 32px;
}
.panel-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(220,38,38,0.04));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 20px; padding: 32px;
}
.result-emoji { font-size: 52px; margin-bottom: 10px; }
.result-verdict {
    font-size: 32px; font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.result-sub { font-size: 14px; color: rgba(255,255,255,0.55); line-height: 1.5; }
.conf-pill {
    display: inline-block; margin-top: 16px;
    padding: 5px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.7);
}

/* ── Summary info grid ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 18px; }
.info-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 14px;
}
.info-key { font-size: 10px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.info-val { font-size: 14px; color: #fff; font-weight: 500; }

/* ── Feature cards ── */
.feat-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.feat-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px; }
.feat-name { font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.85); }
.feat-pct { font-size: 18px; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.feat-val { font-size: 12px; color: rgba(255,255,255,0.35); margin-bottom: 8px; }
.feat-bar-bg { background: rgba(255,255,255,0.06); border-radius: 99px; height: 6px; margin-bottom: 10px; }
.feat-bar { border-radius: 99px; height: 6px; }
.feat-analysis { font-size: 12px; line-height: 1.6; color: rgba(255,255,255,0.5); }
.feat-path-badge {
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 99px; margin-bottom: 6px; font-weight: 600;
}

/* ── Narrative block ── */
.narrative-box {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #6c63ff;
    border-radius: 0 14px 14px 0;
    padding: 20px 24px;
    margin-top: 4px;
}
.narrative-box p { font-size: 14px; line-height: 1.8; color: rgba(255,255,255,0.7); margin: 0 0 10px 0; }
.narrative-box p:last-child { margin-bottom: 0; }
.narrative-highlight { color: #fff; font-weight: 600; }
.narrative-positive { color: #10b981; font-weight: 600; }
.narrative-negative { color: #ef4444; font-weight: 600; }
.narrative-neutral { color: #f59e0b; font-weight: 600; }

/* ── Tree vote viz ── */
.vote-row { display: flex; gap: 3px; flex-wrap: wrap; margin: 12px 0; }
.vote-dot {
    width: 10px; height: 10px; border-radius: 50%;
}

/* ── History ── */
.hist-row {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hist-name { font-weight: 600; color: white; font-size: 14px; }
.hist-sub { font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 2px; }
.badge-ok { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); padding: 4px 12px; border-radius: 99px; font-size: 12px; font-weight: 600; }
.badge-no { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 4px 12px; border-radius: 99px; font-size: 12px; font-weight: 600; }

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6c63ff, #8b5cf6);
    color: white; border: none;
    border-radius: 12px; padding: 14px 24px;
    font-size: 15px; font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #5b52f0, #7c3aed);
    transform: translateY(-1px);
}

/* ── Input widgets ── */
label { color: rgba(255,255,255,0.6) !important; font-size: 13px !important; }
.stSelectbox > div, .stSlider { background: transparent; }
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Divider ── */
.my-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 28px 0; }

/* ── Comparison bar ── */
.cmp-bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.cmp-label { font-size: 12px; color: rgba(255,255,255,0.4); width: 80px; text-align: right; }
.cmp-bar-outer { flex: 1; height: 8px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.cmp-bar-inner { height: 8px; border-radius: 99px; }
.cmp-val { font-size: 12px; color: rgba(255,255,255,0.6); width: 50px; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    try:
        scaler = joblib.load('scaler.pkl')
        model  = joblib.load('credit_default.pkl')
        return scaler, model
    except:
        df = pd.read_csv('df_cleaned.csv')
        feature_names = ['GENDER','Type_Occupation','Type_Income','Marital_status','EDUCATION','AGE','Housing_type','YEAR_EMPLOYED']
        X = df[feature_names]; y = df['label']
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_scaled, y)
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(model, 'credit_default.pkl')
        return scaler, model

@st.cache_data
def load_training_stats():
    df = pd.read_csv('df_cleaned.csv')
    return {
        'age_mean': df['AGE'].mean(),
        'age_median': df['AGE'].median(),
        'year_mean': df['YEAR_EMPLOYED'].mean(),
        'year_median': df['YEAR_EMPLOYED'].median(),
        'approval_rate': (df['label'] == 0).mean(),
        'occ_approval': df.groupby('Type_Occupation')['label'].apply(lambda x: (x==0).mean()).to_dict(),
        'income_approval': df.groupby('Type_Income')['label'].apply(lambda x: (x==0).mean()).to_dict(),
        'edu_approval': df.groupby('EDUCATION')['label'].apply(lambda x: (x==0).mean()).to_dict(),
        'marital_approval': df.groupby('Marital_status')['label'].apply(lambda x: (x==0).mean()).to_dict(),
        'housing_approval': df.groupby('Housing_type')['label'].apply(lambda x: (x==0).mean()).to_dict(),
    }

scaler, model = load_or_train_model()
training_stats = load_training_stats()

# ── Mappings ──────────────────────────────────────────────────────────────────
Housing_type_map    = {'Rumah / Apartemen Milik':0,'Apartemen Sewa':1,'Bersama Orang Tua':2,'Apartemen Kota':3,'Apartemen Koperasi':4,'Apartemen Kantor':5}
EDUCATION_map       = {'Pendidikan Tinggi (S1+)':0,'SMA / Sederajat':1,'SMP / Sederajat':2,'Kuliah Tidak Selesai':3,'Gelar Akademik (S2/S3)':4}
Type_Occupation_map = {'Staf Jasa Privat':0,'Buruh/Tenaga Kasar':1,'Manajer':2,'Tenaga Medis':3,'Staf Memasak':4,'Tenaga Penjualan':5,'Akuntan':6,'Teknisi Ahli / IT Senior':7,'Staf Kebersihan':8,'Pengemudi':9,'Buruh Tidak Terampil':10,'Staf IT':11,'Pelayan / Bartender':12,'Staf Inti/Admin':13,'Satpam / Keamanan':14,'Staf HR':15,'Sekretaris':16,'Agen Properti':17}
GENDER_map          = {'Laki-laki':1,'Perempuan':0}
Marital_status_map  = {'Menikah':0,'Belum Menikah':1,'Nikah Sipil':2,'Pisah (Separated)':3,'Duda / Janda':4}
Type_Income_map     = {'Karyawan Swasta':0,'Pensiunan':1,'Karyawan Umum':2,'PNS / Pegawai Negeri':3}

# reverse maps for display
Housing_type_rmap    = {v:k for k,v in Housing_type_map.items()}
EDUCATION_rmap       = {v:k for k,v in EDUCATION_map.items()}
Type_Occupation_rmap = {v:k for k,v in Type_Occupation_map.items()}
GENDER_rmap          = {v:k for k,v in GENDER_map.items()}
Marital_status_rmap  = {v:k for k,v in Marital_status_map.items()}
Type_Income_rmap     = {v:k for k,v in Type_Income_map.items()}

feature_names   = ['GENDER','Type_Occupation','Type_Income','Marital_status','EDUCATION','AGE','Housing_type','YEAR_EMPLOYED']
feature_labels  = {'GENDER':'Jenis Kelamin','Type_Occupation':'Jenis Pekerjaan','Type_Income':'Sumber Pendapatan','Marital_status':'Status Pernikahan','EDUCATION':'Pendidikan','AGE':'Usia','Housing_type':'Tempat Tinggal','YEAR_EMPLOYED':'Lama Bekerja'}

# ── Analysis Engine ───────────────────────────────────────────────────────────
def analyze_applicant(features, raw_values):
    df_input    = pd.DataFrame([features], columns=feature_names)
    scaled      = scaler.transform(df_input)
    pred        = model.predict(scaled)[0]
    prob        = model.predict_proba(scaled)[0]
    approve_pct = prob[0] * 100

    # 1) Tree vote distribution
    tree_votes      = np.array([t.predict(scaled)[0] for t in model.estimators_])
    approve_votes   = int(np.sum(tree_votes == 0))
    reject_votes    = int(np.sum(tree_votes == 1))

    # 2) Decision path feature usage & direction
    indicator, n_nodes_ptr = model.decision_path(scaled)
    feature_usage     = np.zeros(len(feature_names))
    thresh_direction  = np.zeros(len(feature_names))
    for tree_idx, tree in enumerate(model.estimators_):
        node_ids = indicator[0, n_nodes_ptr[tree_idx]:n_nodes_ptr[tree_idx+1]].indices
        for node_id in node_ids:
            feat = tree.tree_.feature[node_id]
            if feat >= 0:
                feature_usage[feat] += 1
                thresh = tree.tree_.threshold[node_id]
                val    = scaled[0, feat]
                thresh_direction[feat] += (1 if val > thresh else -1)

    # 3) Contribution = importance × |scaled_value| (normalized)
    importances   = model.feature_importances_
    scaled_abs    = np.abs(scaled[0])
    contributions = importances * (scaled_abs + 0.01)
    contributions = contributions / contributions.sum() * 100

    # 4) Per-feature signals (positive / negative / neutral for THIS prediction)
    feature_signals = {}
    stats = training_stats
    for i, fname in enumerate(feature_names):
        raw = raw_values[fname]
        direction = thresh_direction[i]
        occ_rate  = None

        if fname == 'AGE':
            base_rate = stats['approval_rate']
            age = raw
            if 25 <= age <= 55:
                signal = 'positive'; reason = f"Usia {age:.0f} tahun masuk rentang produktif (25–55 thn); historis tingkat approval tertinggi."
            elif age < 25:
                signal = 'negative'; reason = f"Usia {age:.0f} tahun terlalu muda; riwayat kredit belum terbentuk."
            else:
                signal = 'neutral'; reason = f"Usia {age:.0f} tahun mendekati batas senior; pertimbangkan masa pensiun."
            vsAvg = f"Rata-rata pemohon: {stats['age_mean']:.0f} thn | Median: {stats['age_median']:.0f} thn"

        elif fname == 'YEAR_EMPLOYED':
            yr = raw
            if yr >= 7:
                signal = 'positive'; reason = f"{yr:.0f} tahun pengalaman kerja → stabilitas finansial kuat, jauh di atas median ({stats['year_median']:.0f} thn)."
            elif yr >= 3:
                signal = 'neutral'; reason = f"{yr:.0f} tahun pengalaman kerja; cukup, namun belum sangat kuat (median data: {stats['year_median']:.0f} thn)."
            else:
                signal = 'negative'; reason = f"Hanya {yr:.0f} tahun bekerja; risiko instabilitas pendapatan. Median pemohon: {stats['year_median']:.0f} thn."
            vsAvg = f"Rata-rata pemohon: {stats['year_mean']:.1f} thn | Median: {stats['year_median']:.0f} thn"

        elif fname == 'Type_Occupation':
            occ_enc = raw_values['Type_Occupation']
            occ_rate = stats['occ_approval'].get(occ_enc, stats['approval_rate'])
            overall  = stats['approval_rate']
            diff     = (occ_rate - overall) * 100
            if occ_rate >= overall + 0.05:
                signal = 'positive'
                reason = f"Profesi '{Type_Occupation_rmap[occ_enc]}' memiliki tingkat approval {occ_rate*100:.0f}% dalam data historis, lebih tinggi dari rata-rata keseluruhan ({overall*100:.0f}%)."
            elif occ_rate <= overall - 0.05:
                signal = 'negative'
                reason = f"Profesi '{Type_Occupation_rmap[occ_enc]}' memiliki tingkat approval hanya {occ_rate*100:.0f}% dalam data historis, lebih rendah dari rata-rata ({overall*100:.0f}%)."
            else:
                signal = 'neutral'
                reason = f"Profesi '{Type_Occupation_rmap[occ_enc]}' memiliki tingkat approval {occ_rate*100:.0f}%, hampir sama dengan rata-rata ({overall*100:.0f}%)."
            vsAvg = f"Approval rate profesi ini: {occ_rate*100:.0f}% | Rata-rata semua profesi: {overall*100:.0f}%"

        elif fname == 'Type_Income':
            inc_enc  = raw_values['Type_Income']
            inc_rate = stats['income_approval'].get(inc_enc, stats['approval_rate'])
            overall  = stats['approval_rate']
            if inc_enc == 3:  # PNS
                signal = 'positive'; reason = "PNS/Pegawai Negeri memiliki pendapatan paling stabil dan terjamin; risiko kredit rendah."
            elif inc_enc == 0:  # swasta
                signal = 'positive'; reason = "Karyawan swasta dengan pendapatan tetap; profil kredit baik."
            elif inc_enc == 2:  # working
                signal = 'neutral'; reason = "Karyawan umum; pendapatan memadai namun perlu melihat faktor lain."
            else:  # pensiunan
                signal = 'neutral'; reason = "Pensiunan memiliki pendapatan tetap namun terbatas; bergantung pada besar pensiunan."
            vsAvg = f"Approval rate tipe pendapatan ini: {inc_rate*100:.0f}% | Rata-rata: {overall*100:.0f}%"

        elif fname == 'EDUCATION':
            edu_enc  = raw_values['EDUCATION']
            edu_rate = stats['edu_approval'].get(edu_enc, stats['approval_rate'])
            overall  = stats['approval_rate']
            if edu_enc in [0, 4]:
                signal = 'positive'; reason = f"Pendidikan tinggi ({EDUCATION_rmap[edu_enc]}) berkorelasi dengan kapasitas repayment lebih baik dan potensi pendapatan lebih tinggi."
            elif edu_enc == 1:
                signal = 'neutral'; reason = "Pendidikan SMA/sederajat adalah profil yang umum dan cukup memenuhi syarat dasar."
            else:
                signal = 'negative'; reason = f"Tingkat pendidikan ({EDUCATION_rmap[edu_enc]}) berada di bawah rata-rata pemohon yang disetujui."
            vsAvg = f"Approval rate kelompok pendidikan ini: {edu_rate*100:.0f}% | Rata-rata: {overall*100:.0f}%"

        elif fname == 'Marital_status':
            mar_enc  = raw_values['Marital_status']
            mar_rate = stats['marital_approval'].get(mar_enc, stats['approval_rate'])
            overall  = stats['approval_rate']
            if mar_enc in [0, 2]:
                signal = 'positive'; reason = "Status menikah umumnya dikaitkan dengan tanggung jawab finansial yang lebih terstruktur dan terjaga."
            elif mar_enc == 1:
                signal = 'neutral'; reason = "Belum menikah memiliki tanggungan lebih sedikit; netral dalam penilaian kredit."
            else:
                signal = 'negative'; reason = f"Status {Marital_status_rmap[mar_enc]} berpotensi menambah beban finansial; perlu dipertimbangkan."
            vsAvg = f"Approval rate status ini: {mar_rate*100:.0f}% | Rata-rata: {overall*100:.0f}%"

        elif fname == 'Housing_type':
            hou_enc  = raw_values['Housing_type']
            hou_rate = stats['housing_approval'].get(hou_enc, stats['approval_rate'])
            overall  = stats['approval_rate']
            if hou_enc in [0, 4]:
                signal = 'positive'; reason = f"Memiliki {Housing_type_rmap[hou_enc]} menunjukkan kepemilikan aset tetap; indikator stabilitas finansial."
            elif hou_enc in [2, 3]:
                signal = 'neutral'; reason = f"{Housing_type_rmap[hou_enc]} → pengeluaran bulanan lebih terkontrol, netral dalam penilaian."
            else:
                signal = 'negative'; reason = f"{Housing_type_rmap[hou_enc]} → tidak memiliki aset properti; faktor risiko tambahan."
            vsAvg = f"Approval rate tipe hunian ini: {hou_rate*100:.0f}% | Rata-rata: {overall*100:.0f}%"

        elif fname == 'GENDER':
            signal = 'neutral'
            reason = "Jenis kelamin digunakan sebagai salah satu variabel demografis dalam model; pengaruhnya relatif kecil."
            vsAvg  = f"Importance model: {importances[i]*100:.1f}%"

        else:
            signal = 'neutral'; reason = ""; vsAvg = ""

        feature_signals[fname] = {
            'signal': signal,
            'reason': reason,
            'vsAvg': vsAvg,
            'contribution': contributions[i],
            'usage': feature_usage[i],
            'importance': importances[i] * 100,
            'direction': thresh_direction[i],
        }

    # 5) Narrative generation (pure rule-based from ML stats)
    def gen_narrative(name, pred, approve_pct, raw_values, feature_signals, approve_votes, reject_votes):
        age      = raw_values['AGE']
        yr       = raw_values['YEAR_EMPLOYED']
        occ      = Type_Occupation_rmap[raw_values['Type_Occupation']]
        inc      = Type_Income_rmap[raw_values['Type_Income']]
        edu      = EDUCATION_rmap[raw_values['EDUCATION']]
        mar      = Marital_status_rmap[raw_values['Marital_status']]
        hou      = Housing_type_rmap[raw_values['Housing_type']]

        positives = [k for k,v in feature_signals.items() if v['signal']=='positive']
        negatives = [k for k,v in feature_signals.items() if v['signal']=='negative']
        top_feat  = sorted(feature_signals.items(), key=lambda x: x[1]['contribution'], reverse=True)

        # Top 2 most influential features
        top1_name  = feature_labels[top_feat[0][0]]
        top1_info  = top_feat[0][1]
        top2_name  = feature_labels[top_feat[1][0]]
        top2_info  = top_feat[1][1]

        verdict = "DISETUJUI" if pred == 0 else "DITOLAK"
        n_trees  = len(model.estimators_)

        para1 = (
            f"Model Random Forest dengan {n_trees} decision tree telah menganalisis profil <b>{name}</b> "
            f"dan menghasilkan keputusan <b>{verdict}</b> dengan tingkat keyakinan <b>{approve_pct:.1f}%</b>. "
            f"Sebanyak <b>{approve_votes} dari {n_trees} pohon keputusan</b> memilih 'Setuju', "
            f"sementara {reject_votes} pohon memilih 'Tolak'."
        )

        para2 = (
            f"Dua faktor paling berpengaruh dalam keputusan ini adalah <b>{top1_name}</b> "
            f"(kontribusi {top1_info['contribution']:.1f}%) dan <b>{top2_name}</b> "
            f"(kontribusi {top2_info['contribution']:.1f}%). "
        )

        if pred == 0:  # APPROVED
            if positives:
                pos_labels = [feature_labels[p] for p in positives[:3]]
                para2 += f"Faktor pendukung utama adalah: <b>{', '.join(pos_labels)}</b>."
            else:
                para2 += "Profil secara keseluruhan dinilai cukup layak meski tidak ada faktor yang sangat unggul."
        else:  # REJECTED
            if negatives:
                neg_labels = [feature_labels[n] for n in negatives[:3]]
                para2 += f"Faktor risiko yang menarik perhatian model: <b>{', '.join(neg_labels)}</b>."

        # Age & employment specific paragraph
        age_comment = ""
        if age < 25:
            age_comment = f"Usia pemohon yang masih sangat muda ({age:.0f} tahun) menjadi salah satu kekhawatiran; profil kredit belum terbentuk dengan baik."
        elif age > 55:
            age_comment = f"Di usia {age:.0f} tahun, model mempertimbangkan potensi masa pensiun yang memengaruhi kemampuan pembayaran jangka panjang."
        else:
            age_comment = f"Usia {age:.0f} tahun masuk dalam rentang produktif yang dinilai stabil oleh model."

        yr_comment = ""
        if yr < 2:
            yr_comment = f"Lama bekerja yang singkat ({yr:.0f} tahun) meningkatkan ketidakpastian stabilitas pendapatan."
        elif yr >= 7:
            yr_comment = f"Pengalaman kerja selama {yr:.0f} tahun jauh di atas median ({training_stats['year_median']:.0f} tahun), menunjukkan stabilitas karir yang kuat."
        else:
            yr_comment = f"Masa kerja {yr:.0f} tahun dinilai cukup oleh model untuk menilai stabilitas pendapatan."

        para3 = age_comment + " " + yr_comment

        # Recommendation if rejected
        para4 = ""
        if pred == 1:
            recs = []
            if 'YEAR_EMPLOYED' in negatives:
                recs.append("tambah pengalaman kerja minimal 3–5 tahun")
            if 'EDUCATION' in negatives:
                recs.append("tingkatkan jenjang pendidikan atau sertifikasi profesional")
            if 'Housing_type' in negatives:
                recs.append("kepemilikan properti akan memperkuat profil kredit")
            if 'AGE' in negatives:
                recs.append("ajukan kembali setelah memiliki riwayat kredit yang lebih matang")
            if recs:
                para4 = "Untuk meningkatkan peluang di pengajuan berikutnya, pertimbangkan untuk: <b>" + ", ".join(recs) + "</b>."

        return [p for p in [para1, para2, para3, para4] if p.strip()]

    narrative = gen_narrative(
        raw_values.get('Name', ''), pred, approve_pct,
        {k: raw_values[k] for k in feature_names},
        feature_signals, approve_votes, reject_votes
    )

    return {
        'pred': pred,
        'prob': prob,
        'approve_pct': approve_pct,
        'approve_votes': approve_votes,
        'reject_votes': reject_votes,
        'contributions': contributions,
        'feature_signals': feature_signals,
        'narrative': narrative,
        'tree_votes': tree_votes,
    }

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    'history': [], 'last_result': None, 'last_raw': None,
    'Name': '', 'GENDER': 'Laki-laki', 'AGE': 35,
    'Marital_status': 'Menikah', 'Family_Members': 3,
    'Housing_type': 'Rumah / Apartemen Milik',
    'Type_Occupation': 'Teknisi Ahli / IT Senior',
    'Type_Income': 'Karyawan Umum',
    'YEAR_EMPLOYED': 5, 'EDUCATION': 'Pendidikan Tinggi (S1+)',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Guard: reset stale/invalid session values (e.g. from old code cache)
_map_validators = {
    'GENDER':          list(GENDER_map.keys()),
    'Marital_status':  list(Marital_status_map.keys()),
    'Housing_type':    list(Housing_type_map.keys()),
    'Type_Occupation': list(Type_Occupation_map.keys()),
    'Type_Income':     list(Type_Income_map.keys()),
    'EDUCATION':       list(EDUCATION_map.keys()),
}
for _key, _valid_vals in _map_validators.items():
    if st.session_state.get(_key) not in _valid_vals:
        st.session_state[_key] = defaults[_key]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 4px 0 20px 0;'>
        <div style='font-size:22px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;'>💳 Credit Analyzer</div>
        <div style='font-size:12px;color:rgba(255,255,255,0.3);margin-top:4px;'>Sistem Prediksi Kelayakan Kartu Kredit</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Identitas ──
    st.markdown("<div class='sec-label'>👤 Identitas Pemohon</div>", unsafe_allow_html=True)
    Name   = st.text_input("Nama Lengkap", value=st.session_state.Name,
                            placeholder="Contoh: Budi Santoso", key="inp_name")
    GENDER = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True,
                      index=["Laki-laki","Perempuan"].index(st.session_state.GENDER), key="inp_gender")

    # ── Data Pribadi ──
    st.markdown("<div class='sec-label'>🏠 Data Pribadi</div>", unsafe_allow_html=True)

    AGE = st.slider("Usia (Tahun)", min_value=18, max_value=70,
                    value=st.session_state.AGE, step=1, key="inp_age",
                    help="Usia pemohon saat ini")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div style='text-align:center;background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.25);border-radius:10px;padding:10px;margin-top:4px;'><div style='font-size:10px;color:rgba(255,255,255,0.35);'>USIA</div><div style='font-size:22px;font-weight:700;color:#8b7ff7;font-family:Space Grotesk,sans-serif;'>{AGE}</div><div style='font-size:10px;color:rgba(255,255,255,0.3);'>tahun</div></div>", unsafe_allow_html=True)
    with col_b:
        age_cat = "Sangat Muda" if AGE < 25 else ("Produktif" if AGE <= 55 else "Senior")
        age_col = "#10b981" if 25 <= AGE <= 55 else ("#f59e0b" if AGE > 55 else "#ef4444")
        st.markdown(f"<div style='text-align:center;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:10px;margin-top:4px;'><div style='font-size:10px;color:rgba(255,255,255,0.35);'>KATEGORI</div><div style='font-size:14px;font-weight:700;color:{age_col};margin-top:4px;'>{age_cat}</div></div>", unsafe_allow_html=True)

    Marital_status = st.selectbox("Status Pernikahan", list(Marital_status_map.keys()),
                                   index=list(Marital_status_map.keys()).index(st.session_state.Marital_status),
                                   key="inp_marital")
    Family_Members = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=15,
                                      value=st.session_state.Family_Members, step=1, key="inp_family")
    Housing_type   = st.selectbox("Tipe Hunian", list(Housing_type_map.keys()),
                                   index=list(Housing_type_map.keys()).index(st.session_state.Housing_type),
                                   key="inp_housing")

    # ── Pekerjaan ──
    st.markdown("<div class='sec-label'>💼 Pekerjaan & Pendidikan</div>", unsafe_allow_html=True)

    Type_Occupation = st.selectbox("Jenis Pekerjaan", list(Type_Occupation_map.keys()),
                                    index=list(Type_Occupation_map.keys()).index(st.session_state.Type_Occupation),
                                    key="inp_occ")
    Type_Income     = st.selectbox("Sumber Pendapatan", list(Type_Income_map.keys()),
                                    index=list(Type_Income_map.keys()).index(st.session_state.Type_Income),
                                    key="inp_income")

    YEAR_EMPLOYED   = st.slider("Lama Bekerja (Tahun)", min_value=0, max_value=40,
                                 value=st.session_state.YEAR_EMPLOYED, step=1, key="inp_year",
                                 help="Berapa tahun pemohon telah bekerja di pekerjaan saat ini")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(f"<div style='text-align:center;background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.25);border-radius:10px;padding:10px;margin-top:4px;'><div style='font-size:10px;color:rgba(255,255,255,0.35);'>LAMA KERJA</div><div style='font-size:22px;font-weight:700;color:#8b7ff7;font-family:Space Grotesk,sans-serif;'>{YEAR_EMPLOYED}</div><div style='font-size:10px;color:rgba(255,255,255,0.3);'>tahun</div></div>", unsafe_allow_html=True)
    with col_d:
        yr_cat = "Baru Mulai" if YEAR_EMPLOYED < 2 else ("Memadai" if YEAR_EMPLOYED < 5 else ("Baik" if YEAR_EMPLOYED < 10 else "Sangat Senior"))
        yr_col = "#ef4444" if YEAR_EMPLOYED < 2 else ("#f59e0b" if YEAR_EMPLOYED < 5 else "#10b981")
        st.markdown(f"<div style='text-align:center;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:10px;margin-top:4px;'><div style='font-size:10px;color:rgba(255,255,255,0.35);'>LEVEL</div><div style='font-size:14px;font-weight:700;color:{yr_col};margin-top:4px;'>{yr_cat}</div></div>", unsafe_allow_html=True)

    EDUCATION = st.selectbox("Pendidikan Terakhir", list(EDUCATION_map.keys()),
                              index=list(EDUCATION_map.keys()).index(st.session_state.EDUCATION),
                              key="inp_edu")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analisis Kelayakan Kredit")

# ── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:28px;'>
    <div style='font-size:28px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;letter-spacing:-0.5px;'>
        💳 Credit Card Approval System
    </div>
    <div style='font-size:14px;color:rgba(255,255,255,0.4);margin-top:6px;'>
        Analisis kelayakan pengajuan kartu kredit berbasis Random Forest · 100 Decision Trees
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
total    = len(st.session_state.history)
approved = sum(1 for h in st.session_state.history if h['result'] == 'Diterima')
rejected = total - approved
rate_str = f"{approved/total*100:.0f}%" if total > 0 else "—"

kpi_html = f"""
<div class='kpi-grid'>
    <div class='kpi-card' style='--accent:#6c63ff'>
        <div class='kpi-label'>Total Analisis</div>
        <div class='kpi-val' style='color:#8b7ff7'>{total}</div>
    </div>
    <div class='kpi-card' style='--accent:#10b981'>
        <div class='kpi-label'>Disetujui</div>
        <div class='kpi-val' style='color:#10b981'>{approved}</div>
    </div>
    <div class='kpi-card' style='--accent:#ef4444'>
        <div class='kpi-label'>Ditolak</div>
        <div class='kpi-val' style='color:#ef4444'>{rejected}</div>
    </div>
    <div class='kpi-card' style='--accent:#f59e0b'>
        <div class='kpi-label'>Approval Rate</div>
        <div class='kpi-val' style='color:#f59e0b'>{rate_str}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Run prediction ────────────────────────────────────────────────────────────
if predict_btn:
    if not Name.strip():
        st.warning("⚠️ Mohon isi nama pemohon terlebih dahulu.")
    else:
        for k in ['Name','GENDER','AGE','Marital_status','Family_Members','Housing_type','Type_Occupation','Type_Income','YEAR_EMPLOYED','EDUCATION']:
            st.session_state[k] = locals()[k]

        raw_values = {
            'Name': Name,
            'GENDER': GENDER_map[GENDER],
            'Type_Occupation': Type_Occupation_map[Type_Occupation],
            'Type_Income': Type_Income_map[Type_Income],
            'Marital_status': Marital_status_map[Marital_status],
            'EDUCATION': EDUCATION_map[EDUCATION],
            'AGE': AGE,
            'Housing_type': Housing_type_map[Housing_type],
            'YEAR_EMPLOYED': YEAR_EMPLOYED,
        }
        features = [raw_values[f] for f in feature_names]

        with st.spinner("Menjalankan analisis..."):
            result = analyze_applicant(features, raw_values)

        result['raw_display'] = {
            'Name': Name, 'GENDER': GENDER, 'AGE': AGE,
            'Marital_status': Marital_status, 'Housing_type': Housing_type,
            'Type_Occupation': Type_Occupation, 'Type_Income': Type_Income,
            'YEAR_EMPLOYED': YEAR_EMPLOYED, 'EDUCATION': EDUCATION,
        }

        st.session_state.last_result = result
        st.session_state.last_raw    = raw_values

        st.session_state.history.insert(0, {
            'name': Name, 'result': 'Diterima' if result['pred'] == 0 else 'Ditolak',
            'prob': f"{result['approve_pct']:.1f}%",
            'age': AGE, 'occupation': Type_Occupation, 'income': Type_Income,
        })
        st.rerun()

# ── Show result ───────────────────────────────────────────────────────────────
if st.session_state.last_result:
    r    = st.session_state.last_result
    rd   = r['raw_display']
    pred = r['pred']
    n_trees = len(model.estimators_)

    # ── Row 1: Result + Gauge + Tree votes ───────────────────────────────────
    col_res, col_right = st.columns([1, 1], gap="large")

    with col_res:
        if pred == 0:
            st.markdown(f"""
            <div class='panel-approved'>
                <div class='result-emoji'>✅</div>
                <div style='font-size:11px;letter-spacing:3px;color:rgba(255,255,255,0.3);margin-bottom:8px;'>HASIL KEPUTUSAN</div>
                <div class='result-verdict' style='color:#10b981;'>DISETUJUI</div>
                <div class='result-sub'>Pengajuan atas nama <strong style='color:white'>{rd['Name']}</strong> memenuhi kriteria kelayakan kredit.</div>
                <div class='conf-pill'>Tingkat keyakinan model: <strong style='color:#10b981'>{r['approve_pct']:.1f}%</strong></div>
                <div class='conf-pill' style='margin-left:8px;'>🌲 {r['approve_votes']}/{n_trees} pohon setuju</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='panel-rejected'>
                <div class='result-emoji'>❌</div>
                <div style='font-size:11px;letter-spacing:3px;color:rgba(255,255,255,0.3);margin-bottom:8px;'>HASIL KEPUTUSAN</div>
                <div class='result-verdict' style='color:#ef4444;'>DITOLAK</div>
                <div class='result-sub'>Pengajuan atas nama <strong style='color:white'>{rd['Name']}</strong> belum memenuhi kriteria kelayakan kredit saat ini.</div>
                <div class='conf-pill'>Keyakinan penolakan: <strong style='color:#ef4444'>{100-r['approve_pct']:.1f}%</strong></div>
                <div class='conf-pill' style='margin-left:8px;'>🌲 {r['reject_votes']}/{n_trees} pohon menolak</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Profile summary
        st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:3px;color:rgba(255,255,255,0.25);margin-bottom:10px;'>RINGKASAN PEMOHON</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-grid'>
            <div class='info-item'><div class='info-key'>👤 Nama</div><div class='info-val'>{rd['Name']}</div></div>
            <div class='info-item'><div class='info-key'>⚧ Gender</div><div class='info-val'>{rd['GENDER']}</div></div>
            <div class='info-item'><div class='info-key'>🎂 Usia</div><div class='info-val'>{rd['AGE']} tahun</div></div>
            <div class='info-item'><div class='info-key'>💍 Status</div><div class='info-val'>{rd['Marital_status']}</div></div>
            <div class='info-item'><div class='info-key'>💼 Pekerjaan</div><div class='info-val'>{rd['Type_Occupation']}</div></div>
            <div class='info-item'><div class='info-key'>💰 Pendapatan</div><div class='info-val'>{rd['Type_Income']}</div></div>
            <div class='info-item'><div class='info-key'>🎓 Pendidikan</div><div class='info-val'>{rd['EDUCATION']}</div></div>
            <div class='info-item'><div class='info-key'>🏠 Hunian</div><div class='info-val'>{rd['Housing_type']}</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r['approve_pct'],
            title={"text": "Probabilitas Disetujui", "font": {"size": 13, "color": "rgba(255,255,255,0.5)", "family": "DM Sans"}},
            number={"suffix": "%", "font": {"size": 38, "color": "#10b981" if pred == 0 else "#ef4444", "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "rgba(255,255,255,0.2)", "tickfont": {"color": "rgba(255,255,255,0.3)", "size":11}},
                "bar": {"color": "#10b981" if pred == 0 else "#ef4444", "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range":[0,40],  "color":"rgba(239,68,68,0.12)"},
                    {"range":[40,65], "color":"rgba(245,158,11,0.08)"},
                    {"range":[65,100],"color":"rgba(16,185,129,0.12)"},
                ],
                "threshold": {"line":{"color":"rgba(255,255,255,0.3)","width":2}, "thickness":0.8, "value":50},
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40,b=0,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font={"family":"DM Sans"})
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Tree vote visualizer
        vote_dots = ""
        for v in r['tree_votes']:
            col_dot = "#10b981" if v == 0 else "#ef4444"
            title_text = 'Setuju' if v == 0 else 'Tolak'
            vote_dots += f"<div class='vote-dot' style='background:{col_dot};opacity:0.8;' title='{title_text}'></div>"

        st.markdown(f"""
        <div style='background:#111118;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px 18px;'>
            <div style='font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,0.25);margin-bottom:10px;'>DISTRIBUSI SUARA — {n_trees} DECISION TREES</div>
            <div class='vote-row'>{vote_dots}</div>
            <div style='display:flex;gap:16px;margin-top:8px;'>
                <div style='font-size:12px;'><span style='color:#10b981;'>●</span> <span style='color:rgba(255,255,255,0.5);'>{r['approve_votes']} Setuju</span></div>
                <div style='font-size:12px;'><span style='color:#ef4444;'>●</span> <span style='color:rgba(255,255,255,0.5);'>{r['reject_votes']} Tolak</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ANALISIS NARATIF ─────────────────────────────────────────────────────
    st.markdown("<div class='my-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom:16px;'>
        <div style='font-size:18px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;'>📝 Analisis Naratif</div>
        <div style='font-size:13px;color:rgba(255,255,255,0.35);margin-top:4px;'>Penjelasan keputusan model secara menyeluruh — dihasilkan murni dari logika Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    narrative_html = "<div class='narrative-box'>"
    for para in r['narrative']:
        narrative_html += f"<p>{para}</p>"
    narrative_html += "</div>"
    st.markdown(narrative_html, unsafe_allow_html=True)

    # ── ANALISIS PER FITUR ────────────────────────────────────────────────────
    st.markdown("<div class='my-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom:16px;'>
        <div style='font-size:18px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;'>🔬 Analisis Mendalam Per Faktor</div>
        <div style='font-size:13px;color:rgba(255,255,255,0.35);margin-top:4px;'>Kontribusi setiap faktor terhadap keputusan, berurutan dari yang paling berpengaruh</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Bagaimana angka kontribusi & analisis ini dihasilkan?"):
        st.markdown("""
**Kontribusi (%)** = `feature_importance × |nilai_setelah_normalisasi|`, lalu dinormalisasi ke 100%.

**Analisis per fitur** dihasilkan dari:
- **Tingkat approval historis** per kategori (dari data pelatihan)
- **Arah keputusan** di sepanjang decision path di 100 pohon (apakah nilai Anda "tinggi" atau "rendah" relatif terhadap threshold yang digunakan)
- **Perbandingan** dengan median dan rata-rata data pemohon sebelumnya

Semua kalkulasi ini **murni dari model Random Forest**, tanpa AI tambahan.
        """)

    sorted_feats = sorted(r['feature_signals'].items(), key=lambda x: x[1]['contribution'], reverse=True)
    col_f1, col_f2 = st.columns(2)

    for i, (fname, fdata) in enumerate(sorted_feats):
        signal   = fdata['signal']
        contrib  = fdata['contribution']
        imp      = fdata['importance']
        reason   = fdata['reason']
        vsAvg    = fdata['vsAvg']

        sig_color = {'positive':'#10b981','negative':'#ef4444','neutral':'#f59e0b'}[signal]
        sig_bg    = {'positive':'rgba(16,185,129,0.12)','negative':'rgba(239,68,68,0.12)','neutral':'rgba(245,158,11,0.12)'}[signal]
        sig_icon  = {'positive':'✅','negative':'⚠️','neutral':'➡️'}[signal]
        sig_label = {'positive':'Mendukung','negative':'Risiko','neutral':'Netral'}[signal]
        bar_pct   = min(contrib * 3, 100)

        raw_val_display = ''
        rd_full = r['raw_display']
        fname_display = {
            'GENDER': rd_full['GENDER'],
            'AGE': f"{rd_full['AGE']} tahun",
            'YEAR_EMPLOYED': f"{rd_full['YEAR_EMPLOYED']} tahun",
            'Type_Occupation': rd_full['Type_Occupation'],
            'Type_Income': rd_full['Type_Income'],
            'Marital_status': rd_full['Marital_status'],
            'EDUCATION': rd_full['EDUCATION'],
            'Housing_type': rd_full['Housing_type'],
        }.get(fname, '')

        card = f"""
        <div class='feat-card' style='border-color:rgba(255,255,255,0.1);'>
            <div class='feat-header'>
                <div>
                    <span style='font-size:10px;font-weight:700;letter-spacing:1.5px;color:rgba(255,255,255,0.3);display:block;margin-bottom:4px;'>#{i+1} · {feature_labels[fname].upper()}</span>
                    <span style='font-size:14px;font-weight:600;color:white;'>{fname_display}</span>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:22px;font-weight:700;font-family:Space Grotesk,sans-serif;color:{sig_color};'>{contrib:.1f}%</div>
                    <div style='font-size:10px;color:rgba(255,255,255,0.3);'>importance: {imp:.1f}%</div>
                </div>
            </div>
            <div class='feat-bar-bg'>
                <div class='feat-bar' style='width:{bar_pct:.0f}%;background:{sig_color};'></div>
            </div>
            <div style='display:flex;align-items:center;gap:6px;margin-bottom:8px;'>
                <span style='display:inline-block;padding:2px 10px;border-radius:99px;font-size:10px;font-weight:700;background:{sig_bg};color:{sig_color};border:1px solid {sig_color}33;'>{sig_icon} {sig_label}</span>
                <span style='font-size:11px;color:rgba(255,255,255,0.25);'>{vsAvg}</span>
            </div>
            <div class='feat-analysis'>{reason}</div>
        </div>"""

        if i % 2 == 0:
            with col_f1:
                st.markdown(card, unsafe_allow_html=True)
        else:
            with col_f2:
                st.markdown(card, unsafe_allow_html=True)

    # ── Waterfall contribution chart ──────────────────────────────────────────
    st.markdown("<div class='my-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom:16px;'>
        <div style='font-size:18px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;'>📊 Visualisasi Kontribusi Faktor</div>
        <div style='font-size:13px;color:rgba(255,255,255,0.35);margin-top:4px;'>Distribusi bobot setiap faktor terhadap keputusan akhir</div>
    </div>
    """, unsafe_allow_html=True)

    feat_labels_sorted = [feature_labels[f] for f, _ in sorted_feats]
    contribs_sorted    = [d['contribution'] for _, d in sorted_feats]
    signals_sorted     = [d['signal'] for _, d in sorted_feats]
    colors_sorted      = ['#10b981' if s=='positive' else ('#ef4444' if s=='negative' else '#f59e0b') for s in signals_sorted]

    fig_bar = go.Figure(go.Bar(
        x=contribs_sorted[::-1],
        y=feat_labels_sorted[::-1],
        orientation='h',
        marker_color=colors_sorted[::-1],
        marker_line_width=0,
        text=[f"{c:.1f}%" for c in contribs_sorted[::-1]],
        textposition='outside',
        textfont={'color': 'rgba(255,255,255,0.6)', 'size': 12, 'family': 'Space Grotesk'},
    ))
    fig_bar.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False, "range": [0, max(contribs_sorted)*1.35]},
        yaxis={"tickfont": {"color": "rgba(255,255,255,0.6)", "size": 12}, "gridcolor": "rgba(255,255,255,0.04)"},
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<div class='my-divider'></div>", unsafe_allow_html=True)
    col_ht, col_hc = st.columns([4, 1])
    with col_ht:
        st.markdown("<div style='font-size:18px;font-weight:700;font-family:Space Grotesk,sans-serif;color:white;margin-bottom:14px;'>📋 Riwayat Analisis</div>", unsafe_allow_html=True)
    with col_hc:
        if st.button("🗑️ Hapus"):
            st.session_state.history     = []
            st.session_state.last_result = None
            st.session_state.last_raw    = None
            st.rerun()

    for h in st.session_state.history[:10]:
        badge = f"<span class='badge-ok'>✅ Diterima</span>" if h['result']=='Diterima' else f"<span class='badge-no'>❌ Ditolak</span>"
        st.markdown(f"""
        <div class='hist-row'>
            <div>
                <div class='hist-name'>{h['name']}</div>
                <div class='hist-sub'>{h['occupation']} · {h['income']} · {h['age']} thn</div>
            </div>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='font-size:13px;color:rgba(255,255,255,0.3);'>{h['prob']}</span>
                {badge}
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.15);font-size:12px;'>Credit Card Approval System · Random Forest (100 Trees) · Analisis Murni dari ML · Built with Streamlit</p>", unsafe_allow_html=True)
