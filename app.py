import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Early Warning Siswa",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F7F9FC; }
    .block-container { padding: 2rem 3rem; }

    .header-box {
        background: linear-gradient(135deg, #1A2E4A 0%, #2C4A70 100%);
        border-radius: 12px;
        padding: 1.75rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; }
    .header-box p  { margin: 0.3rem 0 0; opacity: 0.7; font-size: 0.9rem; }

    .risk-card {
        border-radius: 12px;
        padding: 1.4rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-label { font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.3rem; }
    .risk-tier  { font-size: 1.9rem; font-weight: 800; }
    .risk-pct   { font-size: 0.95rem; margin-top: 0.2rem; }

    .section-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6B7280;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1.25rem 0 0.6rem;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #1A2E4A; }
    .metric-label { font-size: 0.78rem; color: #6B7280; margin-top: 0.2rem; }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_FILE = "prediction_history.xlsx"
TIER_CONFIG = {
    "Low Risk":      {"color": "#D1FAE5", "text": "#065F46", "emoji": "✅"},
    "Medium Risk":   {"color": "#FEF3C7", "text": "#92400E", "emoji": "⚠️"},
    "High Risk":     {"color": "#FFEDD5", "text": "#9A3412", "emoji": "🔶"},
    "Critical Risk": {"color": "#FEE2E2", "text": "#991B1B", "emoji": "🚨"},
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_case09.pkl", "rb") as f:
        return pickle.load(f)

bundle   = load_model()
model    = bundle["model"]
features = bundle["features"]
encoders = bundle["encoders"]

# ── History helpers ───────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_excel(HISTORY_FILE)
    return pd.DataFrame()

def save_to_history(inputs: dict, prob: float, tier: str):
    row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_tier":         tier,
        "probabilitas_pct":  round(prob * 100, 1),
        **inputs
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_excel(HISTORY_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_excel(HISTORY_FILE, index=False)
    return df_all

# ── ML helpers ────────────────────────────────────────────────────────────────
def get_tier(prob):
    if prob < 0.30:   return "Low Risk"
    elif prob < 0.50: return "Medium Risk"
    elif prob < 0.70: return "High Risk"
    else:             return "Critical Risk"

def predict(inputs: dict):
    row = {}
    for feat in features:
        val = inputs[feat]
        if feat in encoders:
            val = encoders[feat].transform([val])[0]
        row[feat] = val
    X    = pd.DataFrame([row])[features]
    prob = model.predict_proba(X)[0][1]
    return prob

# ── Chart helpers ─────────────────────────────────────────────────────────────
def feature_importance_chart():
    imp = model.feature_importances_
    df  = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ["#2C4A70" if v > df["Importance"].median() else "#94B8D8" for v in df["Importance"]]
    ax.barh(df["Feature"], df["Importance"], color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Importance Score", fontsize=8)
    ax.set_title("Faktor Paling Berpengaruh", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

def tier_donut_chart(df_hist):
    counts = df_hist["risk_tier"].value_counts()
    colors_map = {
        "Low Risk": "#34D399", "Medium Risk": "#FBBF24",
        "High Risk": "#F97316", "Critical Risk": "#EF4444"
    }
    labels = counts.index.tolist()
    vals   = counts.values.tolist()
    colors = [colors_map.get(l, "#CBD5E1") for l in labels]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax.set_title("Distribusi Risk Tier", fontsize=9, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig

def trend_chart(df_hist):
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
    df_hist["tanggal"]   = df_hist["timestamp"].dt.date
    daily = df_hist.groupby(["tanggal", "risk_tier"]).size().unstack(fill_value=0)
    tier_order  = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    colors_list = ["#34D399", "#FBBF24", "#F97316", "#EF4444"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for tier, color in zip(tier_order, colors_list):
        if tier in daily.columns:
            ax.plot(daily.index, daily[tier], marker="o", markersize=4,
                    label=tier, color=color, linewidth=1.8)
    ax.set_xlabel("Tanggal", fontsize=8)
    ax.set_ylabel("Jumlah Prediksi", fontsize=8)
    ax.set_title("Tren Prediksi Harian per Risk Tier", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

def kehadiran_dist_chart(df_hist):
    fig, ax = plt.subplots(figsize=(5, 3))
    tier_order  = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    colors_list = ["#34D399", "#FBBF24", "#F97316", "#EF4444"]
    for tier, color in zip(tier_order, colors_list):
        subset = df_hist[df_hist["risk_tier"] == tier]["persentase_kehadiran"].dropna()
        if len(subset):
            ax.hist(subset, bins=15, alpha=0.6, color=color, label=tier, edgecolor="white")
    ax.set_xlabel("Kehadiran (%)", fontsize=8)
    ax.set_ylabel("Jumlah Siswa", fontsize=8)
    ax.set_title("Distribusi Kehadiran per Risk Tier", fontsize=9, fontweight="bold", pad=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-box">
    <h1>🎓 Early Warning System — Kelulusan Siswa</h1>
    <p>Prediksi risiko tidak lulus berdasarkan faktor proses belajar · Intelligo ID</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard = st.tabs(["🔍 Prediksi Siswa", "📊 Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-title">Data Siswa</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            usia          = st.number_input("Usia (tahun)", min_value=10, max_value=20, value=15)
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])
            jenis_sekolah = st.selectbox("Jenis Sekolah", ["Negeri", "Swasta"])
        with c2:
            pendidikan_ortu = st.selectbox("Pendidikan Orang Tua", ["SD", "SMP", "SMA", "S1", "S2"])
            pekerjaan_ortu  = st.selectbox("Pekerjaan Orang Tua", ["Buruh", "PNS", "Swasta", "Wiraswasta"])
            waktu_tempuh    = st.number_input("Waktu Tempuh (menit)", min_value=1, max_value=120, value=20)

        st.markdown('<div class="section-title">Aktivitas & Dukungan Belajar</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            kehadiran     = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar   = st.slider("Jam Belajar / Minggu", 0, 60, 15)
            skor_dukungan = st.slider("Skor Dukungan (1–10)", 1.0, 10.0, 6.0, step=0.5)
        with c4:
            akses_internet = st.radio("Akses Internet", ["Ya", "Tidak"], horizontal=True)
            les_privat     = st.radio("Les Privat", ["Ya", "Tidak"], horizontal=True)
            ikut_ekskul    = st.radio("Ikut Ekskul", ["Ya", "Tidak"], horizontal=True)

        predict_btn = st.button("🔍 Prediksi Risiko", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

        if predict_btn:
            inputs = {
                "usia":                   usia,
                "jenis_kelamin":          jenis_kelamin,
                "jenis_sekolah":          jenis_sekolah,
                "pendidikan_orang_tua":   pendidikan_ortu,
                "pekerjaan_orang_tua":    pekerjaan_ortu,
                "akses_internet":         akses_internet,
                "les_privat":             les_privat,
                "ikut_ekskul":            ikut_ekskul,
                "persentase_kehadiran":   kehadiran,
                "jam_belajar_per_minggu": jam_belajar,
                "waktu_tempuh_menit":     waktu_tempuh,
                "skor_dukungan":          skor_dukungan,
            }

            prob = predict(inputs)
            tier = get_tier(prob)
            cfg  = TIER_CONFIG[tier]

            # Simpan ke Excel
            save_to_history(inputs, prob, tier)
            st.toast("✅ Data tersimpan ke history", icon="💾")

            # Risk card
            st.markdown(f"""
            <div class="risk-card" style="background:{cfg['color']}; border:2px solid {cfg['text']}22;">
                <div class="risk-label" style="color:{cfg['text']}">RISK TIER</div>
                <div class="risk-tier" style="color:{cfg['text']}">{cfg['emoji']} {tier}</div>
                <div class="risk-pct" style="color:{cfg['text']}">Probabilitas Tidak Lulus: <b>{prob*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            st.markdown("**Skala Risiko**")
            st.markdown(f"""
            <div style="background:#E5E7EB; border-radius:8px; height:14px; margin-bottom:0.3rem;">
                <div style="width:{prob*100:.1f}%; background:{cfg['text']}; height:14px; border-radius:8px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#6B7280;">
                <span>0% Low</span><span>30%</span><span>50%</span><span>70%</span><span>100% Critical</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Rekomendasi
            rekom = {
                "Low Risk":      "✅ Tidak ada intervensi khusus. Pantau rutin per semester.",
                "Medium Risk":   "⚠️ Pantau kehadiran dua mingguan. Dorong ekskul akademik.",
                "High Risk":     "🔶 Konseling bulanan. Tawarkan les tambahan. Hubungi orang tua.",
                "Critical Risk": "🚨 Intervensi segera. Konseling mingguan, remedial wajib, pertemuan orang tua."
            }
            st.info(rekom[tier])

            # Faktor perhatian
            concerns = []
            if kehadiran < 70:
                concerns.append(f"Kehadiran rendah ({kehadiran}%) — idealnya ≥ 80%")
            if jam_belajar < 15:
                concerns.append(f"Jam belajar sedikit ({jam_belajar} jam/minggu)")
            if les_privat == "Tidak" and akses_internet == "Tidak":
                concerns.append("Tidak ada les privat dan tidak ada akses internet")
            if skor_dukungan < 5:
                concerns.append(f"Skor dukungan rendah ({skor_dukungan})")
            if concerns:
                st.markdown("**Faktor yang perlu perhatian:**")
                for c in concerns:
                    st.markdown(f"- {c}")

            st.markdown('<div class="section-title">Feature Importance Model</div>', unsafe_allow_html=True)
            st.pyplot(feature_importance_chart(), use_container_width=True)

        else:
            st.markdown("""
            <div style="background:white; border-radius:10px; padding:2.5rem; text-align:center;
                        color:#9CA3AF; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🎓</div>
                <div style="font-weight:600;">Isi data siswa di sebelah kiri</div>
                <div style="font-size:0.85rem; margin-top:0.3rem;">lalu klik tombol Prediksi Risiko</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    df_hist = load_history()

    if df_hist.empty:
        st.info("Belum ada data prediksi. Jalankan prediksi di tab pertama terlebih dahulu.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        total      = len(df_hist)
        critical   = (df_hist["risk_tier"] == "Critical Risk").sum()
        high       = (df_hist["risk_tier"] == "High Risk").sum()
        avg_prob   = df_hist["probabilitas_pct"].mean()
        at_risk    = critical + high

        st.markdown('<div class="section-title">Ringkasan</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Prediksi</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#EF4444">{critical}</div>
                <div class="metric-label">Critical Risk</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#F97316">{at_risk}</div>
                <div class="metric-label">Butuh Intervensi (High + Critical)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_prob:.1f}%</div>
                <div class="metric-label">Rata-rata Probabilitas Tidak Lulus</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Visualisasi</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.pyplot(tier_donut_chart(df_hist), use_container_width=True)
        with ch2:
            st.pyplot(kehadiran_dist_chart(df_hist), use_container_width=True)

        if len(df_hist) > 1:
            st.pyplot(trend_chart(df_hist), use_container_width=True)

        # ── History table ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Riwayat Prediksi</div>', unsafe_allow_html=True)

        # Filter tier
        tier_filter = st.multiselect(
            "Filter Risk Tier",
            options=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
            default=["High Risk", "Critical Risk"]
        )
        df_show = df_hist[df_hist["risk_tier"].isin(tier_filter)] if tier_filter else df_hist

        st.dataframe(
            df_show.sort_values("timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=300
        )

        # Download button
        st.download_button(
            label="⬇️ Download History (Excel)",
            data=open(HISTORY_FILE, "rb").read(),
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest · Skenario B (tanpa nilai akademik) · Dataset sintetik untuk keperluan bootcamp · Intelligo ID")
