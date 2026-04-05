"""
app.py  –  Smart Soil Health Analyzer
Streamlit UI with farmer-friendly theme, ML predictions, charts and reports.

Run with:
    streamlit run app.py
"""

import io
import os
import textwrap
import time
import base64
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                  # headless backend before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ── Local modules ────────────────────────────────────────────────────────────
from model import predict, train_and_save, FEATURES
from utils import (
    compute_health_score,
    get_crop_recommendations,
    get_suggestions,
    IDEAL,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Soil Health Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom CSS for farmer theme
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Nunito', sans-serif; }

/* ── Background ── */
.stApp { background-color: #f0f7eb; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #2d5a1b 0%, #4a7c2e 50%, #8bc34a 100%);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title  { font-size: 2.4rem; font-weight: 800; color: #fff; margin:0; letter-spacing:-0.5px; }
.hero-sub    { font-size: 1.05rem; color: rgba(255,255,255,0.85); margin-top:4px; }

/* ── Section headers ── */
.section-hdr {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2d5a1b;
    border-left: 4px solid #8bc34a;
    padding-left: 10px;
    margin: 1.5rem 0 0.75rem;
}

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.card-high   { border-top: 5px solid #4caf50; }
.card-medium { border-top: 5px solid #ffc107; }
.card-low    { border-top: 5px solid #ef5350; }

/* ── Fertility badge ── */
.badge-high   { background:#e8f5e9; color:#1b5e20; border-radius:99px; padding:6px 18px; font-weight:700; display:inline-block; font-size:1.05rem; }
.badge-medium { background:#fff8e1; color:#e65100; border-radius:99px; padding:6px 18px; font-weight:700; display:inline-block; font-size:1.05rem; }
.badge-low    { background:#ffebee; color:#b71c1c; border-radius:99px; padding:6px 18px; font-weight:700; display:inline-block; font-size:1.05rem; }

/* ── Score big number ── */
.score-big { font-size:3rem; font-weight:800; color:#2d5a1b; line-height:1; }
.score-grade { font-size:0.9rem; color:#555; margin-top:2px; }

/* ── Crop pill ── */
.crop-pill {
    display:inline-block;
    background:#e8f5e9;
    border:1px solid #a5d6a7;
    color:#1b5e20;
    border-radius:99px;
    padding:5px 14px;
    margin:3px;
    font-size:0.88rem;
    font-weight:600;
}

/* ── Suggestion item ── */
.sug-item {
    background:#f9fbe7;
    border-left:3px solid #8bc34a;
    border-radius:0 8px 8px 0;
    padding:8px 12px;
    margin-bottom:8px;
    font-size:0.9rem;
}
.sug-high   { border-left-color:#ef5350; background:#fff3f3; }
.sug-medium { border-left-color:#ffc107; background:#fffde7; }

/* ── Metric label ── */
.metric-lbl { font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:.5px; margin-bottom:2px; }

/* ── Slider label ── */
.slider-lbl { font-size:0.85rem; font-weight:600; color:#3a3a3a; margin-bottom:-10px; }

/* ── Button override ── */
div.stButton > button {
    background: linear-gradient(135deg, #2d5a1b, #4a7c2e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.9 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode()


def health_score_color(score: int) -> str:
    if score >= 75: return "#4caf50"
    if score >= 55: return "#8bc34a"
    if score >= 35: return "#ffc107"
    return "#ef5350"


def ensure_model():
    """Train models on first run if artifacts are missing."""
    if not os.path.exists("rf_model.joblib"):
        with st.spinner("⚙️ Training ML models for the first time …"):
            metrics = train_and_save()
        st.success(
            f"Models trained! RF accuracy: {metrics['rf_accuracy']}%  |  "
            f"LR accuracy: {metrics['lr_accuracy']}%"
        )


# ════════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════════

def plot_npk(n: float, p: float, k: float) -> plt.Figure:
    labels  = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
    actual  = [n, p, k]
    ideal_v = [IDEAL["Nitrogen"], IDEAL["Phosphorus"], IDEAL["Potassium"]]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#f0f7eb")
    ax.set_facecolor("#f0f7eb")

    bars1 = ax.bar(x - w/2, actual,  w, label="Your Soil",
                   color=["#4a7c2e","#6aab42","#8bc34a"], zorder=3)
    bars2 = ax.bar(x + w/2, ideal_v, w, label="Ideal Level",
                   color="#c8e6c9", edgecolor="#4a7c2e", linewidth=0.8, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("kg / ha", fontsize=9); ax.set_ylim(0, 165)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=8, color="#2d5a1b", fontweight="bold")

    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_radar(sub_scores: dict) -> plt.Figure:
    labels = list(sub_scores.keys())
    values = [sub_scores[l] for l in labels]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_c = values + values[:1]
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),
                           facecolor="#f0f7eb")
    ax.set_facecolor("#f0f7eb")
    ax.plot(angles_c, values_c, "o-", linewidth=2, color="#4a7c2e")
    ax.fill(angles_c, values_c, alpha=0.25, color="#8bc34a")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25","50","75","100"], size=6)
    ax.grid(color="#a5d6a7", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_feature_importance(feat_imp: dict) -> plt.Figure:
    items  = sorted(feat_imp.items(), key=lambda x: x[1])
    labels = [i[0] for i in items]
    vals   = [i[1] for i in items]
    colors = plt.cm.YlGn(np.linspace(0.4, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#f0f7eb")
    ax.set_facecolor("#f0f7eb")
    bars = ax.barh(labels, vals, color=colors, edgecolor="none", height=0.55, zorder=3)
    ax.xaxis.grid(True, alpha=0.4, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_xlabel("Importance (%)", fontsize=9)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=8, color="#2d5a1b")
    fig.tight_layout()
    return fig


def plot_gauge(score: int) -> plt.Figure:
    """Semi-circle gauge for the health score."""
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor="#f0f7eb")
    ax.set_facecolor("#f0f7eb")
    ax.set_aspect("equal")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta) * 1.0, np.sin(theta) * 1.0,
            linewidth=18, color="#e0e0e0", solid_capstyle="round")

    # Filled arc proportional to score
    filled = np.linspace(np.pi, np.pi - (score / 100) * np.pi, 200)
    ax.plot(np.cos(filled) * 1.0, np.sin(filled) * 1.0,
            linewidth=18, color=health_score_color(score), solid_capstyle="round")

    ax.text(0, 0.1, str(score), ha="center", va="center",
            fontsize=34, fontweight="bold", color="#2d5a1b")
    ax.text(0, -0.25, "/ 100", ha="center", va="center",
            fontsize=12, color="#666")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.3)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ════════════════════════════════════════════════════════════════════════════

def build_report(inputs, result, score_data, crops, suggestions) -> bytes:
    lines = [
        "SMART SOIL HEALTH ANALYZER – REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "SOIL PARAMETERS",
        *[f"  {k:<20}: {v}" for k, v in inputs.items()],
        "",
        "FERTILITY PREDICTION",
        f"  Class        : {result['class']}",
        f"  Confidence   : {result['confidence']}%",
        f"  LR Probs     : {result['probabilities']}",
        "",
        "SOIL HEALTH SCORE",
        f"  Score : {score_data['score']} / 100  ({score_data['grade']})",
        "",
        "FEATURE SCORES",
        *[f"  {k:<20}: {v}%" for k, v in score_data["sub_scores"].items()],
        "",
        "CROP RECOMMENDATIONS",
        *[f"  {c.emoji} {c.name:<14} – {c.reason}" for c in crops],
        "",
        "IMPROVEMENT SUGGESTIONS",
        *[f"  [{s.urgency}] {s.icon} {s.title}: {s.detail}" for s in suggestions],
        "",
        "FEATURE IMPORTANCES (Random Forest)",
        *[f"  {k:<20}: {v}%" for k, v in result["feature_importances"].items()],
    ]
    return "\n".join(lines).encode("utf-8")


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_model()

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-banner'>
      <div class='hero-title'>🌱 Smart Soil Health Analyzer</div>
      <div class='hero-sub'>Empowering Farmers with AI · Precision Agriculture · Explainable Insights</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_manual, tab_upload, tab_model = st.tabs(
        ["🎛️ Manual Input", "📂 Upload CSV", "📊 Model Info"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – Manual Input
    # ════════════════════════════════════════════════════════════════════════
    with tab_manual:
        st.markdown("<div class='section-hdr'>Enter Soil Parameters</div>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='slider-lbl'>🌱 Nitrogen (N) – kg/ha</div>",
                        unsafe_allow_html=True)
            N = st.slider("N", 0, 150, 60, key="N", label_visibility="collapsed")

            st.markdown("<div class='slider-lbl'>🌿 Phosphorus (P) – kg/ha</div>",
                        unsafe_allow_html=True)
            P = st.slider("P", 0, 150, 45, key="P", label_visibility="collapsed")

        with col2:
            st.markdown("<div class='slider-lbl'>🍂 Potassium (K) – kg/ha</div>",
                        unsafe_allow_html=True)
            K = st.slider("K", 0, 150, 55, key="K", label_visibility="collapsed")

            st.markdown("<div class='slider-lbl'>⚗️ pH Level</div>",
                        unsafe_allow_html=True)
            pH = st.slider("pH", 4.0, 9.0, 6.5, 0.1, key="pH",
                           label_visibility="collapsed")

        with col3:
            st.markdown("<div class='slider-lbl'>💧 Moisture (%)</div>",
                        unsafe_allow_html=True)
            moisture = st.slider("Moisture", 0, 100, 40, key="moisture",
                                 label_visibility="collapsed")

            st.markdown("<div class='slider-lbl'>🌾 Organic Carbon (%)</div>",
                        unsafe_allow_html=True)
            OC = st.slider("OC", 0.0, 10.0, 2.5, 0.1, key="OC",
                           label_visibility="collapsed")

        inputs = {
            "pH": pH, "Nitrogen": N, "Phosphorus": P,
            "Potassium": K, "Moisture": moisture, "OrganicCarbon": OC
        }

        st.write("")
        run = st.button("🚜 Analyze Soil", key="analyze_manual")
        _render_results(inputs, run)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – Upload CSV
    # ════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.markdown("<div class='section-hdr'>Upload Soil Data CSV</div>",
                    unsafe_allow_html=True)
        st.info(
            "CSV must contain columns: **pH, Nitrogen, Phosphorus, Potassium, "
            "Moisture, OrganicCarbon**  (one sample per row)"
        )

        # Download sample template
        sample_df = pd.DataFrame([{
            "pH": 6.5, "Nitrogen": 75, "Phosphorus": 55,
            "Potassium": 80, "Moisture": 42, "OrganicCarbon": 2.8
        }])
        st.download_button(
            "⬇️ Download Sample CSV Template",
            data=sample_df.to_csv(index=False),
            file_name="sample_soil.csv",
            mime="text/csv"
        )

        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"✅ {len(df_up)} sample(s) loaded.")
                st.dataframe(df_up[FEATURES].head(10), use_container_width=True)

                if st.button("🚜 Analyze All Rows", key="analyze_csv"):
                    results_rows = []
                    prog = st.progress(0, text="Analyzing samples …")
                    for i, row in df_up.iterrows():
                        inp = row[FEATURES].to_dict()
                        res   = predict(inp)
                        score = compute_health_score(inp)
                        results_rows.append({
                            **inp,
                            "FertilityClass": res["class"],
                            "Confidence(%)":  res["confidence"],
                            "HealthScore":    score["score"],
                            "Grade":          score["grade"],
                        })
                        prog.progress((i + 1) / len(df_up))

                    out_df = pd.DataFrame(results_rows)
                    st.success("Analysis complete!")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=out_df.to_csv(index=False),
                        file_name="soil_analysis_results.csv",
                        mime="text/csv"
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – Model Info
    # ════════════════════════════════════════════════════════════════════════
    with tab_model:
        st.markdown("<div class='section-hdr'>Model Details & Retraining</div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='card'>
            <b>Baseline Model</b><br>
            Logistic Regression (L2 regularisation, max_iter=500)<br><br>
            <b>Main Model</b><br>
            Random Forest (200 trees, max_depth=12)<br><br>
            <b>Dataset</b><br>
            2 000 synthetic samples; pH 4–9, NPK 0–150, Moisture 0–100, OC 0–10
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class='card'>
            <b>Label logic</b><br>
            Weighted score across 6 features → thresholds for Low / Medium / High<br><br>
            <b>Health Score formula</b><br>
            N 22% · P 20% · K 20% · pH 18% · Moisture 10% · OC 10%<br><br>
            <b>Explainability</b><br>
            Random Forest native feature_importances_ (Gini importance)
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔄 Retrain Models Now"):
            with st.spinner("Training …"):
                m = train_and_save()
            st.success(
                f"Done! RF: {m['rf_accuracy']}%  |  LR: {m['lr_accuracy']}%"
            )


# ════════════════════════════════════════════════════════════════════════════
# RESULTS RENDERER (shared between manual and CSV tabs)
# ════════════════════════════════════════════════════════════════════════════

def _render_results(inputs: dict, run: bool):
    if not run:
        return

    # Loading animation
    with st.spinner("🌱 Analyzing soil composition …"):
        time.sleep(0.8)
        result     = predict(inputs)
        score_data = compute_health_score(inputs)
        crops      = get_crop_recommendations(result["class"], inputs)
        suggestions = get_suggestions(inputs)

    st.success("✅ Analysis complete! Here are your results:")
    st.write("")

    # ── Row 1: Fertility + Health Score ─────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1, 1.4])

    cls_lower = result["class"].lower()
    with c1:
        st.markdown(f"""
        <div class='card card-{cls_lower}'>
          <div class='metric-lbl'>Fertility Class</div>
          <div class='badge-{cls_lower}'>{result['class']} Fertility</div>
          <br>
          <div style='margin-top:10px;font-size:0.85rem;color:#555'>
            Confidence: <b>{result['confidence']}%</b>
          </div>
          <div style='font-size:0.8rem;color:#888;margin-top:4px'>
            Low: {result['probabilities'].get('Low',0)}% &nbsp;|&nbsp;
            Med: {result['probabilities'].get('Medium',0)}% &nbsp;|&nbsp;
            High: {result['probabilities'].get('High',0)}%
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        fig_g = plot_gauge(score_data["score"])
        st.pyplot(fig_g, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;font-size:0.9rem;color:#555;margin-top:-10px'>"
            f"Grade: <b>{score_data['grade']}</b></div>",
            unsafe_allow_html=True
        )
        plt.close(fig_g)

    with c3:
        fig_r = plot_radar(score_data["sub_scores"])
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    # ── Row 2: Crop Recommendations ─────────────────────────────────────────
    st.markdown("<div class='section-hdr'>🌾 Crop Recommendations</div>",
                unsafe_allow_html=True)
    pills = " ".join(
        f"<span class='crop-pill'>{c.emoji} {c.name}</span>" for c in crops
    )
    st.markdown(f"<div class='card'>{pills}</div>", unsafe_allow_html=True)

    # Crop detail table
    crop_data = [{"Crop": c.name, "Season": c.season, "Why": c.reason}
                 for c in crops]
    st.dataframe(pd.DataFrame(crop_data), use_container_width=True,
                 hide_index=True)

    # ── Row 3: Improvement Suggestions ──────────────────────────────────────
    st.markdown("<div class='section-hdr'>🛠️ Improvement Suggestions</div>",
                unsafe_allow_html=True)
    urgency_cls = {"High": "sug-high", "Medium": "sug-medium", "Low": ""}
    for s in suggestions:
        css = urgency_cls.get(s.urgency, "")
        st.markdown(
            f"<div class='sug-item {css}'>"
            f"<b>{s.icon} {s.title}</b> "
            f"<span style='font-size:0.75rem;background:{'#ffcdd2' if s.urgency=='High' else '#fff9c4' if s.urgency=='Medium' else '#c8e6c9'};border-radius:4px;padding:1px 6px;margin-left:4px'>{s.urgency}</span>"
            f"<br>{s.detail}</div>",
            unsafe_allow_html=True
        )

    # ── Row 4: Charts ────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>📊 Charts</div>",
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**NPK Levels vs Ideal**")
        fig_npk = plot_npk(inputs["Nitrogen"], inputs["Phosphorus"],
                           inputs["Potassium"])
        st.pyplot(fig_npk, use_container_width=True)
        plt.close(fig_npk)

    with ch2:
        st.markdown("**Feature Importance (Explainable AI)**")
        fig_fi = plot_feature_importance(result["feature_importances"])
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)

    # ── Download Report ──────────────────────────────────────────────────────
    st.write("")
    report_bytes = build_report(inputs, result, score_data, crops, suggestions)
    st.download_button(
        "⬇️ Download Full Soil Report (.txt)",
        data=report_bytes,
        file_name=f"soil_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )


if __name__ == "__main__":
    main()
