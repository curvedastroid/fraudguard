"""
FraudGuard — Main Streamlit Entry Point
========================================
Run from the FraudGuard/ project root:

    streamlit run streamlit/app.py

Or from FraudGuard/streamlit/:

    streamlit run app.py

The app auto-detects paths relative to its own location via utils/data_loader.py.
"""

import sys
import os
from pathlib import Path

# Ensure the streamlit/ folder is on sys.path so pages/ can import utils/
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard — Sequential Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Sidebar nav style */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stRadio > label { font-weight: 600; }

/* Metric card tweaks */
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }

/* Badge helpers */
.badge-fraud {
    background:#fee2e2; color:#dc2626;
    padding:2px 10px; border-radius:9999px; font-weight:700; font-size:0.82rem;
}
.badge-legit {
    background:#d1fae5; color:#059669;
    padding:2px 10px; border-radius:9999px; font-weight:700; font-size:0.82rem;
}
.badge-model {
    background:#ede9fe; color:#6d28d9;
    padding:2px 10px; border-radius:9999px; font-weight:600; font-size:0.80rem;
    margin-right:6px;
}
/* Section header */
.section-header {
    font-size:1.15rem; font-weight:700; color:#4338ca;
    border-left: 4px solid #6366f1; padding-left:10px; margin-bottom:6px;
}
/* Info box */
.info-box {
    background:#eff6ff; border:1px solid #bfdbfe;
    border-radius:8px; padding:12px 16px; margin-bottom:12px;
    font-size:0.9rem; color:#1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — logo + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.6rem;">🛡️</div>
        <div style="font-size:1.25rem; font-weight:800; letter-spacing:1px; color:#e0e7ff;">
            FraudGuard
        </div>
        <div style="font-size:0.72rem; color:#a5b4fc; margin-top:2px;">
            Sequential Transaction Fraud Detection
        </div>
    </div>
    <hr style="border-color:#4338ca; margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Overview",
            "📊  Dataset Explorer",
            "🧠  Model Training",
            "📈  Model Comparison",
            "🔮  Live Prediction",
            "📄  Documentation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border-color:#4338ca; margin:16px 0 12px 0;">
    <div style="font-size:0.72rem; color:#818cf8; text-align:center;">
        GBC AIML 2026 · Applied Math for DP I<br>
        Models: LSTM · GRU · BiLSTM<br>
        Dataset: IEEE-CIS Fraud Detection
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Route to the selected page
# ---------------------------------------------------------------------------
# Import page modules lazily to keep startup fast
if page == "🏠  Overview":
    from pages.p01_overview import render
    render()

elif page == "📊  Dataset Explorer":
    from pages.p02_dataset_explorer import render
    render()

elif page == "🧠  Model Training":
    from pages.p03_model_training import render
    render()

elif page == "📈  Model Comparison":
    from pages.p04_model_comparison import render
    render()

elif page == "🔮  Live Prediction":
    from pages.p05_live_prediction import render
    render()

elif page == "📄  Documentation":
    from pages.p06_documentation import render
    render()
