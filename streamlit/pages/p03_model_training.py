"""
Page 3 — Model Training
========================
Architecture cards for LSTM, GRU, BiLSTM plus training curve charts.
Shows demo curves before actual training completes.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


_MODEL_CONFIGS = {
    "lstm": {
        "label": "LSTM",
        "color": "#6366f1",
        "bg":    "#ede9fe",
        "icon":  "🔵",
        "owner": "Pair 1",
        "env":   "Google Colab (GPU T4)",
        "notebook": "01_lstm_pair1.ipynb",
        "layers": [
            ("Input",       "(batch, 5, 15)",    "#e0e7ff"),
            ("Masking",     "mask_value=0.",      "#e0e7ff"),
            ("LSTM(64)",    "return_sequences=True", "#c7d2fe"),
            ("BatchNorm",   "",                   "#ddd6fe"),
            ("Dropout",     "0.3",                "#ddd6fe"),
            ("LSTM(32)",    "return_sequences=False", "#c7d2fe"),
            ("BatchNorm",   "",                   "#ddd6fe"),
            ("Dropout",     "0.3",                "#ddd6fe"),
            ("Dense(1)",    "sigmoid",            "#a5b4fc"),
        ],
        "hyperparams": {
            "Optimizer": "Adam (lr=0.001)",
            "Loss":      "binary_crossentropy",
            "Class weight": "~28:1",
            "Batch size": "512",
            "Epochs":    "50 (EarlyStopping patience=5)",
            "Threshold": "0.45",
        },
    },
    "gru": {
        "label": "GRU",
        "color": "#059669",
        "bg":    "#d1fae5",
        "icon":  "🟢",
        "owner": "Pair 2",
        "env":   "Google Colab (GPU T4)",
        "notebook": "02_gru_pair2.ipynb",
        "layers": [
            ("Input",      "(batch, 5, 15)",    "#d1fae5"),
            ("Masking",    "mask_value=0.",      "#d1fae5"),
            ("GRU(64)",    "return_sequences=True", "#a7f3d0"),
            ("BatchNorm",  "",                   "#6ee7b7"),
            ("Dropout",    "0.3",                "#6ee7b7"),
            ("GRU(32)",    "return_sequences=False", "#a7f3d0"),
            ("BatchNorm",  "",                   "#6ee7b7"),
            ("Dropout",    "0.3",                "#6ee7b7"),
            ("Dense(1)",   "sigmoid",            "#34d399"),
        ],
        "hyperparams": {
            "Optimizer": "Adam (lr=0.001)",
            "Loss":      "binary_crossentropy",
            "Class weight": "~28:1",
            "Batch size": "512",
            "Epochs":    "50 (EarlyStopping patience=5)",
            "Threshold": "0.45",
        },
    },
    "bilstm": {
        "label": "Bidirectional LSTM",
        "color": "#be185d",
        "bg":    "#fce7f3",
        "icon":  "🔴",
        "owner": "Pair 3",
        "env":   "Google Colab (GPU T4)",
        "notebook": "03_bilstm_pair3.ipynb",
        "layers": [
            ("Input",               "(batch, 5, 15)",           "#fce7f3"),
            ("Masking",             "mask_value=0.",             "#fce7f3"),
            ("Bidirectional LSTM(64)", "merge_mode='concat'",   "#fbcfe8"),
            ("BatchNorm",           "",                          "#f9a8d4"),
            ("Dropout",             "0.3",                       "#f9a8d4"),
            ("Bidirectional LSTM(32)", "merge_mode='concat'",   "#fbcfe8"),
            ("BatchNorm",           "",                          "#f9a8d4"),
            ("Dropout",             "0.3",                       "#f9a8d4"),
            ("Dense(1)",            "sigmoid",                   "#f472b6"),
        ],
        "hyperparams": {
            "Optimizer": "Adam (lr=0.001)",
            "Loss":      "binary_crossentropy",
            "Class weight": "~28:1",
            "Batch size": "512",
            "Epochs":    "50 (EarlyStopping patience=5)",
            "Threshold": "0.45",
        },
    },
}


def _arch_card(cfg: dict):
    """Render architecture table + hyperparams side by side."""
    label = cfg["label"]
    color = cfg["color"]
    bg    = cfg["bg"]

    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {color};'
        f'border-radius:8px;padding:10px 16px;margin-bottom:10px;">'
        f'<span style="font-weight:700;font-size:1rem;color:{color};">'
        f'{cfg["icon"]} {label}</span>'
        f'<span style="margin-left:16px;font-size:0.78rem;color:#6b7280;">'
        f'Owner: {cfg["owner"]} · {cfg["env"]} · {cfg["notebook"]}</span></div>',
        unsafe_allow_html=True,
    )

    a_col, h_col = st.columns([3, 2])
    with a_col:
        layers_df = pd.DataFrame(cfg["layers"], columns=["Layer", "Config", "_bg"])
        st.dataframe(
            layers_df[["Layer", "Config"]],
            use_container_width=True, hide_index=True, height=300,
        )
    with h_col:
        st.markdown("**Hyperparameters**")
        for k, v in cfg["hyperparams"].items():
            st.markdown(f"- **{k}:** {v}")


def _training_curves(model_key: str, cfg: dict):
    """Plot loss and AUC curves side-by-side."""
    from utils.data_loader import load_history
    hist = load_history(model_key)

    epochs = list(range(1, len(hist.get("loss", [])) + 1))
    color  = cfg["color"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("loss", []),
                             mode="lines", name="Train Loss",
                             line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("val_loss", []),
                             mode="lines", name="Val Loss",
                             line=dict(color=color, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("auc", []),
                             mode="lines", name="Train AUC",
                             line=dict(color="#f97316", width=2),
                             yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("val_auc", []),
                             mode="lines", name="Val AUC",
                             line=dict(color="#f97316", width=2, dash="dash"),
                             yaxis="y2"))
    fig.update_layout(
        height=280, margin=dict(l=20, r=60, t=20, b=40),
        xaxis_title="Epoch",
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="AUC", side="right", overlaying="y", range=[0.5, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown("## 🧠 Model Training")
    st.markdown("""
    <div class="info-box">
        All three models are trained on <b>Google Colab (T4 GPU)</b> using the
        processed tensors produced by notebook <code>00_data_pipeline.ipynb</code>.
        Training curves below show demo values until actual training completes.
    </div>
    """, unsafe_allow_html=True)

    # Model selector pills
    selected = st.radio(
        "Select model to inspect",
        options=list(_MODEL_CONFIGS.keys()),
        format_func=lambda k: _MODEL_CONFIGS[k]["label"],
        horizontal=True,
    )
    cfg = _MODEL_CONFIGS[selected]

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    _arch_card(cfg)

    st.markdown('<div class="section-header">Training Curves</div>',
                unsafe_allow_html=True)
    _training_curves(selected, cfg)

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

    # Callbacks explanation
    st.markdown('<div class="section-header">Callbacks Used</div>',
                unsafe_allow_html=True)
    cb_cols = st.columns(3)
    with cb_cols[0]:
        st.markdown("""
        **EarlyStopping**
        - Monitor: `val_auc`
        - Patience: 5 epochs
        - Restores best weights
        """)
    with cb_cols[1]:
        st.markdown("""
        **ReduceLROnPlateau**
        - Factor: 0.5
        - Patience: 3 epochs
        - Min LR: 1e-6
        """)
    with cb_cols[2]:
        st.markdown("""
        **ModelCheckpoint**
        - Saves best epoch only
        - Format: `.keras`
        - Monitor: `val_auc`
        """)

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

    # Colab vs local guide
    st.markdown('<div class="section-header">Environment Guide</div>',
                unsafe_allow_html=True)
    env_df = pd.DataFrame([
        {"Notebook": "00_data_pipeline",   "Environment": "Local Anaconda", "Why": "CPU-only; pure pandas/numpy; large file I/O"},
        {"Notebook": "01_lstm_pair1",      "Environment": "Google Colab T4", "Why": "GPU training ~3x faster"},
        {"Notebook": "02_gru_pair2",       "Environment": "Google Colab T4", "Why": "GPU training ~3x faster"},
        {"Notebook": "03_bilstm_pair3",    "Environment": "Google Colab T4", "Why": "GPU training ~3x faster"},
        {"Notebook": "04_evaluation_solo", "Environment": "Local Anaconda", "Why": "Load trained models; CPU-only inference for eval"},
    ])
    st.dataframe(env_df, use_container_width=True, hide_index=True)
