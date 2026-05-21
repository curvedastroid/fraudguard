"""
Page 1 — Overview
==================
Hero banner, pipeline summary, 3-model status cards, and quick-stat metrics.
"""

import streamlit as st
import plotly.graph_objects as go


def render():
    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1b4b,#4338ca);
                border-radius:12px; padding:32px 36px; margin-bottom:28px;">
        <div style="font-size:2.2rem; font-weight:800; color:#fff; letter-spacing:1px;">
            🛡️ FraudGuard
        </div>
        <div style="font-size:1.05rem; color:#c7d2fe; margin-top:6px; max-width:640px;">
            Sequential credit-card fraud detection using recurrent neural networks.
            We model each cardholder's transaction history as a time series and train
            three architectures — LSTM, GRU, and Bidirectional LSTM — to flag
            fraudulent activity in real time.
        </div>
        <div style="margin-top:20px;">
            <span class="badge-model">LSTM</span>
            <span class="badge-model">GRU</span>
            <span class="badge-model">BiLSTM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick stats ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Transactions", "590,540")
    c2.metric("Fraud Rate", "3.5 %")
    c3.metric("Unique Cardholders", "~138 K")
    c4.metric("Sequence Length", "5 steps")

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Model Leaderboard ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Leaderboard (demo values)</div>',
                unsafe_allow_html=True)

    from utils.data_loader import load_all_metrics
    metrics = load_all_metrics()

    rows = [
        {"Model": "🥇 Bidirectional LSTM", "key": "bilstm",
         "AUC-ROC": metrics["bilstm"]["auc"],
         "F1":      metrics["bilstm"]["f1"],
         "Precision": metrics["bilstm"]["precision"],
         "Recall":  metrics["bilstm"]["recall"],
         "Params":  f'{metrics["bilstm"]["params"]:,}'},
        {"Model": "🥈 LSTM", "key": "lstm",
         "AUC-ROC": metrics["lstm"]["auc"],
         "F1":      metrics["lstm"]["f1"],
         "Precision": metrics["lstm"]["precision"],
         "Recall":  metrics["lstm"]["recall"],
         "Params":  f'{metrics["lstm"]["params"]:,}'},
        {"Model": "🥉 GRU", "key": "gru",
         "AUC-ROC": metrics["gru"]["auc"],
         "F1":      metrics["gru"]["f1"],
         "Precision": metrics["gru"]["precision"],
         "Recall":  metrics["gru"]["recall"],
         "Params":  f'{metrics["gru"]["params"]:,}'},
    ]

    import pandas as pd
    df = pd.DataFrame(rows).drop(columns=["key"])
    st.dataframe(
        df.style
          .format({"AUC-ROC": "{:.4f}", "F1": "{:.4f}",
                   "Precision": "{:.4f}", "Recall": "{:.4f}"})
          .background_gradient(subset=["AUC-ROC", "F1"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Pipeline diagram ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Data & Modelling Pipeline</div>',
                unsafe_allow_html=True)

    steps = [
        ("1", "Raw CSV\nDownload", "#6366f1"),
        ("2", "Cardholder\nUID", "#8b5cf6"),
        ("3", "Sort &\nDelta-t", "#a855f7"),
        ("4", "15 Feature\nSelection", "#ec4899"),
        ("5", "Sliding\nWindow (5)", "#f43f5e"),
        ("6", "Scale &\nSplit", "#ef4444"),
        ("7", "Train\nLSTM/GRU/BiLSTM", "#f97316"),
        ("8", "Evaluate &\nCompare", "#eab308"),
        ("9", "Deploy\nStreamlit", "#22c55e"),
    ]

    fig = go.Figure()
    for i, (num, label, color) in enumerate(steps):
        # Node box
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=48, color=color, line=dict(color="#fff", width=2)),
            text=[num],
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="middle center",
            hovertext=label.replace("\n", " "),
            hoverinfo="text",
            showlegend=False,
        ))
        # Label below
        fig.add_annotation(x=i, y=-0.18, text=label,
                           showarrow=False, font=dict(size=10, color="#374151"),
                           align="center")
        # Arrow
        if i < len(steps) - 1:
            fig.add_annotation(
                x=i + 0.62, y=0, ax=i + 0.38, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowcolor="#9ca3af", arrowsize=1.2, arrowwidth=1.5,
            )

    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=10, b=50),
        xaxis=dict(visible=False, range=[-0.6, len(steps) - 0.4]),
        yaxis=dict(visible=False, range=[-0.45, 0.35]),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Team structure ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Team & Ownership</div>',
                unsafe_allow_html=True)

    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    with tcol1:
        st.markdown("""
        <div style='background:#ede9fe;border-radius:10px;padding:14px;text-align:center;'>
            <div style='font-size:1.5rem;'>👤</div>
            <div style='font-weight:700;font-size:0.85rem;color:#4338ca;'>Project Manager</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>Hessam</div>
            <div style='font-size:0.75rem;color:#7c3aed;margin-top:6px;'>Data Pipeline<br>Streamlit App</div>
        </div>""", unsafe_allow_html=True)
    with tcol2:
        st.markdown("""
        <div style='background:#dbeafe;border-radius:10px;padding:14px;text-align:center;'>
            <div style='font-size:1.5rem;'>👥</div>
            <div style='font-weight:700;font-size:0.85rem;color:#1d4ed8;'>Pair 1</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>TBD</div>
            <div style='font-size:0.75rem;color:#2563eb;margin-top:6px;'>LSTM Model<br>Notebook 01</div>
        </div>""", unsafe_allow_html=True)
    with tcol3:
        st.markdown("""
        <div style='background:#d1fae5;border-radius:10px;padding:14px;text-align:center;'>
            <div style='font-size:1.5rem;'>👥</div>
            <div style='font-weight:700;font-size:0.85rem;color:#065f46;'>Pair 2</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>TBD</div>
            <div style='font-size:0.75rem;color:#059669;margin-top:6px;'>GRU Model<br>Notebook 02</div>
        </div>""", unsafe_allow_html=True)
    with tcol4:
        st.markdown("""
        <div style='background:#fce7f3;border-radius:10px;padding:14px;text-align:center;'>
            <div style='font-size:1.5rem;'>👥</div>
            <div style='font-weight:700;font-size:0.85rem;color:#9d174d;'>Pair 3</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>TBD</div>
            <div style='font-size:0.75rem;color:#be185d;margin-top:6px;'>BiLSTM Model<br>Notebook 03</div>
        </div>""", unsafe_allow_html=True)
    with tcol5:
        st.markdown("""
        <div style='background:#fef9c3;border-radius:10px;padding:14px;text-align:center;'>
            <div style='font-size:1.5rem;'>👤</div>
            <div style='font-weight:700;font-size:0.85rem;color:#92400e;'>Solo Evaluator</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:4px;'>TBD</div>
            <div style='font-size:0.75rem;color:#b45309;margin-top:6px;'>Compare All 3<br>Notebook 04</div>
        </div>""", unsafe_allow_html=True)
