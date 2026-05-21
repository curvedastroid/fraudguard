"""
Page 2 — Dataset Explorer
==========================
Class distribution, transaction amount distribution, delta-t histogram,
feature schema table, cardholder UID explanation.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Demo/static dataset statistics (pre-EDA results)
_STATS = {
    "total_transactions": 590_540,
    "fraud_count":         20_663,
    "legit_count":        569_877,
    "unique_cardholders": 138_420,
    "date_range":         "Nov 2017 – Dec 2017 (approx. 2 months)",
    "raw_features":       394,
    "selected_features":  15,
    "seq_len":            5,
}


def render():
    st.markdown("## 📊 Dataset Explorer")
    st.markdown("""
    <div class="info-box">
        <b>Dataset:</b> IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019) ·
        590,540 transactions · 394 raw features · 3.5% fraud rate
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level stats ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{_STATS['total_transactions']:,}")
    c2.metric("Fraud Transactions", f"{_STATS['fraud_count']:,}", "3.5 %")
    c3.metric("Unique Cardholders", f"{_STATS['unique_cardholders']:,}")
    c4.metric("Raw Features", str(_STATS['raw_features']))

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Row 1: class distribution + amount distribution ──────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Class Distribution</div>',
                    unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[_STATS["legit_count"], _STATS["fraud_count"]],
            hole=0.55,
            marker=dict(colors=["#6366f1", "#ef4444"],
                        line=dict(color="#fff", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig_pie.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Class imbalance ≈ 28:1 — addressed via class_weight in training.")

    with col_right:
        st.markdown('<div class="section-header">Transaction Amount Distribution</div>',
                    unsafe_allow_html=True)
        # Simulated log-normal amount distributions
        rng = np.random.default_rng(42)
        legit_amt = rng.lognormal(mean=3.8, sigma=1.2, size=3000)
        fraud_amt = rng.lognormal(mean=4.5, sigma=1.4, size=400)
        legit_amt = np.clip(legit_amt, 0, 2000)
        fraud_amt = np.clip(fraud_amt, 0, 2000)

        fig_amt = go.Figure()
        fig_amt.add_trace(go.Histogram(
            x=legit_amt, name="Legitimate", nbinsx=50,
            marker_color="#6366f1", opacity=0.7,
            histnorm="probability density",
        ))
        fig_amt.add_trace(go.Histogram(
            x=fraud_amt, name="Fraud", nbinsx=50,
            marker_color="#ef4444", opacity=0.7,
            histnorm="probability density",
        ))
        fig_amt.update_layout(
            barmode="overlay", height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Transaction Amount (USD)",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_amt, use_container_width=True)
        st.caption("Fraud transactions skew towards higher amounts.")

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Row 2: delta-t histogram + cardholder UID ────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-header">Delta-t Between Transactions (seconds)</div>',
                    unsafe_allow_html=True)
        rng2 = np.random.default_rng(7)
        # Most gaps are hours; fraud often close together
        legit_dt = rng2.exponential(scale=86_400, size=3000)
        fraud_dt = rng2.exponential(scale=18_000, size=400)
        legit_dt = np.clip(legit_dt, 60, 30 * 86_400)
        fraud_dt = np.clip(fraud_dt, 60, 30 * 86_400)

        fig_dt = go.Figure()
        fig_dt.add_trace(go.Histogram(
            x=np.log10(legit_dt + 1), name="Legitimate", nbinsx=40,
            marker_color="#6366f1", opacity=0.7,
        ))
        fig_dt.add_trace(go.Histogram(
            x=np.log10(fraud_dt + 1), name="Fraud", nbinsx=40,
            marker_color="#ef4444", opacity=0.7,
        ))
        fig_dt.update_layout(
            barmode="overlay", height=270,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="log₁₀(delta_t seconds)",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dt, use_container_width=True)
        st.caption("Fraudulent transactions occur in tighter temporal bursts.")

    with col_r2:
        st.markdown('<div class="section-header">Cardholder UID Construction</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#f5f3ff;border-radius:8px;padding:16px;font-size:0.87rem;color:#3730a3;">
            <b>Why no direct customer ID in the dataset?</b><br><br>
            Vesta anonymised PII. We reconstruct a <b>proxy cardholder UID</b>
            using two consistently-present fields:
            <br><br>
            <code style="background:#ede9fe;padding:3px 8px;border-radius:4px;font-size:0.9em;">
                uid = card1 + "_" + addr1
            </code>
            <br><br>
            <b>card1</b> — masked card number prefix (numeric)<br>
            <b>addr1</b> — billing ZIP code (numeric)<br>
            NaN values filled with <code>-999</code> before concatenation.<br><br>
            This gives <b>~138,420 unique pseudo-cardholders</b>,
            matching community-validated approach on Kaggle.
        </div>
        """, unsafe_allow_html=True)

        st.code(
            "df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)\n"
            "df['uid'] = df['uid'].fillna('-999_-999')",
            language="python"
        )

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Feature schema ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">15 Selected Features per Timestep</div>',
                unsafe_allow_html=True)

    feat_df = pd.DataFrame([
        {"#": 1,  "Feature": "TransactionAmt",   "Source": "Raw",          "Type": "Numeric",  "Description": "Transaction amount in USD"},
        {"#": 2,  "Feature": "delta_t_seconds",  "Source": "Engineered",   "Type": "Numeric",  "Description": "Seconds elapsed since previous transaction for this cardholder"},
        {"#": 3,  "Feature": "D1_days",           "Source": "Raw (Vesta)",  "Type": "Numeric",  "Description": "Days since last transaction — Vesta pre-computed proxy"},
        {"#": 4,  "Feature": "C1",                "Source": "Raw",          "Type": "Count",    "Description": "Count-type feature (meaning masked by Vesta)"},
        {"#": 5,  "Feature": "C2",                "Source": "Raw",          "Type": "Count",    "Description": "Count-type feature"},
        {"#": 6,  "Feature": "C3",                "Source": "Raw",          "Type": "Count",    "Description": "Count-type feature"},
        {"#": 7,  "Feature": "C4",                "Source": "Raw",          "Type": "Count",    "Description": "Count-type feature"},
        {"#": 8,  "Feature": "C5",                "Source": "Raw",          "Type": "Count",    "Description": "Count-type feature"},
        {"#": 9,  "Feature": "M1",                "Source": "Raw",          "Type": "Binary",   "Description": "Match flag (e.g. name/address match)"},
        {"#": 10, "Feature": "M3",                "Source": "Raw",          "Type": "Binary",   "Description": "Match flag"},
        {"#": 11, "Feature": "M6",                "Source": "Raw",          "Type": "Binary",   "Description": "Match flag"},
        {"#": 12, "Feature": "V12",               "Source": "Raw (Vesta)",  "Type": "Numeric",  "Description": "Vesta-engineered Vxxx feature (fraud signal)"},
        {"#": 13, "Feature": "V20",               "Source": "Raw (Vesta)",  "Type": "Numeric",  "Description": "Vesta-engineered Vxxx feature"},
        {"#": 14, "Feature": "V37",               "Source": "Raw (Vesta)",  "Type": "Numeric",  "Description": "Vesta-engineered Vxxx feature"},
        {"#": 15, "Feature": "addr1_norm",        "Source": "Engineered",   "Type": "Numeric",  "Description": "Normalised billing ZIP — encodes geographic signal"},
    ])

    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=440)

    st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)

    # ── Sequence construction walkthrough ─────────────────────────────────────
    st.markdown('<div class="section-header">Sliding Window Sequence Construction</div>',
                unsafe_allow_html=True)
    st.markdown("""
    For each cardholder we sort transactions chronologically by `TransactionDT`
    (elapsed seconds from a reference date).
    A sliding window of size **SEQ_LEN = 5** produces one training sample per step:

    | Window | T-4 | T-3 | T-2 | T-1 | T (target) | Label |
    |--------|-----|-----|-----|-----|------------|-------|
    | 1      | tx₀ | tx₁ | tx₂ | tx₃ | tx₄        | isFraud(tx₄) |
    | 2      | tx₁ | tx₂ | tx₃ | tx₄ | tx₅        | isFraud(tx₅) |
    | …      |  …  |  …  |  …  |  …  |  …         | … |

    Cardholders with fewer than 5 transactions are **zero-padded** on the left.
    A `Masking(mask_value=0.)` layer in Keras ignores padded steps during backprop.
    """)
