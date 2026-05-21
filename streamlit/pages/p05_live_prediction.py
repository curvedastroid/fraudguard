"""
Page 5 — Live Prediction
=========================
Two subtabs:
  • Manual — enter 5 transactions by hand, score against selected model
  • CSV Upload — upload sample_transactions.csv (or any 25-row CSV),
    score all rows against all three models, show tabular results
"""

import streamlit as st
import pandas as pd
import numpy as np


_MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
_MODEL_COLORS_HEX = {"lstm": "#6366f1", "gru": "#059669", "bilstm": "#be185d"}

# Default values for manual entry (typical legitimate transaction)
_DEFAULTS = {
    "TransactionAmt": [45.0, 120.5, 30.0, 89.99, 250.0],
    "delta_t_seconds": [86400, 3600, 172800, 7200, 43200],
    "D1_days": [1.0, 0.04, 2.0, 0.08, 0.5],
    "C1_count": [1, 2, 1, 3, 1],
    "C2_count": [1, 1, 2, 2, 1],
    "M1_match": [1, 1, 0, 1, 1],
    "M3_match": [1, 0, 1, 1, 0],
}


def _fraud_badge(label: str) -> str:
    if label == "FRAUD":
        return '<span class="badge-fraud">FRAUD</span>'
    return '<span class="badge-legit">Legit</span>'


def _prob_bar(prob: float, color: str) -> str:
    pct = int(prob * 100)
    return (
        f'<div style="background:#f3f4f6;border-radius:9999px;height:10px;width:100%;">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:9999px;"></div>'
        f'</div><span style="font-size:0.78rem;color:#374151;">{pct}%</span>'
    )


# ─── Manual tab ─────────────────────────────────────────────────────────────

def _manual_tab():
    st.markdown("""
    <div class="info-box">
        Enter 5 consecutive transactions for a single cardholder.
        The model treats them as a sequence (T-4 → T) and predicts
        whether the final transaction (T) is fraudulent.
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Model to use for prediction",
        options=list(_MODEL_LABELS.keys()),
        format_func=lambda k: _MODEL_LABELS[k],
    )

    st.markdown("#### Enter 5 transactions (oldest → most recent)")
    rows = []
    for i in range(5):
        t_label = f"T-{4-i}" if i < 4 else "T (latest — will be scored)"
        with st.expander(f"Transaction {i+1}: {t_label}", expanded=(i == 4)):
            c1, c2, c3 = st.columns(3)
            amt  = c1.number_input(f"Amount ($)",    key=f"amt_{i}",
                                   value=_DEFAULTS["TransactionAmt"][i],
                                   min_value=0.0, step=1.0)
            dt   = c2.number_input(f"Delta-t (sec)", key=f"dt_{i}",
                                   value=float(_DEFAULTS["delta_t_seconds"][i]),
                                   min_value=0.0, step=60.0)
            d1   = c3.number_input(f"D1 (days)",     key=f"d1_{i}",
                                   value=_DEFAULTS["D1_days"][i],
                                   min_value=0.0, step=0.1)
            c4, c5, c6, c7 = st.columns(4)
            c1_v = c4.number_input(f"C1 count", key=f"c1_{i}",
                                   value=int(_DEFAULTS["C1_count"][i]),
                                   min_value=0, step=1)
            c2_v = c5.number_input(f"C2 count", key=f"c2_{i}",
                                   value=int(_DEFAULTS["C2_count"][i]),
                                   min_value=0, step=1)
            m1   = c6.selectbox(f"M1 match", [1, 0], key=f"m1_{i}",
                                index=0 if _DEFAULTS["M1_match"][i] else 1)
            m3   = c7.selectbox(f"M3 match", [1, 0], key=f"m3_{i}",
                                index=0 if _DEFAULTS["M3_match"][i] else 1)
        rows.append([amt, dt, d1, c1_v, c2_v, m1, m3, 0, 0, 0, 0, 0, 0, 0, 0])

    if st.button("🔮 Predict", type="primary"):
        from utils.model_loader import load_model
        from utils.predict import predict_sequence, SEQ_LEN, N_FEATURES

        seq = np.array(rows, dtype=np.float32)  # (5, 15)
        model = load_model(model_choice)

        prob, label = predict_sequence(model, seq) if model else (
            float(np.clip(rows[-1][0] / 2000 + (1 - rows[-1][5]) * 0.1, 0, 0.99)),
            "FRAUD" if rows[-1][0] > 900 or rows[-1][5] == 0 else "Legit",
        )

        if model is None:
            st.warning(f"Model `{model_choice}` not trained yet — showing demo inference.")

        st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Model", _MODEL_LABELS[model_choice])
        res_col2.metric("Fraud Probability", f"{prob:.1%}")
        res_col3.metric("Prediction", label)

        color = "#ef4444" if label == "FRAUD" else "#10b981"
        st.markdown(
            f'<div style="background:{color}22;border:2px solid {color};'
            f'border-radius:10px;padding:16px 24px;text-align:center;margin-top:10px;">'
            f'<span style="font-size:1.6rem;">'
            f'{"🚨" if label == "FRAUD" else "✅"}</span>'
            f'<span style="font-size:1.3rem;font-weight:800;color:{color};margin-left:12px;">'
            f'{label}</span>'
            f'<br><span style="font-size:0.85rem;color:#6b7280;">Probability: {prob:.4f} '
            f'· Threshold: 0.45</span></div>',
            unsafe_allow_html=True,
        )


# ─── CSV Upload tab ──────────────────────────────────────────────────────────

def _csv_tab():
    st.markdown("""
    <div class="info-box">
        Upload the <b>sample_transactions.csv</b> (25 rows, mixed fraud/legit) or
        any CSV matching the same column schema. All three models will score every
        row and the results appear in a colour-coded table below.
    </div>
    """, unsafe_allow_html=True)

    # Option to load bundled sample file
    col_a, col_b = st.columns([1, 2])
    with col_a:
        use_sample = st.checkbox("Use bundled sample CSV (25 transactions)", value=True)

    if use_sample:
        from utils.data_loader import load_presentation_csv
        df = load_presentation_csv()
        if df is None:
            st.error("Bundled sample CSV not found at `data/presentation/sample_transactions.csv`.")
            return
        st.success(f"Loaded bundled sample: {len(df)} transactions")
    else:
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Required columns: TransactionAmt, delta_t_seconds, D1_days, "
                 "C1_count, C2_count, M1_match, M3_match",
        )
        if uploaded is None:
            st.info("Upload a CSV file to continue.")
            return
        df = pd.read_csv(uploaded)
        st.success(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Show raw data preview
    with st.expander("Preview raw CSV"):
        st.dataframe(df, use_container_width=True, height=220, hide_index=True)

    # Model selection for scoring
    models_to_score = st.multiselect(
        "Models to score",
        options=list(_MODEL_LABELS.keys()),
        default=list(_MODEL_LABELS.keys()),
        format_func=lambda k: _MODEL_LABELS[k],
    )
    threshold = st.slider("Fraud threshold", 0.10, 0.90, 0.45, 0.01)

    if st.button("🔮 Score All Rows", type="primary") and models_to_score:
        from utils.model_loader import load_model
        from utils.predict import predict_csv

        with st.spinner("Running inference..."):
            models_dict = {k: load_model(k) for k in models_to_score}
            scored = predict_csv(df, models_dict, scaler=None, threshold=threshold)

        st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Prediction Results")

        # Build display DataFrame
        display_cols = ["TransactionID", "TransactionAmt", "isFraud", "fraud_label"]
        for k in models_to_score:
            display_cols += [f"fraud_prob_{k}", f"fraud_pred_{k}"]

        # Only keep columns that exist
        display_cols = [c for c in display_cols if c in scored.columns]
        result_df = scored[display_cols].copy()

        # Rename for readability
        rename = {
            "TransactionID": "Txn ID",
            "TransactionAmt": "Amount ($)",
            "isFraud": "Actual",
            "fraud_label": "Actual Label",
        }
        for k in models_to_score:
            rename[f"fraud_prob_{k}"] = f"{_MODEL_LABELS[k]} Prob"
            rename[f"fraud_pred_{k}"] = f"{_MODEL_LABELS[k]} Pred"
        result_df.rename(columns=rename, inplace=True)

        # Style: highlight fraud predictions
        def highlight_fraud(val):
            if val == "FRAUD":
                return "background-color:#fee2e2;color:#dc2626;font-weight:700"
            elif val == "Legit":
                return "background-color:#d1fae5;color:#065f46;"
            return ""

        def highlight_actual(val):
            if val == 1 or val == "FRAUD":
                return "background-color:#fef3c7;color:#92400e;font-weight:700"
            return ""

        pred_cols = [f"{_MODEL_LABELS[k]} Pred" for k in models_to_score
                     if f"{_MODEL_LABELS[k]} Pred" in result_df.columns]
        actual_cols = [c for c in ["Actual", "Actual Label"] if c in result_df.columns]

        styled = result_df.style
        for col in pred_cols:
            styled = styled.applymap(highlight_fraud, subset=[col])
        for col in actual_cols:
            styled = styled.applymap(highlight_actual, subset=[col])
        for col in [f"{_MODEL_LABELS[k]} Prob" for k in models_to_score
                    if f"{_MODEL_LABELS[k]} Prob" in result_df.columns]:
            styled = styled.format({col: "{:.3f}"})

        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

        # Summary stats
        st.markdown("#### Summary")
        sum_cols = st.columns(len(models_to_score) + 1)
        sum_cols[0].metric("Total rows", len(result_df))
        for i, k in enumerate(models_to_score):
            pred_col = f"{_MODEL_LABELS[k]} Pred"
            if pred_col in result_df.columns:
                n_fraud = (result_df[pred_col] == "FRAUD").sum()
                sum_cols[i + 1].metric(
                    f"{_MODEL_LABELS[k]} flagged",
                    f"{n_fraud} / {len(result_df)}",
                    f"{n_fraud/len(result_df):.0%} fraud rate",
                )

        st.download_button(
            "⬇️ Download results CSV",
            data=result_df.to_csv(index=False),
            file_name="fraudguard_predictions.csv",
            mime="text/csv",
        )


# ─── Main render ─────────────────────────────────────────────────────────────

def render():
    st.markdown("## 🔮 Live Prediction")

    tab_manual, tab_csv = st.tabs(["✏️ Manual Input (5 transactions)", "📁 CSV Upload & Batch Score"])

    with tab_manual:
        _manual_tab()

    with tab_csv:
        _csv_tab()
