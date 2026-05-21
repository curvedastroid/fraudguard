"""
predict.py
==========
Inference helpers for FraudGuard.

Two entry points:

1. predict_sequence(model, seq_array, threshold=0.45)
   Feed a (1, SEQ_LEN, n_features) tensor, get back (prob, label).

2. predict_csv(df, model, scaler, meta, threshold=0.45)
   Take a pandas DataFrame of raw transaction rows (like the presentation CSV),
   build mini-sequences, run inference, and return the DataFrame with extra
   columns: fraud_prob_{model_key}, fraud_pred_{model_key}.

For the CSV demo we use a single-row "sequence" (SEQ_LEN=1, padded to SEQ_LEN
with zeros) so each row in the uploaded CSV is scored independently —
this is intentional for presentation purposes.
"""

import numpy as np
import pandas as pd

# Features expected per row (must match notebook 00 feature list)
FEATURES_15 = [
    "TransactionAmt",
    "delta_t_seconds",
    "D1_days",
    "C1_count",
    "C2_count",
    "M1_match",
    "M3_match",
    # remaining 8 — may be absent in presentation CSV, will be zero-filled
    "C3_count", "C4_count", "C5_count",
    "M6_match",
    "V12", "V20", "V37",
    "addr1_norm",
]
SEQ_LEN = 5
N_FEATURES = 15
THRESHOLD = 0.45


def predict_sequence(model, seq_array: np.ndarray, threshold: float = THRESHOLD):
    """
    Parameters
    ----------
    model      : loaded Keras model
    seq_array  : shape (SEQ_LEN, N_FEATURES) or (1, SEQ_LEN, N_FEATURES)
    threshold  : classification threshold

    Returns
    -------
    prob  : float  — fraud probability
    label : str    — "FRAUD" or "Legit"
    """
    if seq_array.ndim == 2:
        seq_array = seq_array[np.newaxis, ...]  # add batch dim
    prob = float(model.predict(seq_array, verbose=0)[0][0])
    label = "FRAUD" if prob >= threshold else "Legit"
    return prob, label


def _row_to_tensor(row: pd.Series, scaler=None) -> np.ndarray:
    """Convert a single DataFrame row to shape (SEQ_LEN, N_FEATURES), zero-padded."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for i, col in enumerate(FEATURES_15):
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    vec[i] = float(val)
                except (ValueError, TypeError):
                    pass
    # Scale if scaler provided
    if scaler is not None:
        try:
            vec = scaler.transform(vec.reshape(1, -1))[0]
        except Exception:
            pass
    # Pad to SEQ_LEN (place the single step at the last position, rest = 0)
    tensor = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    tensor[-1] = vec
    return tensor


def predict_csv(df: pd.DataFrame, models: dict, scaler=None,
                threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Score every row in df against all provided models.

    Parameters
    ----------
    df      : DataFrame — presentation CSV or user-uploaded CSV
    models  : {model_key: keras_model_or_None}
    scaler  : fitted scaler or None
    threshold

    Returns
    -------
    df copy with added columns per model:
        fraud_prob_<key>   (float, 0-1)
        fraud_pred_<key>   ("FRAUD" | "Legit")
    """
    result = df.copy()
    for key, model in models.items():
        probs, preds = [], []
        for _, row in df.iterrows():
            if model is not None:
                tensor = _row_to_tensor(row, scaler)
                p, lbl = predict_sequence(model, tensor, threshold)
            else:
                # Demo mode: derive plausible mock probability from TransactionAmt
                # (higher amount + M1_match=1 → more suspicious)
                amt = float(row.get("TransactionAmt", 50))
                m1  = float(row.get("M1_match", 0))
                base = min(amt / 2000, 0.85)
                p = round(base * (1.3 if m1 else 1.0), 3)
                p = min(p, 0.99)
                lbl = "FRAUD" if p >= threshold else "Legit"
            probs.append(p)
            preds.append(lbl)
        result[f"fraud_prob_{key}"] = probs
        result[f"fraud_pred_{key}"] = preds
    return result
