"""
data_loader.py
==============
Loads processed tensors, scaler, class weights, and per-model metrics
from the FraudGuard data/processed and evaluation directories.

All functions return None (or empty dicts) gracefully when files don't
exist yet — allowing the app to run in "demo mode" before training.
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Root resolution — works whether app.py is run from FraudGuard/ or
# FraudGuard/streamlit/
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    here = Path(__file__).resolve()
    # Walk up until we find a directory that contains "notebooks/"
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (parent / "notebooks").exists():
            return parent
    return here.parent.parent  # fallback


ROOT = _project_root()
DATA_PROC = ROOT / "data" / "processed"
DATA_PRES = ROOT / "data" / "presentation"
EVAL_DIR  = ROOT / "evaluation"
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Tensors & scaler
# ---------------------------------------------------------------------------

def load_tensors():
    """Return (X_test, y_test) numpy arrays, or (None, None)."""
    x_path = DATA_PROC / "X_test.npy"
    y_path = DATA_PROC / "y_test.npy"
    if x_path.exists() and y_path.exists():
        return np.load(str(x_path)), np.load(str(y_path))
    return None, None


def load_scaler():
    """Return fitted MinMaxScaler or None."""
    p = DATA_PROC / "scaler.pkl"
    if p.exists():
        with open(str(p), "rb") as f:
            return pickle.load(f)
    return None


def load_class_weights():
    """Return class_weights dict {0: float, 1: float} or default."""
    p = DATA_PROC / "class_weights.json"
    if p.exists():
        with open(str(p)) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return {0: 1.0, 1: 28.0}


def load_meta():
    """Return meta.json dict (SEQ_LEN, n_features, feature_names, etc.) or {}."""
    p = DATA_PROC / "meta.json"
    if p.exists():
        with open(str(p)) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Per-model metrics  (written by each training notebook)
# ---------------------------------------------------------------------------

_MODEL_KEYS = ["lstm", "gru", "bilstm"]
_METRICS_FILES = {
    "lstm":   MODELS_DIR / "lstm"   / "lstm_metrics.json",
    "gru":    MODELS_DIR / "gru"    / "gru_metrics.json",
    "bilstm": MODELS_DIR / "bilstm" / "bilstm_metrics.json",
}
_HISTORY_FILES = {
    "lstm":   MODELS_DIR / "lstm"   / "lstm_history.json",
    "gru":    MODELS_DIR / "gru"    / "gru_history.json",
    "bilstm": MODELS_DIR / "bilstm" / "bilstm_history.json",
}

# Demo / placeholder data shown before training completes
DEMO_METRICS = {
    "lstm":   {"auc": 0.9312, "f1": 0.7841, "precision": 0.8103, "recall": 0.7595,
               "params": 42_113, "train_time_s": 184},
    "gru":    {"auc": 0.9188, "f1": 0.7703, "precision": 0.7952, "recall": 0.7465,
               "params": 32_449, "train_time_s": 141},
    "bilstm": {"auc": 0.9401, "f1": 0.7989, "precision": 0.8247, "recall": 0.7745,
               "params": 83_009, "train_time_s": 267},
}

DEMO_HISTORY = {
    "loss":     [0.42, 0.31, 0.25, 0.21, 0.19, 0.17, 0.16, 0.155, 0.150, 0.148],
    "val_loss": [0.45, 0.34, 0.28, 0.24, 0.22, 0.21, 0.205, 0.202, 0.200, 0.199],
    "auc":      [0.71, 0.80, 0.85, 0.88, 0.90, 0.916, 0.921, 0.929, 0.931, 0.933],
    "val_auc":  [0.68, 0.77, 0.82, 0.86, 0.88, 0.901, 0.908, 0.915, 0.918, 0.920],
}


def load_metrics(model_key: str) -> dict:
    """Load metrics JSON for a model; fall back to DEMO_METRICS."""
    p = _METRICS_FILES.get(model_key)
    if p and p.exists():
        with open(str(p)) as f:
            return json.load(f)
    return DEMO_METRICS.get(model_key, {})


def load_history(model_key: str) -> dict:
    """Load training history JSON; fall back to DEMO_HISTORY."""
    p = _HISTORY_FILES.get(model_key)
    if p and p.exists():
        with open(str(p)) as f:
            return json.load(f)
    return DEMO_HISTORY.copy()


def load_all_metrics() -> dict:
    """Return {model_key: metrics_dict} for all three models."""
    return {k: load_metrics(k) for k in _MODEL_KEYS}


# ---------------------------------------------------------------------------
# Evaluation artefacts
# ---------------------------------------------------------------------------

def load_comparison_csv():
    """Return pandas DataFrame from evaluation/comparison_results.csv or None."""
    import pandas as pd
    p = EVAL_DIR / "comparison_results.csv"
    if p.exists():
        return pd.read_csv(str(p))
    return None


def load_presentation_csv():
    """Return pandas DataFrame from data/presentation/sample_transactions.csv."""
    import pandas as pd
    p = DATA_PRES / "sample_transactions.csv"
    if p.exists():
        return pd.read_csv(str(p))
    return None
