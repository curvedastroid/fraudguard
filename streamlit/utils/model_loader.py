"""
model_loader.py
===============
Loads trained Keras models from FraudGuard/models/.
Returns None silently when a model file is not yet present,
so the app degrades gracefully to demo mode.
"""

from pathlib import Path
from functools import lru_cache

from .data_loader import MODELS_DIR

_MODEL_PATHS = {
    "lstm":   MODELS_DIR / "lstm"   / "lstm_model.keras",
    "gru":    MODELS_DIR / "gru"    / "gru_model.keras",
    "bilstm": MODELS_DIR / "bilstm" / "bilstm_model.keras",
}

_MODEL_LABELS = {
    "lstm":   "LSTM",
    "gru":    "GRU",
    "bilstm": "Bidirectional LSTM",
}


@lru_cache(maxsize=3)
def load_model(model_key: str):
    """
    Load and cache a Keras model.  Returns None if file doesn't exist.
    Import tensorflow lazily to avoid slow startup when not needed.
    """
    p = _MODEL_PATHS.get(model_key)
    if p is None or not p.exists():
        return None
    try:
        import tensorflow as tf  # noqa: F401 — needed to deserialise .keras
        from tensorflow import keras
        return keras.models.load_model(str(p))
    except Exception as e:
        import streamlit as st
        st.warning(f"Could not load {model_key} model: {e}")
        return None


def models_available() -> dict:
    """Return {model_key: bool} indicating which model files exist."""
    return {k: v.exists() for k, v in _MODEL_PATHS.items()}


def label(model_key: str) -> str:
    return _MODEL_LABELS.get(model_key, model_key.upper())
