# FraudGuard

FraudGuard is a Streamlit app for sequential credit-card fraud detection using
LSTM, GRU, and Bidirectional LSTM models.

## Local Setup

Use Python 3.10 to match the Render deployment environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit/app.py
```

## Render Deployment

This repository includes a `render.yaml` Blueprint. To deploy:

1. Push the latest `main` branch to GitHub.
2. Open the Render dashboard.
3. Choose **New +** then **Blueprint**.
4. Connect `curvedastroid/fraudguard`.
5. Keep the default Blueprint path: `render.yaml`.
6. Deploy the detected `fraudguard-app` service.

Render builds from the repository root so the Streamlit app can read its
supporting `data/`, `models/`, and `evaluation/` folders.

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

and start the app with:

```bash
streamlit run streamlit/app.py --server.port $PORT --server.address 0.0.0.0
```

## Repository Notes

- `data/raw/` is ignored because Kaggle source files are too large for Git.
- `data/processed/*.npy` is ignored because training tensors are large and not
  required for the hosted Streamlit app.
- The small trained `.keras` model files are committed so the deployed app can
  run live inference.
