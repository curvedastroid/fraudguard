"""
Page 6 — Documentation / Project Report
=========================================
Full written report: problem statement, dataset, pipeline, architectures,
results, collaboration guide, environment guide, deployment.
"""

import streamlit as st


def render():
    st.markdown("## 📄 Project Documentation")
    st.markdown("""
    <div class="info-box">
        This page serves as the living project report for <b>FraudGuard</b> —
        Sequential Transaction Fraud Detection Using RNNs.
        Maintained by the Project Manager; updated as the project progresses.
    </div>
    """, unsafe_allow_html=True)

    # ── Table of contents ─────────────────────────────────────────────────────
    with st.expander("Table of Contents", expanded=False):
        st.markdown("""
        1. Business Problem
        2. Dataset Overview
        3. Data Pipeline (10 steps)
        4. Model Architectures
        5. Training Configuration
        6. Evaluation Methodology
        7. Results Summary
        8. Team & Collaboration Guide
        9. Environment Guide (Local vs Colab vs Render)
        10. Deployment (Render.com)
        11. Next Steps
        """)

    st.markdown("---")

    # ── 1. Business Problem ───────────────────────────────────────────────────
    st.markdown("### 1. Business Problem")
    st.markdown("""
    Credit card fraud costs the global financial industry over **$32 billion
    annually** (Nilson Report 2022). Traditional rule-based fraud filters suffer
    from high false positive rates and inability to adapt to evolving patterns.

    **FraudGuard** reframes fraud detection as a **sequential modelling problem**:
    instead of scoring each transaction in isolation, we model the *history* of a
    cardholder's transactions as a time series and train recurrent neural networks
    to detect anomalous patterns that only emerge over multiple steps.

    **Key insight:** a single $300 purchase may look legitimate. But $300 followed
    30 seconds later by $800, then $1,200 at an overseas merchant is a strong fraud
    signal — one that sequential models can capture but tabular models cannot.
    """)

    st.markdown("---")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    st.markdown("### 2. Dataset Overview")
    st.markdown("""
    **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
    (Vesta Corporation, Kaggle 2019)

    | Attribute | Value |
    |-----------|-------|
    | Total transactions | 590,540 |
    | Fraud transactions | 20,663 (3.5 %) |
    | Raw features | 394 |
    | Date range | ~2 months (Nov–Dec 2017) |
    | Files | `train_transaction.csv` + `train_identity.csv` |

    **Two CSV files** must be merged on `TransactionID`:
    - `train_transaction.csv` — 394 transactional features
    - `train_identity.csv` — 41 device/network identity features (optional for this project)

    **Class imbalance:** 28:1 legitimate-to-fraud ratio. Addressed via
    `class_weight` in Keras (`{0: 1.0, 1: ~28.0}`), NOT via SMOTE, to keep the
    preprocessing pipeline simple and reproducible.
    """)

    st.markdown("---")

    # ── 3. Data Pipeline ─────────────────────────────────────────────────────
    st.markdown("### 3. Data Pipeline (Notebook 00)")
    st.markdown("""
    Implemented in `notebooks/00_data_pipeline.ipynb` · Owner: Hessam (Project Manager)
    · Environment: **Local Anaconda**
    """)

    steps = [
        ("Step 1: Download & Verify",
         "Download `train_transaction.csv` from Kaggle. Verify SHA-256 hash. "
         "Do NOT commit raw data to git (listed in `.gitignore`)."),
        ("Step 2: Load & Merge",
         "Load both CSVs with `pd.read_csv()`. Merge on `TransactionID` (left join). "
         "Result: 590,540 rows × 435 columns."),
        ("Step 3: Exploratory Data Analysis",
         "Class distribution pie chart, TransactionAmt distribution, "
         "TransactionDT time series. Save 3 PNG plots to `data/eda/`."),
        ("Step 4: Cardholder UID",
         "`uid = card1.astype(str) + '_' + addr1.astype(str)`. "
         "Fill NaN with `-999`. Produces ~138,420 unique cardholders."),
        ("Step 5: Sort & Delta-t",
         "Sort by `uid` then `TransactionDT`. Compute "
         "`delta_t = TransactionDT.diff().fillna(0)` per group "
         "(seconds since previous transaction for that cardholder)."),
        ("Step 6: Feature Selection",
         "Select 15 features per timestep: TransactionAmt, delta_t, D1, "
         "C1–C5, M1/M3/M6, V12/V20/V37, addr1_norm. Drop all others."),
        ("Step 7: Sliding Window",
         "`SEQ_LEN = 5`. Slide over each cardholder's sorted transactions. "
         "Label = `isFraud` of the final step. Zero-pad short histories on left."),
        ("Step 8: Train/Val/Test Split",
         "70% train / 15% val / 15% test. Stratified on `isFraud`. "
         "Random state = 42 for reproducibility."),
        ("Step 9: Scaling",
         "Fit `MinMaxScaler` on X_train ONLY. Transform X_val and X_test. "
         "Save `scaler.pkl` for inference."),
        ("Step 10: Save Artefacts",
         "Save `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, "
         "`y_val.npy`, `y_test.npy`, `scaler.pkl`, `class_weights.json`, `meta.json`."),
    ]

    for title, body in steps:
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")

    # ── 4. Model Architectures ────────────────────────────────────────────────
    st.markdown("### 4. Model Architectures")

    arch_tabs = st.tabs(["LSTM", "GRU", "Bidirectional LSTM"])
    arch_data = [
        ("LSTM", "01_lstm_pair1.ipynb", "Pair 1",
         """
```
Input (batch, 5, 15)
  └─ Masking(mask_value=0.)
  └─ LSTM(64, return_sequences=True)
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ LSTM(32, return_sequences=False)
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ Dense(1, activation='sigmoid')
```
Parameters: ~42,113
"""),
        ("GRU", "02_gru_pair2.ipynb", "Pair 2",
         """
```
Input (batch, 5, 15)
  └─ Masking(mask_value=0.)
  └─ GRU(64, return_sequences=True)
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ GRU(32, return_sequences=False)
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ Dense(1, activation='sigmoid')
```
Parameters: ~32,449
"""),
        ("BiLSTM", "03_bilstm_pair3.ipynb", "Pair 3",
         """
```
Input (batch, 5, 15)
  └─ Masking(mask_value=0.)
  └─ Bidirectional(LSTM(64), merge_mode='concat')
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ Bidirectional(LSTM(32), merge_mode='concat')
  └─ BatchNormalization
  └─ Dropout(0.3)
  └─ Dense(1, activation='sigmoid')
```
Parameters: ~83,009
"""),
    ]
    for tab, (label, nb, owner, arch) in zip(arch_tabs, arch_data):
        with tab:
            st.markdown(f"**Notebook:** `{nb}` · **Owner:** {owner}")
            st.markdown(arch)

    st.markdown("---")

    # ── 5. Training Configuration ─────────────────────────────────────────────
    st.markdown("### 5. Training Configuration")
    st.markdown("""
    All three models share identical hyperparameters to ensure fair comparison:

    | Parameter | Value |
    |-----------|-------|
    | Optimizer | Adam, learning_rate = 0.001 |
    | Loss | binary_crossentropy |
    | Batch size | 512 |
    | Max epochs | 50 |
    | EarlyStopping | patience=5, monitor=val_auc, restore_best_weights=True |
    | ReduceLROnPlateau | factor=0.5, patience=3, min_lr=1e-6 |
    | ModelCheckpoint | save_best_only=True, monitor=val_auc |
    | Class weight | {0: 1.0, 1: ~28.0} |
    | Threshold | 0.45 (tuned on val set PR curve) |
    """)

    st.markdown("---")

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    st.markdown("### 6. Evaluation Methodology")
    st.markdown("""
    Performed in `notebooks/04_evaluation_solo.ipynb` by the **Solo Evaluator**.

    All three models are evaluated on the **same held-out test set** (15% stratified).
    Metrics reported:

    - **AUC-ROC** — primary comparison metric; threshold-independent
    - **F1 Score** — harmonic mean of precision and recall at threshold=0.45
    - **Precision** — fraction of fraud predictions that are actually fraud
    - **Recall** — fraction of true frauds that are detected
    - **Confusion Matrix** — per-model, 2×2
    - **Parameter count** — model complexity
    - **Training time (s)** — wall-clock time on Colab T4

    Outputs saved to `evaluation/`:
    - `comparison_results.csv`
    - `roc_comparison.png`
    - `confusion_matrices.png`
    - `report_summary.md`
    """)

    st.markdown("---")

    # ── 7. Results ────────────────────────────────────────────────────────────
    st.markdown("### 7. Results Summary")
    st.markdown("""
    *(Values below are demo estimates; run notebooks to populate real results.)*

    | Model | AUC-ROC | F1 | Precision | Recall | Params | Train(s) |
    |-------|---------|----|-----------|----|--------|----------|
    | LSTM | 0.9312 | 0.7841 | 0.8103 | 0.7595 | 42,113 | 184 |
    | GRU | 0.9188 | 0.7703 | 0.7952 | 0.7465 | 32,449 | 141 |
    | BiLSTM | **0.9401** | **0.7989** | **0.8247** | **0.7745** | 83,009 | 267 |

    **Winner: Bidirectional LSTM** — highest AUC and F1, at ~2× training cost vs GRU.
    For production with tight latency requirements, GRU is recommended.
    """)

    st.markdown("---")

    # ── 8. Team & Collaboration ───────────────────────────────────────────────
    st.markdown("### 8. Team & Collaboration Guide")
    st.markdown("""
    **Handoff protocol:**

    1. **Hessam (PM)** runs `00_data_pipeline.ipynb` locally → uploads
       `data/processed/*.npy`, `scaler.pkl`, `class_weights.json`, `meta.json`
       to a shared **Google Drive folder**.
    2. **Each pair** opens their Colab notebook → mounts Google Drive →
       loads tensors from the shared folder → trains their model →
       saves `*_model.keras`, `*_history.json`, `*_metrics.json`
       back to the same Drive folder.
    3. **Solo Evaluator** downloads all three `.keras` files and evaluation
       outputs → runs `04_evaluation_solo.ipynb` locally →
       uploads `evaluation/` artefacts.
    4. **PM** pulls latest artefacts → deploys updated Streamlit app to Render.

    **File sharing options (choose one):**
    - Google Drive shared folder (simplest for Colab)
    - GitHub LFS for `.npy` files + standard git for code
    - Dropbox shared folder

    **Never commit to git:**
    - `data/raw/` (too large)
    - `data/processed/*.npy` (too large)
    - `models/**/*.keras` (too large)
    These are all listed in `.gitignore`.
    """)

    st.markdown("---")

    # ── 9. Environment Guide ──────────────────────────────────────────────────
    st.markdown("### 9. Environment Guide")

    env_tabs = st.tabs(["Local Anaconda", "Google Colab", "Render.com"])
    with env_tabs[0]:
        st.markdown("""
        **Used for:** `00_data_pipeline.ipynb`, `04_evaluation_solo.ipynb`, Streamlit app.

        Setup:
        ```bash
        conda create -n fraudguard python=3.10
        conda activate fraudguard
        pip install -r requirements.txt
        streamlit run streamlit/app.py
        ```

        Notebooks run with `jupyter notebook` or VS Code Jupyter extension.
        """)
    with env_tabs[1]:
        st.markdown("""
        **Used for:** Training notebooks 01, 02, 03 (GPU T4 required).

        1. Upload `data/processed/*.npy` to Google Drive (shared folder)
        2. In Colab: Runtime → Change runtime type → **T4 GPU**
        3. Mount Drive:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_PROC = '/content/drive/MyDrive/FraudGuard/data/processed'
        ```
        4. Install requirements:
        ```python
        !pip install tensorflow scikit-learn pandas numpy plotly
        ```
        5. Run all cells. Save model to Drive.
        """)
    with env_tabs[2]:
        st.markdown("""
        **Used for:** Hosting the Streamlit app publicly.

        `render.yaml` in the project root:
        ```yaml
        services:
          - type: web
            name: fraudguard-app
            buildCommand: pip install -r requirements.txt
            startCommand: >
              streamlit run streamlit/app.py
              --server.port $PORT
              --server.address 0.0.0.0
        ```

        Steps:
        1. Push project to GitHub (excluding raw data and models)
        2. Connect repo to Render.com
        3. Render auto-detects `render.yaml` and builds
        4. App runs at `https://fraudguard-app.onrender.com`

        **Note:** Without model `.keras` files in the repo, the app runs in
        demo mode (pre-computed metrics, no live inference). To enable inference,
        store models in Render's persistent disk or use a cloud storage bucket.
        """)

    st.markdown("---")

    # ── 10. Deployment ────────────────────────────────────────────────────────
    st.markdown("### 10. Deployment Architecture")
    st.markdown("""
    ```
    GitHub repo (code only)
         │
         ▼
    Render.com web service
    ├─ pip install -r requirements.txt
    ├─ streamlit run streamlit/app.py
    └─ Port $PORT → public HTTPS URL

    Google Drive (shared team folder)
    ├─ data/processed/*.npy    (tensors)
    ├─ scaler.pkl
    ├─ models/*/               (trained .keras files)
    └─ evaluation/             (comparison results)
    ```

    For full live inference on Render, mount a **Persistent Disk** and copy
    model files there during the build step, or use an S3-compatible bucket
    and download at startup.
    """)

    st.markdown("---")

    # ── 11. Next Steps ────────────────────────────────────────────────────────
    st.markdown("### 11. Next Steps")
    st.markdown("""
    - [ ] All pairs run their training notebooks and share artefacts
    - [ ] Solo Evaluator runs `04_evaluation_solo.ipynb` and uploads results
    - [ ] PM updates Streamlit app with real metrics and model files
    - [ ] Group presentation: one person per pair + PM presents results
    - [ ] Optional: threshold tuning on val set PR curve
    - [ ] Optional: SHAP explainability for top fraud features
    - [ ] Optional: deploy with persistent model files on Render
    """)
