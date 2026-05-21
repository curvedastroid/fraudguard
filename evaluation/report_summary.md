# FraudGuard — Evaluation Report

## Best Model: GRU

### Summary
The GRU model achieves the highest AUC-ROC of 0.8323 and F1 score of 0.1788
on the held-out test set, making it the recommended model for the FraudGuard production deployment.

### Comparison Table
|   Rank | Model   |   AUC-ROC |   F1 (fraud) |   Precision |   Recall |   Parameters |   Train time (min) |
|-------:|:--------|----------:|-------------:|------------:|---------:|-------------:|-------------------:|
|      1 | GRU     |    0.8323 |       0.1788 |      0.1013 |   0.7619 |        25377 |               49.7 |
|      2 | LSTM    |    0.8305 |       0.1774 |      0.1005 |   0.755  |        33313 |               47.3 |
|      3 | BiLSTM  |    0.8182 |       0.1687 |      0.0952 |   0.7414 |        83009 |               50.2 |

### Methodology
- All models trained on identical 70/15/15 train/val/test splits
- Same 15 features per timestep, same SEQ_LEN=5 window
- Same class weighting (computed from training set)
- Threshold tuned to 0.45 for maximum F1 on validation set
- Evaluated using AUC-ROC, F1, Precision, Recall, and parameter count

### Next Steps
- Deploy GRU in Streamlit app as the default model
- All three models are selectable in the live prediction page
- See evaluation/roc_comparison.png and evaluation/confusion_matrices.png
