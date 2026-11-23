# M-07 URL Sequence Bi-LSTM — Performance Matrix

- **Model ID:** M-07
- **Architecture:** Character-level Bidirectional LSTM + Self-Attentive Temporal Pooling Head
- **Vocabulary:** 100 Character Tokens + Length Embedding
- **Calibrated Decision Boundary:** $\tau = 0.010$
- **Inference Latency:** 0.072 ms / URL

## Holdout Test Performance (N = 4,257 URLs)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **98.69%** |
| **ROC-AUC** | **0.9666** |
| **PR-AUC** | **0.9991** |
| **Macro F1** | **0.7779** |
| **Phishing F1** | **0.9933** |
| **Legitimate F1** | **0.5625** |

## Confusion Matrix Breakdown
- **True Positives (Phishing):** 4,165
- **True Negatives (Legitimate):** 36
- **False Positives:** 46
- **False Negatives:** 10

## Generated Artifacts
- `confusion_matrix.png` — Heatmap of test classifications
- `per_class_metrics_bar.png` — Class-level Precision, Recall, and F1 comparison
- `training_curves.png` — Binary Cross-Entropy loss and Macro F1 trajectory across epochs
- `metrics.json` — Structured JSON metrics schema
