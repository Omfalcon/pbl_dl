# M-01 Lexical URL Multi-Layer Perceptron — Performance Matrix

- **Model ID:** M-01
- **Architecture:** 3-Layer Dense Neural Network built from scratch with custom automatic differentiation
- **Features:** 40 Handcrafted Lexical, Entropy, Brand, and Homoglyph Indicators
- **Selected Training Strategy:** Strategy B (Natural Distribution + Class-Weighted BCE)
- **Calibrated Decision Boundary:** $\tau = 0.050$

## Key Holdout Test Results (N = 4,257 URLs)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **98.66%** |
| **ROC-AUC** | **0.9773** |
| **PR-AUC** | **0.9995** |
| **Macro F1** | **0.8239** |
| **Phishing Class F1** | **0.9932** |
| **Legitimate Class F1** | **0.6545** |

## Confusion Matrix Breakdown
- **True Positives (Phishing):** 4,146
- **True Negatives (Legitimate):** 54
- **False Positives:** 28
- **False Negatives:** 29

## Generated Artifacts
- `confusion_matrix.png` — Heatmap of true vs predicted labels on test set
- `training_curves.png` — Train loss, validation loss, and validation Macro F1 trajectory
- `class_performance_bar.png` — Precision, Recall, and F1 comparison across classes
- `metrics.json` — Structured JSON evaluation schema
