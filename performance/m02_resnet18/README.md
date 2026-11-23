# M-02 ResNet-18 Visual Classifier — Performance Matrix

- **Model ID:** M-02
- **Backbone Architecture:** ResNet-18 (ImageNet Pretrained)
- **Classification Head:** AdaptiveAvgPool2d -> Flatten -> Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2)
- **Input Preprocessing:** TopViewportCrop (1:1 aspect ratio focusing on authentication header)
- **Best Validation Checkpoint:** Epoch 9 (`best_model.pt`)

## Validation Performance Metrics (N = 262 Screenshots)

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **83.59%** |
| **ROC-AUC** | **0.8572** |
| **PR-AUC** | **0.9768** |
| **Macro F1** | **0.7037** |
| **Weighted F1** | **0.8623** |
| **Phishing F1** | **0.9016** |
| **Legitimate F1** | **0.5057** |

## Confusion Matrix Breakdown
- **True Positives (Phishing):** 197
- **True Negatives (Legitimate):** 22
- **False Positives:** 4
- **False Negatives:** 39

## Generated Artifacts
- `confusion_matrix.png` — Visual classification heatmap
- `per_class_metrics_bar.png` — Precision, Recall, and F1 per class
- `confidence_statistics.png` — Prediction probability distribution by outcome
- `metrics.json` — Structured JSON metrics schema
