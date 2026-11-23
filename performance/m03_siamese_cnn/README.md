# M-03 Siamese CNN Brand Matcher — Performance Matrix

- **Model ID:** M-03
- **Architecture:** Shared-Weight Siamese ResNet-18 with Cosine Contrastive Loss
- **Embedding Space:** 256-Dimensional L2-Normalized Representation Vector
- **Decision Boundary:** $\tau = 0.82$ (Cosine Similarity Threshold)
- **Checkpoint Evaluated:** `best_siamese.pt` (Restored from Epoch 1)

## Independent Test Metrics (N = 20 Disjoint Pairs)

| Metric | Test Split Score | Validation Score | Note |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **85.00%** | 86.96% | Evaluated on holdout brands |
| **Precision** | **1.0000** | 1.0000 | **0 False Positives** (Zero brand false alarms) |
| **Recall** | **0.6250** | 0.6667 | Strict match criteria |
| **F1-Score** | **0.7692** | 0.8000 | Balanced harmonic mean |
| **ROC-AUC** | **0.8750** | 0.8095 | Strong separability |
| **PR-AUC** | **0.8629** | 0.8433 | Robust precision-recall curve |

## Holdout Brand Evaluation Breakdown
- `amazon` (11 pairs): **100.0% Accuracy** (Mean Pos Sim: 1.0000, Mean Neg Sim: 0.4897)
- `netflix` (11 pairs): **100.0% Accuracy** (Mean Pos Sim: 1.0000, Mean Neg Sim: 0.4751)
- `google` (10 pairs): **70.0% Accuracy** (3 minor false negatives on modified brand variants)

## Generated Artifacts
- `confusion_matrix.png` — Test pair verification matrix
- `similarity_distribution.png` — Scatter of positive vs negative similarity vs decision threshold
- `per_brand_accuracy.png` — Accuracy across unseen holdout brands
- `metrics.json` — Structured JSON metrics schema
