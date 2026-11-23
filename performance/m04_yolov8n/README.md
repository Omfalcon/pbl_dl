# M-04 YOLOv8n UI Component Detector — Performance Matrix

- **Model ID:** M-04
- **Architecture:** Ultralytics YOLOv8n Nano (3.2 Million Parameters)
- **Target Taxonomy (6 Classes):** `brand_logo`, `login_form`, `username_field`, `password_field`, `submit_button`, `payment_element`
- **Checkpoint:** `models/checkpoints/yolov8n_synthetic/run1/weights/best.pt`

## Executive Benchmark Scores

| Metric | Overall Test Score | Standard |
| :--- | :---: | :--- |
| **mAP@50** | **0.9950** | Mean Average Precision at IoU 0.50 |
| **mAP@50-95** | **0.9699** | COCO-Standard Precision Average |
| **Precision (Mean)** | **0.9973** | True Positive Localization Ratio |
| **Recall (Mean)** | **1.0000** | 100% Component Discovery Rate |

## Generated Artifacts
- `BoxF1_curve.png` — F1 score vs confidence threshold curve across UI classes
- `BoxPR_curve.png` — Precision-Recall trade-off curve
- `confusion_matrix.png` & `confusion_matrix_normalized.png` — Detection classification matrix
- `results.png` — Training trajectory (Box Loss, Class Loss, DFL Loss)
- `per_class_f1_map_bar.png` — Comparative bar chart across all 6 UI classes
- `metrics.json` — Structured JSON metrics schema
