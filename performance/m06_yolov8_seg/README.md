# M-06 YOLOv8n-Seg Instance Segmenter — Performance Matrix

- **Model ID:** M-06
- **Architecture:** Ultralytics YOLOv8n-Seg (Dual Bounding Box + Pixel-Level Mask Heads)
- **Checkpoint:** `models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt`
- **Checkpoint Size:** 6.45 MB

## Dual Localization Metrics

| Evaluation Domain | mAP@50 | mAP@50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Bounding Box Head** | **0.9950** | **0.8889** | **0.9525** | **1.0000** |
| **Instance Mask Head** | **0.9950** | **0.8927** | **0.9525** | **1.0000** |

## Generated Artifacts
- `BoxF1_curve.png` & `MaskF1_curve.png` — F1 score curves for boxes and masks
- `BoxPR_curve.png` & `MaskPR_curve.png` — Precision-Recall curves
- `confusion_matrix.png` & `confusion_matrix_normalized.png` — Detection confusion matrices
- `box_vs_mask_comparison.png` — Side-by-side mAP@50-95 comparison per class
- `results.png` — Multi-task loss training dynamics
- `metrics.json` — Structured JSON metrics schema
