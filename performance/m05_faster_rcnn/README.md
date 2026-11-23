# M-05 Faster R-CNN Object Detector — Performance Matrix

- **Model ID:** M-05
- **Architecture:** Faster R-CNN with ResNet-50-FPN Feature Pyramid Backbone (41.5M params)
- **Checkpoint:** `models/checkpoints/faster_rcnn_synthetic/best_faster_rcnn.pt`
- **Mean Inference Latency:** 129.12 ms (CUDA)

## Benchmark Performance Metrics

| Metric | Score | Note |
| :--- | :---: | :--- |
| **mAP@50** | **1.0000** | Perfect detection at 0.50 IoU |
| **mAP@50-95** | **0.9806** | Primary localization fidelity |
| **Mean Precision** | **0.9925** | High-precision box boundaries |
| **Mean Recall** | **1.0000** | Zero missed UI elements |

## Generated Artifacts
- `confusion_matrix.png` — 7x7 Multi-class detection confusion matrix
- `per_class_f1_map_bar.png` — Per-class F1-Score & mAP@50-95
- `loss_curves.png` — Classifier, Box Reg, Objectness & RPN loss trajectory
- `metrics.json` — Structured JSON metrics schema
