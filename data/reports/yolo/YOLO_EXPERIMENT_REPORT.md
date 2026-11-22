# M-04 YOLOv8n Synthetic UI Elements — Test Evaluation Report

**Evaluation Date:** 2026-11-22T22:43:09.751721  
**Model Architecture:** Ultralytics YOLOv8n (Nano)  
**Evaluated Checkpoint:** `runs\detect\models\checkpoints\yolov8n_synthetic\run1\weights\best.pt`  
**Test Dataset Split:** `test` (`data/synthetic_yolo/images/test`)  
**Pretrained Prior:** Pre-trained COCO initialization + Synthetic Web UI fine-tuning  

---

## 1. Executive Performance Summary

| Metric | Overall Test Score | Description |
| :--- | :---: | :--- |
| **mAP@50** | **0.9950** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@50-95** | **0.9699** | Primary COCO-standard localization & classification accuracy |
| **Precision (Mean)** | **0.9973** | Ratio of true positive detections over all predicted boxes |
| **Recall (Mean)** | **1.0000** | Ratio of detected ground-truth objects over all existing objects |

---

## 2. Per-Class Detection Performance

|   class_id | class_name      |   precision |   recall |   f1_score |   mAP50_95 |
|-----------:|:----------------|------------:|---------:|-----------:|-----------:|
|          0 | brand_logo      |      0.9995 |        1 |     0.9998 |     0.8462 |
|          1 | login_form      |      0.998  |        1 |     0.999  |     0.995  |
|          2 | username_field  |      0.9986 |        1 |     0.9993 |     0.995  |
|          3 | password_field  |      0.9982 |        1 |     0.9991 |     0.9932 |
|          4 | submit_button   |      0.9985 |        1 |     0.9992 |     0.995  |
|          5 | payment_element |      0.991  |        1 |     0.9955 |     0.995  |

---

## 3. Metric Visualizations on Disk

The following visual curves and confusion matrices have been generated under `data\reports\yolo`:
- `results.png` (Training & Validation loss curves for Box loss, Class loss, DFL loss, and mAP)
- `BoxPR_curve.png` (Precision-Recall curve across all 6 classes)
- `confusion_matrix.png` & `confusion_matrix_normalized.png` (Element misclassification matrix)
- `BoxF1_curve.png` (F1 score vs confidence threshold)
- `val_batch0_labels.jpg` vs `val_batch0_pred.jpg` (Visual ground-truth vs model prediction comparison)

---

## 4. Synthetic Experiment Disclaimer & Real-World Roadmap

> [!WARNING]
> **Controlled Synthetic Experiment Notice:**  
> This experiment measures bounding box localization on programmatically rendered synthetic webpage templates. While it establishes that YOLOv8n effectively learns UI layout hierarchies (login cards, input fields, submit buttons, and logos), **these synthetic benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations. In Phase 7.5+, human-verified bounding boxes from the 1,000 catalogued screenshots will be incorporated to measure true real-world generalization.
