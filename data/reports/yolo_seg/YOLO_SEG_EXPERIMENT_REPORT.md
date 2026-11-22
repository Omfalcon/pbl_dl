# M-06 YOLOv8n-Seg Synthetic UI Elements — Instance Segmentation Report

**Phase:** Phase 7.7 YOLOv8 Instance Segmentation  
**Architecture:** Ultralytics YOLOv8n-seg (Nano Segmentation)  
**Evaluated Checkpoint:** `models\checkpoints\yolov8n_seg_synthetic\run1\weights\best.pt`  
**Test Split:** `test` (`data/synthetic_yolo_seg/images/test`, 120 synthetic images)  
**Evaluation Date:** 2026-11-23 02:41:34  

---

## 1. Dual Evaluation: Bounding Box Detection vs. Pixel Instance Segmentation

| Metric Tier | mAP@50 | mAP@50-95 | Mean Precision | Mean Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Bounding Box Detection** | **0.995** | **0.8889** | 0.9525 | 1.0 |
| **Instance Mask Segmentation** | **0.995** | **0.8927** | 0.9525 | 1.0 |

---

## 2. Per-Class Mask Segmentation Accuracy

| Class ID | Class Name | Mask Precision | Mask Recall | Mask mAP@50 | Mask mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `brand_logo` | 0.9988 | 1.0 | 0.995 | 0.8457 |
| 1 | `login_form` | 0.9945 | 1.0 | 0.995 | 0.995 |
| 2 | `username_field` | 0.9893 | 1.0 | 0.995 | 0.8955 |
| 3 | `password_field` | 0.9936 | 1.0 | 0.995 | 0.8654 |
| 4 | `submit_button` | 0.9936 | 1.0 | 0.995 | 0.7828 |
| 5 | `payment_element` | 0.745 | 1.0 | 0.995 | 0.9716 |

---

## 3. Operational & Latency Telemetry

- **Mean Inference Latency:** N/A ms / image
- **Checkpoint Model Size:** 6.45 MB
- **Hardware Profile:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA)

---

## 4. Controlled Synthetic Experiment & Generalization Disclaimer

> [!WARNING]
> **Synthetic Dataset Notice:**  
> This benchmark evaluates bounding box and pixel segmentation mask accuracy on programmatically rendered synthetic webpage templates. While it establishes that YOLOv8n-seg learns exact geometric contours (rounded card containers, input fields, pill buttons, and brand logo symbols), **these synthetic benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations.
