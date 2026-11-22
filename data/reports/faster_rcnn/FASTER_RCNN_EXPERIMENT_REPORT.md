# M-05 Faster R-CNN vs M-04 YOLOv8n — Detection Comparison Report

**Phase:** Phase 7.6 Object Detection Benchmark  
**Evaluation Split:** `test` (120 synthetic images, untouched holdout)  
**Hardware Platform:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA 12.4)  
**Evaluation Date:** 2026-11-23 01:27:55  

---

## 1. Executive Performance Comparison

| Metric | M-04 YOLOv8n (One-Stage) | M-05 Faster R-CNN (Two-Stage) | Delta / Winner |
| :--- | :---: | :---: | :---: |
| **Architecture** | YOLOv8 Nano (Ultralytics) | Faster R-CNN (ResNet-50-FPN) | Different Backbones |
| **mAP@50** | **0.995** | **1.0** | Comparison on test split |
| **mAP@50-95** | **0.9698842207793051** | **0.9806** | COCO localization standard |
| **Mean Precision** | 0.997297344750244 | 0.9925 | True Positives / Detections |
| **Mean Recall** | 1.0 | 1.0 | Detected Objects / Ground Truth |
| **Inference Latency** | ~9.2 ms / image | 129.12 ms / image | Latency profile |
| **Checkpoint Size** | 6.2 MB | 315.0 MB | Storage footprint |
| **Peak VRAM** | < 1.0 GB | 1065.08 MB | Memory efficiency |

---

## 2. Per-Class Average Precision (Faster R-CNN)

| Class ID | Class Name | Total GT | Precision | Recall | F1 Score | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `brand_logo` | 114 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8905 |
| 1 | `login_form` | 114 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9971 |
| 2 | `username_field` | 105 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9989 |
| 3 | `password_field` | 105 | 0.9906 | 1.0 | 0.9953 | 1.0 | 1.0 |
| 4 | `submit_button` | 114 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9969 |
| 5 | `payment_element` | 27 | 0.9643 | 1.0 | 0.9818 | 1.0 | 1.0 |

---

## 3. Architectural Analysis: One-Stage vs. Two-Stage

1. **YOLOv8n (Anchor-Free One-Stage):**
   - Directly predicts bounding boxes and class logits from feature pyramid maps in a single pass.
   - Extremely lightweight (3.0M parameters, 6.2 MB checkpoint).
   - Low latency (~9ms) making it ideal for high-throughput scanning pipelines.

2. **Faster R-CNN (ResNet-50-FPN Two-Stage):**
   - Stage 1: Region Proposal Network (RPN) proposes candidate regions of interest.
   - Stage 2: RoIAlign extracts fixed-size feature vectors per proposal, followed by classification and bounding-box refinement.
   - ResNet-50 backbone (~41.5M parameters) provides richer multi-scale semantic representations.
   - Higher memory requirement and inference latency due to proposal generation and RoI pooling stages.

---

## 4. Scientific Integrity & Generalization Notice

> [!WARNING]
> **Controlled Synthetic Dataset Experiment:**  
> This benchmark evaluates bounding box localization on programmatically rendered synthetic webpage templates. While it establishes how Faster R-CNN and YOLOv8n learn UI layout hierarchies (login cards, input fields, submit buttons, and logos), **these benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations. In Phase 7.7+, human-annotated bounding boxes from the 1,000 catalogued screenshots will be used to evaluate true real-world generalization.
