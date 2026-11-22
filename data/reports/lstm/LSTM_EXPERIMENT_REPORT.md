# M-07 URL Sequence LSTM — Phishing Detection Experiment Report

**Phase:** Phase 7.8 Sequential Pattern Analysis  
**Model Identifier:** `M-07`  
**Architecture:** Bidirectional URL Sequence LSTM with Attentive Pooling  
**Evaluated Checkpoint:** `models\checkpoints\lstm\best_lstm.pt`  
**Test Split:** `data/splits/test/urls.parquet` (4257 records: 4175 Phishing, 82 Legitimate)  
**Calibrated Decision Threshold:** $\tau = 0.01$ (Tuned strictly on Validation split)  
**Evaluation Date:** 2026-11-23 03:17:54  

---

## 1. Executive Summary & Comparative Performance

| Performance Metric | M-07 URL Sequence LSTM | M-01 Lexical Baseline (MLP) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Model Paradigm** | **End-to-End Sequence (Char BiLSTM)** | Handcrafted 40-d Lexical Vector | Semantic vs. Statistical |
| **Overall Accuracy** | **0.9869** | 0.9866102889358703 | +0.0002 |
| **Macro F1-Score** | **0.7779** | 0.8238591448077615 | -0.0459 |
| **ROC-AUC** | **0.9666** | 0.9772893237914415 | -0.0107 |
| **PR-AUC** | **0.9991** | 0.9995242693074335 | — |
| **Class 0 (Legitimate) F1** | **0.5625** | 0.6545454545454545 | -0.0920 |
| **Class 1 (Phishing) F1** | **0.9933** | 0.9931728350700683 | +0.0001 |
| **Inference Latency** | **0.07 ms / URL** | ~0.12 ms / URL | Sequence graph overhead |
| **Checkpoint Size** | **2.41 MB** | ~0.08 MB (NumPy NPZ) | Deep sequential parameters |

---

## 2. Detailed Test Split Performance

### 2.1 Confusion Matrix

| Ground Truth \ Prediction | Predicted Legitimate (0) | Predicted Phishing (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate (0)** | **TN = 36** | FP = 46 | 82 |
| **Actual Phishing (1)** | FN = 10 | **TP = 4165** | 4175 |
| **Total Predicted** | 46 | 4211 | 4257 |

### 2.2 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Legitimate)** | 0.7826 | 0.4390 | 0.5625 | 82 |
| **Class 1 (Phishing)** | 0.9891 | 0.9976 | 0.9933 | 4175 |
| **Macro Average** | **0.8858** | **0.7183** | **0.7779** | **4257** |

---

## 3. Scientific Integrity & Data Limitation Disclosures

> [!IMPORTANT]
> **Redirect-Chain Training Data Disclosure:**  
> A comprehensive dataset audit of `data/splits/` and `data/processed/` confirmed that **no historical labelled redirect sequence datasets exist in the corpus**. The dataset contains 44,129 static URL records. While `PlaywrightScanner` captures real-time `RedirectHop` telemetry during live browser crawls, and the `URLSequenceLSTM` architecture includes an explicit redirect sequence processing branch, **redirect-chain learning has NOT yet been trained on empirical multi-hop data**. The model was trained and evaluated strictly on raw sequential URL patterns.

> [!NOTE]
> **Domain-Level Split Isolation:**  
> Zero domain leakage was verified between splits. All 16 training brands and over 32,000 domains are strictly disjoint from the validation and test sets. Preprocessing vocabularies were fitted strictly on training data records.
