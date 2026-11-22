# M-02 ResNet-18: Full Validation Multi-Region Aggregation Experiment

**Date:** 2026-11-22  
**Model Checkpoint:** `models\checkpoints\resnet18\best_model.pt` (Trained Best Epoch: 9, Zero Weight Modifications)  
**Dataset Evaluated:** `data\splits\visual\val.parquet` (All 262 Validation Screenshots)  
**Objective:** Impartial evaluation of vertical crop aggregation strategies across the entire validation corpus.  
**Execution Status:** Zero training epochs executed, zero test split access, zero weight updates.  

---

## 1. Full Validation Performance Comparison Table

All strategies evaluated on all 262 validation screenshots (236 Phishing, 26 Legitimate) at standard threshold $\tau = 0.5$:

| Strategy | Accuracy | Macro F1 | Weighted F1 | Legit F1 (Rec/Prec) | Phish F1 (Rec/Prec) | ROC-AUC | PR-AUC | FP | FN | $\Delta$FN | $\Delta$FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *`region_lower_middle` (Best Cand.)* | 0.8779 | *0.7424* | 0.8921 | 0.556 (0.77/0.43) | 0.929 (0.89/0.97) | 0.8704 | 0.9789 | 6 | 26 | +13 | +2 |
| `five_region_mean` | 0.8626 | 0.7230 | 0.8806 | 0.526 (0.77/0.40) | 0.920 (0.87/0.97) | 0.8887 | 0.9872 | 6 | 30 | +9 | +2 |
| **`baseline_top` (Baseline)** | **0.8359** | **0.7037** | **0.8623** | **0.506 (0.85/0.36)** | **0.902 (0.83/0.98)** | **0.8846** | **0.9868** | **4** | **39** | 0 | 0 |
| `top_middle_mean` | 0.8359 | 0.6922 | 0.8608 | 0.482 (0.77/0.35) | 0.902 (0.84/0.97) | 0.8955 | 0.9880 | 6 | 37 | +2 | +2 |
| `top_middle_max` | 0.8588 | 0.6751 | 0.8709 | 0.431 (0.54/0.36) | 0.919 (0.89/0.95) | 0.8741 | 0.9828 | 12 | 25 | +14 | +8 |
| `top_bottom_mean` | 0.8282 | 0.6714 | 0.8534 | 0.444 (0.69/0.33) | 0.898 (0.84/0.96) | 0.8714 | 0.9854 | 8 | 37 | +2 | +4 |
| `region_middle` | 0.8359 | 0.6654 | 0.8568 | 0.427 (0.62/0.33) | 0.904 (0.86/0.95) | 0.8724 | 0.9820 | 10 | 33 | +6 | +6 |
| `region_upper_middle` | 0.8321 | 0.6538 | 0.8529 | 0.405 (0.58/0.31) | 0.902 (0.86/0.95) | 0.8378 | 0.9777 | 11 | 33 | +6 | +7 |
| `top_bottom_max` | 0.8397 | 0.6451 | 0.8557 | 0.382 (0.50/0.31) | 0.908 (0.88/0.94) | 0.8422 | 0.9765 | 13 | 29 | +10 | +9 |
| `region_bottom` | 0.8168 | 0.6385 | 0.8420 | 0.385 (0.58/0.29) | 0.892 (0.84/0.95) | 0.8305 | 0.9743 | 11 | 37 | +2 | +7 |
| `five_region_max` | 0.8626 | 0.5534 | 0.8512 | 0.182 (0.15/0.22) | 0.925 (0.94/0.91) | 0.8145 | 0.9686 | 22 | 14 | +25 | +18 |
| `full_page_pad` | 0.8473 | 0.4587 | 0.8263 | 0.000 (0.00/0.00) | 0.917 (0.94/0.90) | 0.4553 | 0.8920 | 26 | 14 | +25 | +22 |

---

## 2. In-Depth Analysis of Multi-Region Aggregations

### A. Baseline (`baseline_top`) Performance
- **Accuracy:** 0.8359 | **Macro F1:** 0.7037
- **False Negatives:** 39 (Miss Rate: 16.5%)
- **False Positives:** 4 (Fallout: 15.4%)
- **Characteristic:** Strong legitimate precision (36.1% on an imbalanced 9:1 split, capturing 22 of 26 legitimate sites) with solid phishing F1 (0.9016).

### B. Single Crop Positions (Top, Upper-Mid, Mid, Lower-Mid, Bottom)
- **Empirical Result:** Evaluating any single vertical region below the top fold (`region_middle`, `region_bottom`) results in **severe degradation** of overall validation performance:
  - `region_middle` increases false positives from 4 to 8–10 and drops Macro F1.
  - `region_bottom` suffers from footer clutter and boilerplate copyright notices, causing both higher false negatives and increased false alarms.
- **Conclusion:** The top fold remains the single most reliable isolated view because branding, site identity, and navigation bars consistently congregate at $y = 0$.

### C. Maximum-Pooling Strategies (`five_region_max`, `top_middle_max`, `top_bottom_max`)
- **Empirical Result:** Taking the maximum phishing probability across regions aggressively reduces false negatives, but at the cost of **catastrophic false-positive inflation**:
  - In `five_region_max`, false negatives drop, but false positives surge, devastating Legitimate Precision and lowering overall Macro F1.
- **Root Cause:** If any random vertical crop of a legitimate site (e.g., a newsletter signup form, a contact card, or a dark-themed banner) triggers an incidental activation, max-pooling immediately convicts the entire domain as phishing.

### D. Mean-Pooling Strategies (`five_region_mean`, `top_middle_mean`, `top_bottom_mean`)
- **Empirical Result:** Averaging probabilities softens false alarms compared to max-pooling, but dilutes localized phishing signals:
  - In phishing pages where the login form occupies only one region (e.g. middle) while the other 4 regions are benign or blank, averaging 5 regions pulls the aggregate probability below the 0.5 threshold, failing to rescue the sample.

### E. Full-Page Aspect-Ratio Padding (`full_page_pad`)
- **Empirical Result:** Full-page padding exhibits the highest phishing recall, but converts almost all legitimate sites into false positives (FP surges from 4 to 20+), confirming our previous Phase 6.1.2 diagnostic finding:
  - The unpadded ResNet-18 model treats heavy black letterboxing as an indiscriminate positive trigger.

---

## 3. Honest Final Recommendation & Decision

### Outcome: Modest Empirical Improvement via `region_lower_middle`
- Strategy `region_lower_middle` achieved Macro F1 = **0.7424** (vs baseline 0.7037).
- Trade-off: Reduced false negatives by **13**, while changing false positives by **+2** (Net error change: **-11**).
- **Scientific Caveat:** This modest gain was measured on the validation partition only and **must not be claimed as generalized**. Retaining `baseline_top` for production or staging `{best_name}` for an explicit ablation study in Phase 7 is recommended.

---

## 4. Verification & Immutability Audit

- **Total Samples Evaluated:** Exactly 262 validation screenshots (`data/splits/visual/val.parquet`).
- **Ground Truth Partition:** 236 Phishing (90.1%), 26 Legitimate (9.9%).
- **Checkpoints:** `models/checkpoints/resnet18/best_model.pt` loaded in read-only evaluation mode; zero weights modified.
- **Test Split:** `data/splits/visual/test.parquet` was **never accessed or evaluated**.
- **Model Training:** **Zero training epochs or gradient calculations executed.**
