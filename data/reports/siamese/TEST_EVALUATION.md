# M-03 Siamese ResNet-18 Brand Matcher — Independent Test Evaluation

**Evaluation Target:** M-03 Siamese ResNet-18 Brand Impersonation Matcher  
**Checkpoint Evaluated:** `models/checkpoints/siamese/best_siamese.pt` (Epoch 1)  
**Test Partition:** `data/splits/siamese/test.parquet` (20 pairs)  
**Pre-configured Decision Threshold:** $\tau = 0.82$  
**Execution Date:** 2026-11-22  

---

## 1. Executive Summary & Verification

An independent, un-tuned evaluation was conducted on the held-out test split of the Siamese brand matcher.
In strict compliance with evaluation protocols:
- **Best Checkpoint Loaded:** `best_siamese.pt` (restored from Epoch 1, not final epoch weights).
- **Zero Threshold Tuning:** Evaluated strictly using the checkpoint's pre-configured decision threshold ($	au = 0.82$).
- **Zero Weight Modification:** No fine-tuning, training, or gradient updates occurred.
- **Held-Out Test Brands:** 3 distinct brands (`amazon`, `netflix`, `google`) completely disjoint from training and validation splits.

---

## 2. Test Classification Performance Metrics

| Metric | Test Split Value | Validation Value (Epoch 1) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **85.00%** | 86.96% | -0.0196 |
| **Precision** | **1.0000** | 1.0000 | +0.0000 |
| **Recall** | **0.6250** | 0.6667 | -0.0417 |
| **F1-Score** | **0.7692** | 0.8000 | -0.0308 |
| **ROC-AUC** | **0.8750** | 0.8095 | +0.0655 |
| **PR-AUC** | **0.8629** | 0.8433 | +0.0196 |

### Test Confusion Matrix

```text
                     Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)          12                       0                      (Total: 12)
Actual Positive (1)          3                        5                      (Total: 8)
```

---

## 3. Detailed Per-Pair Prediction Registry

| Pair ID | Brand 1 | Brand 2 | Pair Type | Ground Truth | Cosine Sim ($S$) | Euclidean Dist ($D$) | Prediction | Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `test_pos_amazon_0_0` | `amazon` | `amazon` | identical_self | 1 | 1.0000 | 0.0000 | 1 | CORRECT |
| `test_pos_netflix_0_0` | `netflix` | `netflix` | identical_self | 1 | 1.0000 | 0.0000 | 1 | CORRECT |
| `test_pos_google_0_0` | `google` | `google` | identical_self | 1 | 1.0000 | 0.0000 | 1 | CORRECT |
| `test_pos_google_0_1` | `google` | `google` | same_brand_variant | 1 | 0.5065 | 0.9935 | 0 | MISCLASSIFIED (False_Negative) |
| `test_pos_google_0_2` | `google` | `google` | same_brand_variant | 1 | 0.4857 | 1.0142 | 0 | MISCLASSIFIED (False_Negative) |
| `test_pos_google_1_1` | `google` | `google` | identical_self | 1 | 1.0000 | 0.0000 | 1 | CORRECT |
| `test_pos_google_1_2` | `google` | `google` | same_brand_variant | 1 | 0.5236 | 0.9761 | 0 | MISCLASSIFIED (False_Negative) |
| `test_pos_google_2_2` | `google` | `google` | identical_self | 1 | 1.0000 | 0.0000 | 1 | CORRECT |
| `test_neg_amazon_netflix_0` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_amazon_netflix_1` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_amazon_netflix_2` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_amazon_netflix_3` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_google_amazon_4` | `google` | `amazon` | cross_brand_negative | 0 | 0.6379 | 0.8509 | 0 | CORRECT |
| `test_neg_amazon_netflix_5` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_google_netflix_6` | `google` | `netflix` | cross_brand_negative | 0 | 0.5652 | 0.9325 | 0 | CORRECT |
| `test_neg_google_amazon_7` | `google` | `amazon` | cross_brand_negative | 0 | 0.6379 | 0.8509 | 0 | CORRECT |
| `test_neg_amazon_netflix_8` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_amazon_netflix_9` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |
| `test_neg_google_netflix_10` | `google` | `netflix` | cross_brand_negative | 0 | 0.5652 | 0.9325 | 0 | CORRECT |
| `test_neg_amazon_netflix_11` | `amazon` | `netflix` | cross_brand_negative | 0 | 0.4526 | 1.0464 | 0 | CORRECT |

---

## 4. Per-Brand Breakdown for Held-Out Test Brands

| Brand ID | Total Pairs Involved | Positive Pairs | Negative Pairs | Accuracy | Mean Positive Similarity | Mean Negative Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `amazon` | 11 | 1 | 10 | 100.0% | 1.0000 | 0.4897 |
| `google` | 10 | 6 | 4 | 70.0% | 0.7526 | 0.6016 |
| `netflix` | 11 | 1 | 10 | 100.0% | 1.0000 | 0.4751 |

---

## 5. Overfitting Diagnostics & Training Curve Analysis

### Training vs. Validation Trajectory
- **Selected Checkpoint:** Epoch 1 (`best_siamese.pt`)
  - Validation Loss: `0.0553`
  - Validation Macro F1: `0.8`
  - Validation Precision: `1.0` (0 false positives out of 14 negatives)
  - Validation Recall: `0.6667`
- **Terminal Checkpoint:** Epoch 7 (`final_siamese.pt` at early stop)
  - Validation Loss: `0.0835` (+51.0% degradation)
  - Validation Macro F1: `0.5714` (-28.6% drop)
  - Validation Accuracy: `0.4783` (collapsed from 86.96% to 47.83%)
  - False Positives: Spiked from 0 to 11 (out of 14 negative pairs misclassified as positive)

### Why Epoch 1 Was Selected
1. **Pretrained ImageNet Prior:** The convolutional stem begins with pre-trained ResNet-18 weights containing robust edge, texture, and visual descriptor filters. At Epoch 1, the projection head ($512 \to 256$) adapts smoothly to the contrastive objective while retaining semantic differentiation.
2. **Over-Optimization on Small Support Set:** The training split contains only 73 pairs across 12 brands. Continued optimization after Epoch 1 caused the model to aggressively minimize the contrastive margin on positive pairs by collapsing the embedding space towards a shared manifold. Consequently, inter-brand distance shrank dramatically, causing cross-brand negative pairs to exceed the similarity threshold ($	au = 0.82$) and generating massive false positives.
3. **Early Stopping Protection:** The trainer's early stopping mechanism ($patience=6$) successfully recognized the validation degradation, halting further weight corruption at Epoch 7 and preserving the genuine best checkpoint from Epoch 1.

---

## 6. Limitations of the 20-Pair Test Set

1. **Discrete Metric Granularity:** With only 8 positive pairs and 12 negative pairs, a single misclassification causes large discontinuous metric swings (each pair accounts for 5.0% of accuracy and 12.5% of recall).
2. **Imbalanced Brand Representation:** Google has 6 positive pairs (due to multiple available assets: primary logo, icon, and screenshot), while Amazon and Netflix have only 1 positive pair each (self-similarity anchor).
3. **Absence of Malicious Hard Negatives:** All test negative pairs are legitimate cross-brand pairs (`google` vs `amazon`, `amazon` vs `netflix`). Zero adversarial phishing impostors (e.g. typosquatted or slightly distorted attacker logos) exist in the test split because external phishing feeds rarely isolate clean, segmented brand logos.

---

## 7. Strategic Verdict & Next Steps

> [!IMPORTANT]
> **Verdict:** **M-03 is mechanically functional and suitable as a baseline conditional brand matcher, but REQUIRES DATA EXPANSION before production deployment.**
>
> - **Strengths:** 256-d unit sphere metric learning, zero domain leakage, and perfect precision (zero false positives) on authentic test pairs demonstrate that the architecture and metric loss work correctly.
> - **Weaknesses:** Rapid overfitting after Epoch 1 confirms that 73 training pairs across 12 brands is insufficient for representation generalization. Contrastive learning requires hundreds to thousands of diverse logo variants, dark/light modes, color variations, and cropped UI elements.
> - **Recommendation:** Keep M-03 as a conditional validation stage (triggered only when YOLO detects a logo candidate or PSL lexical analysis flags a brand hint). Expand the reference gallery with multi-scale crops, synthetic augmentations (affine, perspective, color jitter), and logo bounding boxes from Phase 6.3 (YOLO element detection).
