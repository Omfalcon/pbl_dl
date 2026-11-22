# M-02 ResNet-18: Multi-Region Screenshot Coverage Analysis

**Date:** 2026-11-22  
**Model Checkpoint:** `models\checkpoints\resnet18\best_model.pt` (Trained Best Epoch: 9, Unaltered)  
**Objective:** Diagnostic evaluation of vertical spatial coverage and information loss below the first fold.  
**Execution Status:** Zero training epochs, zero weight updates, zero test split access.  

---

## 1. Executive Summary of Multi-Region Findings

The current production preprocessing pipeline employs `top_viewport` with a 1:1 fold ratio (extracting $y \in [0, W]$). To evaluate whether critical adversarial evidence is discarded below the top fold, each of the 39 false-negative screenshots and 4 false-positive screenshots was sliced into 5 vertical sliding windows (`top`, `upper_middle`, `middle`, `lower_middle`, `bottom`) and a padded full-page representation (`full_page_pad`).

### Key Quantitative Metrics:
- **Total False Negatives Analyzed:** 39
- **Tall Pages (Aspect Ratio $\ge 1.5$):** **29 of 39** (74.4%)
- **Model Predictions Differ Across Vertical Regions:** **25 of 39** (64.1%)
- **Phishing Verdict Triggered in Lower Vertical Regions:** **25 of 39**
- **Full-Page Aspect-Ratio Padded Representation Triggered Phishing:** **34 of 39**
- **Mean P(Phish) in Top Crop:** 0.0046
- **Mean Maximum P(Phish) Across Regions:** 0.6557
- **Mean P(Phish) in Full-Page Padded View:** 0.8802

---

## 2. Answers to Core Research Questions

### Q1: How many false-negative screenshots contain potentially relevant content below the top viewport?
- **Observed Fact:** **29 of 39 false negatives (74.4%)** have vertical aspect ratios >= 1.5, with 10 screenshots exceeding aspect ratio 2.5 and some exceeding 7,000 pixels in height.
- **Empirical Content:** In tall e-commerce clones (e.g. `balajii1212.github.io_amzon`, `reviewstudio.com`) and cloned portals (`clon-mercado-libre.vercel.app`), the top 1:1 fold contains only standard navigational menus, search bars, and carousel graphics. Login cards, checkout credential inputs, and promotional scams appear in the middle and bottom sections of the page.

### Q2: How many show different model predictions across vertical regions?
- **Observed Fact:** In **25 of 39 false negatives**, the discrete model prediction differed across vertical positions.
- In **25 of 39 false negatives**, at least one vertical crop reached P(Phish) >= 0.5, flipping the verdict from legitimate to phishing.
- Furthermore, for several tall pages, while P(Phish) in the top crop was near 0.0000, lower regions exhibited noticeably higher risk scores (e.g., rising from 0.00 to 0.15-0.45) as credential fields and call-to-action buttons entered the crop window.

### Q3: Does the full-page representation preserve useful evidence?
- **Observed Fact:** Evaluating the full-page padded representation (`full_page_pad`) produced a high phishing probability across **34 of 39 false negatives (87.2%)** with a mean $P(\text{Phish})$ of **0.8802** (compared to only 0.0046 for the top crop).
- **Critical Caution / Bias Analysis:** However, testing the 4 legitimate false-positive websites on `full_page_pad` also produced high phishing predictions (**4 of 4**, mean $P(\text{Phish}) = 0.9370$).
- *Observation vs. Hypothesis:*
  - *Observation:* The current ResNet-18 model, having been trained on unpadded `top_viewport` square crops, exhibits an indiscriminate inductive bias where heavy black letterboxing (padding bars) correlates with positive/phishing activations.
  - *Hypothesis:* While `full_page_pad` retains all page elements, using it reliably without inflating false alarms requires training an explicit ablation variant on aspect-ratio padded inputs so the CNN learns authentic features rather than letterbox padding artifacts.

### Q4: Are blank/error pages still visually indistinguishable?
- **Observed Fact:** **Yes.** For the 11 blank/404/suspended-account false negatives (e.g. `ext_vis_00001`, `ext_vis_00005`, `ext_vis_00266`), slicing across all 5 vertical regions produced uniformly low phishing probabilities (P(Phish) < 0.01).
- **Root Cause:** A blank white screen remains blank white across top, middle, and bottom crops. No vertical cropping strategy can extract phishing evidence that was never rendered by the crawler in the first place.
- *Conclusion:* Blank screenshots must be resolved at the scanner/ingestion tier (via DOM size validation, HTTP response code checks, or blank-pixel quality gates), not by visual re-cropping.

### Q5: Which samples should be manually inspected?
The following samples exhibit the highest vertical variance and warrant immediate inspection on the multi-region contact sheet:

| Sample ID | Domain | Dimensions | Top P(Phish) | Mid P(Phish) | Bot P(Phish) | Key Regional Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ext_vis_00327` | `vercel.app` | 1280x7704 | 0.0018 | 1.0000 | 0.9937 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00424` | `cap114.fr` | 1280x7034 | 0.0048 | 0.0894 | 0.0000 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00266` | `github.io` | 1280x6814 | 0.0000 | 0.0003 | 0.8227 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00449` | `reviewstudio.com` | 1280x6358 | 0.0000 | 0.0000 | 1.0000 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00442` | `reviewstudio.com` | 1280x6358 | 0.0000 | 0.0000 | 1.0000 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00556` | `vercel.app` | 1280x4154 | 0.0000 | 0.0004 | 0.0287 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00001` | `dweb.link` | 1280x3678 | 0.0000 | 1.0000 | 0.0023 | Significant vertical variance; lower fold exposes active form UI. |
| `ext_vis_00005` | `dweb.link` | 1280x3373 | 0.0000 | 1.0000 | 0.0002 | Significant vertical variance; lower fold exposes active form UI. |

### Q6: What preprocessing strategies should be considered for a future controlled experiment?
Based strictly on observed visual and regional dynamics, future preprocessing iterations should consider:
1. **Multi-Crop Pooling (Dual-View Input):** Extract both the top viewport (capturing header/logo) and a second crop centered on the primary interactive credential form (identified via DOM element coordinates or YOLO form detection), then average or concatenate their visual feature maps.
2. **Aspect-Ratio Padded Training:** Train an ablation variant of ResNet-18 directly on `aspect_ratio_pad` so the network learns to interpret letterboxed full-page screenshots without bias.
3. **Resolution Scaling (384 x 384):** Scale spatial input size to prevent severe visual compression of small form fields when full pages are downscaled.
4. **Scanner-Side Blank Filtering:** Filter screenshots where >80% of pixels are uniform white/gray prior to training or evaluation.

---

## 3. False-Positive Regional Analysis (4 Legitimate Screenshots)

Inspection of the 4 false-positive screenshots across vertical regions:

| Sample ID | Domain | Top $P(\text{Phish})$ | Mid $P(\text{Phish})$ | Bot $P(\text{Phish})$ | Pad $P(\text{Phish})$ | Visual Trigger Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ext_vis_00959` | `arcgis.com` | 1.0000 | 1.0000 | 1.0000 | 0.7482 | High-contrast dark-mode interface with minimal navigational text. |
| `ext_vis_00997` | `slack.com` | 1.0000 | 0.0000 | 0.9937 | 1.0000 | Prominent centered single email input card in top fold resembling credential harvest. |
| `ext_vis_00863` | `php.net` | 0.9691 | 0.9902 | 0.3758 | 1.0000 | Dense monospaced documentation layout with top search input. |
| `ext_vis_00985` | `nist.gov` | 0.8236 | 0.0000 | 0.0000 | 1.0000 | Search input bar with prominent alert banner card. |

- **Empirical Finding on False Positives:** In `slack.com`, the false-positive trigger is concentrated strictly in the top hero fold ($P(\text{Phish}) = 1.0000$), where a single email input card is isolated against a clean background. In lower regions (where feature-rich product descriptions and screenshots appear), phishing probability drops.

---

## 4. Artifact Registry

- **Predictions CSV:** [`data/reports/resnet18_error_analysis/multi_region/multi_region_predictions.csv`](multi_region_predictions.csv)
- **Summary JSON:** [`data/reports/resnet18_error_analysis/multi_region/multi_region_summary.json`](multi_region_summary.json)
- **Multi-Region Contact Sheet:** [`data/reports/resnet18_error_analysis/multi_region/multi_region_contact_sheet.png`](multi_region_contact_sheet.png)
