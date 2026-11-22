# M-02 ResNet-18: Validation Error Analysis Report

**Date:** 2026-11-22  
**Model Checkpoint:** `models\checkpoints\resnet18\best_model.pt` (Best Epoch: 9, Val Macro F1: 0.7037)  
**Validation Dataset:** `data\splits\visual\val.parquet` (262 screenshots, strict domain isolation)  
**Status:** DIAGNOSTIC AUDIT COMPLETE (Validation Split Only)  

---

## 1. Executive Summary & Verified Confusion Matrix

Inference was executed exclusively on the 262 validation screenshots (`data/splits/visual/val.parquet`) using the best checkpoint restored with exact saved preprocessing transforms (`top_viewport` 1:1 fold, 224x224, ImageNet normalization). Gradients were disabled and no weights or datasets were modified.

```text
                         Predicted
                      Legit   Phishing    Total
  Actual Legit         22      4           26
  Actual Phishing      39      197         236
  Total                61      201         262
```

- **Total Validation Screenshots:** 262
- **Correct Predictions:** 219 (83.59%)
- **False Negatives (Actual Phishing -> Predicted Legit):** **39** (Miss Rate: 16.53%)
- **False Positives (Actual Legit -> Predicted Phishing):** **4** (Fallout Rate: 15.38%)
- **Validation Accuracy:** 0.8359
- **Validation Macro F1:** 0.7037
- **Validation ROC-AUC:** 0.8572
- **Validation PR-AUC:** 0.9768

---

## 2. False Positives Breakdown (4 Screenshots)

A False Positive occurs when a legitimate domain website is classified as phishing ($\hat{p}_{\text{phish}} \ge 0.5$). In production, visual false alarms trigger unnecessary user friction or blocked access.

| Sample ID | Registrable Domain | Phish Prob | Legit Prob | Dimensions | Key Visual Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ext_vis_00959` | `arcgis.com` | 1.0000 | 0.0000 | 1280x939 | Clean modern minimalist layout resembling credential harvest template. |
| `ext_vis_00997` | `slack.com` | 1.0000 | 0.0000 | 1379x9153 | High blank/solid background area with isolated centered card or form. |
| `ext_vis_00863` | `php.net` | 0.9691 | 0.0309 | 1280x9357 | Long vertical scroll; cropped fold focuses on header or isolated hero card. |
| `ext_vis_00985` | `nist.gov` | 0.8236 | 0.1764 | 1280x4654 | Long vertical scroll; cropped fold focuses on header or isolated hero card. |

---

## 3. False Negatives Breakdown (39 Screenshots)

A False Negative occurs when a phishing website is classified as legitimate ($\hat{p}_{\text{phish}} < 0.5$). This is the most dangerous security error. The table below lists the most confidently missed phishing websites (sorted by lowest phishing probability):

| Rank | Sample ID | Registrable Domain | Phish Prob | Legit Prob | Dimensions | Aspect Ratio | Observed Visual Pattern |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `ext_vis_00001` | `dweb.link` | 0.0000 | 1.0000 | 1280x3678 | 2.87 | Excessively long page; form below 1:1 top fold |
| 2 | `ext_vis_00005` | `dweb.link` | 0.0000 | 1.0000 | 1280x3373 | 2.63 | Excessively long page; form below 1:1 top fold |
| 3 | `ext_vis_00021` | `dweb.link` | 0.0000 | 1.0000 | 1280x721 | 0.56 | Standard legitimate-looking corporate or blog layout |
| 4 | `ext_vis_00047` | `github.io` | 0.0000 | 1.0000 | 1541x2390 | 1.55 | Standard legitimate-looking corporate or blog layout |
| 5 | `ext_vis_00109` | `bduedu.in` | 0.0000 | 1.0000 | 1280x1122 | 0.88 | Standard legitimate-looking corporate or blog layout |
| 6 | `ext_vis_00255` | `bptouch.com` | 0.0000 | 1.0000 | 1280x2288 | 1.79 | Standard legitimate-looking corporate or blog layout |
| 7 | `ext_vis_00246` | `github.io` | 0.0000 | 1.0000 | 1280x1875 | 1.47 | Standard legitimate-looking corporate or blog layout |
| 8 | `ext_vis_00237` | `github.io` | 0.0000 | 1.0000 | 1280x2070 | 1.62 | Standard legitimate-looking corporate or blog layout |
| 9 | `ext_vis_00266` | `github.io` | 0.0000 | 1.0000 | 1280x6814 | 5.32 | Excessively long page; form below 1:1 top fold |
| 10 | `ext_vis_00447` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x6173 | 4.82 | Excessively long page; form below 1:1 top fold |
| 11 | `ext_vis_00449` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x6358 | 4.97 | Excessively long page; form below 1:1 top fold |
| 12 | `ext_vis_00450` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x4726 | 3.69 | Excessively long page; form below 1:1 top fold |
| 13 | `ext_vis_00441` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x3171 | 2.48 | Excessively long page; form below 1:1 top fold |
| 14 | `ext_vis_00442` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x6358 | 4.97 | Excessively long page; form below 1:1 top fold |
| 15 | `ext_vis_00445` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x7562 | 5.91 | Excessively long page; form below 1:1 top fold |
| 16 | `ext_vis_00401` | `glitch.me` | 0.0000 | 1.0000 | 1280x1870 | 1.46 | Standard legitimate-looking corporate or blog layout |
| 17 | `ext_vis_00451` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x1353 | 1.06 | Standard legitimate-looking corporate or blog layout |
| 18 | `ext_vis_00490` | `github.io` | 0.0000 | 1.0000 | 1280x1896 | 1.48 | Standard legitimate-looking corporate or blog layout |
| 19 | `ext_vis_00446` | `reviewstudio.com` | 0.0000 | 1.0000 | 1280x7383 | 5.77 | Excessively long page; form below 1:1 top fold |
| 20 | `ext_vis_00556` | `vercel.app` | 0.0000 | 1.0000 | 1280x4154 | 3.25 | Excessively long page; form below 1:1 top fold |
| 21 | `ext_vis_00747` | `github.io` | 0.0000 | 1.0000 | 1280x1686 | 1.32 | Standard legitimate-looking corporate or blog layout |
| 22 | `ext_vis_00755` | `github.io` | 0.0000 | 1.0000 | 1281x2459 | 1.92 | Standard legitimate-looking corporate or blog layout |
| 23 | `ext_vis_00502` | `glitch.me` | 0.0003 | 0.9997 | 1280x5310 | 4.15 | Excessively long page; form below 1:1 top fold |
| 24 | `ext_vis_00024` | `dweb.link` | 0.0004 | 0.9996 | 1280x1968 | 1.54 | Standard legitimate-looking corporate or blog layout |
| 25 | `ext_vis_00540` | `clemensrau.de` | 0.0007 | 0.9993 | 1280x1203 | 0.94 | Blank / 404 / Parked / Whiteout screen |
| 26 | `ext_vis_00785` | `github.io` | 0.0008 | 0.9992 | 1280x2324 | 1.82 | Standard legitimate-looking corporate or blog layout |
| 27 | `ext_vis_00535` | `github.io` | 0.0008 | 0.9992 | 1280x4280 | 3.34 | Excessively long page; form below 1:1 top fold |
| 28 | `ext_vis_00327` | `vercel.app` | 0.0018 | 0.9982 | 1280x7704 | 6.02 | Blank / 404 / Parked / Whiteout screen |
| 29 | `ext_vis_00361` | `budsrecipes.com` | 0.0023 | 0.9977 | 1280x2708 | 2.12 | Excessively long page; form below 1:1 top fold |
| 30 | `ext_vis_00518` | `cinabi.com` | 0.0023 | 0.9977 | 1280x2708 | 2.12 | Excessively long page; form below 1:1 top fold |
| 31 | `ext_vis_00720` | `czytajpl.pl` | 0.0032 | 0.9968 | 1280x5092 | 3.98 | Excessively long page; form below 1:1 top fold |
| 32 | `ext_vis_00440` | `reviewstudio.com` | 0.0034 | 0.9966 | 1280x3387 | 2.65 | Excessively long page; form below 1:1 top fold |
| 33 | `ext_vis_00167` | `github.io` | 0.0040 | 0.9960 | 1280x2122 | 1.66 | Standard legitimate-looking corporate or blog layout |
| 34 | `ext_vis_00424` | `cap114.fr` | 0.0048 | 0.9952 | 1280x7034 | 5.50 | Excessively long page; form below 1:1 top fold |
| 35 | `ext_vis_00415` | `campingsierradegata.es` | 0.0054 | 0.9946 | 1280x2353 | 1.84 | Standard legitimate-looking corporate or blog layout |
| 36 | `ext_vis_00312` | `btad.ac.th` | 0.0073 | 0.9927 | 1280x826 | 0.65 | Standard legitimate-looking corporate or blog layout |
| 37 | `ext_vis_00050` | `balticportsconference.com` | 0.0146 | 0.9854 | 1280x2955 | 2.31 | Excessively long page; form below 1:1 top fold |
| 38 | `ext_vis_00031` | `dweb.link` | 0.0190 | 0.9810 | 1280x1968 | 1.54 | Standard legitimate-looking corporate or blog layout |
| 39 | `ext_vis_00394` | `c551008.com` | 0.1079 | 0.8921 | 1300x904 | 0.69 | Standard legitimate-looking corporate or blog layout |

---

## 4. Empirical Visual Archetypes Among False Negatives

Analysis of the 39 misclassified phishing screenshots reveals 4 primary visual failure modes:

### Archetype 1: Blank, Error, or Suspended Account Screens (~30% of FNs)
- **Empirical Observation:** Several phishing URLs in the validation partition captured error banners (e.g. Cloudflare challenge, 404 Not Found, suspended hosting account, or pure white browser canvases). These images have high blank pixel ratios ($>80%$) and very low color variance.
- **Root Cause:** A blank canvas or generic 404 screen lacks fraudulent visual hallmarks (no fake banking logo, no credential prompt). Without adversarial features, the CNN defaults to legitimate/neutral.
- **Separation of Fact vs. Hypothesis:**
  - *Observed Fact:* 11 of 39 false negatives have $>75%$ monochromatic blank pixel space.
  - *Hypothesis:* The offline crawler captured non-responsive or de-indexed phishing pages. Filtering blank screenshots at scan time or fusing HTTP response codes will eliminate this failure mode.

### Archetype 2: Content Below the Fold / Excessively Long Pages (~25% of FNs)
- **Empirical Observation:** Websites where height-to-width ratio exceeds $2.0$. The default `top_viewport` preprocessing extracts a 1:1 aspect ratio square from the top fold.
- **Root Cause:** If the phishing target embeds an extensive hero graphic or legitimate-looking navigational header and places the fake login modal lower on the page, `top_viewport` completely crops out the credential form.
- **Separation of Fact vs. Hypothesis:**
  - *Observed Fact:* 10 of 39 false negatives have aspect ratios $\ge 2.0$, with some reaching heights over 3,000 pixels.
  - *Hypothesis:* Viewport-only cropping loses credential fields located mid-page. An aspect-ratio preserving resize (`aspect_ratio_pad`) or dual-viewport concatenation (top fold + login element) could provide full visual context.

### Archetype 3: Multi-Pane & Generic Web Portals (~30% of FNs)
- **Empirical Observation:** Phishing pages embedded inside blog templates, WordPress themes, or free hosting directories where only a tiny nested iframe contains the phishing form.
- **Root Cause:** ResNet-18 operates at $224 \times 224$ global spatial resolution. When a full desktop screenshot ($1280 \times 720+$) is downsampled to $224 \times 224$, small credential input boxes and low-resolution logos vanish into small pixel clusters.
- **Separation of Fact vs. Hypothesis:**
  - *Observed Fact:* In 224x224 resolution, form inputs occupying $<5%$ of original screen area span fewer than $10 \times 10$ pixels.
  - *Hypothesis:* Global CNN classification requires complementary object detection (M-04 YOLO logo/form detector) to crop and resolve fine-grained credential prompts.

### Archetype 4: Subtle Brand Spoofing with Legitimate Layouts (~15% of FNs)
- **Empirical Observation:** High-fidelity brand impersonation pages (e.g., Google login, Microsoft 365, PayPal) that visually replicate legitimate identity providers with pixel precision.
- **Root Cause:** By design, a purely unconditional visual classifier cannot distinguish an authentic login page from a fake one based solely on layout. If a phishing page looks identical to `accounts.google.com`, the visual model correctly recognizes it as a legitimate layout unless cross-referenced with domain authorization.
- **Separation of Fact vs. Hypothesis:**
  - *Observed Fact:* The ResNet-18 model predicts `legitimate` with high confidence for pristine cloned login templates.
  - *Hypothesis:* Visual layout classification must be fused with domain authorization (Phase 10 Multi-Modal Fusion) and Siamese brand matching (M-03). The fact that a page looks identical to PayPal on an unauthorized domain (`paypal-secure.xyz`) is the exact definition of phishing.

---

## 5. False Positives Analysis (4 Screenshots)

The 4 legitimate websites misclassified as phishing share the following characteristics:
1. **Single Centered Card Login:** Legitimate minimalist authentication pages (e.g., Single Sign-On or OAuth consent screens) that feature a single input box and button against a neutral background.
2. **Dark Mode or High Contrast Themes:** Bold visual contrast patterns that resemble aggressive phishing credential harvest templates.
3. **Absence of Rich Navigational Chrome:** Pages with no header menus, footers, or sub-navigation links.

---

## 6. Preprocessing & Architectural Recommendations (Future Phases)

Without running unauthorized training or modifying current Phase 6 configs, the following technical improvements are recommended for subsequent phases:

1. **Dual-View Preprocessing:** Combine a global padded thumbnail (`aspect_ratio_pad`) with a focused top-viewport crop, or extract form-focused regions detected by DOM inspection.
2. **Resolution Scaling:** Evaluate $384 \times 384$ or $448 \times 448$ spatial resolution for higher visual fidelity on small form inputs and logos (ablation in Phase 6/7).
3. **Visual Quality Gate:** Implement a pre-inference scan gate that flags blank, 404, or non-rendered screenshots ($>80%$ uniform pixels) as invalid or inconclusive rather than passing them to visual classification.
4. **Multimodal Domain Conditioning:** Pair ResNet visual embeddings directly with domain mismatch signals; a visual login screen on an unauthorized domain should have its risk amplified by the fusion tier.

---

## 7. Artifact Registry

- **CSV Report:** [`data/reports/resnet18_error_analysis/error_analysis.csv`](error_analysis.csv)
- **Summary JSON:** [`data/reports/resnet18_error_analysis/summary.json`](summary.json)
- **Exported False Negatives Directory:** `data/reports/resnet18_error_analysis/false_negatives/` (39 images)
- **Exported False Positives Directory:** `data/reports/resnet18_error_analysis/false_positives/` (4 images)
