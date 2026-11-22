# DATASET_QUALITY_REPORT.md — Empirical Data Quality, Balance & Leakage Proof

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Date:** 2026-11-22  
**Telemetry File:** `data/reports/dataset_statistics.json` and `data/reports/preprocessing_report.json`

---

## 1. Executive Quality Summary

This report establishes the empirical quality, distributional characteristics, and mathematical integrity of PhishVision AI's prepared datasets prior to model development.

- **Total Ingested Raw URLs:** 94,344
- **Deduplication Rate:** 0.23% (219 exact duplicate URLs eliminated)
- **Final Cleaned Corpus:** **94,125 unique URLs**
- **Unique Registrable Domains Covered:** **55,364 domains**
- **Overall Class Balance:** 53.7% Legitimate ($N = 50,589$) vs 46.3% Phishing ($N = 43,536$) — **Highly balanced binary distribution**.
- **Cross-Split Domain Leakage:** **0.00% (Strictly zero overlapping domains)**.

---

## 2. Data Cleaning & Integrity Metrics

### 2.1 URL Cleansing & Parameter Stripping
All raw URLs were passed through `scripts/datasets/normalize_urls.py`:
- **Tracking Noise Removed:** Stripped 15+ marketing and session tracking parameters (`utm_source`, `utm_medium`, `fbclid`, `gclid`, `_ga`) which otherwise cause spurious lexical correlations in ML models.
- **Protocol & Port Sanitization:** Default HTTP `:80` and HTTPS `:443` ports were normalized out; mixed-case hostnames converted to uniform lowercase.
- **PSL Domain Resolution:** 100% of URLs resolved to public suffix-compliant registrable domains using `tldextract` (e.g. `login.paypal.co.uk` correctly identified as `paypal.co.uk`, not `co.uk`).

### 2.2 Missing Value Profile
Across all 94,125 records in `data/processed/urls/corpus_normalized.parquet`:

| Field Name | Completeness Rate | Missing Values | Handling Strategy |
| :--- | :--- | :--- | :--- |
| `original_url` | 100.0% | 0 | Mandatory field |
| `canonical_url` | 100.0% | 0 | Mandatory field |
| `url_hash_sha256` | 100.0% | 0 | Computed for 100% of records |
| `scheme` | 100.0% | 0 | Defaults to `http` if unspecified |
| `hostname` | 100.0% | 0 | Mandatory |
| `registrable_domain`| 100.0% | 0 | 100% resolved via Public Suffix List |
| `label` | 100.0% | 0 | Binary integer (`0` or `1`) |
| `source` | 100.0% | 0 | Provenance tracked |
| `target_brand` | 32.4% present | 63,618 (N/A) | Populated for PhishTank & PhreshPhish; legitimate domains do not carry impersonation tags. |
| `collection_timestamp`| 46.8% present | 50,076 (N/A) | Recorded for PhishTank & PhreshPhish feeds; Tranco static list does not carry timestamps. |

---

## 3. Lexical Distribution Characteristics

From `data/reports/dataset_statistics.json`:

- **Minimum URL Length:** 12 characters (e.g., `http://t.co/`)
- **Maximum URL Length:** 2,048 characters (obfuscated, deeply nested phishing redirect strings)
- **Mean URL Length:** **37.93 characters**
- **Median URL Length:** **23.0 characters**
- **Standard Deviation:** **48.12 characters**

Phishing URLs exhibit significant right-skewed length distributions compared to benign base domains, reflecting subdomains, path-based token stuffing, and encoded credential parameters.

---

## 4. Brand Impersonation Representation

Among the labeled phishing records ($N = 43,536$), the top targeted brand targets identified from PhishTank and PhreshPhish include:
1. **Generic / Unclassified Campaigns (`Other`):** 11,288 records
2. **Facebook / Meta Platforms:** 2,410 records
3. **Internal Revenue Service (IRS / Tax):** 1,842 records
4. **Amazon.com:** 1,215 records
5. **PayPal Inc.:** 1,180 records
6. **Microsoft / Office365:** 985 records
7. **Apple Inc. / iCloud:** 840 records
8. **WhatsApp:** 620 records
9. **Chase Bank:** 510 records
10. **State Bank of India (SBI):** 430 records

This aligns directly with the 18 target brands curated in `data/brands/reference_gallery/manifest_brands.json`.

---

## 5. Domain Leakage Mathematical Verification Proof

To ensure PhishVision AI models generalize to **previously unseen websites** rather than simply memorizing specific domains, all splits were partitioned strictly at the domain level:

$$\text{DomainSet}(\text{Train}) \cap \text{DomainSet}(\text{Validation}) = \emptyset$$
$$\text{DomainSet}(\text{Train}) \cap \text{DomainSet}(\text{Test}) = \emptyset$$
$$\text{DomainSet}(\text{Validation}) \cap \text{DomainSet}(\text{Test}) = \emptyset$$

### Empirical Split Verification Results:
- Total Domains: 55,364
- Training Set Domains: 38,754
- Validation Set Domains: 8,304
- Test Set Domains: 8,306
- **Overlapping Domains Count:** **0**
- **Verification Script:** [`scripts/datasets/generate_splits.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/generate_splits.py) executed and passed assertion.
- **Automated Regression Test:** [`tests/test_dataset_splitting.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/tests/test_dataset_splitting.py) passed.

---

## 6. Visual Preprocessing & Perceptual Hashing (pHash)

- **Engine:** [`scripts/datasets/validate_images.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/validate_images.py) & [`scripts/datasets/deduplicate.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/deduplicate.py).
- **Blank Detection Threshold:** Grayscale pixel standard deviation $< 3.0$ successfully identifies all-white, all-black, or solid error screenshots.
- **Perceptual Near-Duplicate Clustering:** 64-bit DCT perceptual hash with Hamming distance threshold $\le 4$ clusters visual duplicates and eliminates mirrored screenshot copies.
