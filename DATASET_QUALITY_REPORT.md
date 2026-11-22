# DATASET_QUALITY_REPORT.md — Empirical Data Quality, Balance & Leakage Proof

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Date:** 2026-11-22  
**Telemetry File:** `data/reports/dataset_statistics.json` and `data/reports/preprocessing_report.json`

---

## 1. Executive Quality Summary

This report establishes the empirical quality, distributional characteristics, and mathematical integrity of PhishVision AI's prepared datasets following Phase 2 code review and corrections.

- **Total Acquired Raw Records:** 94,344 URLs & domains + 1,000 visual screenshots (`data/raw/dataset.zip`)
- **Total Ingested Processed Records:** **94,129 URL records** + **1,000 catalogued screenshots**
  - **Verified Ground-Truth URL Corpus:** **44,129 URLs** (PhishTank, OpenPhish, PhreshPhish)
  - **Unverified Candidate Pool:** **50,000 domains** (Tranco Top 50k, strictly isolated)
  - **Catalogued Visual Screenshot Corpus:** **1,000 screenshots** (800 Phishing, 200 Legitimate) in `data/processed/screenshots/visual_corpus.parquet`
- **Total Valid Screenshots on Disk:** **1,008 images** (1,000 external + 8 scanner baseline; 0 corrupt, 0 blank)
- **Unique Registrable Domains in Verified URL Corpus:** **6,036 domains**
- **Unique Registrable Domains in Visual Corpus:** **517 domains**
- **Verified URL Class Balance:** 1.35% Legitimate ($N = 594$) vs 98.65% Phishing ($N = 43,535$)
- **Visual Dataset Class Balance:** 20.00% Legitimate ($N = 200$) vs 80.00% Phishing ($N = 800$)
- **Cross-Split Domain Leakage:** **0.00% (Strictly zero overlapping domains for both URL splits and visual splits)**.
- **Record Conservation:** **100.0% (Zero input records lost or duplicated across splits)**.

---

## 2. Label Integrity & Provenance Guarantees

### 2.1 Tranco Candidate Isolation
- **Rule Enforced:** Popularity ranking is not proof of website security. Threat actors routinely host phishing pages on compromised or high-traffic services, cloud providers, and legitimate domains.
- **Implementation:** Tranco domains are marked `verification_status="UNVERIFIED_CANDIDATE"` with `label=-1` and stored in `data/processed/urls/tranco_candidate_pool.parquet`.
- **Result:** Zero Tranco domains are injected into the ground-truth training corpus without independent visual or security verification.

### 2.2 Rejection of Unknown and Malformed Labels
- **Rule Enforced:** Missing, null, unknown, or malformed label values must never default to phishing or benign.
- **Implementation:** `URLNormalizer.parse_label()` maps `None`, `""`, `"unknown"`, `"pending"`, `"nan"`, and unmapped strings directly to `(None, "REJECTED_UNKNOWN")`.
- **Result:** Guaranteed ground-truth label integrity across all ingestion pipelines.

---

## 3. URL Normalization & Parser Hardening

All input URLs pass through `scripts/datasets/normalize_urls.py`:
- **Userinfo Stripping:** Usernames and credentials embedded in URL authorities (e.g. `https://user:pass@victim.com/`) are cleanly excised to produce `https://victim.com/` while preserving the full input in `original_url`.
- **Tracking Parameter Sanitization:** 15+ marketing and session tracking tokens (`utm_source`, `utm_medium`, `fbclid`, `gclid`, `_ga`, `mc_eid`) are stripped to prevent spurious non-semantic correlations.
- **Port Normalization:** Default `:80` and `:443` are stripped; non-standard service ports (`:8080`, `:8443`) are strictly preserved.
- **IPv6 & IDN Parsing:** IPv6 bracketed hosts (`http://[2001:db8::1]:8080/`) and internationalized domain names (IDN punycode `https://xn--...`) are correctly canonicalized without parse errors.
- **Public Suffix List (PSL) Resolution:** Registrable domains resolved via `tldextract` (e.g. `login.paypal.co.uk` -> `paypal.co.uk`). Any record with unparseable domain receives an explicit synthetic key (`__missing_domain_<idx>__`) ensuring 0% row loss.

---

## 4. Lexical Distribution Characteristics

From `data/reports/dataset_statistics.json` (computed over 44,129 verified records):

- **Minimum URL Length:** 13 characters
- **Maximum URL Length:** 3,695 characters
- **Mean URL Length:** **56.77 characters**
- **Median URL Length:** **36.00 characters**
- **Standard Deviation:** **69.50 characters**

Phishing URLs display substantial right-skewed length distributions due to subdomains, path-based brand keyword stuffing, and obfuscated redirect parameters.

### Top Targeted Brands in Ground-Truth Phishing:
1. **Generic / Unclassified Campaigns (`Other`):** 39,246 records
2. **Allegro:** 1,065 records
3. **Internal Revenue Service (IRS / Tax):** 752 records
4. **Facebook / Meta Platforms:** 372 records (combined cases)
5. **Amazon.com:** 229 records
6. **Microsoft / Office365:** 214 records
7. **Optus:** 115 records
8. **Netflix:** 77 records
9. **Adobe Inc.:** 70 records
10. **Google LLC:** 41 records

---

## 5. Domain-Aware Splitting & Mathematical Zero-Leakage Proof

To prevent deep learning models from memorizing specific hostnames or domains, dataset splitting is executed strictly at the **registrable domain** level with class-aware stratification:

$$\text{DomainSet}(\text{Train}) \cap \text{DomainSet}(\text{Validation}) = \emptyset$$
$$\text{DomainSet}(\text{Train}) \cap \text{DomainSet}(\text{Test}) = \emptyset$$
$$\text{DomainSet}(\text{Validation}) \cap \text{DomainSet}(\text{Test}) = \emptyset$$

### Empirical Partitioning Metrics:
- **Total Unique Domains:** 6,036
- **Training Partition (70.0% domains):** 4,224 domains | 32,157 records (435 Legit, 31,722 Phish)
- **Validation Partition (15.0% domains):** 904 domains | 7,715 records (77 Legit, 7,638 Phish)
- **Test Partition (15.0% domains):** 908 domains | 4,257 records (82 Legit, 4,175 Phish)
- **Domain Overlaps Detected:** **0**
- **Record Conservation:** $32,157 + 7,715 + 4,257 = 44,129$ records ($100.0\%$ rows conserved).

---

## 6. Visual & Annotation Quality Readiness

### 6.1 Acquired Screenshot Corpus
- **Total Screenshots on Disk:** **1,008 images** across `data/raw/external_dataset/` and `data/raw/screenshots/`.
  - **Newly Acquired Visual Dataset (`DatasetV1`):** **1,000 images** (800 Phishing, 200 Legitimate) extracted safely from `data/raw/dataset.zip`.
  - **Existing Playwright Scanner Screenshots:** **8 images** (4 Phishing, 4 Legitimate).
- **Image Integrity Verification:** 1,008/1,008 images valid (0 corrupt, 0 zero-byte, 0 blank solid screens).
- **Deduplication Analysis (via SHA-256 and 64-bit DCT pHash):**
  - **Exact Duplicates:** 65 images.
  - **Near-Duplicates (Hamming distance $\le 4$):** 161 images.
- **Visual Cluster-Aware Partitioning (`data/splits/visual/`):**
  - **Train:** 624 screenshots (62.4% records, 70.0% clusters | 362 domains | 147 Legit, 477 Phish)
  - **Validation:** 262 screenshots (26.2% records, 14.7% clusters | 76 domains | 26 Legit, 236 Phish)
  - **Test:** 114 screenshots (11.4% records, 15.3% clusters | 79 domains | 27 Legit, 87 Phish)
  - **Zero-Leakage Proof:** **0 overlapping domains**, **0 overlapping SHA-256 hashes**, and **0 pHash near-duplicates (Hamming $\le 4$)**.
  - **Conservation:** $624 + 262 + 114 = 1,000$ screenshots ($100.0\%$ conserved).

### 6.2 Missing Modalities & Remaining Work
- **Brand Reference Images on Disk:** **0 images** (18 brand folder workspaces scaffolded; actual reference SVG/PNG logos and canonical screenshots are pending acquisition).
- **Brand Similarity Pairs:** **0 pairs** (strictly 0 until brand reference images are curated).
- **YOLO UI Detection Annotations:** **0 files** (annotation engine and independent bbox validator verified via unit tests; annotation pass pending).
- **Semantic Segmentation Masks:** **0 files** (`SegmentationMaskManager` verified with image-mask pair validation; mask generation pending).
- **Human Verification Distinction:** Automated report generator strictly tracks manual verification (`"is_verified": true`); 0 unverified files are counted as verified.
