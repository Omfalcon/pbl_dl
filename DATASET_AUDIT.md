# DATASET_AUDIT.md — PhishVision AI Dataset Audit & Inventory

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Date:** 2026-11-22  
**Audit Standard:** Strict empirical verification of filesystem state, row counts, cryptographic hashes, and domain sets.

---

## 1. Executive Modality Inventory

| Modality / Category | Acquired Records | Verified Ground-Truth | Unverified Candidate Pool | Actual On-Disk Assets | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **URL Records (All Sources)** | **94,129** | 44,129 | 50,000 | `data/processed/urls/` | 100% normalized, tracking tokens stripped, PSL domains extracted. |
| **Verified URL Labels** | **44,129** | 44,129 (43,535 Phish, 594 Legit) | 0 | `corpus_normalized.parquet` | Derived from PhishTank, OpenPhish, and PhreshPhish feeds. |
| **Tranco Candidate URLs** | **50,000** | 0 (Isolated) | 50,000 | `tranco_candidate_pool.parquet` | Tranco popularity rank is NOT treated as ground-truth legitimacy. |
| **Website Screenshots (Total)** | **1,008** | **1,008** (804 Phish, 204 Legit) | 0 | `data/raw/external_dataset/`, `data/raw/screenshots/` | **1,000 newly acquired** (800 Phish, 200 Legit) + **8 baseline scanner screenshots**. |
| **Catalogued Visual Corpus** | **1,000** | **1,000** (800 Phish, 200 Legit) | 0 | `data/processed/screenshots/visual_corpus.parquet` | Fully catalogued with URL, dimensions, SHA256, and 64-bit DCT pHash. |
| **Brand Reference Images** | **0** | 0 | 0 | `data/brands/` | Empty directory scaffolding does NOT count as acquired images. |
| **Brand Similarity Pairs** | **0** | 0 | 0 | `data/brands/` | 0 pairs curated on disk. |
| **YOLO Bounding Box Annotations** | **0** | 0 | 0 | `data/annotations/detection/` | 0 detection annotations on disk. |
| **Semantic Segmentation Masks** | **0** | 0 | 0 | `data/annotations/segmentation/`| 0 segmentation mask PNGs on disk. |
| **Human-Verified Annotations** | **0** | 0 | 0 | `data/annotations/` | Strict zero count: unverified files are never marked as verified. |

---

## 2. Raw Dataset Feeds Inventory (`data/raw/`)

All raw datasets acquired during Milestone 2A are preserved intact without modification. Cryptographic provenance is tracked in `data/metadata/provenance.json`.

| Feed / Source | File Path | Format | Raw Record Count | File Size | SHA-256 Checksum | Modality | Ground-Truth Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhishTank** | `data/raw/urls/phishtank_online_valid_20261122.csv` | CSV | 43,044 rows | 8.30 MB | `a59c884ab7...` | Verified Phishing URLs + Targets | Verified Ground-Truth Phishing |
| **Tranco Top 1M** | `data/raw/urls/tranco_top1m.csv` | CSV | 1,000,000 domains | 22.69 MB | `60b943d04f...` | Popular Web Domains | Unverified Candidate Domains |
| **Tranco Filtered** | `data/raw/urls/tranco_top_50000.csv` | CSV | 50,000 domains | 1.00 MB | `7cba20ec0b...` | Top 50,000 Domains | Unverified Candidate Domains |
| **PhreshPhish** | `data/raw/urls/phreshphish_train-000.parquet` | Parquet | 1,000 rows | 47.34 MB | `9a9c693a95...` | Full HTML source + URLs + Brands | Verified (Phish & Benign) |
| **OpenPhish** | `data/raw/urls/openphish_20261122.txt` | TXT | 300 rows | 15.1 KB | `9d2f4617fa...` | Active Zero-Day Phishing | Verified Ground-Truth Phishing |
| **Zenodo (8041387)**| `data/raw/metadata/brands.csv` | CSV | 86 brands | 17.0 KB | `038b324c03...` | Target Brand Mapping Table | Metadata Reference |
| **External Visual Dataset** | `data/raw/dataset.zip` & `data/raw/external_dataset/` | ZIP / PNG / CSV | 1,000 screenshots (800 Phish, 200 Legit) | 803.25 MB | `2e08cb670b...` | Full Webpage Screenshots + HTML + Metadata | Verified Visual Ground-Truth |

---

## 3. Processed & Normalized Corpus Inventory (`data/processed/`)

All raw records passed through the hardened Milestone 2B preprocessing pipeline:
- **URL Canonicalization & Hardening:** Stripping tracking query parameters (`utm_*`, `fbclid`, `gclid`), stripping userinfo (`user:pass@host`), parsing IPv6 literal hosts (`[2001:db8::1]`), and IDN punycode normalization.
- **Port Normalization:** Default `:80` and `:443` removed; custom ports (`:8080`, `:8443`) preserved.
- **PSL Domain Extraction:** Registrable domain extracted via `tldextract`. Records without valid registrable domains receive synthetic fallback domain keys (`__missing_domain_<idx>__`) to strictly guarantee **zero row loss**.
- **Label Integrity Isolation:** Tranco is isolated completely from the verified URL corpus. Unlabeled, missing, or malformed labels are rejected (`REJECTED_UNKNOWN`) and never defaulted to phishing.

| Artifact | File Path | Format | Records Count | Columns | Verification Status | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Verified URL Corpus (Parquet)** | `data/processed/urls/corpus_normalized.parquet` | Parquet | **44,129** | 16 | `VERIFIED` | Cleaned, deduplicated verified ground-truth URL corpus (43,535 Phishing, 594 Legitimate). |
| **Verified URL Corpus (CSV)** | `data/processed/urls/corpus_normalized.csv` | CSV | **44,129** | 16 | `VERIFIED` | CSV export of the verified ground-truth corpus. |
| **Tranco Unverified Candidates** | `data/processed/urls/tranco_candidate_pool.parquet`| Parquet | **50,000** | 16 | `UNVERIFIED_CANDIDATE` | Isolated candidate pool for potential future semi-supervised or dynamic scanning. |
| **Catalogued Visual Corpus** | `data/processed/screenshots/visual_corpus.parquet` | Parquet | **1,000** | 16 | `VERIFIED` | 1,000 validated screenshots (800 Phish, 200 Legit) with SHA256, 64-bit DCT pHash, URL, and dimensions. |

### Verified Corpus Schema:
1. `original_url`: Exact raw input URL.
2. `canonical_url`: Normalized URL (lowercase scheme/host, stripped userinfo/noise).
3. `url_hash_sha256`: Unique 64-character hex identifier.
4. `scheme`: Protocol (`http` or `https`).
5. `hostname`: Fully qualified domain name.
6. `registrable_domain`: Domain under public suffix (e.g. `paypal.co.uk`, `google.com`).
7. `subdomain`: Extracted subdomain prefix (e.g. `login`, `secure`).
8. `tld`: Top-level public suffix (`com`, `co.uk`, `in`).
9. `port`: Non-standard port or null.
10. `path`: Normalized path string.
11. `has_query`: Boolean indicator of query parameters.
12. `raw_label`: Preserved raw source label (provenance tracking).
13. `label`: Binary classification target (`0` = Legitimate, `1` = Phishing, `-1` = Candidate).
14. `verification_status`: Status tag (`VERIFIED`, `UNVERIFIED_CANDIDATE`, `REJECTED_UNKNOWN`).
15. `source`: Provenance feed name (`PhishTank`, `PhreshPhish`, `OpenPhish`, `Tranco`).
16. `target_brand`: Target company if impersonation was identified.
17. `collection_timestamp`: ISO 8601 acquisition timestamp.

---

## 4. Curated Brand Reference Gallery (`data/brands/reference_gallery/`)

Master registry: `data/brands/reference_gallery/manifest_brands.json` tracking **18 high-value target brands**:
- **Banking / Finance:** State Bank of India (`sbi`), HDFC Bank (`hdfc`), PayPal (`paypal`), JPMorgan Chase (`chase`), Wells Fargo (`wellsfargo`), Bank of America (`bofa`).
- **Technology & Cloud:** Microsoft (`microsoft`), Google (`google`), Apple (`apple`), Adobe (`adobe`).
- **E-Commerce & Logistics:** Amazon (`amazon`), DHL (`dhl`), FedEx (`fedex`).
- **Social Media & Streaming:** Meta / Facebook (`facebook`), Instagram (`instagram`), WhatsApp (`whatsapp`), LinkedIn (`linkedin`), Netflix (`netflix`).

> [!NOTE]
> Directory scaffolding exists for all 18 brands under `data/brands/reference_gallery/<brand_slug>/`. However, **0 brand reference images are currently on disk**. In accordance with strict empirical reporting, empty directory structures are recorded as 0 acquired brand assets.

---

## 5. Leakage-Safe Splits Inventory (`data/splits/`)

The verified URL corpus was partitioned strictly by **registrable domain** using class-aware stratification and deterministic random seed (`seed=42`). Total record conservation is mathematically enforced ($|\text{Train}| + |\text{Validation}| + |\text{Test}| = |\text{Total}|$).

| Partition | Record Count | Record % | Unique Domains | Domain % | Class 0 (Legit) | Class 1 (Phish) | Phishing % | Parquet Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **32,157** | **72.87%** | **4,224** | **70.0%** | 435 | 31,722 | 98.65% | `data/splits/train/urls.parquet` |
| **Validation** | **7,715** | **17.48%** | **904** | **15.0%** | 77 | 7,638 | 99.00% | `data/splits/validation/urls.parquet` |
| **Test** | **4,257** | **9.65%** | **908** | **15.0%** | 82 | 4,175 | 98.07% | `data/splits/test/urls.parquet` |
| **Total** | **44,129** | **100.0%** | **6,036** | **100.0%** | **594** | **43,535** | **98.65%** | — |

> [!IMPORTANT]
> **Domain Leakage Audit Verification:**
> - $\text{Train} \cap \text{Validation} = \emptyset$ (0 overlapping domains)
> - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping domains)
> - $\text{Validation} \cap \text{Test} = \emptyset$ (0 overlapping domains)
> **Empirical Leakage Count: 0.** Zero cross-split domain leakage verified.
> **Record Conservation:** $32,157 + 7,715 + 4,257 = 44,129$ rows (Zero rows dropped).

---

## 6. Leakage-Safe Visual Splits Inventory (`data/splits/visual/`)

The 1,000 validated screenshots from `visual_corpus.parquet` were partitioned strictly by **connected visual cluster** (combining registrable domain, SHA-256 exact match, and 64-bit DCT perceptual hash Hamming distance $\le 4$) using class-aware stratification and deterministic random seed (`seed=42`).

| Partition | Record Count | Record % | Visual Clusters | Cluster % | Unique Domains | Class 0 (Legit) | Class 1 (Phish) | Phishing % | Parquet Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **624** | **62.4%** | **328** | **70.0%** | **362** | 147 | 477 | 76.44% | `data/splits/visual/train.parquet` |
| **Validation** | **262** | **26.2%** | **69** | **14.7%** | **76** | 26 | 236 | 90.08% | `data/splits/visual/val.parquet` |
| **Test** | **114** | **11.4%** | **72** | **15.3%** | **79** | 27 | 87 | 76.32% | `data/splits/visual/test.parquet` |
| **Total** | **1,000** | **100.0%** | **469** | **100.0%** | **517** | **200** | **800** | **80.00%** | — |

> [!IMPORTANT]
> **Visual Tri-Factor Leakage Audit Verification:**
> - $\text{Train} \cap \text{Validation} = \emptyset$ (0 overlapping domains, 0 overlapping SHA-256 hashes, 0 pHash matches $\le 4$)
> - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping domains, 0 overlapping SHA-256 hashes, 0 pHash matches $\le 4$)
> - $\text{Validation} \cap \text{Test} = \emptyset$ (0 overlapping domains, 0 overlapping SHA-256 hashes, 0 pHash matches $\le 4$)
> **Empirical Leakage Count: 0.** Zero cross-split domain and zero visual duplicate leakage verified.
> **Record Conservation:** $624 + 262 + 114 = 1,000$ screenshots (Zero screenshots lost).

