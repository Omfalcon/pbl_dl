# DATASET_AUDIT.md — PhishVision AI Dataset Audit & Inventory

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Date:** 2026-11-22  
**Audit Standard:** Empirical inspection of filesystem state, row counts, cryptographic hashes, and domain sets.

---

## 1. Raw Dataset Feeds Inventory (`data/raw/`)

All raw datasets acquired during Milestone 2A are preserved intact without modification. Cryptographic provenance is tracked in `data/metadata/provenance.json`.

| Feed / Source | File Path | Format | Record Count | File Size | SHA-256 Checksum | Modality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PhishTank** | `data/raw/urls/phishtank_online_valid_20261122.csv` | CSV | 43,044 rows | 8.30 MB | `a59c884ab7...` | Verified Phishing URLs + Targets |
| **Tranco Top 1M** | `data/raw/urls/tranco_top1m.csv` | CSV | 1,000,000 domains | 22.69 MB | `60b943d04f...` | Popular Benign Research Domains |
| **Tranco Filtered** | `data/raw/urls/tranco_top_50000.csv` | CSV | 50,000 domains | 1.00 MB | `7cba20ec0b...` | Top 50,000 Benign Domains |
| **PhreshPhish** | `data/raw/urls/phreshphish_train-000.parquet` | Parquet | 1,000 rows | 47.34 MB | `9a9c693a95...` | Full HTML source + URLs + Brand Tags |
| **OpenPhish** | `data/raw/urls/openphish_20261122.txt` | TXT | 300 rows | 15.1 KB | `9d2f4617fa...` | Active Community Zero-Day Phish |
| **Zenodo (8041387)**| `data/raw/metadata/brands.csv` | CSV | 86 brands | 17.0 KB | `038b324c03...` | Target Brand Mapping Table |

---

## 2. Processed & Normalized Corpus Inventory (`data/processed/`)

All raw records passed through the Milestone 2B preprocessing pipeline:
- URL canonicalization and parameter stripping (removing tracking tokens like `utm_*`, `fbclid`).
- Default port stripping (`:80`, `:443`).
- Public Suffix List (PSL) domain extraction via `tldextract`.
- Exact deduplication via SHA-256 URL hashing.

| Artifact | File Path | Format | Records Count | Columns | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Master Normalized Corpus** | `data/processed/urls/corpus_normalized.parquet` | Parquet | **94,125** | 15 | Cleaned, deduplicated, and balanced URL features dataset |
| **Master Normalized CSV** | `data/processed/urls/corpus_normalized.csv` | CSV | **94,125** | 15 | CSV export of the master corpus |

### Corpus Column Schema:
1. `original_url`: Raw input URL string.
2. `canonical_url`: Normalized URL (lowercase scheme/host, stripped noise).
3. `url_hash_sha256`: Unique 64-character hex identifier.
4. `scheme`: Protocol (`http` or `https`).
5. `hostname`: Fully qualified domain name.
6. `registrable_domain`: Domain under public suffix (e.g. `paypal.co.uk`, `sbi.co.in`).
7. `subdomain`: Extracted subdomain prefix (e.g. `login`, `secure`).
8. `tld`: Top-level public suffix (`com`, `co.uk`, `sbi`).
9. `port`: Non-standard port or null.
10. `path`: Normalized path string.
11. `has_query`: Boolean indicator of query parameters.
12. `label`: Binary classification target (`0` = Legitimate, `1` = Phishing).
13. `source`: Provenance feed name (`PhishTank`, `Tranco`, `PhreshPhish`, `OpenPhish`).
14. `target_brand`: Target company if impersonation was identified (e.g. `Facebook`, `Amazon.com`, `PayPal`).
15. `collection_timestamp`: ISO 8601 acquisition timestamp.

---

## 3. Curated Brand Reference Gallery (`data/brands/reference_gallery/`)

Master registry: `data/brands/reference_gallery/manifest_brands.json` tracking **18 high-value target brands**:
- **Banking / Finance:** State Bank of India (`sbi`), HDFC Bank (`hdfc`), PayPal (`paypal`), JPMorgan Chase (`chase`), Wells Fargo (`wellsfargo`), Bank of America (`bofa`).
- **Technology & Cloud:** Microsoft (`microsoft`), Google (`google`), Apple (`apple`), Adobe (`adobe`).
- **E-Commerce & Logistics:** Amazon (`amazon`), DHL (`dhl`), FedEx (`fedex`).
- **Social Media & Streaming:** Meta / Facebook (`facebook`), Instagram (`instagram`), WhatsApp (`whatsapp`), LinkedIn (`linkedin`), Netflix (`netflix`).

Each brand contains dedicated workspaces for `logos/`, `canonical_screenshots/`, and `embeddings/`.

---

## 4. Leakage-Safe Splits Inventory (`data/splits/`)

Partitioned strictly by **registrable domain** using a deterministic random seed (`seed=42`):

| Partition | Proportion Target | URL Records | Unique Domains | Class 0 (Legit) | Class 1 (Phish) | Parquet Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **70%** | **66,828** (71.0%) | **38,754** (70.0%) | 35,393 (53.0%) | 31,435 (47.0%) | `data/splits/train/urls.parquet` |
| **Validation** | **15%** | **10,374** (11.0%) | **8,304** (15.0%) | 7,602 (73.3%) | 2,772 (26.7%) | `data/splits/validation/urls.parquet` |
| **Test** | **15%** | **16,923** (18.0%) | **8,306** (15.0%) | 7,594 (44.9%) | 9,329 (55.1%) | `data/splits/test/urls.parquet` |
| **Total** | **100%** | **94,125** | **55,364** | **50,589** (53.7%) | **43,536** (46.3%) | — |

> [!IMPORTANT]
> **Domain Overlap Audit:**
> - $\text{Train} \cap \text{Validation} = \emptyset$ (0 overlapping domains)
> - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping domains)
> - $\text{Validation} \cap \text{Test} = \emptyset$ (0 overlapping domains)
> **Empirical Leakage Count: 0.** Zero cross-split domain leakage verified.
