# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Phase Status:** REVIEW & CORRECTIONS COMPLETED  
**Transitioning To:** Phase 3 — Neural Network Mathematical Sandbox from Scratch  
**Timestamp:** 2026-11-22  

---

## 1. Phase 2 Deliverables & Final Review Status

| Item | Scope | File / Artifact | Review Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Label Integrity** | Label Parsing & Candidate Isolation | `scripts/datasets/normalize_urls.py`, `tranco_candidate_pool.parquet` | COMPLETED | Tranco domains isolated into candidate pool (`label=-1`, `verification_status="UNVERIFIED_CANDIDATE"`). Rejection of unknown/missing labels (`REJECTED_UNKNOWN`). |
| **2. Zenodo Visual Acquisition** | Selective Download & Safe Extraction | `scripts/datasets/download_sources/zenodo.py` | COMPLETED | Selective downloading of specific archives, MD5 checksum verification, Zip-Slip traversal protection, actual extracted image counting. |
| **3. Visual Dataset Readiness** | Empirical Local Asset Audit | `scripts/datasets/dataset_statistics.py` | COMPLETED | 8/8 valid local screenshots. Brand assets and pairs accurately reported as 0 on disk (empty folders not counted as acquired datasets). |
| **4. Annotation Improvements** | Independent Box Validation & Segmentation | `scripts/datasets/validate_annotations.py` | COMPLETED | Fixed global errors-list bug; each box evaluated independently. Added `SegmentationMaskManager` (save, load, pair dimension validation). Strict separation of human-verified vs unverified annotations. |
| **5. Domain-Aware Splitting** | Class-Stratified Zero-Leakage Partitions | `scripts/datasets/generate_splits.py` | COMPLETED | Stratified domain splitting with mathematical zero leakage. Total record conservation asserted (44,129 rows in -> 44,129 rows split). Temporal holdout splitter implemented. |
| **6. URL Normalization** | Parser Hardening & Edge Cases | `scripts/datasets/normalize_urls.py` | COMPLETED | Strips userinfo, handles custom ports, parses IPv6 brackets, normalizes IDN punycode, preserves raw original URL and source label. |
| **7. Final Verification** | Unit Tests & Documentation | `tests/`, `DATASET_AUDIT.md`, `DATASET_QUALITY_REPORT.md` | COMPLETED | 127/127 unit tests passing. Audits updated with truthful, empirically measured numbers. |
| **8. Visual Dataset Integration** | Safe Extraction, Cataloging & Splits | `data/raw/external_dataset/`, `data/processed/screenshots/visual_corpus.parquet`, `data/splits/visual/` | COMPLETED | Extracted 2,894 files from `dataset.zip` safely (0 zip-slip, 0 forbidden binaries). Catalogued 1,000 valid screenshots (800 Phish, 200 Legit) across 517 domains. Created zero-leakage visual splits (Train: 547, Val: 274, Test: 179). 131 unit tests passing. |

---

## 2. Phase 2 Exit Criteria Verification

- [x] **Source Research & Provenance:** 6 target sources audited; cryptographic hashes logged in `provenance.json`.
- [x] **Automated Acquisition:** Modular streaming downloaders with resume support and integrity verification.
- [x] **Verified vs Unverified Data Separation:** 44,129 verified URLs in `corpus_normalized.parquet`; 50,000 unverified candidate domains in `tranco_candidate_pool.parquet`.
- [x] **Visual Screenshot Acquisition & Validation:** 1,008 total valid screenshots on disk (1,000 newly acquired + 8 baseline; 804 Phish, 204 Legit; 0 corrupt, 0 blank).
- [x] **Visual Dataset Preprocessing & Cataloging:** `data/processed/screenshots/visual_corpus.parquet` tracking dimensions, SHA-256, 64-bit DCT pHash, and canonical URLs.
- [x] **Domain Leakage Prevention:** Verified 0 overlapping domains across Train, Val, and Test splits for both URL corpus and visual screenshot dataset.
- [x] **Documentation & Empirical Reports:** `DATASET_MANIFEST.md`, `DATASET_AUDIT.md`, `DATASET_SOURCES.md`, and `DATASET_QUALITY_REPORT.md` reflect actual measured numbers.
- [x] **Zero Automated Model Training:** As strictly mandated, NO models were automatically trained. Ready for Phase 3 ResNet screenshot classifier.
- [x] **Comprehensive Testing:** 131 unit tests pass with zero failures.
