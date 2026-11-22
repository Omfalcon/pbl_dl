# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Phase Status:** COMPLETED  
**Transitioning To:** Phase 3 — Neural Network Mathematical Sandbox from Scratch  
**Timestamp:** 2026-11-22  

---

## 1. Phase 2 Deliverables & Progress Checklist

| Milestone | Deliverable | File Path | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- |
| **2A** | **Dataset Source Research** | [`DATASET_SOURCES.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/DATASET_SOURCES.md) | COMPLETED | Verified Zenodo, PhreshPhish, PhishTank, OpenPhish, URLhaus, Tranco. |
| **2A** | **Auth & Secrets Template** | [`.env.example`](file:///c:/Users/hp/PycharmProjects/pbl_dl/.env.example) | COMPLETED | Keys documented for PhishTank, URLhaus, Zenodo. |
| **2A** | **Curated Brand Gallery** | [`data/brands/reference_gallery/manifest_brands.json`](file:///c:/Users/hp/PycharmProjects/pbl_dl/data/brands/reference_gallery/manifest_brands.json) | COMPLETED | Master registry and directories for 18 monitored brands. |
| **2A** | **Central Acquisition CLI** | [`scripts/datasets/download_all.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/download_all.py) | COMPLETED | Multi-source streaming downloaders with SHA-256 provenance tracking. |
| **2A** | **Actual Raw Data Ingested**| `data/raw/urls/`, `data/raw/metadata/` | COMPLETED | 94,344 raw records ingested across PhishTank, Tranco, PhreshPhish, OpenPhish, Zenodo brands. |
| **2B** | **Image Validation Engine** | [`scripts/datasets/validate_images.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/validate_images.py) | COMPLETED | Corrupt header detection, zero-byte detection, blank solid screenshot detection (std dev < 3.0), and Lanczos resampling. |
| **2B** | **URL Normalizer** | [`scripts/datasets/normalize_urls.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/normalize_urls.py) | COMPLETED | Tracking parameter stripping (utm, fbclid), default port removal, PSL registrable domain extraction via `tldextract`. |
| **2B** | **Deduplication Engine** | [`scripts/datasets/deduplicate.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/deduplicate.py) | COMPLETED | Cryptographic SHA-256 URL/image matching and 64-bit DCT perceptual hash (pHash) near-duplicate clustering. |
| **2B** | **Unified Pipeline & Corpus** | [`scripts/datasets/preprocess_pipeline.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/preprocess_pipeline.py) | COMPLETED | Cleaned 94,125 URLs across 55,364 domains; saved to `corpus_normalized.parquet` & `corpus_normalized.csv`. |
| **2B** | **Preprocessing Report** | [`data/reports/preprocessing_report.json`](file:///c:/Users/hp/PycharmProjects/pbl_dl/data/reports/preprocessing_report.json) | COMPLETED | Full telemetry of raw vs cleaned records, duplicate drop counts, class distribution (50.5k legit / 43.5k phish). |
| **2C** | **Annotation Validator** | [`scripts/datasets/validate_annotations.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/validate_annotations.py) | COMPLETED | Validates YOLO detection bounding boxes (5 classes) and semantic segmentation masks (6 classes). |
| **2C** | **Annotation Visualizer**| [`scripts/datasets/visualize_annotations.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/visualize_annotations.py) | COMPLETED | OpenCV/PIL visual overlay renderer for human review of bounding boxes and semantic segmentation masks. |
| **2C** | **Annotation Manager** | [`scripts/datasets/generate_annotations.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/generate_annotations.py) | COMPLETED | Generates YOLO annotations, marks auto-generated labels as unverified, and creates previews. |
| **2D** | **Domain-Aware Splitter** | [`scripts/datasets/generate_splits.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/generate_splits.py) | COMPLETED | Generates Train (70%), Val (15%), Test (15%) strictly partitioned by registrable domain with zero leakage proof. |
| **2D** | **Dataset Profiler** | [`scripts/datasets/dataset_statistics.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/scripts/datasets/dataset_statistics.py) | COMPLETED | Computes statistical overview, URL length distribution, top brand targets; saved to `dataset_statistics.json`. |
| **2D** | **Audit & Quality Reports**| [`DATASET_AUDIT.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/DATASET_AUDIT.md), [`DATASET_QUALITY_REPORT.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/DATASET_QUALITY_REPORT.md) | COMPLETED | Complete empirical audit and quality verification reports. |
| **2D** | **Full Unit Test Suite** | `tests/test_dataset_*.py`, `tests/test_data_*.py`, `tests/test_annotations.py` | COMPLETED | 33 Phase 2 unit tests passing; 117 total unit tests passing in ~2.1s. |

---

## 2. Phase 2 Exit Criteria Verification

- [x] **Source Research & Provenance:** 6 target sources audited; cryptographic hashes logged in `provenance.json`.
- [x] **Automated Acquisition:** Modular streaming downloaders with resume support and integrity verification.
- [x] **Actual Datasets Acquired:** 94,344 raw records (PhishTank 43k, Tranco 50k, PhreshPhish 1k, OpenPhish 300, Zenodo brands).
- [x] **Data Preprocessing & Cleaning:** Tracking noise stripped, default ports removed, PSL domains extracted.
- [x] **Deduplication Engine:** SHA-256 for exact duplicates; 64-bit DCT pHash for near-duplicate visual screenshots.
- [x] **Annotation Infrastructure:** Strict validators and visualizers for YOLO bounding boxes and segmentation masks.
- [x] **Domain Leakage Prevention:** Verified 0 overlapping domains across Train, Val, and Test splits.
- [x] **Documentation & Empirical Reports:** `DATASET_AUDIT.md`, `DATASET_SOURCES.md`, and `DATASET_QUALITY_REPORT.md` reflect actual measured numbers.
- [x] **Zero Automated Model Training:** As strictly mandated, NO models were automatically trained. Training remains under manual user control.
- [x] **Comprehensive Testing:** All unit tests pass with zero regressions.
