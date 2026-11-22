# PROJECT_STATE.md — Persistent Project State

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Current Phase:** 2 — Dataset Acquisition, Preparation, Annotation & Validation  
**Phase Status:** COMPLETED  
**Next Phase:** Phase 3 — Neural Network Mathematical Sandbox from Scratch  
**Last Updated:** 2026-11-22  

---

## 1. Executive Summary

PhishVision AI is an AI-powered phishing website detection and brand impersonation platform combining lexical URL intelligence, deep computer vision, behavioral and sequence analysis, and calibrated multi-evidence risk fusion.

- **Phase 0 (System Architecture & Requirements):** COMPLETED.
- **Phase 1 & Phase 1.1 (Repository Foundation, Scanner Implementation & Security Hardening):** COMPLETED.
- **Phase 2 (Dataset Acquisition, Preparation, Annotation & Validation):** COMPLETED (INCLUDING FINAL VISUAL DATASET INTEGRATION).
  - All 4 Milestones (2A, 2B, 2C, 2D) fully implemented, validated, and hardened through code review and empirical visual acquisition.
  - Raw feeds ingested: PhishTank (43k), Tranco (50k), PhreshPhish (1k), OpenPhish (300), Zenodo brands (86), and External Visual Dataset (1,000 screenshots).
  - Ground-truth separation: **44,129 verified URL records** (43,535 Phishing, 594 Legitimate) in `corpus_normalized.parquet`; **50,000 unverified candidate domains** isolated in `tranco_candidate_pool.parquet`.
  - Visual Dataset: **1,008 total valid screenshots on disk** (804 Phishing, 204 Legitimate; 1,000 external + 8 scanner baseline). Catalogued into `data/processed/screenshots/visual_corpus.parquet`.
  - Curated Brand Reference Gallery established for 18 target brands (0 local image assets on disk accurately audited).
  - Annotation validation and visualization suite operational for YOLO bounding boxes (independent box evaluation) and semantic segmentation masks (`SegmentationMaskManager` for image-mask pairing).
  - Domain-aware leakage-safe splitting generated:
    - URL Splits: Train (32,157 records), Val (7,715 records), Test (4,257 records) with **0 domain leakage**.
    - Visual Splits: Train (624 screenshots), Val (262 screenshots), Test (114 screenshots) with **0 domain leakage, 0 exact duplicate leakage, and 0 near-duplicate leakage**.
  - Comprehensive statistical profile exported to `data/reports/dataset_statistics.json`.
  - **Phase 2 Committed in git:** `56d8565` (`feat(dataset): finalize Phase 2 visual dataset integration`).
- **Phase 3 (Neural Network Mathematical Sandbox from Scratch):** COMPLETED.
  - Linear algebra primitives (dot product, norms, matrix multiply, cosine similarity).
  - Activations & analytical derivatives (ReLU, Sigmoid, Tanh, Softmax).
  - Loss functions & gradients (MSE, Binary Cross Entropy with stability clamping).
  - Single artificial neuron with analytic forward/backward propagation.
  - Fully vectorized Dense layer with Xavier/He initialization and backpropagation.
  - Sequential multi-layer MLP pipeline.
  - Dynamic reverse-mode automatic differentiation computational graph.
  - Central finite differences numerical gradient checker (asserting relative error $< 10^{-6}$).
  - Full mathematical documentation published in `docs/PHASE_3_MATH_FOUNDATIONS.md`.
- **Phase 4 (Feedforward Deep Neural Network for Lexical Phishing Detection):** IN PROGRESS.
  - Implemented 4 custom optimizers from scratch: SGD, SGD with Momentum, RMSProp, and Adam (`app/ml/from_scratch/optimizers.py`).
  - Implemented 3 learning rate schedulers: StepLR, ExponentialLR, and CosineAnnealingLR.
  - Verified updates against hand-calculated analytical values and proven convergence on regression & classification.
  - Optimizer convergence benchmark exported to `data/reports/optimizer_benchmark.json`.
  - **Full test suite: 156 unit tests passing in ~4.8s**. Zero regressions.

---

## 2. Current State Snapshot

- **Active Phase:** `Phase 4` (Feedforward Deep Neural Network for Lexical Phishing Detection)
- **Target Next Phase:** `Phase 5` (Lexical Feature Extraction Pipeline & MLP Training)
- **Working Modules:**
  - `app.scanner.anti_ssrf`: Hardened URL parsing, port validation, DNS resolution, and IP classification.
  - `app.scanner.proxy`: Validating local egress forward proxy with connection-level IP pinning and TLS SNI preservation.
  - `app.scanner.playwright_scanner`: Isolated Playwright Chromium scanner with context-level routing, popup containment, service worker blocking, and secure TLS checking.
  - `app.scanner.models`: Pydantic structured scan telemetry schemas.
  - `scripts.datasets.download_sources`: Modular streaming downloaders for OpenPhish, PhishTank, Tranco, Zenodo, and PhreshPhish.
  - `scripts.datasets.download_all`: Central CLI acquisition orchestrator.
  - `scripts.datasets.fetch_brand_references`: Brand gallery scaffolding and metadata management.
  - `scripts.datasets.validate_images`: Image integrity checker, blank screenshot filter, and standardization resizer.
  - `scripts.datasets.normalize_urls`: URL canonicalizer, tracking stripper, and PSL registrable domain extractor.
  - `scripts.datasets.deduplicate`: Exact SHA-256 and perceptual pHash deduplicator.
  - `scripts.datasets.preprocess_pipeline`: Master data ingestion and corpus generator.
  - `scripts.datasets.validate_annotations`: YOLO detection and U-Net segmentation annotation validator.
  - `scripts.datasets.visualize_annotations`: Visual overlay renderer for bounding boxes and segmentation masks.
  - `scripts.datasets.generate_annotations`: Annotation manager and preview generator.
  - `scripts.datasets.generate_splits`: Domain-aware leakage-safe train/val/test partitioner.
  - `scripts.datasets.dataset_statistics`: Comprehensive dataset profiler.
- **Working Environment:**
  - Python: `3.13.3` (Located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe`)
  - Deep Learning Framework: `PyTorch 2.6.0+cu124` (Hardware acceleration: CUDA 12.4 operational)
  - Key Libraries: `playwright 1.50+`, `beautifulsoup4 4.14.3`, `tldextract 5.3.1`, `pydantic 2.x`, `torchvision 0.21.0`, `ultralytics 8.4.60`, `scikit-learn 1.8.0`, `imagehash 4.3.2`
- **Git State:** Tracking `main`, clean repository structure with `.gitignore` excluding transient scan artifacts.

---

## 3. Completed Work

### Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
1. Authored all 10 context management documents: `AGENTS.md`, `PROJECT_STATE.md`, `CURRENT_PHASE.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `TECH_STACK.md`, `API_CONTRACTS.md`, `DATASET_MANIFEST.md`, `MODEL_REGISTRY.md`, `DECISIONS.md`, and `PHASE_ROADMAP.md`.
2. Verified Python virtual environment with CUDA 12.4 PyTorch support.
3. Formally mandated Python + React as exclusive project technology (ADR-001) and established Visual Intelligence as central research pillar (ADR-009).

### Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
1. **Package Hierarchy:** Created `app/core/`, `app/scanner/`, `app/intelligence/`, `app/fusion/`, `app/api/`, `config/`, `tests/`, `data/`, `models/` with `__init__.py` modules.
2. **Hardened Anti-SSRF Module (`app/scanner/anti_ssrf.py`):**
   - Scheme and authority sanitization, port boundaries, obfuscated IP parsing (dword/hex/octal), IPv4-mapped IPv6 literal inspection.
3. **Validating Egress Forward Proxy (`app/scanner/proxy.py`):**
   - Intercepts all outbound HTTP/CONNECT requests from Chromium and binds outbound TCP sockets directly to validated public `pinned_ip`, preserving TLS SNI.
4. **Hardened Playwright Scanner (`app/scanner/playwright_scanner.py`):**
   - Removed insecure `--no-sandbox` flags, service worker blocking, popup containment, secure TLS certificate checking.
5. **Interactive CLI (`scripts/manual_scan.py`):**
   - Interactive manual scan CLI tool supporting both headless and `--headful` execution.

### Phase 2: Dataset Acquisition, Preparation & Annotation (IN PROGRESS)
1. **Source Research & Verification (`DATASET_SOURCES.md`):**
   - Empirically audited Zenodo, PhreshPhish, PhishTank, OpenPhish, URLhaus, and Tranco.
2. **Curated Brand Reference Gallery (`data/brands/reference_gallery/`):**
   - Populated `manifest_brands.json` and workspaces for 18 monitored target brands.
3. **Automated Downloader Architecture (`scripts/datasets/download_sources/`):**
   - Modular streaming downloaders with provenance logging (`data/metadata/provenance.json`).
4. **Data Cleaning & Image Validation (`scripts/datasets/validate_images.py`):**
   - Image integrity decoding, blank screenshot rejection (std dev < 3.0), and Lanczos standardization.
5. **URL Canonicalization & Normalization (`scripts/datasets/normalize_urls.py`):**
   - Marketing tracking parameter stripping, default port removal, PSL registrable domain extraction.
6. **Perceptual Deduplication Engine (`scripts/datasets/deduplicate.py`):**
   - Exact SHA-256 matching and 64-bit DCT pHash near-duplicate clustering.
7. **Production Corpus Generation (`scripts/datasets/preprocess_pipeline.py`):**
   - Processed **94,125 unique, balanced records** (50,589 Legitimate vs 43,536 Phishing) across **55,364 registrable domains**.
8. **Testing & Validation:**
   - 21 dataset unit tests passing. Total suite: **105 passed in 1.88s**.

---

## 4. Pending Work (Master Plan)

- **Phase 2:** Dataset acquisition, Curated Brand Reference Gallery population, annotation, and domain-based splitting.
- **Phase 3:** Neural network mathematical sandbox (pure NumPy tensor ops, neurons, backprop, numerical gradient checking).
- **Phase 4:** Optimizers (SGD, Momentum, Nesterov) and learning algorithms (Hebbian, Oja's, Competitive, Boltzmann).
- **Phase 5:** URL intelligence (PSL parsing, homoglyphs, typosquatting, entropy) and MLP baseline.
- **Phase 6:** Visual Core 1 & 2: ResNet screenshot classifier (unconditional) and Siamese brand visual matcher (conditional).
- **Phase 7:** Visual Core 3 & 4: YOLO logo/UI detector, Faster R-CNN comparative benchmark, U-Net and YOLO-Seg segmentation.
- **Phase 8:** LSTM and Transformer redirect sequence learning.
- **Phase 9:** VAE latent-space experiments (GAN excluded).
- **Phase 10:** Multi-modal calibrated risk fusion engine synthesizing visual evidence vectors with lexical and behavioral telemetry.
- **Phase 11:** React dashboard and full application integration.
- **Phase 12:** Robustness, latency, and error analysis.
- **Phase 13:** Final documentation, report, demo, and submission.

---

## 5. Technology Mandate & Operational Notes

1. **Approved Technology Decision (ADR-001):**  
   The project is officially and permanently a **Python + React** project. Legacy C/C++/Java implementation requirements, along with CMake/Make/Maven/Gradle build systems, are intentionally excluded requirements. All mathematical and ML objectives are fully preserved and executed in pure Python/NumPy/PyTorch.
2. **Virtual Environment Filesystem Location (ADR-007):**  
   The Python venv is located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`. Terminal tools executing Python within this environment require either direct execution via the full binary path or `BypassSandbox: true` if cross-directory permissions are enforced.
3. **Environment Security Boundaries:**  
   The scanner enforces connection-level IP safety via the `ValidatingEgressProxy` and context-level routing. While this provides application and network-layer anti-SSRF and DNS rebinding immunity, true OS-level process and kernel isolation requires containerized deployment (e.g., Docker container with network namespaces and non-root execution), which will be packaged for production.

---

## 6. Last Verified Tests & Checks

- **Date:** 2026-11-22
- **Command:** `& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -m pytest -v tests/`
- **Result:** `85 passed in 3.00s` (Exit Code 0). All Anti-SSRF rules, proxy IP pinning, TLS error handling, and live navigation checks verified.
- **Offline Suite:** `84 passed in 0.68s` with `pytest -m "not integration"`.

---

## 7. Exact Next Action

**Begin Phase 2:**  
Execute dataset acquisition, Curated Brand Reference Gallery population (`data/brands/reference_gallery/` for 15–20 brands), image deduplication via pHash, and leakage-safe domain-based dataset partitioning.
