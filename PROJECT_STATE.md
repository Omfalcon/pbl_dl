# PROJECT_STATE.md — Persistent Project State

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Current Phase:** Phase 4 — Feedforward Deep Neural Network for Lexical Phishing Detection  
**Phase Status:** IN PROGRESS (Optimizers and Schedulers Completed; Feature Extraction & MLP Remaining)  
**Next Milestone:** Lexical Feature Extraction Pipeline & MLP Training  
**Last Updated:** 2026-11-22  

---

## 1. Executive Summary

PhishVision AI is an AI-powered phishing website detection and brand impersonation platform combining lexical URL intelligence, deep computer vision, behavioral and sequence analysis, and calibrated multi-evidence risk fusion.

- **Phase 0 (System Architecture & Requirements):** COMPLETED.
- **Phase 1 & Phase 1.1 (Repository Foundation, Scanner Implementation & Security Hardening):** COMPLETED.
- **Phase 2 (Dataset Acquisition, Preparation, Annotation & Validation):** COMPLETED (COMMITTED in `56d8565`).
  - Ingested multi-source raw feeds: PhishTank, Tranco, PhreshPhish, OpenPhish, Zenodo brands, and External Visual Dataset (1,000 screenshots).
  - Ground-truth separation: **44,129 verified URL records** (43,535 Phishing, 594 Legitimate) in `corpus_normalized.parquet`; **50,000 unverified candidate domains** isolated in `tranco_candidate_pool.parquet`.
  - Visual Dataset: **1,008 total valid screenshots on disk** (804 Phishing, 204 Legitimate). Catalogued in `data/processed/screenshots/visual_corpus.parquet`.
  - Brand Reference Gallery established for 18 target brands.
  - Annotation validation and visualization suite operational for YOLO bounding boxes (independent box evaluation) and semantic segmentation masks.
  - Domain-aware leakage-safe splitting generated:
    - URL Splits: Train (32,157 records), Val (7,715 records), Test (4,257 records) with **0 domain leakage**.
    - Visual Splits: Train (624 screenshots), Val (262 screenshots), Test (114 screenshots) with **0 domain leakage, 0 exact duplicate leakage, and 0 near-duplicate leakage**.
  - Statistical profile exported to `data/reports/dataset_statistics.json`.
- **Phase 3 (Neural Network Mathematical Sandbox from Scratch):** COMPLETED.
  - Linear algebra primitives (dot product, norms, matrix multiply, cosine similarity).
  - Activations & analytical derivatives (ReLU, Sigmoid, Tanh, Softmax).
  - Loss functions & gradients (MSE, Binary Cross Entropy with stability clamping).
  - Single artificial neuron with analytic forward/backward propagation.
  - Fully vectorized Dense layer with Xavier/He initialization and backpropagation.
  - Dynamic reverse-mode automatic differentiation computational graph.
  - Central finite differences numerical gradient checker (asserting relative error $< 10^{-6}$).
  - Full mathematical documentation published in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md).
- **Phase 4 (Feedforward Deep Neural Network for Lexical Phishing Detection):** IN PROGRESS.
  - Implemented 4 custom optimizers from scratch: SGD, SGD with Momentum, RMSProp, and Adam (`app/ml/from_scratch/optimizers.py`).
  - Implemented 3 learning rate schedulers: StepLR, ExponentialLR, and CosineAnnealingLR.
  - Hardened with input length validation, shape checking, gradient finiteness checks, state consistency, and scheduler rate synchronization.
  - Verified updates against hand-calculated analytical values and proven convergence on regression & classification.
  - Optimizer convergence benchmark exported to `data/reports/optimizer_benchmark.json`.
  - **Full test suite: 167 unit tests passing (1 deselected) in ~6.39s**. Zero regressions.

---

## 2. Current State Snapshot

- **Active Phase:** `Phase 4` (Feedforward Deep Neural Network for Lexical Phishing Detection)
- **Target Next Step:** Lexical URL feature extraction pipeline & MLP model training
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
  - `app.ml.from_scratch.primitives`: Mathematical primitives (dot product, matrix multiply, cosine similarity, vector norms).
  - `app.ml.from_scratch.activations`: Activation functions and analytic derivatives (ReLU, Sigmoid, Tanh, Softmax).
  - `app.ml.from_scratch.losses`: Loss functions and analytic gradients (MSE, Binary Cross Entropy).
  - `app.ml.from_scratch.neuron`: Single neuron forward and backward pass.
  - `app.ml.from_scratch.layers`: Dense layer with weight/bias initialization, forward and backward propagation.
  - `app.ml.from_scratch.computational_graph`: Reverse-mode automatic differentiation tape and DAG topological execution.
  - `app.ml.from_scratch.gradient_check`: Finite-differences numerical gradient validation.
  - `app.ml.from_scratch.optimizers`: SGD, SGDMomentum, RMSProp, Adam, StepLR, ExponentialLR, CosineAnnealingLR.
- **Working Environment & Reproducibility:**
  - Python: `3.11 - 3.13` support (Active local environment: `Python 3.13.3`)
  - Deep Learning Framework: `PyTorch 2.6.0+cu124` (Hardware acceleration: CUDA 12.4 operational)
  - Key Libraries: `playwright 1.50+`, `beautifulsoup4 4.14.3`, `tldextract 5.3.1`, `pydantic 2.x`, `torchvision 0.21.0`, `ultralytics 8.4.60`, `scikit-learn 1.8.0`, `imagehash 4.3.2`
  - Automated CI: GitHub Actions (`.github/workflows/ci.yml`)
- **Git State:** Tracking `main`, clean repository structure with `.gitignore` excluding transient scan artifacts, raw archives, and datasets.

---

## 3. Completed Milestones

### Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
1. Authored all 10 context management documents: `AGENTS.md`, `PROJECT_STATE.md`, `CURRENT_PHASE.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `TECH_STACK.md`, `API_CONTRACTS.md`, `DATASET_MANIFEST.md`, `MODEL_REGISTRY.md`, `DECISIONS.md`, and `PHASE_ROADMAP.md`.
2. Verified Python virtual environment with CUDA 12.4 PyTorch support.
3. Formally mandated Python + React as exclusive project technology (ADR-001) and established Visual Intelligence as central research pillar (ADR-009).

### Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
1. **Package Hierarchy:** Created `app/core/`, `app/scanner/`, `app/intelligence/`, `app/fusion/`, `app/api/`, `config/`, `tests/`, `data/`, `models/` with `__init__.py` modules.
2. **Hardened Anti-SSRF Module (`app/scanner/anti_ssrf.py`):** Scheme and authority sanitization, port boundaries, obfuscated IP parsing (dword/hex/octal), IPv4-mapped IPv6 literal inspection.
3. **Validating Egress Forward Proxy (`app/scanner/proxy.py`):** Intercepts outbound HTTP/CONNECT requests from Chromium and binds outbound TCP sockets directly to validated public `pinned_ip`, preserving TLS SNI.
4. **Hardened Playwright Scanner (`app/scanner/playwright_scanner.py`):** Removed insecure `--no-sandbox` flags, service worker blocking, popup containment, secure TLS certificate checking.
5. **Interactive CLI (`scripts/manual_scan.py`):** Interactive manual scan CLI tool supporting both headless and `--headful` execution.

### Phase 2: Dataset Acquisition, Preparation & Annotation (COMPLETED)
1. Ingested and canonicalized 44,129 verified URLs and isolated 50,000 Tranco domains in unverified candidate pool.
2. Extracted and validated 1,008 screenshots (804 Phishing, 204 Legitimate) with zero-leakage domain partitions.
3. Implemented YOLO bounding box and semantic segmentation annotation validation toolchain.
4. Exported complete statistical profile to `data/reports/dataset_statistics.json`.
5. Committed in git: `56d8565` (`feat(dataset): finalize Phase 2 visual dataset integration`).

### Phase 3: Neural Network Mathematical Sandbox from Scratch (COMPLETED)
1. Pure NumPy tensor ops, activations, loss functions, dense layers, sequential model, autograd DAG graph, and numerical gradient checking.
2. Comprehensive documentation in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md).

### Phase 4: Feedforward Deep Neural Network for Lexical Phishing Detection (IN PROGRESS)
1. Custom optimizers from scratch: SGD, SGD with Momentum, RMSProp, Adam (`app/ml/from_scratch/optimizers.py`).
2. Custom learning rate schedulers: StepLR, ExponentialLR, CosineAnnealingLR.
3. Hardened parameter validation, shape checking, non-finite gradient rejection, and scheduler synchronization.
4. Empirical convergence benchmarked in `data/reports/optimizer_benchmark.json`.

---

## 4. Pending Work (Master Plan)

- **Phase 4 (Remaining):** URL lexical feature extraction (30+ features) and Feedforward MLP training and evaluation.
- **Phase 5:** URL intelligence (PSL parsing, homoglyphs, typosquatting, entropy) and baseline comparison.
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
   The project is officially and permanently a **Python + React** project. Legacy C/C++/Java implementation requirements are excluded.
2. **Reproducibility:**  
   Standard installation via `requirements.txt`. Development environment uses Python 3.11–3.13.
3. **Environment Security Boundaries:**  
   The scanner enforces connection-level IP safety via the `ValidatingEgressProxy` and context-level routing.

---

## 6. Last Verified Tests & Checks

- **Date:** 2026-11-22
- **Command:** `pytest -v -m "not integration"`
- **Result:** `167 passed, 1 deselected in 6.39s` (Exit Code 0). All anti-SSRF, scanner models, dataset processing, neural network math foundations, parameter finiteness/dtype/writability checks, and custom optimizer edge cases verified.
