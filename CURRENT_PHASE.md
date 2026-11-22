# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 8.1 — Backend Reliability and Security Audit  
**Phase Status:** AUDIT COMPLETE & EMPIRICALLY VERIFIED (386/386 TESTS PASSING)  
**Previous Phases:**
- Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
- Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
- Phase 2: Dataset Acquisition, Brand Gallery & Partitioning (COMPLETED & COMMITTED in `56d8565`)
- Phase 3: Neural Network Mathematical Sandbox from Scratch (COMPLETED & DOCUMENTED in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md))  
- Phase 4: Optimizers, Ablations & Learning Algorithms from Scratch (COMPLETED & BENCHMARKED in `data/reports/optimizer_benchmark.json`)
- Phase 5: URL Intelligence, Homoglyph Detection, Typosquatting & MLP Baseline (COMPLETED & EVALUATED in `data/reports/lexical_mlp_evaluation.json`)
- Phase 6.1: M-02 ResNet-18 Classifier Implementation, Training & Multi-Region Diagnostics (COMPLETED)
- Phase 6.2, 6.2.1 & 6.2.2: M-03 Siamese CNN Brand Impersonation Matcher Architecture, Authentic Acquisition, Training & Independent Test Evaluation (COMPLETED)
- Phase 7.1: M-04 YOLO UI Component & Logo Detection Pipeline Scaffolding, Class Schema & Dataset Audit (COMPLETED)
- Phase 7.2–7.4: Synthetic Dataset Generation & YOLOv8n Object Detection Experiment (COMPLETED, mAP50=0.9950, mAP50-95=0.9699)
- Phase 7.5 & 7.5.1: YOLOv8n + Siamese + Phishing Pipeline Integration & Multimodal Validation (COMPLETED)
- Phase 7.6: M-05 Faster R-CNN Object Detection Implementation & Benchmark Scaffolding (COMPLETED & VERIFIED)
- Phase 7.7: M-06 YOLOv8n-Seg Instance Segmentation Training & Test Evaluation (COMPLETED, Box mAP50=0.9950, Mask mAP50=0.9950)
- Phase 7.8: M-07 URL Sequence LSTM Training, Evaluation & Targeted Audit (COMPLETED, Accuracy=0.9869, ROC-AUC=0.9666, PR-AUC=0.9991, Macro F1=0.7779)
- Phase 8: Seven-Model Multimodal Integration, Backend Completion, and Calibrated Risk Fusion (COMPLETED)
- Phase 8.1: Backend Reliability and Security Audit (COMPLETED)
**Target Next Milestone:** Phase 9 — Interactive Dashboard / Forensic UI & Production Deployment  
**Timestamp:** 2026-11-23  

---

## 1. Phase 8.1 Backend Reliability and Security Audit Summary

- **Scan Lifecycle Failure Handling (`app/api/v1/endpoints.py`):**
  - Moved pipeline initialization (`pipeline = get_pipeline()`) inside the protected `try` block in `_run_scan_job()`.
  - Guaranteed every scan job strictly transitions to a terminal state (`COMPLETED`, `FAILED`, or `BLOCKED_SSRF`).
  - Added regression test (`test_scan_lifecycle_initialization_failure_reaches_terminal_failed`) proving that initialization failure marks scan record as `FAILED` and surfaces a 500 status on report retrieval instead of hanging in `IN_PROGRESS`.
- **CORS Configuration Hardening (`app/main.py`):**
  - Removed insecure wildcard origin `["*"]` combined with `allow_credentials=True`.
  - Implemented `resolve_cors_configuration()` reading `PHISHVISION_CORS_ORIGINS`.
  - Provided safe default origins for local development: `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:8000`, `http://127.0.0.1:8000`.
  - Enforced automatic disabling of credentials (`allow_credentials=False`) if wildcard `*` is configured.
  - Added regression test (`test_cors_resolution_security`).
- **Image Upload DOS & Decompression Bomb Protection (`app/api/v1/endpoints.py`):**
  - Replaced unconstrained in-memory reads (`await image.read()`) with bounded 64KB chunk streaming capped by `PHISHVISION_MAX_UPLOAD_BYTES` (default 10 MB), returning HTTP 413 `PAYLOAD_TOO_LARGE` if exceeded.
  - Rejection of empty 0-byte uploads with HTTP 400 `EMPTY_PAYLOAD`.
  - Strict format verification with Pillow (`ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP"}`).
  - Dimension limits: $\text{width}, \text{height} \le 4096$ px and $\text{total pixels} \le 16,000,000$ px, returning HTTP 400 `IMAGE_DIMENSIONS_TOO_LARGE`.
  - Pillow decompression bomb protection (`Image.MAX_IMAGE_PIXELS = 16_000_000`) catching `Image.DecompressionBombError` and returning HTTP 400 `DECOMPRESSION_BOMB_DETECTED`.
  - Corrupted payload handling returning HTTP 400 `INVALID_IMAGE_PAYLOAD`.
  - Added regression test (`test_image_upload_security_protections`).
- **Checkpoint Download Integrity (`scripts/download_checkpoints.py`):**
  - Added `expected_sha256: None` to all registry entries (no synthetic or invented hashes).
  - Enforced maximum download size limit `PHISHVISION_MAX_DOWNLOAD_BYTES` (default 500 MB).
  - Streamed download into PID-unique temporary file (`.tmp_{pid}`) with incremental SHA-256 calculation.
  - Atomically replaces destination file (`temp_path.replace(dest_path)`) only upon successful completion and verification.
  - On failure or mismatch, temporary file is removed and existing checkpoint destination is strictly preserved.
  - Added test suite [`tests/test_download_checkpoints.py`](tests/test_download_checkpoints.py) covering size limits, checksum match, mismatch preservation, and unverified downloads.
- **Accurate Health Check Status Reporting (`app/api/v1/endpoints.py`):**
  - Updated `/api/v1/health` logic:
    - `OPERATIONAL`: All 7/7 models loaded (HTTP 200).
    - `DEGRADED`: 1 to 6 models loaded (HTTP 200).
    - `SERVICE_UNAVAILABLE`: 0 models loaded or pipeline initialization error (HTTP 503).
  - Added regression test (`test_health_check_status_levels`).
- **Scan Store Limitation & Production Scaling Requirements:**
  - Documented that `ScanStore` is process-local and in-memory.
  - Formulated architectural requirements for production multi-worker deployment:
    1. Distributed datastore (PostgreSQL / Redis with persistence) for scan records.
    2. Distributed task queue (Celery / ARQ / Redis Streams) for scanning jobs.
    3. Shared object storage (S3 / GCS / MinIO) for screenshot and preview artifacts.
- **API Abuse Protection Assessment:**
  - Dynamic browser scanning is resource-intensive. Recommendations documented for public deployment:
    1. Authentication: API Keys (`X-API-Key`) or Bearer tokens (`Authorization: Bearer <jwt>`).
    2. Rate Limiting: Token-bucket or sliding-window rate limiter (e.g. 10 scans/minute per IP/API key).
- **Automated Verification:**
  - **386/386 tests passing (100%)** across the entire repository. Zero regressions.

- **Unified Pipeline Architecture (`app/ml/unified_pipeline.py`):**
  - Unifies all 7 trained neural model pipelines (M-01 through M-07) behind a single production inferencer.
  - Zero model retraining, zero modification to existing trained checkpoints.
  - Transparent model availability tracking and physical checkpoint existence inspection (`get_checkpoint_status()`).
  - Supports environment variable overrides for all weights (`PHISHVISION_M01_WEIGHTS` through `PHISHVISION_M07_CHECKPOINT`).
- **Scanner-to-ML Adapter (`app/scanner/adapter.py`):**
  - Bridges Playwright scanner runtime data structures (`ScanResult`, `RedirectHop`, `FormInfo`) with `UnifiedPhishingPipeline`.
  - Chronologically extracts redirect hostnames for M-07 Bi-LSTM sequential encoding.
  - Formats DOM forms with password field and external submission indicators.
  - Gracefully handles blocked SSRF scans (`ScanStatus.BLOCKED_SSRF`), timeouts, and headless failures without uncaught exceptions.
- **FastAPI REST API (`app/main.py`, `app/api/v1/`):**
  - Production-ready implementation of endpoints specified in `API_CONTRACTS.md`:
    - `POST /api/v1/scan`: Fast and dynamic scanning with RFC 7807 problem details for scheme errors and Anti-SSRF blocks.
    - `GET /api/v1/scan/{scan_id}`: Polls scan lifecycle status (`IN_PROGRESS`, `COMPLETED`, `FAILED`, `BLOCKED_SSRF`).
    - `GET /api/v1/scan/{scan_id}/report`: Returns full sanitized explainable reports without internal path exposure.
    - `GET /api/v1/scan/{scan_id}/screenshot`: Streams captured PNG screenshots.
    - `GET /api/v1/brands`: Returns list of 18 protected reference brands and authorized domains.
    - `GET /api/v1/health`: Surfaces system health, device, and transparent checkpoint status for all 7 models.
    - `POST /api/v1/analyze/url`: Fast synchronous lexical intelligence.
    - `POST /api/v1/analyze/brand-similarity`: Fast image upload brand matching.
  - Non-blocking execution using `BackgroundTasks` and `asyncio.to_thread` guarantees API event loop responsiveness.
- **External Checkpoint Deployment (`scripts/download_checkpoints.py`):**
  - Structured CLI tool verifying physical existence, file sizes, and paths for all 9 model weights and auxiliary preprocessors.
  - Automated verification (`--verify`) and remote download (`--download`) prevents committing large binaries to git.
- **Risk Fusion False Positive Audit & Resolution:**
  - Audited and fixed false positive triggers:
    1. `cloudsupport-knowledge.org` (Synthetic 404/empty page): Single detector low-confidence edge hallucinations no longer bypass non-harvesting capping. Correctly classified as `SUSPICIOUS` (risk 0.40, previously falsely PHISHING 0.8582).
    2. `t.me` (Real-world legitimate Telegram preview): Presence of lone input fields/buttons without a password field or login card is no longer treated as credential harvesting. Correctly classified as `LEGITIMATE` (risk 0.30, previously falsely PHISHING 0.8500).
- **Spatial Consensus Engine (`app/ml/vision/spatial_consensus.py`):**
  - Fuses spatial detections across M-04 (YOLOv8n), M-05 (Faster R-CNN), and M-06 (YOLOv8-Seg) via IoU clustering ($\text{IoU} \ge 0.45$).
  - Filters oversized background boxes (> 45% image area).
  - Preserves exact polygon segmentation masks and pixel area from YOLO-Seg.
  - Crops logo candidates with ~10% padding from deduplicated consensus logo clusters for M-03 Siamese verification.
- **Calibrated Multi-Evidence Risk Fusion:**
  - Normalizes raw sigmoid scores for cost-sensitive models (M-01 threshold 0.05, M-07 threshold 0.01) onto an aligned $[0.0, 1.0]$ risk scale.
  - Cross-verifies DOM form submissions and password inputs with visual detections.
  - Preserves critical brand impersonation override ($risk \ge 0.95$), official brand whitelist pass ($risk \le 0.05$), and non-harvesting page risk capping ($\le 0.40$).
  - Preserves `UNKNOWN / INSUFFICIENT EVIDENCE` fallbacks when visual capture is absent and URL confidence is ambiguous.
- **Automated Verification Suite:**
  - [`tests/test_seven_model_integration.py`](tests/test_seven_model_integration.py): **16/16 PASSED (100%)**
  - [`tests/test_scanner_adapter.py`](tests/test_scanner_adapter.py): **7/7 PASSED (100%)**
  - [`tests/test_api.py`](tests/test_api.py): **8/8 PASSED (100%)**
  - Full suite passing: **89/89 tests passing across all active modules**.

- **M-07 Architecture (`app/ml/sequence/lstm_model.py`):**
  - Bidirectional LSTM with self-attentive temporal pooling (`AttentionPooling`).
  - Processes character-level sequences of canonical URLs up to 256 characters.
  - Dual-branch architecture: URL character LSTM branch (hidden_dim=64, num_layers=2, bidirectional -> 128-d output) + conditional redirect sequence LSTM branch (hidden_dim=32, num_layers=1).
  - Total parameters: **210,689 parameters** (~0.84 MB weights), peak VRAM < 350 MiB on RTX 2050 (4GB).
- **Data Integrity & Sequence Audit:**
  - Audit of `data/splits/` confirmed **no historical labelled redirect sequence datasets exist in the corpus**. The dataset contains 44,129 static URL records.
  - Redirect sequence learning is explicitly marked as not yet trained on empirical multi-hop data. The model operates on raw sequential URL patterns with zero-padded fallback for redirect inputs.
  - Reuses domain-isolated splits with verified **zero domain and zero URL leakage**:
    - Train: `data/splits/train/urls.parquet` (32,157 samples: 31,722 Phishing, 435 Legitimate).
    - Validation: `data/splits/validation/urls.parquet` (7,715 samples: 7,638 Phishing, 77 Legitimate).
    - Test: `data/splits/test/urls.parquet` (4,257 samples: 4,175 Phishing, 82 Legitimate).
- **Class Imbalance Strategy:**
  - Implements balanced class-weighted BCE loss ($w_0 = 36.96$ for Legitimate, $w_1 = 0.5068$ for Phishing) to counter the 72.9:1 imbalance without synthesizing fake samples.
  - Calibrates decision threshold strictly on the validation set ($\tau \in [0.01, 0.99]$) to maximize validation Macro F1 before test evaluation.
- **Inference Engine (`app/ml/sequence/lstm_inference.py`):**
  - Production `LSTMInferencer` for single and batched URL predictions, returning calibrated probabilities, labels, and latency benchmarks.
- **Automated Verification Suite (`tests/test_lstm.py`):**
  - **17/17 unit tests passing (100%)** covering preprocessing, leakage isolation, attentive pooling, forward/backward passes, weighted loss, inferencer, and metric calculation.
- **Pre-Evaluation Reports:**
  - Generated `PENDING_TRAINING_EXECUTION` manifest at `data/reports/lstm/metrics.json` and `data/reports/lstm/LSTM_EXPERIMENT_REPORT.md`.

---

## 2. Phase 7.8 Deliverables Matrix

| Deliverable | Specification | Implementation File | Verification & Test Results | Status |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM Model Architecture** | BiLSTM with Attention Pooling & conditional redirect branch | [`app/ml/sequence/lstm_model.py`](app/ml/sequence/lstm_model.py) | Verified forward/backward pass, attention weights, 210,689 parameters. | PASSED |
| **Sequence Preprocessing** | Zero-leakage character vocabulary & sequence dataset | [`app/ml/sequence/sequence_preprocessing.py`](app/ml/sequence/sequence_preprocessing.py) | Verified padding, truncation, UNK handling, JSON serialization. | PASSED |
| **Production Inferencer** | Single/batch sequential prediction engine with latency tracking | [`app/ml/sequence/lstm_inference.py`](app/ml/sequence/lstm_inference.py) | Verified predict and predict_batch with structured telemetry. | PASSED |
| **Training & Evaluation Script** | Config-driven training harness with balanced BCE & threshold tuning | [`app/ml/train_lstm.py`](app/ml/train_lstm.py) | Verified `--eval-only` and pre-eval manifest generation. | PASSED |
| **Configuration** | Hardware-aware hyperparameters (batch=128, FP32/CUDA) | [`configs/lstm.yaml`](configs/lstm.yaml) | Configured for RTX 2050 4GB GPU & CPU fallback. | PASSED |
| **Unit Test Suite** | 17 automated tests for preprocessing, isolation, model, and loss | [`tests/test_lstm.py`](tests/test_lstm.py) | **17 passed in 6.26s** (100% pass rate). | PASSED |
| **Pre-Evaluation Reports** | Status manifest and training instructions | [`data/reports/lstm/`](data/reports/lstm/) | Emitted `PENDING_TRAINING_EXECUTION` manifest and report. | PASSED |
