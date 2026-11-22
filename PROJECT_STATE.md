# PROJECT_STATE.md — Persistent Project State

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Current Phase:** Phase 8 — Seven-Model Integration: Implementation  
**Phase Status:** INTEGRATION COMPLETE & EMPIRICALLY VERIFIED (74/74 TESTS PASSING)  
**Next Milestone:** Phase 9 — Interactive Dashboard / Forensic UI & Production API Deployment  
**Last Updated:** 2026-11-23  

---

## 1. Executive Summary

PhishVision AI is an AI-powered phishing website detection and brand impersonation platform combining lexical URL intelligence, deep computer vision, behavioral and sequence analysis, and calibrated multi-evidence risk fusion.

- **Phase 0 (System Architecture & Requirements):** COMPLETED.
- **Phase 1 & Phase 1.1 (Repository Foundation, Scanner Implementation & Security Hardening):** COMPLETED.
- **Phase 2 (Dataset Acquisition, Preparation, Annotation & Validation):** COMPLETED (COMMITTED in `56d8565`).
  - Ingested multi-source raw feeds: PhishTank, Tranco, PhreshPhish, OpenPhish, Zenodo brands, and External Visual Dataset (1,008 screenshots).
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
- **Phase 4 (Optimizers, Ablations & Learning Algorithms from Scratch):** COMPLETED.
  - Custom optimizers from scratch: SGD, SGD with Momentum, RMSProp, and Adam (`app/ml/from_scratch/optimizers.py`).
  - Learning rate schedulers: StepLR, ExponentialLR, and CosineAnnealingLR.
  - Hardened with input length, tensor shape, parameter dtype/writability, gradient finiteness checks, state consistency, and scheduler rate synchronization.
  - Non-linear benchmark report exported to `data/reports/optimizer_benchmark.json`.
- **Phase 5 (URL Intelligence, Homoglyph Detection, Typosquatting & MLP Baseline):** COMPLETED.
  - Public Suffix List (PSL) domain extraction engine (`app/intelligence/url_domain/psl_extractor.py`) supporting multi-label ccTLDs (`sbi.co.in`, `paypal.co.uk`), IPv4/IPv6, subdomains, and Punycode.
  - Unicode homoglyph, mixed-script, and IDN confusable detection (`app/intelligence/url_domain/homoglyph.py`) with zero false positives on pure-script IDNs (`президент.рф`).
  - Typosquatting, combosquatting, and subdomain impersonation detector (`app/intelligence/url_domain/typosquatting.py`) computing Levenshtein, Damerau-Levenshtein distance, and similarity scores against 18 protected brands.
  - Strict preservation of official brand domains (`sbi.co.in`, `paypal.co.uk`, `google.com`).
  - Unified `DomainIntelligenceAnalyzer` producing structured, explainable evidence schemas.
  - Reusable 40-feature lexical URL extractor (`app/intelligence/lexical.py`) covering lengths, counts, tokens, Shannon entropy, and brand impersonation cues.
  - Zero-leakage standardization preprocessor fitted strictly on training data (`app/intelligence/preprocessing.py`, saved to `models/lexical/preprocessor.json`).
  - Controlled class imbalance strategy comparison: Strategy A (Balanced Batches + Ordinary BCE, Val Macro F1: 0.5065) vs. Strategy B (Natural Batches + Class-Weighted BCE, Val Macro F1: 0.6450). Selected Strategy B strictly on validation metrics without double-compensation.
  - Validation/test discrepancy resolved via domain-cluster analysis (accounting for 65.9% Google Docs/Forms concentration in validation) and best-validation-checkpoint restoration at epoch 17 (calibrated $\tau = 0.050$).
  - 3-Layer Lexical MLP (M-01) trained from scratch with Adam and CosineAnnealingLR (`models/lexical/model_weights.npz`).
  - Holdout test metrics on 4,257 unseen records across 908 unseen domains: **ROC-AUC: 0.9773, PR-AUC: 0.9995, Macro F1: 0.8239, Phishing F1: 0.9932, Legit F1: 0.6545, Accuracy: 98.66%**. Full report exported to `data/reports/lexical_mlp_evaluation.json`.
  - **Full test suite: 210 unit tests passing (1 deselected) in ~4.76s**. Zero regressions.

- **Phase 6.1 (Visual Core 1: ResNet-18 Screenshot Classifier):** IMPLEMENTED & TRAINED.
  - Implemented `ResNet18ScreenshotClassifier`, configurable viewport cropping (`TopViewportCrop`, `AspectRatioPadResize`), and `ScreenshotDataset`.
  - Trained M-02 on visual splits (`train.parquet`: 624, `val.parquet`: 262, `test.parquet`: 114) reaching best validation Macro F1 = 0.7037 at epoch 9 (`models/checkpoints/resnet18/best_model.pt`).
  - Diagnostic error analysis conducted: evaluated 39 false negatives and 4 false positives with visual contact sheets and multi-region viewport cropping experiments across all 262 validation screenshots.
- **Phase 6.2, 6.2.1 & 6.2.2 (Visual Core 2: Siamese CNN Brand Matcher & Evaluation):** TRAINED & EVALUATED ON TEST SET.
  - Shared-weight Siamese CNN architecture (`SiameseResNet18`) with 256-d L2-normalized embeddings, cosine similarity, ContrastiveLoss, CosineContrastiveLoss, and TripletLoss ($\alpha=0.3$).
  - Curated brand reference gallery: **33 authentic assets acquired across all 18 target brands** with cryptographic SHA-256 provenance tracking (`brand_assets_registry.json`).
  - Generated **116 authentic Siamese pairs** (46 positive, 70 negative) in `data/brands/manifest_siamese_pairs.csv` and partitioned disjointly into Train (73 pairs across 12 brands), Val (23 pairs across 3 brands), and Test (20 pairs across 3 brands) with verified zero-leakage (`app/ml/vision/brand_pairs.py`).
  - Production inference interface (`SiameseBrandMatcher`) with graceful `INSUFFICIENT_REFERENCE_DATA` handling when assets are absent and domain authorization verification against official domain lists.
  - User trained model; best validation checkpoint saved at Epoch 1 (`models/checkpoints/siamese/best_siamese.pt`) with Val F1: 0.8000, Val Loss: 0.0553.
  - Independent test evaluation performed on 20 held-out pairs (`data/splits/siamese/test.parquet`) across brands `amazon`, `netflix`, and `google` using $\tau = 0.82$: Accuracy = 85.00%, Precision = 1.0000 (0 false positives), Recall = 0.6250, F1 = 0.7692, ROC-AUC = 0.8750, PR-AUC = 0.8629. Exported to [`data/reports/siamese/TEST_EVALUATION.md`](data/reports/siamese/TEST_EVALUATION.md).

- **Phase 7.1 (Visual Core 3: M-04 YOLO Object Detection Pipeline & Audit):** COMPLETED.
  - Standardized 6-class detection taxonomy: `brand_logo`, `login_form`, `username_field`, `password_field`, `submit_button`, `payment_element`.
  - Audited visual dataset: 1,008 total assets (1,000 in `visual_corpus.parquet` + 8 scanner screenshots); verified 100% physical existence on disk (1,000/1,000), 0 domain overlap, and 0 sample overlap across splits (Train: 624, Val: 262, Test: 114).
  - Built canonical annotation manifest `data/annotations/detection/annotation_manifest.csv` and `annotation_manifest.json` tracking all 1,000 screenshots with strict initial `UNANNOTATED` status (0 fake bounding boxes, 0 synthetic labels).
  - Implemented robust `YOLOAnnotationValidator` with empty-image handling (0-byte text files), boundary overshoots, positive dimensions, duplicate box warnings, and image pairing checks (`app/ml/vision/yolo_annotations.py`).
  - Authored comprehensive human annotation guidelines [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md).
  - Hardware audit: NVIDIA RTX 2050 (4GB VRAM), CUDA 12.4, PyTorch 2.6.0, Ultralytics 8.4.60. Recommended YOLOv8n / YOLO11n @ 640x640 with FP16.
  - Full test suite: 26 unit tests passing (100% pass rate).

- **Phase 7.2–7.4 (Synthetic Dataset & YOLOv8n Training / Evaluation Pipeline):** MODEL TRAINED & TEST EVALUATED.
  - Implemented `SyntheticWebpageGenerator` (`app/ml/vision/synthetic_generator.py`) producing varied templates (standard login, banking, social, ecommerce, payment checkout, and empty 404/blog pages) with exact bounding boxes.
  - Generated 800 synthetic screenshots across Train (560), Val (120), and Test (120) with **zero brand leakage** (16 train brands, 4 val brands, 4 test brands). 3,779 exact bounding boxes; 62 empty-label negative samples.
  - User executed training for 30 epochs on RTX 2050 (4 GB VRAM) in ~7.7 mins (`models/checkpoints/yolov8n_synthetic/run1/weights/best.pt`).
  - Evaluated on untouched 120 test screenshots: **mAP@50 = 0.9950, mAP@50-95 = 0.9699, Precision = 0.9973, Recall = 1.0000, Latency = 7.2 ms**.
  - Generated full visual diagnostics in `data/reports/yolo/`: loss curves (`results.png`), PR curves (`BoxPR_curve.png`), confusion matrices (`confusion_matrix.png`), `class_metrics.csv`, `metrics.json`, and `YOLO_EXPERIMENT_REPORT.md`.
  - 34 unit tests passing across `tests/test_yolo_synthetic.py`, `tests/test_yolo_annotations.py`, and `tests/test_annotations.py`.
- **Phase 7.6 (M-05 Faster R-CNN Object Detector):** COMPLETED.
  - Faster R-CNN with ResNet-50-FPN backbone and customized 7-class head. Hardware-aware RTX 2050 config (`models/checkpoints/faster_rcnn_synthetic/best_faster_rcnn.pt`).
- **Phase 7.7 (M-06 YOLOv8-Seg Instance Segmenter):** COMPLETED.
  - Programmatic geometric polygon dataset (`data/synthetic_yolo_seg/`). Trained YOLOv8n-seg (`models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt`) achieving Box mAP50 = 0.9950, Mask mAP50 = 0.9950.
- **Phase 7.8 (M-07 URL Sequence Bi-LSTM):** COMPLETED.
  - Bidirectional LSTM with AttentionPooling (`models/checkpoints/lstm/best_lstm.pt`) achieving test Accuracy = 0.9869, ROC-AUC = 0.9666, PR-AUC = 0.9991, Macro F1 = 0.7779.
- **Phase 8 (Seven-Model Multimodal Integration & Spatial Consensus):** COMPLETED & VERIFIED.
  - Unified pipeline (`app/ml/unified_pipeline.py`) integrating all 7 models.
  - Non-duplicative spatial consensus engine (`app/ml/vision/spatial_consensus.py`) merging M-04, M-05, and M-06 via IoU clustering ($\text{IoU} \ge 0.45$).
  - Calibrated risk normalization for cost-sensitive models (M-01 $\tau=0.05$, M-07 $\tau=0.01$).
  - Preserved UNKNOWN fallback, anti-SSRF protections, brand whitelist, and critical brand impersonation override.
  - Automated verification: **74/74 unit and regression tests passing (100%)**.

---

## 2. Current State Snapshot

- **Active Phase:** `Phase 8` (Seven-Model Multimodal Integration Completed)
- **Target Next Step:** Phase 9 (Interactive Dashboard / Forensic UI & Production API Deployment)
- **Working Modules:**
  - `app.ml.unified_pipeline`: Unified production engine integrating all 7 neural pipelines (M-01 through M-07) with calibrated multi-evidence risk fusion.
  - `app.ml.vision.spatial_consensus`: Spatial clustering and deduplication engine merging M-04, M-05, and M-06 bounding boxes and instance segmentation masks.
  - `app.ml.vision.multimodal_pipeline`: Master multimodal phishing detection pipeline with 100% backward compatibility for Phase 7.5 API contracts.
  - `app.ml.vision.siamese_inference`: Production Siamese matcher evaluating cosine similarity against 18-brand reference gallery and domain authorization.
  - `app.intelligence.url_domain.psl_extractor`: Multi-label PSL domain extractor using tldextract.
  - `app.intelligence.url_domain.homoglyph`: Unicode confusable and mixed-script homoglyph detector.
  - `app.intelligence.url_domain.typosquatting`: Levenshtein/Damerau-Levenshtein distance and brand impersonation detector.
  - `app.intelligence.url_domain.analyzer`: Unified domain intelligence evidence orchestrator.
  - `app.scanner.anti_ssrf`: Hardened URL parsing, port validation, DNS resolution, and IP classification.
  - `app.scanner.proxy`: Validating local egress forward proxy with connection-level IP pinning and TLS SNI preservation.
  - `app.scanner.playwright_scanner`: Isolated Playwright Chromium scanner with context-level routing, popup containment, service worker blocking, and secure TLS checking.
  - `app.scanner.models`: Pydantic structured scan telemetry schemas.
  - `app.intelligence.lexical`: 40-feature lexical URL extractor with Shannon entropy and brand impersonation detection.
  - `app.intelligence.preprocessing`: Zero-leakage feature standardizer with outlier clipping and serialization.
  - `app.intelligence.lexical_model`: M-01 Lexical MLP classifier with model weights serialization and metrics evaluation.
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
  - `app.ml.from_scratch.losses`: Loss functions and analytic gradients (MSE, class-weighted Binary Cross Entropy).
  - `app.ml.from_scratch.neuron`: Single neuron forward and backward pass.
  - `app.ml.from_scratch.layers`: Dense layer with weight/bias initialization, forward and backward propagation.
  - `app.ml.from_scratch.computational_graph`: Reverse-mode automatic differentiation tape and DAG topological execution.
  - `app.ml.from_scratch.gradient_check`: Finite-differences numerical gradient validation.
  - `app.ml.from_scratch.optimizers`: SGD, SGDMomentum, RMSProp, Adam, StepLR, ExponentialLR, CosineAnnealingLR.
  - `app.ml.vision.models`: ResNet-18 screenshot classification architecture with custom 2-class head (`Linear(512,128) -> ReLU -> Dropout(0.3) -> Linear(128,2)`).
  - `app.ml.vision.preprocessing`: Viewport cropping (`TopViewportCrop`, `AspectRatioPadResize`), deterministic transforms, and safe PIL loading.
  - `app.ml.vision.dataset`: Parquet/DataFrame screenshot dataset loader with dynamic class-weight computation.
  - `app.ml.vision.evaluation`: Multiclass and binary metrics (Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix).
  - `app.ml.vision.trainer`: Modular ResNet-18 training engine with early stopping, dynamic class weighting, and best-checkpoint restoration.
  - `app.ml.vision.inference`: Production inference engine with checkpoint restoration and calibrated probabilities.
  - `app.ml.train_resnet18`: Reproducible CLI entry point supporting YAML config and argument overrides.
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

### Phase 4: Optimizers, Ablations & Learning Algorithms from Scratch (COMPLETED)
1. Custom optimizers from scratch: SGD, SGD with Momentum, RMSProp, Adam (`app/ml/from_scratch/optimizers.py`).
2. Custom learning rate schedulers: StepLR, ExponentialLR, CosineAnnealingLR.
3. Hardened parameter validation, shape checking, floating-point dtype/writability, non-finite gradient rejection, and scheduler synchronization.
4. Empirical convergence benchmarked on non-linear ring dataset in `data/reports/optimizer_benchmark.json`.

### Phase 5: URL Intelligence, Homoglyph Detection, Typosquatting & MLP Baseline (COMPLETED)
1. **Public Suffix List (PSL) Domain Extractor (`app/intelligence/url_domain/psl_extractor.py`):** Multi-label ccTLD parsing (`sbi.co.in`, `paypal.co.uk`), subdomain isolation (`sbi.co.in.attacker.com`), IPv4/IPv6 literals, ports, and Punycode.
2. **Unicode Homoglyph & Mixed-Script Detector (`app/intelligence/url_domain/homoglyph.py`):** Cross-script visual confusable detection (Cyrillic/Greek lookalikes), IDN normalization, NFKD diacritic separation, and zero false positives on pure-script IDNs (`президент.рф`, `café.com`, `münchen.de`).
3. **Typosquatting & Brand Impersonation Detector (`app/intelligence/url_domain/typosquatting.py`):** Pure-Python Levenshtein and Damerau-Levenshtein edit distances, normalized similarity scores, brand protection, and attack classification (substitution, transposition, insertion, omission, combosquatting, subdomain spoofing).
4. **Unified Domain Intelligence Analyzer (`app/intelligence/url_domain/analyzer.py`):** Structured evidence schema (`DomainIntelligenceEvidence`) emitting domain metrics, confusables, closest brand, and human-readable explanations.
5. **Lexical URL Feature Extractor (`app/intelligence/lexical.py`):** 40 engineered numerical features with zero-leakage preprocessor.
6. **Controlled Imbalance Training & MLP Baseline (`models/lexical/model_weights.npz`):** Evaluated Strategy A vs Strategy B; selected Strategy B strictly on validation metrics. Calibrated $\tau = 0.050$. Unseen test holdout metrics: **ROC-AUC: 0.9773, PR-AUC: 0.9995, Macro F1: 0.8239, Phishing F1: 0.9932, Accuracy: 98.66%**.

### Phase 6.1: Visual Core 1 — ResNet-18 Screenshot Classifier Pipeline (IMPLEMENTATION & TESTING COMPLETE)
1. **Modular Architecture (`app/ml/vision/models.py`):** ResNet-18 backbone with custom 2-class classification head (`AdaptiveAvgPool2d(1) -> Flatten -> Linear(512,128) -> ReLU -> Dropout(0.3) -> Linear(128,2)`). Explicit class mapping `0: legitimate, 1: phishing`.
2. **Configurable Preprocessing (`app/ml/vision/preprocessing.py`):** `TopViewportCrop` focusing on fold 1:1 aspect ratio where authentication/brand UI elements concentrate; `AspectRatioPadResize` preventing layout distortion; deterministic validation/test standardization using ImageNet stats.
3. **Dataset Loader (`app/ml/vision/dataset.py`):** `ScreenshotDataset` supporting parquet and DataFrame inputs with dynamic inverse-frequency class weights $w_c = N / (2 \cdot N_c)$. Verified on visual splits (`train.parquet`: 624, `val.parquet`: 262, `test.parquet`: 114) with 0 domain overlap.
4. **Evaluation Engine (`app/ml/vision/evaluation.py`):** Full vision metrics suite including Accuracy, per-class Prec/Rec/F1, Macro/Weighted F1, Confusion Matrix, ROC-AUC, and PR-AUC with single-class edge case safeguards.
5. **Training Pipeline (`app/ml/vision/trainer.py`):** `ResNet18Trainer` supporting AdamW, Cosine Annealing/StepLR, dynamic class-weighted CrossEntropyLoss, early stopping, and best-validation-checkpoint restoration.
6. **Production Inference (`app/ml/vision/inference.py`):** Single-image and batched prediction engine restoring weights and preprocessing configurations to yield calibrated probabilities.
7. **CLI & Config (`configs/resnet18.yaml`, `app/ml/train_resnet18.py`):** Declarative configuration with comprehensive CLI override flags.
8. **Execution Mandate Honored:** In strict compliance with instructions, zero training epochs were run. Training is staged for manual execution via:
   `& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -m app.ml.train_resnet18 --config configs/resnet18.yaml`
9. **M-03 Blocker Documented:** Siamese brand matcher is held pending acquisition of canonical brand logo assets for `data/brands/reference_gallery/`.

---

## 4. Pending Work (Master Plan)

- **Phase 6:** Visual Core 1 & 2: User manual execution of M-02 ResNet-18 training, followed by M-03 Siamese brand matcher once reference assets are acquired.
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

- **Date:** 2026-11-23
- **Command:** `pytest tests/test_lstm.py -v`
- **Result:** `17 passed in 6.26s` (Exit Code 0). All M-07 URL Sequence LSTM components, character and token vocabulary encoding, padding, truncation, UNK handling, zero domain and zero URL leakage verification, self-attentive temporal pooling, forward and backward pass gradients, balanced class-weighted BCE loss, inferencer predictions, threshold tuning, and comprehensive metric evaluation verified. Regression test suite (`tests/test_lstm.py`, `tests/test_yolo_seg.py`, `tests/test_faster_rcnn.py`) also passed (45 passed in 13.06s). Zero regressions.
