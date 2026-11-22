# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 6.1 — Visual Core: ResNet-18 Screenshot Classifier Pipeline Implementation  
**Phase Status:** IMPLEMENTATION & TESTING COMPLETE (TRAINING PENDING USER EXECUTION)  
**Previous Phases:**
- Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
- Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
- Phase 2: Dataset Acquisition, Brand Gallery & Partitioning (COMPLETED & COMMITTED in `56d8565`)
- Phase 3: Neural Network Mathematical Sandbox from Scratch (COMPLETED & DOCUMENTED in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md))  
- Phase 4: Optimizers, Ablations & Learning Algorithms from Scratch (COMPLETED & BENCHMARKED in `data/reports/optimizer_benchmark.json`)
- Phase 5: URL Intelligence, Homoglyph Detection, Typosquatting & MLP Baseline (COMPLETED & EVALUATED in `data/reports/lexical_mlp_evaluation.json`)
**Target Next Milestone:** Manual Execution of M-02 ResNet-18 Training & Phase 6.2 (Siamese Brand Matcher upon brand asset availability)  
**Timestamp:** 2026-11-22  

---

## 1. Phase 6.1 Deliverables Matrix (M-02 ResNet-18 Screenshot Classifier)

| Deliverable | Specification | Implementation File | Verification & Test Results | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | ResNet-18 backbone with custom 2-class head: `AdaptiveAvgPool2d(1) -> Flatten -> Linear(512,128) -> ReLU -> Dropout(0.3) -> Linear(128,2)` | [`app/ml/vision/models.py`](app/ml/vision/models.py) | Tested synthetic forward/backward pass, tensor output shapes $(B, 2)$, class mapping `0: legitimate, 1: phishing`. | PASSED |
| **Preprocessing & Transforms** | Configurable viewport cropping (`TopViewportCrop`, `AspectRatioPadResize`), ImageNet standardization, deterministic eval transforms | [`app/ml/vision/preprocessing.py`](app/ml/vision/preprocessing.py) | Tested top viewport 1:1 fold crop, aspect-ratio padding to 224x224, RGB conversion, and missing file exception handling. | PASSED |
| **Dataset Loader** | `ScreenshotDataset` loading from parquet splits or DataFrame, dynamic class weight calculation $w_c = N / (2 \cdot N_c)$ | [`app/ml/vision/dataset.py`](app/ml/vision/dataset.py) | Verified on synthetic DataFrames and disk splits (`data/splits/visual/`: 624 train, 262 val, 114 test). | PASSED |
| **Evaluation Metrics** | Accuracy, per-class Precision/Recall/F1, Macro/Weighted F1, Confusion Matrix, ROC-AUC, PR-AUC, single-class edge cases | [`app/ml/vision/evaluation.py`](app/ml/vision/evaluation.py) | Verified with perfect predictions, random predictions, and single-class edge case handling without exceptions. | PASSED |
| **Training Pipeline** | `ResNet18Trainer` with AdamW, Cosine Annealing/StepLR, dynamic class-weighted CrossEntropyLoss, early stopping, best checkpoint restoration | [`app/ml/vision/trainer.py`](app/ml/vision/trainer.py) | Validated training mechanics with mock/unit tests. **CRITICAL RESTRICTION HONORED: Zero training epochs executed.** | PASSED (Ready to Train) |
| **Inference Engine** | `ResNet18Inference` production engine restoring checkpoint weights and preprocessing config, predicting single images or batches | [`app/ml/vision/inference.py`](app/ml/vision/inference.py) | Verified checkpoint save/restore roundtrip, predict single, predict batch, and calibrated probabilities. | PASSED |
| **Training CLI & Config** | Reproducible YAML config and CLI runner with command-line argument overrides | [`configs/resnet18.yaml`](configs/resnet18.yaml), [`app/ml/train_resnet18.py`](app/ml/train_resnet18.py) | Config syntax validated against Pydantic-like dataclass schema, CLI `--help` and argument parser verified. | PASSED |
| **Unit Test Suite** | 15 targeted unit tests covering architecture, preprocessing, dataset loader, metrics, checkpointing, and inference | [`tests/test_resnet18.py`](tests/test_resnet18.py) | **15 passed in 9.90s**. Full suite: **225 passed in 13.19s** (0 regressions). | PASSED |

---

## 2. Visual Dataset Verified Partitions (`data/splits/visual/`)

| Partition | Total Samples | Phishing (`1`) | Legitimate (`0`) | Unique Domains | Class Imbalance Ratio | Split Integrity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (`train.parquet`) | 624 | 477 (76.4%) | 147 (23.6%) | 362 | 3.24 : 1 | 0 domain overlap with Val/Test |
| **Validation** (`val.parquet`) | 262 | 236 (90.1%) | 26 (9.9%) | 76 | 9.08 : 1 | 0 domain overlap with Train/Test |
| **Test** (`test.parquet`) | 114 | 87 (76.3%) | 27 (23.7%) | 79 | 3.22 : 1 | 0 domain overlap with Train/Val |
| **Total Corpus** | 1,000 | 800 (80.0%) | 200 (20.0%) | 517 | 4.00 : 1 | 0 SHA256 or pHash duplicates |

---

## 3. Execution Mandate & Manual Training Command

In strict compliance with user instructions:
- **NO training was executed.**
- **NO training epochs or GPU workloads were started.**
- **NO synthetic or fabricated metrics were generated.**

### Exact Manual Training Command:
To train the M-02 ResNet-18 classifier using the pre-configured environment, run:

```powershell
& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -m app.ml.train_resnet18 --config configs/resnet18.yaml
```

To run with CLI overrides (e.g., using top-viewport crop, batch size 16, epochs 20):
```powershell
& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -m app.ml.train_resnet18 --config configs/resnet18.yaml --epochs 20 --batch-size 16 --crop-strategy top_viewport
```

---

## 4. Phase 6 Status & Blockers (M-03 Siamese Brand Matcher)

- **M-02 (ResNet-18 Screenshot Classifier):** Complete, modular, fully tested, and waiting for manual execution.
- **M-03 (Siamese Brand Visual Matcher):** BLOCKED pending brand reference image acquisition. Disk inspection revealed 0 image files in `data/brands/reference_gallery/` and no pair manifests. Scaffolding is deferred until assets are populated to avoid non-reproducible dummy training.
