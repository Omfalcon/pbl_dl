# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 4 — Feedforward Deep Neural Network for Lexical Phishing Detection  
**Phase Status:** COMPLETED  
**Previous Phases:**
- Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
- Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
- Phase 2: Dataset Acquisition & Validation (COMPLETED & COMMITTED in `56d8565`)
- Phase 3: Neural Network Mathematical Foundations (COMPLETED & DOCUMENTED in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md))  
**Target Next Phase:** Phase 5 — Advanced Homoglyph & Typosquatting Analysis / Fast-Path Deployment  
**Timestamp:** 2026-11-22  

---

## 1. Phase 4 Optimizers & Schedulers Deliverables (`app/ml/from_scratch/optimizers.py`)

| Component | Mathematical Formulation | Module | Test / Verification | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. SGD** | $\theta_t = \theta_{t-1} - \eta g_t$ | [`optimizers.py::SGD`](app/ml/from_scratch/optimizers.py) | Hand-calculated step verified; converges on linear regression ($L < 0.05$) and binary classification ($100\%$ acc). Input validation & shape checking hardened. | PASSED |
| **2. SGD with Momentum** | $v_t = \mu v_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta v_t$ | [`optimizers.py::SGDMomentum`](app/ml/from_scratch/optimizers.py) | Hand-calculated 2-step verification; $97.53\%$ loss reduction on benchmark. Hyperparameter bounds validated. | PASSED |
| **3. RMSProp** | $s_t = \beta s_{t-1} + (1-\beta)g_t^2$, $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t+\epsilon}}g_t$ | [`optimizers.py::RMSProp`](app/ml/from_scratch/optimizers.py) | Hand-calculated step verified; $99.80\%$ loss reduction on benchmark. Gradient finiteness enforced. | PASSED |
| **4. Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\hat{m}_t, \hat{v}_t$ bias-corrected | [`optimizers.py::Adam`](app/ml/from_scratch/optimizers.py) | Hand-calculated step verified; $99.77\%$ loss reduction on benchmark. Finiteness & parameter shape matching validated. | PASSED |
| **5. StepLR Scheduler** | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$ | [`optimizers.py::StepLR`](app/ml/from_scratch/optimizers.py) | Verified step decay at step boundaries. Synchronized with optimizer learning rate. | PASSED |
| **6. ExponentialLR Scheduler**| $\eta_t = \eta_0 \cdot \gamma^t$ | [`optimizers.py::ExponentialLR`](app/ml/from_scratch/optimizers.py) | Verified continuous geometric decay. Synchronized with optimizer learning rate. | PASSED |
| **7. CosineAnnealingLR** | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\frac{t}{T_{\max}}\pi))$ | [`optimizers.py::CosineAnnealingLR`](app/ml/from_scratch/optimizers.py) | Verified smooth decay from $\eta_{\max}$ to $\eta_{\min}$. Synchronized with optimizer learning rate. | PASSED |
| **8. Optimizer Benchmark** | Synthetic non-linear ring dataset comparison | [`scripts/experiments/compare_optimizers.py`](scripts/experiments/compare_optimizers.py) | Report exported to [`data/reports/optimizer_benchmark.json`](data/reports/optimizer_benchmark.json). | PASSED |

---

## 2. Phase 4 Lexical Pipeline & MLP Deliverables

| Deliverable | Specification | Implementation File | Verification & Holdout Results | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lexical Feature Extractor** | 40 engineered numerical features (lengths, tokens, counts, Shannon entropy, brand impersonation flags, ratios) with multi-label PSL parsing (`tldextract`) | [`app/intelligence/lexical.py`](app/intelligence/lexical.py) | Verified across multi-label TLDs (`.co.uk`, `.co.in`), IPv4/IPv6, punycode, brand impersonation, and malformed URLs. Fast vectorized extraction (32k URLs in ~1.3s). | PASSED |
| **Standardization Preprocessor** | Zero-leakage standardization with outlier clipping | [`app/intelligence/preprocessing.py`](app/intelligence/preprocessing.py) | Fitted strictly on `X_train`. Zero leakage verified on validation and test holdout splits. Saved to `models/lexical/preprocessor.json`. | PASSED |
| **Imbalance Strategy Comparison** | Controlled evaluation: Strategy A (Balanced Batches + Ordinary BCE) vs Strategy B (Natural Batches + Class-Weighted BCE) | [`scripts/train_lexical_model.py`](scripts/train_lexical_model.py) | No double-compensation. Selected Strategy B strictly on validation metrics (Val Macro F1: 0.6450 vs 0.5065). | PASSED |
| **Lexical MLP (M-01)** | 3-Layer Perceptron: `Dense(40,64) -> ReLU -> Dense(64,32) -> ReLU -> Dense(32,1) -> Sigmoid` | [`app/intelligence/lexical_model.py`](app/intelligence/lexical_model.py) | Trained using custom `Adam` + `CosineAnnealingLR` with best-validation-checkpoint restoration (restored epoch 17, $\tau=0.050$). Weights saved to `models/lexical/model_weights.npz`. | PASSED |
| **Holdout Evaluation** | Unseen domain-isolated test partition (4,257 records across 908 unseen domains) | [`scripts/train_lexical_model.py`](scripts/train_lexical_model.py) | **ROC-AUC: 0.9773**, **PR-AUC: 0.9995**, **Macro F1: 0.8239**, Phishing F1: 0.9932, Legit F1: 0.6545, Accuracy: 98.66%. Full report in `data/reports/lexical_mlp_evaluation.json`. | PASSED |

---

## 3. Phase 4 Exit Criteria & Quality Rules

- [x] **No PyTorch Optimizers:** All 4 optimizers, 3 schedulers, and the MLP training pipeline built from scratch using NumPy.
- [x] **Hardened Optimizer Safety:** Input length, tensor shape, parameter dtype/writability, gradient finiteness, and synchronized scheduler learning rates validated.
- [x] **No Imbalance Double-Compensation:** Evaluated Strategy A and Strategy B without combining balanced sampling with weighted BCE.
- [x] **Strict Zero Leakage:** Preprocessor fitted on training split only; strategy selection and threshold calibration on validation split only; final metrics measured on holdout test partition.
- [x] **Domain Cluster & Discrepancy Audited:** Explained validation skew (65.9% Google Docs/Forms PhishTank records) vs test split distribution.
- [x] **Robust Metrics Reported:** Full confusion matrix, per-class metrics, Macro Precision/Recall/F1, ROC-AUC, and PR-AUC reported (not just accuracy).
- [x] **Comprehensive Testing:** 191 unit tests passing in 4.54s (1 deselected).
- [x] **No Scope Bleed:** ResNet, YOLO, Siamese CNN, and Transformer models remain isolated for later phases.
