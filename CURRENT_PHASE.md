# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 4 — Feedforward Deep Neural Network for Lexical Phishing Detection  
**Phase Status:** IN PROGRESS (Optimizers from Scratch & Convergence Benchmarking Completed)  
**Previous Phases:**
- Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
- Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
- Phase 2: Dataset Acquisition & Validation (COMPLETED & COMMITTED in `56d8565`)
- Phase 3: Neural Network Mathematical Foundations (COMPLETED & DOCUMENTED in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md))  
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

## 2. Phase 4 Remaining Deliverables

- [ ] **Lexical URL Feature Extractor:** Extract 30+ engineered domain, path, query, entropy, and character-level statistical features.
- [ ] **Standardization & Preprocessing:** Fit robust scaler / min-max normalization on domain-isolated training splits with zero validation/test leakage.
- [ ] **Feedforward MLP Classifier:** Multi-layer perceptron architecture trained using custom from-scratch optimizers and schedulers.
- [ ] **Evaluation Metrics:** Empirical classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) on domain-aware holdout test splits.

---

## 3. Phase 4 Exit Criteria & Quality Rules

- [x] **No PyTorch Optimizers:** All 4 optimizers and 3 schedulers built from scratch using NumPy.
- [x] **Hardened Optimizer Safety:** Input length validation, tensor shape validation, gradient finiteness checks, and synchronized scheduler learning rates.
- [x] **Hand-Calculated Precision:** Step updates verified against deterministic analytical proofs.
- [x] **Empirical Convergence:** All optimizers achieve $\ge 95\%$ accuracy on synthetic classification.
- [x] **Comprehensive Testing:** 20 optimizer unit tests passing; 167 total project unit tests passing in 6.39s (1 deselected).
- [x] **No Scope Bleed:** ResNet, YOLO, Siamese CNN, and Transformer models remain isolated for later phases.
