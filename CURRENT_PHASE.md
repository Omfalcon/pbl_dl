# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 5 — URL Intelligence, Homoglyph Detection, Typosquatting & MLP Baseline  
**Phase Status:** COMPLETED  
**Previous Phases:**
- Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
- Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
- Phase 2: Dataset Acquisition, Brand Gallery & Partitioning (COMPLETED & COMMITTED in `56d8565`)
- Phase 3: Neural Network Mathematical Sandbox from Scratch (COMPLETED & DOCUMENTED in [`docs/PHASE_3_MATH_FOUNDATIONS.md`](docs/PHASE_3_MATH_FOUNDATIONS.md))  
- Phase 4: Optimizers, Ablations & Learning Algorithms from Scratch (COMPLETED & BENCHMARKED in `data/reports/optimizer_benchmark.json`)
**Target Next Phase:** Phase 6 — Visual Core 1 & 2: ResNet Screenshot Classifier & Siamese Brand Matcher  
**Timestamp:** 2026-11-22  

---

## 1. Phase 5 Deliverables Matrix

| Deliverable | Specification | Implementation File | Verification & Test Results | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PSL Domain Extractor** | Multi-label Public Suffix List parsing via `tldextract` with offline cache (`data/cache/tldextract`) | [`app/intelligence/url_domain/psl_extractor.py`](app/intelligence/url_domain/psl_extractor.py) | Verified for multi-part ccTLDs (`sbi.co.in`, `paypal.co.uk`), IPv4/IPv6 address literals, ports, subdomains (`sbi.co.in.attacker.com`), and IDN/Punycode. | PASSED |
| **Unicode Homoglyph Detector** | Cross-script visual spoofing, IDN homographs, and confusable character detection | [`app/intelligence/url_domain/homoglyph.py`](app/intelligence/url_domain/homoglyph.py) | Verified on Cyrillic/Greek lookalikes (`pаypal.com`), mixed scripts, Punycode (`xn--googl-3we.com`), and legitimate single-script IDNs (`президент.рф`). | PASSED |
| **Typosquatting & Brand Impersonation** | Levenshtein & Damerau-Levenshtein distances, normalized similarity, and brand comparison | [`app/intelligence/url_domain/typosquatting.py`](app/intelligence/url_domain/typosquatting.py) | Tested on official brand domains (`sbi.co.in`, `paypal.co.uk`), substitution (`sb1.co.in`), combosquatting (`sbi-secure-login-example.com`), and subdomain spoofing. | PASSED |
| **Domain Intelligence Analyzer** | Unified engine producing structured evidence dataclass for multi-evidence risk fusion | [`app/intelligence/url_domain/analyzer.py`](app/intelligence/url_domain/analyzer.py) | Emits original/normalized hostnames, registrable domain, detected scripts, confusables, closest brand, and explainable summary. | PASSED |
| **Lexical Feature Extractor** | 40 engineered numerical features (lengths, tokens, counts, Shannon entropy, brand cues) | [`app/intelligence/lexical.py`](app/intelligence/lexical.py) | Vectorized extraction over 32,157 URLs in ~1.3s; integrated with `PSLExtractor` and `DomainIntelligenceAnalyzer`. | PASSED |
| **Standardization Preprocessor** | Zero-leakage standardization with outlier clipping | [`app/intelligence/preprocessing.py`](app/intelligence/preprocessing.py) | Fitted strictly on `X_train`. Zero leakage verified across validation and test partitions. Saved to `models/lexical/preprocessor.json`. | PASSED |
| **Imbalance Strategy Comparison** | Strategy A (Balanced Batches + Ordinary BCE) vs Strategy B (Natural Batches + Weighted BCE) | [`scripts/train_lexical_model.py`](scripts/train_lexical_model.py) | Zero double-compensation. Selected Strategy B strictly on validation metrics (Val Macro F1: 0.6450 vs 0.5065). | PASSED |
| **Lexical MLP (M-01)** | 3-Layer Perceptron: `Dense(40,64) -> ReLU -> Dense(64,32) -> ReLU -> Dense(32,1) -> Sigmoid` | [`app/intelligence/lexical_model.py`](app/intelligence/lexical_model.py) | Trained from scratch with custom Adam + Cosine Annealing LR and best-validation-checkpoint restoration (restored epoch 17, $\tau=0.050$). | PASSED |
| **Holdout Evaluation** | Unseen domain-isolated test partition (4,257 records across 908 unseen domains) | [`scripts/train_lexical_model.py`](scripts/train_lexical_model.py) | **ROC-AUC: 0.9773**, **PR-AUC: 0.9995**, **Macro F1: 0.8239**, Phishing F1: 0.9932, Legit F1: 0.6545, Accuracy: 98.66%. Full report in `data/reports/lexical_mlp_evaluation.json`. | PASSED |

---

## 2. Phase 5 Verification Test Suite (`tests/test_domain_intelligence.py`)

All core test scenarios specified in `PHASE_ROADMAP.md` and phase requirements pass:
- `sbi.co.in`: Extracted registrable domain `sbi.co.in`, verified as official brand domain (`is_official_brand_domain=True`, `is_suspicious_typosquat=False`).
- `sb1.co.in`: Flagged as suspicious substitution typosquatting targeting `sbi` (Damerau-Levenshtein distance 1, `is_suspicious_typosquat=True`).
- `sbi-secure-login-example.com`: Flagged as combosquatting embedding `sbi` with security keywords (`attack_type="combosquatting"`).
- `sbi.co.in.attacker.com`: Extracted registrable domain `attacker.com` with `sbi.co.in` in subdomain; flagged as subdomain impersonation (`attack_type="subdomain_impersonation"`).
- `paypal.co.uk`: Extracted registrable domain `paypal.co.uk`, verified as official PayPal brand domain (`is_official_brand_domain=True`).
- `Cyrillic/Latin homoglyphs`: Detected confusable substitutions in `pаypal.com` and `xn--googl-3we.com` (`\u0435` Cyrillic e).
- `Mixed-script domains`: Detected labels combining Latin with non-Latin scripts (`has_mixed_scripts=True`).
- `Character transposition`: Detected swapped adjacent letters in `paypla.com` and `googel.com` (Damerau distance 1, Levenshtein distance 2).
- `Character insertion & deletion`: Detected `paypaal.com` and `goggle.com` (insertion) and `paypl.com` and `gogle.com` (omission).
- `Legitimate IDNs`: Pure Cyrillic domains (`президент.рф`) correctly classified without false positive mixed-script flags.

---

## 3. Phase 5 Exit Criteria & Quality Rules

- [x] **No PyTorch Optimizers in Scratch Pipeline:** All 4 optimizers, 3 schedulers, and the MLP training pipeline built from scratch using NumPy.
- [x] **Hardened Domain Intelligence:** Public Suffix List parsing, Unicode script classification, confusable mapping, and Damerau-Levenshtein distance implemented with pure Python reproducibility.
- [x] **Official Brand Protection:** Official domains for protected brands are strictly recognized and never falsely flagged as typosquatting attacks.
- [x] **Explainable Output:** Domain intelligence engine returns structured evidence with script breakdown, confusables, closest brand, distance, and human-readable explanation (ready for Phase 10 Risk Fusion).
- [x] **Zero Data Leakage:** Preprocessor fitted on training split only; threshold calibrated on validation split only; final metrics measured on holdout test partition.
- [x] **Comprehensive Testing:** **209 unit tests passing in 4.76s** (1 deselected). Zero regressions.
- [x] **No Scope Bleed:** ResNet, YOLO, Siamese CNN, and Transformer models remain isolated for later phases.
