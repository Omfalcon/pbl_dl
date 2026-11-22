# PhishVision AI: Phishing Website and Brand Impersonation Detector

**PhishVision AI** is an advanced multimodal cybersecurity platform designed to detect zero-day phishing websites and visual brand impersonation attacks. By integrating lexical URL and domain intelligence, dynamic behavioral sandboxing, and deep computer vision (layout classification, logo/form detection, visual Siamese matching, and region segmentation), PhishVision AI provides explainable, probabilistic risk assessments without relying on static global whitelists.

---

## 📌 Project Overview & Status

- **Current Active Phase:** **Phase 6** — Visual Core 1 & 2: ResNet Screenshot Classifier & Siamese Brand Matcher
- **Historical Completed Milestones:**
  - ✅ **Phase 0:** Architecture, specifications, risk models, and API contracts finalized.
  - ✅ **Phase 1 & 1.1:** Secure Playwright website scanner, validating egress forward proxy, connection-level IP pinning, and anti-SSRF protections.
  - ✅ **Phase 2:** Multi-source dataset ingestion (PhishTank, OpenPhish, Tranco, PhreshPhish), external visual dataset integration (1,008 screenshots), deduplication (SHA-256 / pHash), annotation toolchain, and domain-aware leakage-safe splits.
  - ✅ **Phase 3:** Neural network mathematical sandbox implemented from scratch (pure NumPy linear algebra, dense layers, activation functions, losses, reverse-mode autodiff computational graph, and finite-difference gradient checking).
  - ✅ **Phase 4:** Optimizers, Ablations & Learning Algorithms from Scratch (SGD, SGDMomentum, RMSProp, Adam, StepLR, ExponentialLR, CosineAnnealingLR, ring benchmark).
  - ✅ **Phase 5:** URL Intelligence, Homoglyph Detection, Typosquatting Analysis & Lexical MLP Baseline (multi-label PSL domain extractor, Unicode mixed-script/confusable detector, Damerau-Levenshtein brand typosquatting, 40-feature lexical extractor, zero-leakage preprocessor, and 3-Layer Lexical MLP M-01 achieving ROC-AUC: 0.9773, PR-AUC: 0.9995, Macro F1: 0.8239, Phishing F1: 0.9932, Accuracy: 98.66% on unseen holdout domains).
- **Core Technology Stack:** Python 3.11–3.13, PyTorch 2.6 (CUDA 12.4 compatible), Torchvision, Ultralytics YOLO, FastAPI, Playwright, React, Vite, Tailwind CSS.

---

## 📚 Project Documentation

| Document | Purpose |
| :--- | :--- |
| [**`AGENTS.md`**](AGENTS.md) | Permanent operating guidelines, reproducibility rules, and environment instructions for AI agents. |
| [**`PROJECT_STATE.md`**](PROJECT_STATE.md) | Single source of truth for active project state, completed milestones, verified tests, and next actions. |
| [**`CURRENT_PHASE.md`**](CURRENT_PHASE.md) | Detailed checklist and exit criteria for the active development phase. |
| [**`REQUIREMENTS.md`**](REQUIREMENTS.md) | Complete functional and non-functional requirements, dataset specifications, and security constraints. |
| [**`ARCHITECTURE.md`**](ARCHITECTURE.md) | End-to-end system architecture, component breakdown, data flow, and sequence diagrams. |
| [**`TECH_STACK.md`**](TECH_STACK.md) | Detailed technology choices, runtime configurations, library versions, and Python+React mandate. |
| [**`API_CONTRACTS.md`**](API_CONTRACTS.md) | REST API endpoints, Pydantic schemas, and data exchange contracts. |
| [**`DATASET_MANIFEST.md`**](DATASET_MANIFEST.md) | Dataset acquisition strategy, candidate source evaluations, deduplication, and domain-splitting protocols. |
| [**`MODEL_REGISTRY.md`**](MODEL_REGISTRY.md) | Registry of all 10 neural network models, architectures, training regimes, and GAN exclusion policy. |
| [**`DECISIONS.md`**](DECISIONS.md) | Architectural Decision Records (ADRs) tracking design choices and justifications. |
| [**`PHASE_ROADMAP.md`**](PHASE_ROADMAP.md) | Complete 14-phase development roadmap from inception through final submission. |
| [**`docs/PHASE_3_MATH_FOUNDATIONS.md`**](docs/PHASE_3_MATH_FOUNDATIONS.md) | Detailed mathematical formulations, autograd engine mechanics, and numerical gradient checking proofs. |

---

## 📊 Dataset Statistics & Ground Truth Separation

| Dataset Entity | Records | Composition / Storage | Leakage Protection |
| :--- | :--- | :--- | :--- |
| **Verified URL Corpus** | 44,129 | 43,535 Phishing, 594 Legitimate (`data/processed/corpus_normalized.parquet`) | Strict domain-aware split: Train (32,157), Val (7,715), Test (4,257) with **0 domain overlap**. |
| **Tranco Candidate Pool** | 50,000 | Unverified candidate domains isolated in `data/processed/tranco_candidate_pool.parquet`. | Never auto-labeled as legitimate without verification. |
| **Visual Screenshot Corpus** | 1,008 | 804 Phishing, 204 Legitimate (`data/processed/screenshots/visual_corpus.parquet`) | Domain-aware split: Train (624), Val (262), Test (114) with **0 domain and 0 perceptual hash overlap**. |
| **Brand Reference Gallery** | 18 Brands | Targeted high-value brands tracked in `data/brands/reference_gallery/`. | Catalogued with official domain definitions and asset metadata. |

---

## 🚀 Getting Started & Reproducibility

### 1. Prerequisites
- **Python:** 3.11, 3.12, or 3.13
- **Operating System:** Windows, Linux, or macOS

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Omfalcon/pbl_dl.git
cd pbl_dl

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser binaries
playwright install chromium
```

### 3. Optional Hardware Acceleration (PyTorch with CUDA)
For GPU acceleration on NVIDIA hardware:
```bash
# Example for CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Running Tests
The hermetic test suite runs offline without requiring external network connectivity or multi-gigabyte raw dataset archives:
```bash
pytest -v -m "not integration"
```
*(All 210 unit tests pass in 4.76s, 1 deselected).*

To run the complete test suite including integration tests (requires local network loopback for scanner proxy tests):
```bash
pytest -v
```

### 5. Continuous Integration
All pull requests and commits to `main` are automatically validated via GitHub Actions (`.github/workflows/ci.yml`), ensuring cross-platform reproducibility, strict test passing, and clean dependency management.

---

## 🔒 Security Architecture
PhishVision AI processes untrusted, adversarial websites with multiple layers of defense:
- **Anti-SSRF:** Canonical URL parsing, port validation, scheme restriction (HTTP/HTTPS only), obfuscated IP decoding (hex/octal/dword/IPv6 literals), and RFC 1918 / RFC 4193 / link-local / loopback blocking.
- **Validating Forward Proxy:** Local egress proxy enforcing DNS resolution pinning directly to the verified socket IP, mitigating DNS rebinding attacks while preserving TLS SNI.
- **Sandboxed Browser Scanner:** Isolated Playwright Chromium execution with context-level routing, popup containment, service worker blocking, and strict execution timeouts.
