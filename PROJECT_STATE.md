# PROJECT_STATE.md — Persistent Project State

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Current Phase:** 0 — Project Blueprint, Requirements & Architecture  
**Phase Status:** COMPLETED  
**Next Phase:** 1 — Repository Foundation & Secure Playwright Website Scanner  
**Last Updated:** 2026-11-22  

---

## 1. Executive Summary

PhishVision AI is an AI-powered phishing website detection and brand impersonation platform combining lexical URL intelligence, deep computer vision, behavioral and sequence analysis, and calibrated multi-evidence risk fusion. 

Phase 0 (System Architecture & Requirements) has been completed. The project architecture explicitly establishes **Visual Intelligence** as the primary scientific research contribution, decomposed into four decoupled, specialized vision modules:
1. ResNet-18/50 General Screenshot Classifier (Unconditional global visual layout patterns).
2. Siamese CNN Brand Visual Matcher with Triplet Loss (Conditional brand similarity metric learning).
3. Ultralytics YOLOv8/v11 UI Element & Logo Detector (Real-time spatial component localization).
4. U-Net & YOLO-Seg Layout Segmenters (Semantic and instance layout topology).

Zero premature implementation code has been written.

---

## 2. Current State Snapshot

- **Active Phase:** `Phase 0` (COMPLETED)
- **Target Next Phase:** `Phase 1` (Repository Foundation & Secure Website Scanner)
- **Working Modules:** None (Design & Architecture Phase only; no application modules implemented yet).
- **Working Environment:**
  - Python: `3.13.3` (Located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe`)
  - Deep Learning Framework: `PyTorch 2.6.0+cu124` (Hardware acceleration: CUDA 12.4 verified operational)
  - Key Libraries: `torchvision 0.21.0`, `ultralytics 8.4.60`, `transformers 4.57.6`, `fastapi`, `scikit-learn 1.8.0`, `tldextract 5.3.1`, `beautifulsoup4 4.14.3`
- **Git State:** Clean branch `main`, tracking `origin/main`, `.gitignore` configured to keep repository clean.

---

## 3. Completed Work (Phase 0)

1. **Repository Inspection:** Examined initial Git repository state (`4ea70f9 first commit`, bare `README.md`).
2. **Environment Verification:** Verified pre-existing host virtual environment at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`, confirmed Python 3.13.3 runtime, PyTorch 2.6.0 with active CUDA 12.4 acceleration.
3. **Permanent Context Documentation:**
   - [`AGENTS.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/AGENTS.md): Established permanent agent instructions, rules against GANs, anti-SSRF mandate, and reproducibility protocols.
   - [`REQUIREMENTS.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/REQUIREMENTS.md): Documented functional requirements (FR-1 to FR-11), non-functional requirements (NFR-1 to NFR-6), dataset criteria, mathematical sandbox, learning paradigms, and security boundaries. Updated with Visual Intelligence as central research pillar.
   - [`ARCHITECTURE.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/ARCHITECTURE.md): Designed high-level architecture, detailed the four independent visual modules (ResNet, Siamese CNN, YOLO, U-Net/YOLO-Seg), visual evidence synthesis, zero-day unknown website handling, and Curated Brand Reference Gallery.
   - [`TECH_STACK.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/TECH_STACK.md): Detailed technology selections, library versions, virtual environment usage instructions, and Python+React mandate.
   - [`API_CONTRACTS.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/API_CONTRACTS.md): Created REST endpoint definitions and formal JSON schemas for scan requests, forensic reports, visual intelligence breakdowns, and error objects.
   - [`DATASET_MANIFEST.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/DATASET_MANIFEST.md): Formulated visual dataset strategy (~10k screenshots, $\ge 1k$ Siamese pairs, ~1.5k detection samples), Curated Brand Reference Gallery schema (`data/brands/reference_gallery/`), and leakage-safe domain-split protocols.
   - [`MODEL_REGISTRY.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/MODEL_REGISTRY.md): Cataloged all 10 planned neural network models with in-depth specifications for the four visual modules, conditional triggers, inputs/outputs, and loss targets; strictly excluded GANs.
   - [`DECISIONS.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/DECISIONS.md): Recorded ADR-001 through ADR-009 (including ADR-009 on Visual Intelligence decomposition).
   - [`PHASE_ROADMAP.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/PHASE_ROADMAP.md): Detailed the complete 14-phase development lifecycle with visual core milestones in Phases 2, 6, 7, and 10.
   - [`.gitignore`](file:///c:/Users/hp/PycharmProjects/pbl_dl/.gitignore): Ignored `.idea`, cache, model weights, virtual environments, and raw dataset directories.
   - [`CURRENT_PHASE.md`](file:///c:/Users/hp/PycharmProjects/pbl_dl/CURRENT_PHASE.md): Recorded Phase 0 deliverables and exit criteria checklist.

---

## 4. Pending Work (Master Plan)

- **Phase 1:** Repository foundation and secure Playwright website scanner (Anti-SSRF, DNS resolution pinning, screenshot & DOM capture).
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
   The project is officially and permanently a **Python + React** project. Legacy C/C++/Java implementation requirements, along with CMake/Make/Maven/Gradle build systems, are intentionally excluded requirements. All mathematical and ML objectives are fully preserved and executed in pure Python/NumPy/PyTorch. No open technical risks or unresolved questions remain regarding the technology stack.
2. **Virtual Environment Filesystem Location (ADR-007):**  
   The Python venv is located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`. Terminal tools executing Python within this environment require either direct execution via the full binary path or `BypassSandbox: true` if cross-directory permissions are enforced.

---

## 6. Last Verified Tests & Checks

- **Date:** 2026-11-22
- **Command:** `& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -c "import torch, torchvision, ultralytics, sklearn, tldextract, bs4, fastapi; print('Torch CUDA:', torch.cuda.is_available(), 'Torch version:', torch.__version__)"`
- **Result:** `Torch CUDA: True Torch version: 2.6.0+cu124` (Exit Code 0). All primary deep learning and ML packages verified functional with GPU acceleration.

---

## 7. Exact Next Action

**Begin Phase 1:**  
Initialize the repository directory structure (`app/`, `config/`, `tests/`) and implement the secure, sandboxed Playwright website scanner (`app/scanner/anti_ssrf.py` and `app/scanner/playwright_scanner.py`) with comprehensive unit tests blocking private IP access.
