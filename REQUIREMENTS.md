# REQUIREMENTS.md — System Requirements Specification

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Status:** Approved for Baseline Architecture (Phase 0)  
**Last Updated:** 2026-11-22  

---

## 1. Industry Context & System Use Case

### 1.1 Problem Statement
Phishing and brand impersonation attacks remain among the primary attack vectors in cybercrime, accounting for substantial financial losses, identity theft, and corporate credential compromises. Attackers continually adapt their tactics by leveraging homoglyphs, zero-day domains, obfuscated JavaScript, dynamic DOM rendering, and cloaking techniques that bypass traditional static blocklists.

### 1.2 System Mission & Central Research Contribution
**PhishVision AI** delivers a multimodal, deep-learning-powered security analysis engine that evaluates any submitted URL in real time.

> [!IMPORTANT]
> **Central Research Contribution: Visual Intelligence**  
> While traditional anti-phishing solutions rely almost exclusively on reactive URL blocklists or superficial lexical heuristics, **Visual Intelligence** represents the primary scientific and architectural pillar of PhishVision AI. Attackers can effortlessly register novel domains and alter URL strings, but to deceive users, they must faithfully recreate or visually simulate the graphical layout, styling, and brand identity of targeted organizations. PhishVision AI captures and decodes these visual signatures through four independent visual intelligence modules.

The system combines:
1. Lexical and structural URL/domain intelligence (incorporating Public Suffix List parsing, homoglyph detection, and typosquatting analysis).
2. **Visual Intelligence Suite (Central Research Contribution):**
   - **Module 1:** ResNet-based webpage screenshot classification (global layout patterns).
   - **Module 2:** Siamese CNN-based visual brand similarity metric learning.
   - **Module 3:** YOLO-based UI element (logos, login fields, submit buttons) localization.
   - **Module 4:** U-Net semantic segmentation and YOLO-Seg instance segmentation for spatial layout topology.
3. Dynamic behavioral telemetry (redirect chains, domain transitions, dynamic DOM snapshots, and static JavaScript metadata).
4. Sequence modeling of navigation paths (LSTM and Transformer).
5. A calibrated risk fusion engine that outputs an explainable, probabilistic threat score with confidence boundaries and an explicit fallback for ambiguous or unobservable targets (`UNKNOWN / INSUFFICIENT EVIDENCE`).

### 1.3 Key Stakeholders
- **End Users / Consumers:** Seek immediate, easy-to-understand verdicts on suspicious links encountered in emails, SMS, or social media.
- **Enterprise Security Operations Centers (SOC):** Require structured telemetry, indicators of compromise (IOCs), and forensic evidence (screenshots, DOM dumps, redirect paths) to accelerate incident triage.
- **Brand Protection & Legal Teams:** Monitor brand logo and visual layout misuse across unauthorized infrastructure.
- **Academic & Security Researchers:** Require reproducible experiments, clear dataset provenance, and inspectable mathematical fundamentals of the underlying machine learning models.

---

## 2. Approved Technology Architecture & Language Mandate (ADR-001)

> [!IMPORTANT]
> **Official Technology Decision — Python + React Only:**  
> The project is officially and permanently a **Python + React** project.
> - **Backend, Machine Learning, Mathematical Fundamentals & Testing:** Exclusively **Python 3.13** (PyTorch, Torchvision, Ultralytics, scikit-learn, FastAPI, Playwright, NumPy, OpenCV).
> - **Frontend User Interface:** **React** (Vite, Tailwind CSS, JavaScript).
> - **Intentionally Excluded Languages & Build Systems:** C, C++, Java, CMake, Make, Maven, and Gradle are permanently excluded from the project under this approved technology decision.
> - **Academic Requirements Mapping:** All original mathematical, pedagogical, and machine learning learning objectives—including tensor operations, broadcasting, analytical backpropagation, numerical gradient checking, optimization algorithms, Hebbian learning, CNNs, YOLO object detection, U-Net/YOLO-Seg segmentation, sequence models (LSTM/Transformer), and VAE—remain 100% intact and will be implemented strictly using pure Python and NumPy. This language decision is final and approved, and is not an open technical risk or defect.

---

## 3. Functional Requirements (FR)

### 3.1 Ingress & Lexical Intelligence
- **FR-1: Safe URL Ingestion & Parsing**  
  Accept any standard web URL (`http://`, `https://`). Perform Unicode normalization (NFC/NFKC), Punycode/IDNA decoding, canonicalization, and extract registrable domains strictly using the Public Suffix List (PSL).

- **FR-2: Lexical & Homoglyph Intelligence**  
  Detect homoglyphs (Cyrillic, Greek, lookalike Latin characters), calculate Shannon entropy of hostname/path, generate character n-grams, and compute Levenshtein and Damerau-Levenshtein edit distances against a reference dictionary of protected brand names. Correctly identify subdomain masking (e.g., recognizing that `sbi.co.in.attacker.com` belongs to `attacker.com`, not `sbi.co.in`).

- **FR-3: Isolated Webpage Scanning & Dynamic Rendering**  
  Spin up a sandboxed, ephemeral Playwright browser session with strict network egress controls. Safely capture:
  - High-resolution viewport screenshot ($1280 \times 720$).
  - Final rendered HTML/DOM tree.
  - Navigation redirect chain (HTTP status codes, intermediate URLs, domain transition timestamps).
  - Script source locations and form submission targets (`action` attributes).

---

### 3.2 Visual Intelligence Suite (Core Research Requirements)

- **FR-4: Visual Module 1 — General Screenshot Classification (ResNet-18/50)**  
  Classify full-page screenshots into general phishing vs. legitimate visual layouts.  
  - *Zero-Day Independence Requirement:* Must evaluate generalized layout aesthetics, styling entropy, and visual composition **completely independently of brand-reference availability**. Must function reliably on completely unknown websites without requiring a reference screenshot for every domain on the internet.

- **FR-5: Visual Module 2 — Visual Brand Similarity (Siamese CNN)**  
  Extract deep visual embeddings from detected logos or candidate brand regions and measure cosine/Euclidean distance against the Curated Brand Reference Gallery.  
  - *Conditional Execution Requirement:* Brand matching must be **strictly conditional** on detected or suspected brand identity (triggered when Module 3 detects a logo or when URL lexical analysis detects brand typosquatting). If no brand is detected or claimed, this module outputs a neutral signal ($0.0$) and does not force false positives on novel, unbranded websites.

- **FR-6: Visual Module 3 — UI Component & Logo Detection (Ultralytics YOLO)**  
  Detect and localize brand logos, username input fields, password input fields, submit buttons, and payment QR codes with precise bounding boxes and class confidences. Compute spatial co-occurrence between logos and credential submission fields.

- **FR-7: Visual Module 4 — Webpage Region Segmentation (U-Net & YOLO-Seg)**  
  Perform semantic and instance segmentation of webpage functional zones (header banner, navigation bar, credential form container, footer, copyright zone). Quantify form surface dominance and layout abnormalities.

---

### 3.3 Behavioral, Sequence & Generative Requirements
- **FR-8: Redirect Sequence Classification (LSTM / Transformer)**  
  Encode redirect sequences (URL lexical features, domain changes, HTTP status transitions) to differentiate standard CDNs/OAuth flows from phishing evasion tactics.

- **FR-9: Generative Latent Modeling (VAE)**  
  Train a Variational Autoencoder (VAE) on webpage layout features and embeddings to characterize the latent distribution of legitimate designs and synthesize hard-negative boundary examples. *(GAN is strictly excluded).*

- **FR-10: Calibrated Risk Fusion & Explainability**  
  Combine evidence from all operational modules using calibrated logistic regression / Bayesian fusion. Output a final risk category (`PHISHING`, `SUSPICIOUS`, `LEGITIMATE`, or `UNKNOWN / INSUFFICIENT EVIDENCE`), a continuous risk index (0.00 – 1.00), and an itemized breakdown of contributing signals.

- **FR-11: Interactive User Dashboard**  
  Provide a React-based single-page application with a submission input, real-time scanning progress indicator, visual bounding-box overlay on webpage screenshots, and graphical radar/pie charts of threat vectors.

---

## 4. Non-Functional Requirements (NFR)

- **NFR-1: Execution Safety & Isolation (Security)**  
  The website scanner must never access RFC 1918 (private IPv4: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), RFC 4193 (unique local IPv6), loopback (`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`, `fe80::/10`), or internal cloud metadata services (`169.254.169.254`). All outbound requests must resolve DNS at connect time with anti-rebinding checks.

- **NFR-2: Scan Latency**  
  Live URL scanning and inference must complete within:
  - Fast-path (lexical URL analysis only): < 150 ms.
  - Full-pipeline (browser navigation, screenshot, vision inference, fusion): < 8 seconds (under normal network conditions; hard timeout at 15 seconds).

- **NFR-3: Explainability**  
  Every classification must provide human-interpretable rationale (e.g., "Mismatched domain: detected PayPal logo on unverified domain `paypa1-security.com`", "Excessive redirect hops through 4 distinct registrable domains").

- **NFR-4: Graceful Degradation & Missing Signal Resilience**  
  If dynamic browsing fails (e.g., target host down, TLS handshake error, timeout), the system must fallback gracefully to static URL/domain intelligence and output `INSUFFICIENT EVIDENCE (DYNAMIC SCAN FAILED)` rather than raising an unhandled exception or returning a false-negative "Legitimate" score.

- **NFR-5: Modularity & Decoupling**  
  Production inference components (URL analysis, ResNet classifier, YOLO detector, Siamese matcher, Risk Fusion) must run independently of offline academic sandbox modules (mathematical sandbox, comparative optimizer ablations, Hebbian learning experiments, Faster R-CNN comparisons).

- **NFR-6: Reproducibility**  
  All dataset splits, preprocessing scripts, and neural network training pipelines must execute deterministically using fixed random seeds and versioned configuration manifests.

---

## 5. Visual Dataset Requirements & Brand Reference Strategy

### 5.1 Curated Visual Brand Reference Dataset
A dedicated reference gallery for 15–20 high-value, frequently impersonated brands (e.g., SBI, HDFC, PayPal, Microsoft, Google, Apple, Amazon, Chase, Netflix) will be maintained at `data/brands/reference_gallery/`:
- Multi-resolution logos (transparent PNG, vector SVG, monochrome variants).
- Canonical high-resolution login and home page screenshots.
- Authorized registrable domains and subdomains mapping.
- Pre-computed 256-dimensional Siamese unit embedding vectors.

### 5.2 Leakage-Safe Partitioning Protocols
- **Split by Registrable Domain:** To prevent data leakage, URLs belonging to the same registrable domain (e.g., `login.example.com` and `support.example.com`) must reside strictly in the same partition (Train, Validation, or Test).
- **Temporal Holdout:** Where timestamps exist, the test set must be drawn from the most recent temporal window to simulate real-world zero-day generalization.
- **Deduplication:** Remove exact URL duplicates and perceptual screenshot duplicates (using pHash Hamming distance threshold < 6).

---

## 6. Academic Fundamentals & Learning Algorithms Scope

### 6.1 Mathematical Sandbox Requirements
Offline academic demonstration module implementing from first principles:
- Tensor creation, broadcasting, inner products, reshaping, Singular Value Decomposition (SVD).
- Numerical gradient checking against analytical backpropagation.
- McCulloch-Pitts neuron and Sigmoid neuron simulations.
- Multi-Layer Perceptron (MLP) forward and backward passes from scratch.
- Gradient descent variants: Standard GD, Mini-batch SGD, Momentum, and Nesterov Accelerated Gradient (NAG).
- Weight initializations (Xavier/Glorot, He/Kaiming) and normalization layers (Batch Norm, Layer Norm).
- Regularization ($L_1$, $L_2$, Dropout) and data augmentation impact on generalization error.

### 6.2 Learning Paradigms Demonstration
Implementation and conceptual comparison of:
1. **Hebbian Learning:** Activity-dependent synaptic plasticity ($\Delta w_{ij} = \eta x_i x_j$).
2. **Modified Hebbian (Oja's Rule):** Normalized synaptic growth preventing unbounded weights ($\Delta w = \eta y (x - y w)$).
3. **Competitive Learning:** Self-organizing feature maps / winner-take-all dynamics.
4. **Error-Correction Learning:** Delta rule and perceptron learning algorithms.
5. **Boltzmann / Energy-Based Learning:** Stochastic Hopfield networks and contrastive divergence principles.

---

## 7. Security & Responsible-Use Boundaries

- **Zero-Credential Policy:** The scanner must never interactively submit credentials, click reCAPTCHAs, or attempt authentication bypasses.
- **Egress Firewalling:** Headless browser instances must be restricted to standard HTTP/HTTPS ports (80, 443, 8080, 8443) and blocked from LAN subnets.
- **Malware Protection:** Automatic file downloads must be disabled in the browser context (`downloadsPath: null`, mime-type interception).
- **Legal Compliance:** Scanning is performed for passive threat intelligence only. The platform will include clear terms indicating that results are advisory risk assessments.
