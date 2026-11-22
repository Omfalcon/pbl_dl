# DECISIONS.md — Architecture Decision Records (ADR)

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Status:** Approved for Baseline Architecture (Phase 0)  
**Last Updated:** 2026-11-22  

---

## ADR-001: Adoption of Python 3.13 and React as Exclusive Project Stack

- **Context:** The legacy academic syllabus suggested implementing certain neural network components using C/C++ or Java with Make/Maven build scripts.
- **Decision:** The project is officially and permanently a **Python + React** project. Python 3.13 is the exclusive language for all backend services, machine learning pipelines, mathematical fundamentals, training, inference, and test suites. React/Vite/Tailwind CSS/JavaScript is the exclusive frontend stack. C, C++, Java, CMake, Make, Maven, and Gradle are permanently excluded from the project under this approved technology decision.
- **Justification:**
  1. The contemporary state-of-the-art for computer vision (YOLOv8/v11, ResNet, Siamese networks) and dynamic web security automation (Playwright) provides first-class support and pre-built bindings in Python.
  2. Python enables direct integration with CUDA 12.4 GPU acceleration without custom JNI or C++ FFI wrappers.
  3. The educational goals of the academic curriculum (demonstrating tensor manipulation, analytical backpropagation, numerical gradient checking, optimization algorithms, and biological learning paradigms) will be strictly satisfied in Phase 3 through pure Python/NumPy implementations from first principles.
- **Consequences:** All deliverables, codebases, manifests, and experiments will be in Python and React. C/C++/Java requirements are intentionally excluded; this decision is final and approved, and will not be treated as a pending risk or deviation.

---

## ADR-002: Variational Autoencoders (VAE) Selected; Generative Adversarial Networks (GAN) Excluded

- **Context:** Previous preliminary outlines considered using GANs for synthetic data augmentation.
- **Decision:** Generative Adversarial Networks (GANs) are permanently removed from the project scope. The generative modeling module (Phase 9) will exclusively employ Variational Autoencoders (VAE).
- **Justification:**
  1. GAN training is notoriously unstable (mode collapse, non-convergence) and requires delicate minimax optimization that provides poor probabilistic likelihood estimates.
  2. VAEs provide a continuous, smooth, and probabilistically principled latent space with an explicit Evidence Lower Bound (ELBO) formulation.
  3. The smooth latent space of a VAE is superior for interpolating between legitimate design embeddings and synthesizing hard-negative boundary examples.
- **Consequences:** No GAN code, models, or dependencies will be introduced at any phase.

---

## ADR-003: Public Suffix List (PSL) Domain Parsing over Naive String Splitting

- **Context:** Naive string splitting (e.g., splitting on `.` and taking the last two tokens) fails on multi-part effective TLDs like `.co.in`, `.com.au`, `.gov.uk`, leading to catastrophic misattributions (e.g., treating `attacker.co.in` as domain `co.in` or missing subdomain masking attacks like `sbi.co.in.attacker.com`).
- **Decision:** Mandate the use of the Public Suffix List via the `tldextract` library for all domain resolution.
- **Consequences:** Guarantees that `sbi.co.in.attacker.com` is definitively identified as belonging to `attacker.com` with `sbi.co.in` identified as a deceptive subdomain prefix.

---

## ADR-004: Decoupled Multi-Model Architecture with Calibrated Risk Fusion

- **Context:** Phishing detection cannot rely on a single modal feature (e.g., a legitimate-looking domain might host a compromised login page, or a suspicious domain might be a harmless parking page).
- **Decision:** Implement separate, decoupled specialist engines (URL/Domain Lexical, ResNet Layout Classifier, YOLO Element Detector, Siamese Brand Matcher, LSTM Redirect Modeler) whose outputs are combined by a calibrated Bayesian / logistic risk fusion engine.
- **Consequences:** System remains resilient to missing modalities (e.g., if a page fails to load, lexical models still provide an assessment marked `UNKNOWN / INSUFFICIENT EVIDENCE`).

---

## ADR-005: Playwright Isolated Headless Scanner with Pre-Connection Anti-SSRF

- **Context:** User-submitted URLs pose severe security risks to the scanning backend, including Server-Side Request Forgery (SSRF) targeting internal subnets (`10.0.0.0/8`, `192.168.0.0/16`), cloud metadata endpoints (`169.254.169.254`), and DNS rebinding attacks.
- **Decision:** Wrap Playwright in an ephemeral, isolated container/process context. Enforce pre-connection DNS resolution pinning to ensure the resolved IP does not belong to private/reserved ranges (RFC 1918 / RFC 4193 / link-local / loopback). Intercept all network redirects and re-validate destination IPs before following.
- **Consequences:** Eliminates SSRF vectors and browser escape hazards before rendering.

---

## ADR-006: Registrable Domain and Temporal Split Strategy to Prevent Data Leakage

- **Context:** Standard random train/test splitting on web URLs causes severe data leakage because pages from the same website appear in both train and test sets, artificially inflating evaluation metrics.
- **Decision:** Mandate strict dataset splitting by registrable domain. All URLs from domain $X$ must reside exclusively in train, val, or test. In addition, temporal holdout splits will be applied for time-stamped feeds.
- **Consequences:** Guarantees realistic evaluation of zero-day detection capabilities.

---

## ADR-007: Reuse of Verified Virtual Environment (`blansole_footscan\.venv`)

- **Context:** A pre-existing Python virtual environment containing Python 3.13.3, PyTorch 2.6.0 with CUDA 12.4 GPU acceleration, Torchvision, Ultralytics, and related dependencies is available on the host at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`.
- **Decision:** Leverage this existing virtual environment rather than creating a duplicate local environment or re-downloading multi-gigabyte PyTorch/CUDA binaries.
- **Consequences:** Conserves bandwidth and disk space, and provides immediate GPU acceleration. Python commands can be invoked using `& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe"`.

---

## ADR-008: Curated Brand Knowledge Graph vs. Internet-Wide Whitelist

- **Context:** Attempting to maintain a whitelist of every legitimate website on the internet is impossible and architecturally flawed.
- **Decision:** Maintain a targeted, curated reference gallery of 15–20 high-risk, frequently impersonated brands (e.g., SBI, PayPal, Microsoft, Google) with their official domain associations. Visual similarity to these brands on non-associated domains triggers brand impersonation alarms, while general phishing detection on all other sites is handled by visual layout and lexical models.
- **Consequences:** Scales to previously unseen websites without requiring an exhaustive global domain whitelist.


---

## ADR-009: Visual Intelligence as Central Research Contribution & Four Decoupled Visual Modules

- **Context:** While attackers can easily register arbitrary domain names and alter lexical structures, they cannot execute a credible credential theft campaign without visually deceiving the victim. Thus, visual semantics, element layout, and brand impersonation constitute the most resilient detection signals. However, the system must not require an exhaustive whitelist of all legitimate websites on the internet, nor should unbranded, novel small-business websites be penalized simply because their logos do not appear in a brand registry.
- **Decision:** Establish **Visual Intelligence** as the primary scientific research contribution of PhishVision AI, structured into four decoupled, specialized modules:
  1. **Module 1 (ResNet-18/50):** Unconditional global visual layout classifier evaluating generalized visual phishing primitives (layout entropy, font clumping, badge replication) across any website without reference templates.
  2. **Module 2 (Siamese CNN):** Metric learning engine with Triplet Loss comparing detected logo crops against a Curated Brand Reference Gallery of 15–20 high-value brands. Operates **strictly conditionally** upon candidate logo detection.
  3. **Module 3 (Ultralytics YOLO):** Real-time spatial object detector identifying brand logos, username fields, password inputs, submit buttons, and payment QR codes.
  4. **Module 4 (U-Net & YOLO-Seg):** Semantic and instance layout segmenters computing functional region dominance (e.g. form-to-viewport ratio).
- **Consequences:** The system delivers zero-day generalization on unknown domains via Modules 1, 3, and 4, while achieving targeted precision against brand impersonation via Module 2 without requiring an exhaustive global domain whitelist.
