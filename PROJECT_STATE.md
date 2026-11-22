# PROJECT_STATE.md — Persistent Project State

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Current Phase:** 1 — Repository Foundation, Secure Playwright Scanner & Security Hardening (Phase 1.1)  
**Phase Status:** COMPLETED  
**Next Phase:** 2 — Dataset Acquisition, Brand Gallery, Annotation & Splitting  
**Last Updated:** 2026-11-22  

---

## 1. Executive Summary

PhishVision AI is an AI-powered phishing website detection and brand impersonation platform combining lexical URL intelligence, deep computer vision, behavioral and sequence analysis, and calibrated multi-evidence risk fusion.

- **Phase 0 (System Architecture & Requirements):** COMPLETED.
- **Phase 1 & Phase 1.1 (Repository Foundation, Scanner Implementation & Security Hardening):** COMPLETED.
  - Package foundation established (`app/`, `config/`, `tests/`, `data/`, `models/`).
  - Production Anti-SSRF engine (`app/scanner/anti_ssrf.py`) hardened against malformed ports, authorities, control characters, obfuscated IPs (dword, hex, octal), and IPv4-mapped IPv6 literals.
  - **Connection-Level IP Safety Enforced:** Implemented `ValidatingEgressProxy` (`app/scanner/proxy.py`), ensuring that all outbound HTTP and HTTPS CONNECT requests are intercepted, validated, and bound strictly to the pre-validated public `pinned_ip`, preserving end-to-end TLS SNI while neutralizing DNS rebinding attacks.
  - **Browser Hardening:** Removed insecure `--no-sandbox` flags from standard Windows execution; configured context-level route interception, service worker blocking (`service_workers="block"`), popup containment, and default secure TLS certificate verification (`ignore_https_errors=False`).
  - Structured Pydantic telemetry models (`app/scanner/models.py`) with `tls_error`, `pinned_ip`, and `proxy_blocked_count`.
  - Comprehensive test suite: **85 tests passing** (84 unit tests in 0.68s + 1 live integration test).

---

## 2. Current State Snapshot

- **Active Phase:** `Phase 1 / 1.1` (COMPLETED)
- **Target Next Phase:** `Phase 2` (Dataset Acquisition, Brand Gallery, Annotation & Splitting)
- **Working Modules:**
  - `app.scanner.anti_ssrf`: Hardened URL parsing, port validation, DNS resolution, and IP classification.
  - `app.scanner.proxy`: Validating local egress forward proxy with connection-level IP pinning and TLS SNI preservation.
  - `app.scanner.playwright_scanner`: Isolated Playwright Chromium scanner with context-level routing, popup containment, service worker blocking, and secure TLS checking.
  - `app.scanner.models`: Pydantic structured scan telemetry schemas.
- **Working Environment:**
  - Python: `3.13.3` (Located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe`)
  - Deep Learning Framework: `PyTorch 2.6.0+cu124` (Hardware acceleration: CUDA 12.4 operational)
  - Key Libraries: `playwright 1.50+`, `beautifulsoup4 4.14.3`, `tldextract 5.3.1`, `pydantic 2.x`, `torchvision 0.21.0`, `ultralytics 8.4.60`, `scikit-learn 1.8.0`
- **Git State:** Tracking `main`, clean repository structure with `.gitignore` excluding transient scan artifacts.

---

## 3. Completed Work

### Phase 0: Project Blueprint, Requirements & Architecture (COMPLETED)
1. Authored all 10 context management documents: `AGENTS.md`, `PROJECT_STATE.md`, `CURRENT_PHASE.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `TECH_STACK.md`, `API_CONTRACTS.md`, `DATASET_MANIFEST.md`, `MODEL_REGISTRY.md`, `DECISIONS.md`, and `PHASE_ROADMAP.md`.
2. Verified Python virtual environment with CUDA 12.4 PyTorch support.
3. Formally mandated Python + React as exclusive project technology (ADR-001) and established Visual Intelligence as central research pillar (ADR-009).

### Phase 1 & 1.1: Scanner Implementation & Security Hardening (COMPLETED)
1. **Package Hierarchy:** Created `app/core/`, `app/scanner/`, `app/intelligence/`, `app/fusion/`, `app/api/`, `config/`, `tests/`, `data/`, `models/` with `__init__.py` modules.
2. **Hardened Anti-SSRF Module (`app/scanner/anti_ssrf.py`):**
   - Scheme and authority sanitization (rejecting control characters, unescaped whitespace, multiple `@` delimiters).
   - Port validation enforcing strict $1 \le \text{port} \le 65535$ and blocking administrative services.
   - Parsing obfuscated IP literals (decimal dwords, hex literals, dotted octals).
   - IPv4-mapped IPv6 literal inspection (`[::ffff:127.0.0.1]`).
   - Socket `getaddrinfo` resolution inspecting all returned IPv4/IPv6 addresses; rejects mixed public/private answers.
   - Comprehensive subnet blacklisting: RFC 1918, RFC 4193, loopback, link-local, cloud metadata (`169.254.169.254`), CGNAT (`100.64.0.0/10`), broadcast, and reserved addresses.
3. **Validating Egress Forward Proxy (`app/scanner/proxy.py`):**
   - Intercepts all outbound HTTP and HTTPS CONNECT requests from Chromium.
   - Revalidates destination against Anti-SSRF policies at the network layer.
   - Binds outbound TCP sockets strictly to the validated `pinned_ip`, completely defeating DNS rebinding.
   - Preserves TLS Server Name Indication (SNI) and Host headers end-to-end.
4. **Hardened Playwright Scanner (`app/scanner/playwright_scanner.py`):**
   - Removed insecure `--no-sandbox` flags from standard Windows execution.
   - Configured `service_workers="block"` to eliminate route interception bypasses.
   - Integrated popup containment listener closing unsolicited popups automatically.
   - Switched to secure default TLS certificate verification (`ignore_https_errors=False`), capturing handshake failures as structured `CERT_ERROR` evidence.
   - Context-level request routing (`context.route("**/*")`) re-validating all subresources and redirects.
   - Ensured reliable `try/finally` cleanup of pages, contexts, browsers, and proxy server sockets.
5. **Testing & Validation:**
   - 72 unit tests in `tests/test_anti_ssrf.py` covering all edge cases.
   - 13 scanner tests in `tests/test_scanner.py` covering proxy blocking, DOM parsing, TLS failure capture, exception cleanup, and live scan.
   - All 84 unit tests pass in 0.68s; 1 live integration test passes against `https://example.com`.

---

## 4. Pending Work (Master Plan)

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
   The project is officially and permanently a **Python + React** project. Legacy C/C++/Java implementation requirements, along with CMake/Make/Maven/Gradle build systems, are intentionally excluded requirements. All mathematical and ML objectives are fully preserved and executed in pure Python/NumPy/PyTorch.
2. **Virtual Environment Filesystem Location (ADR-007):**  
   The Python venv is located at `C:\Users\hp\PycharmProjects\blansole_footscan\.venv`. Terminal tools executing Python within this environment require either direct execution via the full binary path or `BypassSandbox: true` if cross-directory permissions are enforced.
3. **Environment Security Boundaries:**  
   The scanner enforces connection-level IP safety via the `ValidatingEgressProxy` and context-level routing. While this provides application and network-layer anti-SSRF and DNS rebinding immunity, true OS-level process and kernel isolation requires containerized deployment (e.g., Docker container with network namespaces and non-root execution), which will be packaged for production.

---

## 6. Last Verified Tests & Checks

- **Date:** 2026-11-22
- **Command:** `& "C:\Users\hp\PycharmProjects\blansole_footscan\.venv\Scripts\python.exe" -m pytest -v tests/`
- **Result:** `85 passed in 3.00s` (Exit Code 0). All Anti-SSRF rules, proxy IP pinning, TLS error handling, and live navigation checks verified.
- **Offline Suite:** `84 passed in 0.68s` with `pytest -m "not integration"`.

---

## 7. Exact Next Action

**Begin Phase 2:**  
Execute dataset acquisition, Curated Brand Reference Gallery population (`data/brands/reference_gallery/` for 15–20 brands), image deduplication via pHash, and leakage-safe domain-based dataset partitioning.
