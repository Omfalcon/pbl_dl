# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 1 — Repository Foundation, Secure Playwright Scanner & Security Hardening (Phase 1.1)  
**Phase Status:** COMPLETED  
**Transitioning To:** Phase 2 — Dataset Acquisition, Brand Gallery, Annotation & Splitting  
**Timestamp:** 2026-11-22  

---

## 1. Phase 1 & 1.1 Objectives & Deliverables Checklist

| Deliverable | File Path | Status | Verification Notes |
| :--- | :--- | :--- | :--- |
| **Package Architecture Foundation** | `app/`, `config/`, `tests/`, `data/`, `models/` | COMPLETED | Clean Python package hierarchy with `__init__.py` modules established. |
| **Hardened Anti-SSRF Engine** | [`app/scanner/anti_ssrf.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/anti_ssrf.py) | COMPLETED | Hardened against out-of-range ports, control chars, malformed authorities, obfuscated IP literals (dword, hex, octal), and IPv4-mapped IPv6. |
| **Validating Egress Forward Proxy** | [`app/scanner/proxy.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/proxy.py) | COMPLETED | Enforces actual connection-level IP safety by intercepting HTTP/CONNECT requests and binding outbound sockets strictly to validated public IPs while preserving TLS SNI. |
| **Pydantic Scanner Output Models** | [`app/scanner/models.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/models.py) | COMPLETED | Structured telemetry schemas for `ScanResult`, `ScanOptions`, `FormInfo`, `ScriptInfo`, `RedirectHop`, `tls_error`, and `pinned_ip`. |
| **Hardened Playwright Scanner** | [`app/scanner/playwright_scanner.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/playwright_scanner.py) | COMPLETED | Context-level route interception, popup containment, service worker blocking, default secure TLS verification (`ignore_https_errors=False`), and robust cleanup. |
| **Anti-SSRF Unit Test Suite** | [`tests/test_anti_ssrf.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/tests/test_anti_ssrf.py) | COMPLETED | 72 test cases verifying schemes, ports, authorities, obfuscated IPs, IPv4-mapped IPv6, mixed DNS answers, and DNS rebinding simulation. |
| **Scanner & Security Hardening Tests**| [`tests/test_scanner.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/tests/test_scanner.py) | COMPLETED | 13 test cases verifying proxy CONNECT/GET blocking, DOM parsing, TLS failure capture, exception cleanup, and live proxy scan. |
| **Pytest Configuration** | [`pytest.ini`](file:///c:/Users/hp/PycharmProjects/pbl_dl/pytest.ini) | COMPLETED | Registered `integration` test marker to decouple offline unit tests from internet-dependent suites. |

---

## 2. Phase 1 & 1.1 Exit Criteria Verification

- [x] **Connection-Level DNS/IP Enforcement:** Enforced through `ValidatingEgressProxy`. The browser communicates with target hosts exclusively through a local forward proxy that validates destinations and binds TCP sockets directly to the pre-validated `pinned_ip`. TLS SNI and Host headers are preserved end-to-end.
- [x] **Browser Isolation & Sandbox Safety:** Removed insecure `--no-sandbox` flags from standard Windows execution. Only flags strictly necessary for headless automation are passed. Documented that true OS-level network namespace isolation requires Docker container deployment.
- [x] **Context-Level Request Interception:** Interception registered via `context.route("**/*")` to cover all frames, workers, and popups.
- [x] **Service Worker Blocking:** Context configured with `service_workers="block"` to eliminate interception gaps.
- [x] **Popup Containment:** Context intercepts unsolicited popups and closes them automatically without interrupting the primary scan.
- [x] **Secure Default TLS Verification:** `ignore_https_errors` defaults to `False`. Certificate verification failures are captured as structured scan evidence (`CERT_ERROR` status, `tls_error` details) rather than treated as successful navigations.
- [x] **URL Parser Hardening:** Hardened against control characters, malformed authorities (`user@pass@host`), out-of-range ports (<1 or >65535), hex/octal/dword IP representations, and IPv4-mapped IPv6 literals (`[::ffff:127.0.0.1]`).
- [x] **Reliable Resource Cleanup:** Browser context, browser process, and proxy server sockets are guaranteed closed via `try/finally` blocks.
- [x] **Regression & Unit Testing:** All 84 unit tests pass in 0.68s without internet dependencies; live controlled scan with proxy passes in integration test suite.

---

## 3. Hand-off Instructions to Phase 2

When Phase 2 is initiated:
1. Review `DATASET_MANIFEST.md`, `MODEL_REGISTRY.md`, and `ARCHITECTURE.md`.
2. Construct acquisition scripts for candidate feeds (Zenodo, PhishTank, OpenPhish, Tranco).
3. Populate the Curated Brand Reference Gallery (`data/brands/reference_gallery/`) for 15–20 high-value target brands.
4. Implement perceptual hashing (pHash) deduplication for screenshot images.
5. Generate leakage-safe train/val/test splits strictly partitioned by registrable domain.
