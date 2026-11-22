# CURRENT_PHASE.md — Active Phase Status

**Active Phase:** Phase 1 — Repository Foundation & Secure Playwright Website Scanner  
**Phase Status:** COMPLETED  
**Transitioning To:** Phase 2 — Dataset Acquisition, Brand Gallery, Annotation & Splitting  
**Timestamp:** 2026-11-22  

---

## 1. Phase 1 Objectives & Deliverables Checklist

| Deliverable | File Path | Status | Verification Notes |
| :--- | :--- | :--- | :--- |
| **Package Architecture Foundation** | `app/`, `config/`, `tests/`, `data/`, `models/` | COMPLETED | Clean Python package hierarchy with `__init__.py` modules established. |
| **Anti-SSRF & DNS Pinning Engine** | [`app/scanner/anti_ssrf.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/anti_ssrf.py) | COMPLETED | Complete validation of schemes, ports, hostnames, DNS resolution, and IP classification. |
| **Pydantic Scanner Output Models** | [`app/scanner/models.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/models.py) | COMPLETED | Structured telemetry schemas for `ScanResult`, `FormInfo`, `ScriptInfo`, `RedirectHop`. |
| **Playwright Isolated Scanner** | [`app/scanner/playwright_scanner.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/app/scanner/playwright_scanner.py) | COMPLETED | Ephemeral Chromium runner with screenshot capture, DOM serialization, and redirect tracing. |
| **Anti-SSRF Unit Test Suite** | [`tests/test_anti_ssrf.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/tests/test_anti_ssrf.py) | COMPLETED | 51 test cases verifying scheme, localhost, private IPv4/IPv6, link-local, and metadata blocking. |
| **Scanner & Live Navigation Tests**| [`tests/test_scanner.py`](file:///c:/Users/hp/PycharmProjects/pbl_dl/tests/test_scanner.py) | COMPLETED | 4 test cases verifying DOM parsing, SSRF rejection without browser launch, schema dump, and live scan. |
| **Live Scan Verification** | Safe Public Website (`https://example.com`) | COMPLETED | Verified live capture: screenshot rendered, DOM serialized, status 200, duration 1429ms. |

---

## 2. Phase 1 Exit Criteria Verification

- [x] **Strict Anti-SSRF Enforcement:** Out-of-band DNS resolution inspecting every returned IP address. Private subnets (RFC 1918), loopback (`127.0.0.0/8`, `::1`), link-local (`169.254.0.0/16`), cloud metadata (`169.254.169.254`), and CGNAT (`100.64.0.0/10`) reliably blocked.
- [x] **Zero Credential Submission:** Scanner operates in passive observation mode only; never enters credentials or triggers forms.
- [x] **Disabled File Downloads:** Browser context rejects automatic file downloads; download events are immediately cancelled.
- [x] **Evidence Preservation:** High-resolution viewport screenshot, full-page screenshot, and rendered HTML DOM tree saved with unique UUID-based filenames.
- [x] **Deprecations & Warnings Resolved:** Public Suffix List resolution updated to `top_domain_under_public_suffix` with backwards compatibility.
- [x] **Automated Tests Passing:** All 55 test cases in `tests/` pass with zero failures (`55 passed in 2.92s`).
- [x] **Controlled Live Scan Succeeded:** Tested against `https://example.com`; verified HTML and PNG artifacts saved and accessible.

---

## 3. Hand-off Instructions to Phase 2

When Phase 2 is initiated:
1. Review `DATASET_MANIFEST.md`, `MODEL_REGISTRY.md`, and `ARCHITECTURE.md`.
2. Construct acquisition scripts for candidate feeds (Zenodo, PhishTank, OpenPhish, Tranco).
3. Populate the Curated Brand Reference Gallery (`data/brands/reference_gallery/`) for 15–20 high-value target brands.
4. Implement perceptual hashing (pHash) deduplication for screenshot images.
5. Generate leakage-safe train/val/test splits strictly partitioned by registrable domain.
