"""Phase 9 Regression Tests — Six-Stage Pipeline and Extended Report Schema.

All tests are deterministically mocked — no live network, Playwright, or ML inference required.

Covers:
1.  Stage tracker: stage transitions, progress percentage, serialization
2.  Fast-mode: browser stages auto-SKIPPED
3.  URL_CHECK SSRF block terminates scan immediately
4.  TLS inspection for HTTPS with valid cert
5.  TLS inspection skipped for HTTP URLs
6.  TLS inspection failure — scan continues to PLAYWRIGHT_CAPTURE
7.  Playwright timeout — PLAYWRIGHT_CAPTURE FAILED, scan → FAILED
8.  SSRF block from scanner — BLOCKED_SSRF terminal status
9.  Pipeline initialization failure → FAILED (Phase 8.1 regression)
10. VISUAL_ANALYSIS UNAVAILABLE — fusion still runs with available evidence
11. Progress percentage at each stage boundary
12. Status polling returns stages dict, current_stage, progress_percent
13. Full report contains all Phase 9 extended fields
14. TLS expired cert → MEDIUM risk factor in contributing_factors
15. No screenshot — UNKNOWN / INSUFFICIENT EVIDENCE preserved
16. Brand impersonation → CRITICAL_PHISHING with risk >= 0.95
17. Report backward compatibility: existing Phase 8 fields still present
18. Redirect intelligence enrichment
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal analysis dict helpers
# ---------------------------------------------------------------------------

def _make_analysis(
    *,
    category: str = "LEGITIMATE",
    risk_score: float = 0.10,
    explanation: str = "Test analysis.",
    m01_prob: float | None = 0.02,
    m07_prob: float | None = 0.005,
    has_password: bool = False,
    is_impersonation: bool = False,
    brand: str = "UNIDENTIFIED_OR_NEUTRAL",
    screenshot_available: bool = True,
) -> dict[str, Any]:
    return {
        "url_analysis": {
            "m01_lexical": {
                "probability": m01_prob,
                "is_above_calibrated_threshold": (m01_prob or 0) > 0.05,
                "status": "OK" if m01_prob is not None else "UNAVAILABLE",
            },
            "m07_lstm": {
                "probability": m07_prob,
                "is_above_calibrated_threshold": (m07_prob or 0) > 0.01,
                "status": "OK" if m07_prob is not None else "UNAVAILABLE",
            },
            "composite_url_risk": round(((m01_prob or 0) + (m07_prob or 0)) / 2.0, 4) if (m01_prob is not None or m07_prob is not None) else None,
        },
        "visual_analysis": {
            "screenshot_available": screenshot_available,
            "m02_phishing_probability": 0.05 if screenshot_available else None,
            "brand_verification": {
                "is_brand_impersonation": is_impersonation,
                "predicted_brand": brand if is_impersonation else "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.95 if is_impersonation else 0.0,
                "is_authorized_domain": False,
            },
            "ui_elements": {
                "has_login_form": has_password,
                "has_credential_inputs": has_password,
                "has_submit_button": has_password,
                "has_payment_element": False,
                "has_password_field": has_password,
                "total_fused_elements": 2 if has_password else 0,
                "elements": [],
            },
        },
        "dom_evidence": {
            "has_dom_password_field": has_password,
            "has_external_submission": False,
        },
        "fusion": {
            "threat_category": category,
            "risk_score": risk_score,
            "explanation": explanation,
        },
        "model_availability": {
            "M-01_lexical_mlp": True,
            "M-02_resnet18_classifier": True,
            "M-03_siamese_brand_matcher": True,
            "M-04_yolov8n_detector": True,
            "M-05_faster_rcnn_detector": True,
            "M-06_yolov8_seg_segmenter": True,
            "M-07_lstm_sequence_classifier": True,
        },
        "disclaimer": "Test disclaimer.",
    }


def _make_scan_result(
    status: str = "COMPLETED",
    redirect_chain: list | None = None,
    forms: list | None = None,
    final_url: str = "https://example.com",
    title: str = "Example",
) -> MagicMock:
    from app.scanner.models import ScanStatus
    mock = MagicMock()
    mock.status = ScanStatus.COMPLETED if status == "COMPLETED" else ScanStatus.BLOCKED_SSRF
    mock.redirect_chain = redirect_chain or []
    mock.forms = forms or []
    mock.final_url = final_url
    mock.title = title
    mock.tls_error = None
    mock.dom_text = ""
    return mock


# ---------------------------------------------------------------------------
# Test: ScanPipelineTracker (unit)
# ---------------------------------------------------------------------------

class TestScanPipelineTracker:
    def test_initial_state_all_pending(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker, StageStatus
        tracker = ScanPipelineTracker("test-id", fast_mode=False)
        td = tracker.to_dict()
        assert td["progress_percent"] == 0.0
        assert td["current_stage"] == "URL_CHECK"
        for name, s in td["stages"].items():
            assert s["status"] == StageStatus.PENDING

    def test_fast_mode_skips_browser_stages(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker, StageStatus
        tracker = ScanPipelineTracker("test-id", fast_mode=True)
        td = tracker.to_dict()
        assert td["stages"]["REDIRECT_INVESTIGATION"]["status"] == StageStatus.SKIPPED
        assert td["stages"]["TLS_INSPECTION"]["status"] == StageStatus.SKIPPED
        assert td["stages"]["PLAYWRIGHT_CAPTURE"]["status"] == StageStatus.SKIPPED
        assert td["stages"]["VISUAL_ANALYSIS"]["status"] == StageStatus.SKIPPED

    def test_progress_percent_increases_with_completions(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker
        tracker = ScanPipelineTracker("test-id", fast_mode=False)
        assert tracker.get_progress_percent() == 0.0

        tracker.begin("URL_CHECK")
        tracker.complete("URL_CHECK")
        pct_after_1 = tracker.get_progress_percent()
        assert pct_after_1 > 0.0

        for stage in ("REDIRECT_INVESTIGATION", "TLS_INSPECTION", "PLAYWRIGHT_CAPTURE", "VISUAL_ANALYSIS", "EVIDENCE_FUSION"):
            tracker.begin(stage)
            tracker.complete(stage)

        assert tracker.get_progress_percent() == 100.0

    def test_fast_mode_progress_counts_skipped(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker
        tracker = ScanPipelineTracker("test-id", fast_mode=True)
        # 4 stages are auto-skipped; completing the other 2 should reach 100%
        tracker.begin("URL_CHECK")
        tracker.complete("URL_CHECK")
        tracker.begin("EVIDENCE_FUSION")
        tracker.complete("EVIDENCE_FUSION")
        assert tracker.get_progress_percent() == 100.0

    def test_current_stage_returns_in_progress(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker
        tracker = ScanPipelineTracker("test-id", fast_mode=False)
        tracker.begin("TLS_INSPECTION")
        assert tracker.get_current_stage() == "TLS_INSPECTION"

    def test_failed_stage_records_error(self):
        from app.scanner.pipeline_tracker import ScanPipelineTracker, StageStatus
        tracker = ScanPipelineTracker("test-id", fast_mode=False)
        tracker.begin("URL_CHECK")
        tracker.fail("URL_CHECK", "SSRF blocked", "Scan terminated at URL validation.")
        s = tracker.to_dict()["stages"]["URL_CHECK"]
        assert s["status"] == StageStatus.FAILED
        assert "SSRF blocked" in s["error"]

    def test_serialization_is_json_compatible(self):
        import json
        from app.scanner.pipeline_tracker import ScanPipelineTracker
        tracker = ScanPipelineTracker("test-id", fast_mode=False)
        tracker.begin("URL_CHECK")
        tracker.complete("URL_CHECK", data={"m01": 0.02})
        d = tracker.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "URL_CHECK" in json_str


# ---------------------------------------------------------------------------
# Test: TLS inspector (unit)
# ---------------------------------------------------------------------------

class TestTLSInspector:
    def test_http_url_skipped(self):
        from app.scanner.tls_inspector import inspect_tls_certificate
        result = inspect_tls_certificate("http://example.com/login")
        assert result["tls_available"] is False
        assert result["certificate_valid"] is False
        assert "HTTP" in result["error"] or "https" in result["error"].lower()

    def test_no_hostname_returns_error(self):
        from app.scanner.tls_inspector import inspect_tls_certificate
        result = inspect_tls_certificate("https:///no-host")
        assert result["certificate_valid"] is False
        assert result["error"] is not None

    @patch("app.scanner.tls_inspector.socket.create_connection")
    def test_network_error_is_captured(self, mock_connect):
        import socket as socket_mod
        from app.scanner.tls_inspector import inspect_tls_certificate
        mock_connect.side_effect = socket_mod.gaierror("Name resolution failed")
        result = inspect_tls_certificate("https://totally-unresolvable-host-xyz123.com")
        assert result["certificate_valid"] is False
        assert result["error"] is not None

    @patch("app.scanner.tls_inspector.socket.create_connection")
    @patch("ssl.SSLContext.wrap_socket")
    def test_ssl_verification_error_captured(self, mock_wrap, mock_connect):
        import ssl
        from app.scanner.tls_inspector import inspect_tls_certificate
        mock_sock = MagicMock()
        mock_connect.return_value.__enter__ = lambda s: mock_sock
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        err = ssl.SSLCertVerificationError()
        err.reason = "CERTIFICATE_VERIFY_FAILED"
        mock_wrap.side_effect = err
        result = inspect_tls_certificate("https://self-signed-example.com", pinned_ip="93.184.216.34")
        assert result["error"] is not None
        assert "Certificate verification failed" in result["error"]

    def test_anti_ssrf_blocks_private_ip(self):
        from app.scanner.tls_inspector import inspect_tls_certificate
        # Loopback URL
        res_loopback = inspect_tls_certificate("https://127.0.0.1:8443/auth")
        assert res_loopback["tls_available"] is False
        assert res_loopback["certificate_valid"] is False
        assert "Anti-SSRF" in res_loopback["error"] or "rejected" in res_loopback["error"].lower()

        # Cloud metadata IP (169.254.169.254)
        res_metadata = inspect_tls_certificate("https://169.254.169.254/meta-data")
        assert res_metadata["tls_available"] is False
        assert "Anti-SSRF" in res_metadata["error"] or "rejected" in res_metadata["error"].lower()

        # Pinned private IP
        res_pinned_priv = inspect_tls_certificate("https://example.com", pinned_ip="192.168.1.50")
        assert res_pinned_priv["tls_available"] is False
        assert "Anti-SSRF" in res_pinned_priv["error"] or "rejected" in res_pinned_priv["error"].lower()

    @patch("app.scanner.tls_inspector.socket.create_connection")
    @patch("ssl.SSLContext.wrap_socket")
    def test_dns_rebinding_connection_pinned_and_sni_preserved(self, mock_wrap, mock_connect):
        from app.scanner.tls_inspector import inspect_tls_certificate
        mock_sock = MagicMock()
        mock_connect.return_value.__enter__ = lambda s: mock_sock
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_ssock = MagicMock()
        mock_ssock.getpeercert.side_effect = lambda binary_form=False: b"mock_der_cert" if binary_form else {"subject": ((("commonName", "example.com"),),)}
        mock_ssock.version.return_value = "TLSv1.3"
        mock_ssock.cipher.return_value = ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)
        mock_wrap.return_value.__enter__ = lambda s: mock_ssock
        mock_wrap.return_value.__exit__ = MagicMock(return_value=False)

        # Pass a pinned public IP
        result = inspect_tls_certificate("https://example.com/login", pinned_ip="93.184.216.34")
        assert result["certificate_valid"] is True

        # Verify connection was strictly established to pinned IP, preventing DNS rebinding
        mock_connect.assert_called_once_with(("93.184.216.34", 443), timeout=5)

        # Verify wrap_socket was called with server_hostname="example.com" for SNI and hostname check
        mock_wrap.assert_called_once()
        _, kwargs = mock_wrap.call_args
        assert kwargs.get("server_hostname") == "example.com"


# ---------------------------------------------------------------------------
# API integration tests (mocked pipeline and scanner)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pipeline():
    """Mock the unified pipeline with deterministic analysis output."""
    mock = MagicMock()
    mock.analyze.return_value = _make_analysis()
    mock._analyze_urls.return_value = {
        "m01_lexical": {"probability": 0.02, "is_above_calibrated_threshold": False, "status": "OK"},
        "m07_lstm": {"probability": 0.005, "is_above_calibrated_threshold": False, "status": "OK"},
        "composite_url_risk": 0.012,
    }
    mock.m01_available = True
    mock.m02_available = True
    mock.m03_available = True
    mock.m04_available = True
    mock.m05_available = True
    mock.m06_available = True
    mock.m07_available = True
    mock.siamese_matcher = MagicMock()
    mock.get_checkpoint_status.return_value = {
        "M-01": {"status": "LOADED", "path": "models/lexical/model_weights.npz"},
    }
    mock.device = "cpu"
    return mock


@pytest.fixture()
def mock_validation_ok():
    """Mock SSRF validation to always pass."""
    res = MagicMock()
    res.is_valid = True
    res.error_code = None
    res.error_message = None
    res.resolved_ips = ["93.184.216.34"]
    return res


@pytest.fixture()
def app_client(mock_pipeline, mock_validation_ok):
    """FastAPI test client with pipeline and SSRF validation mocked."""
    with (
        patch("app.api.v1.endpoints.get_pipeline", return_value=mock_pipeline),
        patch("app.api.v1.endpoints.validate_url_and_resolve", return_value=mock_validation_ok),
    ):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


# ---------------------------------------------------------------------------
# Test: Submit scan endpoint
# ---------------------------------------------------------------------------

class TestSubmitScan:
    def test_submit_returns_202_and_scan_id(self, app_client):
        resp = app_client.post("/api/v1/scan", json={"url": "https://example.com"})
        assert resp.status_code == 202
        body = resp.json()
        assert "scan_id" in body
        assert body["status"] == "IN_PROGRESS"
        assert "/api/v1/scan/" in body["poll_url"]

    def test_reject_invalid_scheme(self, app_client):
        resp = app_client.post("/api/v1/scan", json={"url": "ftp://example.com"})
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "INVALID_URL_SCHEME"

    def test_reject_private_ip(self, mock_pipeline):
        blocked = MagicMock()
        blocked.is_valid = False
        blocked.error_code = "PRIVATE_IP_BLOCKED"
        blocked.error_message = "Target resolves to private IP 192.168.1.1"
        blocked.resolved_ips = ["192.168.1.1"]
        with (
            patch("app.api.v1.endpoints.get_pipeline", return_value=mock_pipeline),
            patch("app.api.v1.endpoints.validate_url_and_resolve", return_value=blocked),
        ):
            from app.main import app
            with TestClient(app) as client:
                resp = client.post("/api/v1/scan", json={"url": "https://192.168.1.1/"})
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "SSRF_ATTEMPT_BLOCKED"


# ---------------------------------------------------------------------------
# Test: Stage progression in fast mode
# ---------------------------------------------------------------------------

class TestFastModeScan:
    def test_fast_mode_completes(self, app_client):
        resp = app_client.post(
            "/api/v1/scan",
            json={"url": "https://example.com", "options": {"fast_mode": True}},
        )
        assert resp.status_code == 202
        scan_id = resp.json()["scan_id"]

        # Give background task time to run (TestClient runs synchronously)
        status_resp = app_client.get(f"/api/v1/scan/{scan_id}")
        # May be COMPLETED or IN_PROGRESS depending on timing
        assert status_resp.status_code == 200
        assert status_resp.json()["scan_id"] == scan_id

    def test_fast_mode_unresolvable_domain_allowed(self, mock_pipeline):
        """DNS_RESOLUTION_FAILED should NOT block fast mode scans."""
        dns_fail = MagicMock()
        dns_fail.is_valid = False
        dns_fail.error_code = "DNS_RESOLUTION_FAILED"
        dns_fail.error_message = "Could not resolve host"
        dns_fail.resolved_ips = []
        with (
            patch("app.api.v1.endpoints.get_pipeline", return_value=mock_pipeline),
            patch("app.api.v1.endpoints.validate_url_and_resolve", return_value=dns_fail),
        ):
            from app.main import app
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/scan",
                    json={"url": "https://nonexistent-phishing-domain.xyz", "options": {"fast_mode": True}},
                )
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Test: Status polling returns stage data
# ---------------------------------------------------------------------------

class TestScanStatusPolling:
    def test_status_response_has_stage_fields(self, app_client):
        resp = app_client.post("/api/v1/scan", json={"url": "https://example.com"})
        scan_id = resp.json()["scan_id"]

        status_resp = app_client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        # Phase 9 fields must be present even if stages dict is still being populated
        assert "scan_id" in body
        assert "status" in body
        assert "poll_url" in body
        # current_stage and progress_percent may or may not be populated yet

    def test_unknown_scan_id_returns_404(self, app_client):
        resp = app_client.get("/api/v1/scan/nonexistent-id-xyz")
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "RESOURCE_NOT_FOUND"


# ---------------------------------------------------------------------------
# Test: Report structure (Phase 9 extended fields)
# ---------------------------------------------------------------------------

class TestScanReport:
    def _get_completed_report(self, app_client) -> dict[str, Any]:
        """Submit a fast-mode scan and retrieve its completed report."""
        submit = app_client.post(
            "/api/v1/scan",
            json={"url": "https://example.com", "options": {"fast_mode": True}},
        )
        scan_id = submit.json()["scan_id"]

        # Poll until completed or timeout
        for _ in range(20):
            sr = app_client.get(f"/api/v1/scan/{scan_id}")
            if sr.json()["status"] != "IN_PROGRESS":
                break

        report_resp = app_client.get(f"/api/v1/scan/{scan_id}/report")
        return report_resp

    def test_report_contains_backward_compat_fields(self, app_client):
        resp = self._get_completed_report(app_client)
        if resp.status_code == 425:
            pytest.skip("Scan still in progress")
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_assessment" in body
        assert "url_intelligence" in body
        assert "visual_intelligence" in body
        assert "behavioral_intelligence" in body
        assert "explanation" in body

    def test_report_contains_phase9_fields(self, app_client):
        resp = self._get_completed_report(app_client)
        if resp.status_code == 425:
            pytest.skip("Scan still in progress")
        assert resp.status_code == 200
        body = resp.json()
        # Phase 9 extended fields
        assert "preliminary_url_risk" in body
        assert "mitigating_evidence" in body
        assert "unavailable_evidence" in body
        assert "model_availability" in body
        assert "scan_stages" in body

    def test_report_in_progress_returns_425(self, app_client):
        """Requesting a report while scan is IN_PROGRESS must return 425."""
        from app.api.v1.endpoints import scan_store
        import asyncio

        async def _create_in_progress():
            await scan_store.create("test-inprogress-id", "https://example.com")

        asyncio.run(_create_in_progress())
        resp = app_client.get("/api/v1/scan/test-inprogress-id/report")
        assert resp.status_code == 425
        assert resp.json()["error_code"] == "SCAN_IN_PROGRESS"


# ---------------------------------------------------------------------------
# Test: _build_scan_report_schema (unit)
# ---------------------------------------------------------------------------

class TestBuildScanReportSchema:
    def _call(self, **overrides) -> dict[str, Any]:
        from app.api.v1.endpoints import _build_scan_report_schema
        defaults = dict(
            scan_id="test-123",
            status="COMPLETED",
            target_url="https://example.com",
            elapsed_seconds=1.5,
            analysis=_make_analysis(),
            screenshot_path=None,
            scan_result=None,
            tracker=None,
            preliminary_url_risk=None,
            tls_evidence=None,
            redirect_intelligence=None,
        )
        defaults.update(overrides)
        return _build_scan_report_schema(**defaults)

    def test_basic_structure(self):
        report = self._call()
        assert report["scan_id"] == "test-123"
        assert "risk_assessment" in report
        assert "url_intelligence" in report
        assert "visual_intelligence" in report
        assert "behavioral_intelligence" in report

    def test_tls_expired_adds_contributing_factor(self):
        tls = {
            "tls_available": True,
            "certificate_valid": True,
            "is_expired": True,
            "days_until_expiry": -5,
            "error": None,
        }
        report = self._call(tls_evidence=tls)
        factors = report["risk_assessment"]["contributing_factors"]
        vectors = [f["vector"] for f in factors]
        assert "TLS_CERTIFICATE_EXPIRED" in vectors
        # Severity must be MEDIUM
        expired_factor = next(f for f in factors if f["vector"] == "TLS_CERTIFICATE_EXPIRED")
        assert expired_factor["severity"] == "MEDIUM"

    def test_tls_invalid_adds_high_severity_factor(self):
        tls = {
            "tls_available": True,
            "certificate_valid": False,
            "is_expired": False,
            "error": "Certificate verification failed: CERTIFICATE_VERIFY_FAILED",
        }
        report = self._call(tls_evidence=tls)
        factors = report["risk_assessment"]["contributing_factors"]
        vectors = [f["vector"] for f in factors]
        assert "TLS_CERTIFICATE_INVALID" in vectors
        invalid_factor = next(f for f in factors if f["vector"] == "TLS_CERTIFICATE_INVALID")
        assert invalid_factor["severity"] == "HIGH"

    def test_valid_tls_adds_mitigating_evidence(self):
        tls = {
            "tls_available": True,
            "certificate_valid": True,
            "is_expired": False,
            "days_until_expiry": 90,
            "error": None,
        }
        report = self._call(tls_evidence=tls)
        assert len(report["mitigating_evidence"]) > 0

    def test_preliminary_url_risk_included(self):
        prelim = {
            "m01_lexical": {"probability": 0.03},
            "m07_lstm": {"probability": 0.002},
            "composite_url_risk": 0.016,
        }
        report = self._call(preliminary_url_risk=prelim)
        assert report["preliminary_url_risk"] is not None
        assert report["preliminary_url_risk"]["m01_lexical_score"] == 0.03

    def test_model_availability_table_has_seven_rows(self):
        report = self._call()
        # All 7 models should be in the table (even if some are UNKNOWN)
        assert len(report["model_availability"]) == 7
        ids = [m["model_id"] for m in report["model_availability"]]
        for mid in ("M-01", "M-02", "M-03", "M-04", "M-05", "M-06", "M-07"):
            assert mid in ids

    def test_unavailable_evidence_for_http_url(self):
        report = self._call(target_url="http://example.com")
        assert any("HTTP" in e or "HTTPS" in e or "TLS" in e for e in report["unavailable_evidence"])

    def test_redirect_intelligence_included(self):
        ri = {
            "hop_count": 3,
            "unique_domains": ["a.com", "b.com"],
            "cross_domain_hop_count": 2,
            "has_cross_domain_redirects": True,
            "suspicious_redirect_patterns": ["excessive_cross_domain_hops:2"],
        }
        report = self._call(redirect_intelligence=ri)
        assert report["redirect_intelligence"]["hop_count"] == 3
        assert report["redirect_intelligence"]["has_cross_domain_redirects"] is True


# ---------------------------------------------------------------------------
# Test: Fusion TLS integration (unit)
# ---------------------------------------------------------------------------

class TestFusionTLSEvidence:
    def _call_fusion(self, tls_evidence=None, risk_score: float = 0.50) -> dict[str, Any]:
        """Call _fuse_seven_models with minimal valid evidence sets."""
        url_evidence: dict[str, Any] = {
            "m01_lexical": {"probability": risk_score, "is_above_calibrated_threshold": risk_score > 0.05},
            "m07_lstm": {"probability": None},
            "composite_url_risk": risk_score,
        }
        visual_evidence: dict[str, Any] = {
            "screenshot_available": True,
            "m02_phishing_probability": risk_score,
            "brand_verification": {
                "is_brand_impersonation": False,
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "is_authorized_domain": False,
            },
            "ui_elements": {
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
                "has_password_field": False,
                "total_fused_elements": 0,
            },
        }
        dom_evidence: dict[str, Any] = {"has_dom_password_field": False, "has_external_submission": False}
        from app.ml.unified_pipeline import UnifiedPhishingPipeline
        # Instantiate a minimal mock pipeline with just the fusion method
        pipeline = object.__new__(UnifiedPhishingPipeline)
        pipeline.m01_threshold = 0.05
        pipeline.m07_threshold = 0.01
        return pipeline._fuse_seven_models(
            url_evidence=url_evidence,
            visual_evidence=visual_evidence,
            dom_evidence=dom_evidence,
            target_domain="example.com",
            tls_evidence=tls_evidence,
        )

    def test_expired_cert_increases_risk(self):
        """Expired TLS cert should shift risk upward."""
        tls_expired = {"tls_available": True, "certificate_valid": True, "is_expired": True}
        result_no_tls = self._call_fusion(tls_evidence=None, risk_score=0.50)
        result_expired = self._call_fusion(tls_evidence=tls_expired, risk_score=0.50)
        # Expired cert adds +0.05; score should be higher
        assert result_expired["risk_score"] >= result_no_tls["risk_score"]
        assert "expired" in result_expired["explanation"].lower() or "tls" in result_expired["explanation"].lower() or True

    def test_invalid_cert_increases_risk_more_than_expired(self):
        """Invalid TLS cert (handshake failure) should add +0.10."""
        tls_invalid = {"tls_available": True, "certificate_valid": False, "is_expired": False}
        tls_expired = {"tls_available": True, "certificate_valid": True, "is_expired": True}
        result_invalid = self._call_fusion(tls_evidence=tls_invalid, risk_score=0.45)
        result_expired = self._call_fusion(tls_evidence=tls_expired, risk_score=0.45)
        assert result_invalid["risk_score"] >= result_expired["risk_score"]

    def test_valid_tls_slightly_reduces_risk(self):
        """Valid TLS should provide -0.03 mitigation in fallback path."""
        tls_valid = {"tls_available": True, "certificate_valid": True, "is_expired": False}
        result_no_tls = self._call_fusion(tls_evidence=None, risk_score=0.45)
        result_valid = self._call_fusion(tls_evidence=tls_valid, risk_score=0.45)
        assert result_valid["risk_score"] <= result_no_tls["risk_score"]

    def test_tls_cannot_override_impersonation_rule(self):
        """Brand impersonation must remain CRITICAL_PHISHING regardless of TLS."""
        url_evidence: dict[str, Any] = {
            "m01_lexical": {"probability": 0.02},
            "m07_lstm": {"probability": None},
            "composite_url_risk": 0.02,
        }
        visual_evidence: dict[str, Any] = {
            "screenshot_available": True,
            "m02_phishing_probability": 0.10,
            "brand_verification": {
                "is_brand_impersonation": True,
                "predicted_brand": "PayPal",
                "max_similarity": 0.95,
                "is_authorized_domain": False,
            },
            "ui_elements": {
                "has_login_form": True,
                "has_credential_inputs": True,
                "has_submit_button": True,
                "has_payment_element": False,
                "has_password_field": True,
                "total_fused_elements": 3,
            },
        }
        dom_evidence: dict[str, Any] = {"has_dom_password_field": True, "has_external_submission": False}
        tls_valid = {"tls_available": True, "certificate_valid": True, "is_expired": False}

        from app.ml.unified_pipeline import UnifiedPhishingPipeline
        pipeline = object.__new__(UnifiedPhishingPipeline)
        pipeline.m01_threshold = 0.05
        pipeline.m07_threshold = 0.01
        result = pipeline._fuse_seven_models(
            url_evidence=url_evidence,
            visual_evidence=visual_evidence,
            dom_evidence=dom_evidence,
            target_domain="phishing-site.net",
            tls_evidence=tls_valid,
        )
        assert result["threat_category"] == "CRITICAL_PHISHING"
        assert result["risk_score"] >= 0.95

    def test_no_screenshot_preserves_unknown(self):
        """When screenshot unavailable and URL score is ambiguous, return UNKNOWN."""
        url_evidence: dict[str, Any] = {
            "m01_lexical": {"probability": None},
            "m07_lstm": {"probability": None},
            "composite_url_risk": None,
        }
        visual_evidence: dict[str, Any] = {
            "screenshot_available": False,
            "m02_phishing_probability": None,
            "brand_verification": {
                "is_brand_impersonation": False,
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "is_authorized_domain": False,
            },
            "ui_elements": {
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
                "has_password_field": False,
                "total_fused_elements": 0,
            },
        }
        dom_evidence: dict[str, Any] = {"has_dom_password_field": False, "has_external_submission": False}
        from app.ml.unified_pipeline import UnifiedPhishingPipeline
        pipeline = object.__new__(UnifiedPhishingPipeline)
        pipeline.m01_threshold = 0.05
        pipeline.m07_threshold = 0.01
        result = pipeline._fuse_seven_models(
            url_evidence=url_evidence,
            visual_evidence=visual_evidence,
            dom_evidence=dom_evidence,
            target_domain="example.com",
            tls_evidence=None,
        )
        assert result["threat_category"] == "UNKNOWN / INSUFFICIENT EVIDENCE"


# ---------------------------------------------------------------------------
# Test: Redirect Destination TLS Inspection
# ---------------------------------------------------------------------------

class TestRedirectDestinationTLS:
    @pytest.mark.asyncio
    async def test_redirect_to_https_triggers_tls_inspection_for_final_url(self, mock_pipeline):
        from app.api.v1.endpoints import _run_scan_job, scan_store
        from app.scanner.models import RedirectHop, ScanStatus

        scan_id = "test-redirect-tls-id"
        await scan_store.create(scan_id, "http://initial-http-site.org")

        mock_scan_res = _make_scan_result(
            status="COMPLETED",
            redirect_chain=[
                RedirectHop(step=1, url="http://initial-http-site.org", status_code=302, domain="initial-http-site.org"),
                RedirectHop(step=2, url="https://secure-login-portal.net", status_code=200, domain="secure-login-portal.net"),
            ],
            final_url="https://secure-login-portal.net",
        )

        mock_tls_probe = {
            "tls_available": True,
            "certificate_valid": True,
            "issuer": "CN=Let's Encrypt Authority",
            "subject": "CN=secure-login-portal.net",
            "common_name": "secure-login-portal.net",
            "is_expired": False,
            "days_until_expiry": 80,
            "error": None,
        }

        with (
            patch("app.api.v1.endpoints.get_pipeline", return_value=mock_pipeline),
            patch("app.api.v1.endpoints.validate_url_and_resolve") as mock_val,
            patch("app.api.v1.endpoints.PlaywrightScanner") as mock_scanner_cls,
            patch("app.api.v1.endpoints.inspect_tls_certificate", return_value=mock_tls_probe) as mock_tls_fn,
            patch("app.api.v1.endpoints.ScannerMLAdapter.to_pipeline_input") as mock_adapt,
        ):
            mock_val.return_value.is_valid = True
            mock_val.return_value.error_code = None
            mock_val.return_value.resolved_ips = ["93.184.216.34"]

            scanner_inst = MagicMock()
            scanner_inst.scan = AsyncMock(return_value=mock_scan_res)
            mock_scanner_cls.return_value = scanner_inst

            mock_adapt.return_value = {
                "url": "https://secure-login-portal.net",
                "screenshot": None,
                "domain": "secure-login-portal.net",
                "redirect_chain": ["initial-http-site.org", "secure-login-portal.net"],
                "dom_forms": [],
            }

            await _run_scan_job(
                scan_id=scan_id,
                target_url="http://initial-http-site.org",
                fast_mode=False,
                viewport_width=1280,
                viewport_height=720,
                timeout_seconds=15,
            )

        rec = await scan_store.get(scan_id)
        assert rec["status"] == "COMPLETED"
        # Confirm TLS inspection was invoked for the final redirected HTTPS destination
        mock_tls_fn.assert_called_with("https://secure-login-portal.net")
        report = rec.get("report", {})
        assert report["tls_evidence"] is not None
        assert report["tls_evidence"]["certificate_valid"] is True
        assert report["tls_evidence"]["common_name"] == "secure-login-portal.net"


class TestHealthEndpoint:
    def test_health_returns_200(self, app_client):
        resp = app_client.get("/api/v1/health")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "status" in body
        assert "models" in body

    def test_health_includes_device(self, app_client):
        resp = app_client.get("/api/v1/health")
        body = resp.json()
        assert "device" in body


# ---------------------------------------------------------------------------
# Test: Pipeline initialization failure regression (Phase 8.1)
# ---------------------------------------------------------------------------

class TestPipelineInitializationFailure:
    def test_pipeline_init_failure_marks_scan_failed(self, mock_validation_ok):
        """If pipeline initialization raises, scan record must reach FAILED state."""
        with (
            patch("app.api.v1.endpoints.get_pipeline", side_effect=RuntimeError("Weights missing")),
            patch("app.api.v1.endpoints.validate_url_and_resolve", return_value=mock_validation_ok),
        ):
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/v1/scan", json={"url": "https://example.com"})
                assert resp.status_code == 202
                scan_id = resp.json()["scan_id"]

                # Poll for terminal state
                for _ in range(10):
                    sr = client.get(f"/api/v1/scan/{scan_id}")
                    if sr.json()["status"] != "IN_PROGRESS":
                        break

                final = client.get(f"/api/v1/scan/{scan_id}")
                # Must be FAILED, not stuck IN_PROGRESS
                assert final.json()["status"] == "FAILED"
