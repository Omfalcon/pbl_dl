"""Integration and Unit Tests for PhishVision AI FastAPI REST API.

Tests API routes according to API_CONTRACTS.md:
- POST /api/v1/scan (validation, Anti-SSRF, execution)
- GET  /api/v1/scan/{scan_id}
- GET  /api/v1/scan/{scan_id}/report
- GET  /api/v1/scan/{scan_id}/screenshot
- GET  /api/v1/brands
- GET  /api/v1/health
- POST /api/v1/analyze/url
"""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate FastAPI TestClient."""
    return TestClient(app)


class TestPhishVisionAPI:
    """Test suite for PhishVision AI REST API endpoints."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Verify root discovery endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "PhishVision AI"
        assert data["status"] == "OPERATIONAL"
        assert "/api/v1/health" in data["health"]

    def test_health_endpoint(self, client: TestClient) -> None:
        """Verify /api/v1/health returns explicit per-model checkpoint status."""
        response = client.get("/api/v1/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "models" in data
        assert "M-01_lexical_mlp" in data["models"]
        assert "M-07_lstm" in data["models"]
        assert "status" in data
        # Models should have weights_exist and status keys
        for m_key, m_info in data["models"].items():
            assert "weights_exist" in m_info
            assert "status" in m_info

    def test_brands_endpoint(self, client: TestClient) -> None:
        """Verify /api/v1/brands returns monitored reference brands."""
        response = client.get("/api/v1/brands")
        assert response.status_code == 200
        data = response.json()
        assert "brands" in data
        assert data["total_brands"] >= 0
        if data["total_brands"] > 0:
            brand_ids = [b["brand_id"] for b in data["brands"]]
            assert "sbi" in brand_ids or "google" in brand_ids or "paypal" in brand_ids

    def test_scan_invalid_scheme_blocked(self, client: TestClient) -> None:
        """Verify URLs with non-HTTP/HTTPS schemes are rejected with 400 Bad Request."""
        payload = {"url": "ftp://attacker.com/payload.exe"}
        response = client.post("/api/v1/scan", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "INVALID_URL_SCHEME"

    def test_scan_anti_ssrf_blocks_private_ip(self, client: TestClient) -> None:
        """Verify Anti-SSRF preflight blocks private and loopback destinations immediately."""
        private_urls = [
            "http://127.0.0.1/admin",
            "http://localhost:8080/internal",
            "http://192.168.1.1/router",
            "http://10.0.0.1/secret",
            "http://169.254.169.254/latest/meta-data/",
        ]
        for target in private_urls:
            response = client.post("/api/v1/scan", json={"url": target})
            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == "SSRF_ATTEMPT_BLOCKED"

    def test_scan_fast_mode_submission_and_report(self, client: TestClient) -> None:
        """Verify fast-mode scan submission, non-blocking execution, and report retrieval."""
        target_url = "https://legitimate-knowledge-portal.org/article"
        submit_res = client.post(
            "/api/v1/scan",
            json={
                "url": target_url,
                "options": {"fast_mode": True},
            },
        )
        assert submit_res.status_code == 202
        submit_data = submit_res.json()
        assert "scan_id" in submit_data
        scan_id = submit_data["scan_id"]
        assert submit_data["status"] == "IN_PROGRESS"

        # Poll status until completed (should be fast in fast_mode)
        completed = False
        for _ in range(30):
            status_res = client.get(f"/api/v1/scan/{scan_id}")
            assert status_res.status_code == 200
            s_data = status_res.json()
            if s_data["status"] == "COMPLETED":
                completed = True
                break
            time.sleep(0.2)

        assert completed, "Fast mode scan did not complete in time"

        # Retrieve report
        report_res = client.get(f"/api/v1/scan/{scan_id}/report")
        assert report_res.status_code == 200
        report = report_res.json()
        assert report["scan_id"] == scan_id
        assert "risk_assessment" in report
        assert "category" in report["risk_assessment"]
        assert "risk_score" in report["risk_assessment"]
        assert "url_intelligence" in report

        # Verify no local filesystem paths were leaked in screenshot_url or report
        vis = report.get("visual_intelligence", {})
        if vis.get("screenshot_url"):
            assert ":\\" not in vis["screenshot_url"]
            assert vis["screenshot_url"].startswith("/api/v1/scan/")

    def test_nonexistent_scan_id_returns_404(self, client: TestClient) -> None:
        """Verify querying unknown scan_id returns RFC 7807 problem details 404."""
        response = client.get("/api/v1/scan/unknown-uuid-00000")
        assert response.status_code == 404
        assert response.json()["error_code"] == "RESOURCE_NOT_FOUND"

        report_response = client.get("/api/v1/scan/unknown-uuid-00000/report")
        assert report_response.status_code == 404
        assert report_response.json()["error_code"] == "RESOURCE_NOT_FOUND"

    def test_analyze_url_fast_path(self, client: TestClient) -> None:
        """Verify synchronous /api/v1/analyze/url fast intelligence path."""
        target_url = "https://secure-update.titanvault-portal.net/login?token=abc"
        response = client.post("/api/v1/analyze/url", json={"url": target_url})
        assert response.status_code == 200
        data = response.json()
        assert data["target_url"] == target_url
        assert data["registrable_domain"] == "titanvault-portal.net"
        assert "url_intelligence" in data

    def test_scan_lifecycle_initialization_failure_reaches_terminal_failed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that pipeline initialization failure transitions scan to terminal FAILED state."""
        import app.api.v1.endpoints as endpoints_mod

        def fail_get_pipeline():
            raise RuntimeError("Simulated pipeline initialization error")

        monkeypatch.setattr(endpoints_mod, "get_pipeline", fail_get_pipeline)

        submit_res = client.post(
            "/api/v1/scan",
            json={"url": "https://example.com/test", "options": {"fast_mode": True}},
        )
        assert submit_res.status_code == 202
        scan_id = submit_res.json()["scan_id"]

        # Poll until terminal state
        terminal_status = None
        for _ in range(30):
            status_res = client.get(f"/api/v1/scan/{scan_id}")
            assert status_res.status_code == 200
            s_val = status_res.json()["status"]
            if s_val in ("FAILED", "COMPLETED", "BLOCKED_SSRF"):
                terminal_status = s_val
                break
            time.sleep(0.1)

        assert terminal_status == "FAILED"

        # Verify report returns 500 indicating the recorded failure
        report_res = client.get(f"/api/v1/scan/{scan_id}/report")
        assert report_res.status_code == 500
        assert "Simulated pipeline initialization error" in report_res.json()["message"]

    def test_cors_resolution_security(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify CORS resolution enforces safe defaults and prevents wildcard credentials."""
        from app.main import resolve_cors_configuration

        # Default configuration
        monkeypatch.delenv("PHISHVISION_CORS_ORIGINS", raising=False)
        origins, allow_creds = resolve_cors_configuration()
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
        assert allow_creds is True
        assert "*" not in origins

        # Wildcard should automatically disable credentials
        monkeypatch.setenv("PHISHVISION_CORS_ORIGINS", "*")
        origins, allow_creds = resolve_cors_configuration()
        assert origins == ["*"]
        assert allow_creds is False

        # Custom explicit origins
        monkeypatch.setenv("PHISHVISION_CORS_ORIGINS", "https://app.phishvision.local,https://portal.phishvision.local")
        origins, allow_creds = resolve_cors_configuration()
        assert origins == ["https://app.phishvision.local", "https://portal.phishvision.local"]
        assert allow_creds is True

    def test_image_upload_security_protections(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify upload size limits, format validation, and malformed payload protections."""
        import io
        from PIL import Image

        # 1. Empty file
        empty_upload = {"image": ("empty.png", b"", "image/png")}
        res = client.post("/api/v1/analyze/brand-similarity", files=empty_upload)
        assert res.status_code == 400
        assert res.json()["error_code"] == "EMPTY_PAYLOAD"

        # 2. Oversized payload (simulate 1 KB limit)
        monkeypatch.setenv("PHISHVISION_MAX_UPLOAD_BYTES", "1024")
        oversized_data = b"X" * 2048
        res = client.post(
            "/api/v1/analyze/brand-similarity",
            files={"image": ("big.png", oversized_data, "image/png")},
        )
        assert res.status_code == 413
        assert res.json()["error_code"] == "PAYLOAD_TOO_LARGE"

        # Reset upload limit
        monkeypatch.delenv("PHISHVISION_MAX_UPLOAD_BYTES", raising=False)

        # 3. Malformed image bytes
        malformed_bytes = b"not-a-real-png-or-jpeg-image-bytes"
        res = client.post(
            "/api/v1/analyze/brand-similarity",
            files={"image": ("fake.png", malformed_bytes, "image/png")},
        )
        assert res.status_code == 400
        assert res.json()["error_code"] == "INVALID_IMAGE_PAYLOAD"

        # 4. Valid image format and dimensions (small 32x32 PNG)
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        valid_bytes = buf.getvalue()

        res = client.post(
            "/api/v1/analyze/brand-similarity",
            files={"image": ("logo.png", valid_bytes, "image/png")},
            data={"target_domain": "google.com"},
        )
        # Should either succeed (200) or report M-03 not loaded (503), but not 400 or 413
        assert res.status_code in (200, 503)

    def test_health_check_status_levels(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify health check returns OPERATIONAL (7/7), DEGRADED (1-6), and SERVICE_UNAVAILABLE (0)."""
        import app.api.v1.endpoints as endpoints_mod

        class MockPipeline:
            device = "cpu"
            def __init__(self, loaded_count: int):
                self.loaded_count = loaded_count
            def get_checkpoint_status(self):
                models = {}
                for i in range(7):
                    key = f"M-0{i+1}"
                    status_val = "LOADED" if i < self.loaded_count else "MISSING"
                    models[key] = {"status": status_val, "weights_exist": i < self.loaded_count}
                return models

        # All 7 loaded -> OPERATIONAL (200)
        monkeypatch.setattr(endpoints_mod, "get_pipeline", lambda: MockPipeline(7))
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        assert res.json()["status"] == "OPERATIONAL"

        # 4 loaded -> DEGRADED (200)
        monkeypatch.setattr(endpoints_mod, "get_pipeline", lambda: MockPipeline(4))
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        assert res.json()["status"] == "DEGRADED"

        # 0 loaded -> SERVICE_UNAVAILABLE (503)
        monkeypatch.setattr(endpoints_mod, "get_pipeline", lambda: MockPipeline(0))
        res = client.get("/api/v1/health")
        assert res.status_code == 503
        assert res.json()["status"] == "SERVICE_UNAVAILABLE"

        # Pipeline init throws -> SERVICE_UNAVAILABLE (503)
        def broken_pipeline():
            raise RuntimeError("Fatal hardware failure")
        monkeypatch.setattr(endpoints_mod, "get_pipeline", broken_pipeline)
        res = client.get("/api/v1/health")
        assert res.status_code == 503
        assert res.json()["status"] == "SERVICE_UNAVAILABLE"
