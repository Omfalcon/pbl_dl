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
