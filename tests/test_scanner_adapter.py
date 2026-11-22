"""Unit Tests for Scanner-to-ML Adapter (app/scanner/adapter.py).

Verifies conversion of ScanResult, RedirectHop, and FormInfo objects into
the exact input types expected by UnifiedPhishingPipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.scanner.adapter import ScannerMLAdapter
from app.scanner.models import FormField, FormInfo, RedirectHop, ScanResult, ScanStatus


class TestScannerMLAdapter:
    """Tests for ScannerMLAdapter data conversion and error handling."""

    def test_extract_redirect_hostnames(self) -> None:
        """Verify chronological extraction of registrable domains from RedirectHop objects."""
        hops = [
            RedirectHop(step=1, url="http://bit.ly/3xY8a", status_code=301, domain="bit.ly"),
            RedirectHop(step=2, url="https://redirect.phish-tracker.com/gate", status_code=302, domain="phish-tracker.com"),
            RedirectHop(step=3, url="https://secure-login.portal.net/login", status_code=200, domain="portal.net"),
        ]
        result = ScannerMLAdapter.extract_redirect_hostnames(hops)
        assert result == ["bit.ly", "phish-tracker.com", "portal.net"]

    def test_extract_redirect_hostnames_fallback_to_url(self) -> None:
        """Verify domain extraction from URL when hop.domain is empty."""
        hops = [
            RedirectHop(step=1, url="https://login.example.org/path", status_code=200, domain=""),
        ]
        result = ScannerMLAdapter.extract_redirect_hostnames(hops)
        assert result == ["example.org"]

    def test_extract_redirect_hostnames_empty(self) -> None:
        """Verify empty input returns empty list."""
        assert ScannerMLAdapter.extract_redirect_hostnames(None) == []
        assert ScannerMLAdapter.extract_redirect_hostnames([]) == []

    def test_extract_dom_forms(self) -> None:
        """Verify conversion of FormInfo models to DOM dictionaries."""
        forms = [
            FormInfo(
                form_id="login-form",
                action_url="https://attacker.com/collect.php",
                action_domain="attacker.com",
                method="POST",
                is_external_submission=True,
                has_password_field=True,
                fields=[
                    FormField(name="user", field_type="text"),
                    FormField(name="pass", field_type="password", is_password=True),
                ],
            ),
            FormInfo(
                form_id="search-form",
                action_url="/search",
                action_domain="example.com",
                method="GET",
                is_external_submission=False,
                has_password_field=False,
            ),
        ]

        dom_forms = ScannerMLAdapter.extract_dom_forms(forms)
        assert len(dom_forms) == 2

        f1 = dom_forms[0]
        assert f1["form_id"] == "login-form"
        assert f1["action_domain"] == "attacker.com"
        assert f1["is_external_submission"] is True
        assert f1["has_password_field"] is True
        assert f1["field_count"] == 2

        f2 = dom_forms[1]
        assert f2["has_password_field"] is False
        assert f2["is_external_submission"] is False

    def test_to_pipeline_input_successful_scan(self) -> None:
        """Verify complete conversion of a successful ScanResult."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            scan_res = ScanResult(
                scan_id="test-scan-uuid-1234",
                requested_url="http://short.ly/login",
                final_url="https://secure.target-site.com/auth/login",
                status=ScanStatus.COMPLETED,
                status_code=200,
                viewport_screenshot_path=str(tmp_path),
                redirect_chain=[
                    RedirectHop(step=1, url="http://short.ly/login", status_code=301, domain="short.ly"),
                ],
                forms=[
                    FormInfo(form_id="login", action_url="/post", action_domain="target-site.com", has_password_field=True),
                ],
            )

            pipe_input = ScannerMLAdapter.to_pipeline_input(scan_res)

            assert pipe_input["url"] == "https://secure.target-site.com/auth/login"
            assert pipe_input["domain"] == "target-site.com"
            assert pipe_input["screenshot"] == str(tmp_path)
            assert pipe_input["sample_id"] == "test-scan-uuid-1234"
            assert pipe_input["redirect_chain"] == ["short.ly"]
            assert len(pipe_input["dom_forms"]) == 1
            assert pipe_input["dom_forms"][0]["has_password_field"] is True
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_to_pipeline_input_missing_screenshot(self) -> None:
        """Verify graceful fallback when screenshot does not exist on disk."""
        scan_res = ScanResult(
            scan_id="test-scan-uuid-5678",
            requested_url="https://legitimate.org",
            final_url="https://legitimate.org",
            status=ScanStatus.COMPLETED,
            viewport_screenshot_path="/nonexistent/path/shot.png",
        )

        pipe_input = ScannerMLAdapter.to_pipeline_input(scan_res)
        assert pipe_input["screenshot"] is None
        assert pipe_input["domain"] == "legitimate.org"

    def test_build_blocked_or_failed_report_ssrf(self) -> None:
        """Verify building a blocked SSRF fallback report."""
        scan_res = ScanResult(
            scan_id="ssrf-blocked-001",
            requested_url="http://169.254.169.254/latest/meta-data",
            final_url="http://169.254.169.254/latest/meta-data",
            status=ScanStatus.BLOCKED_SSRF,
            errors=["Cloud metadata address blocked"],
        )

        rep = ScannerMLAdapter.build_blocked_or_failed_report(scan_res)
        assert rep["threat_category"] == "BLOCKED_SSRF"
        assert rep["risk_score"] == 1.0
        assert rep["status"] == "BLOCKED_SSRF"
        assert rep["scan_id"] == "ssrf-blocked-001"
        assert len(rep["contributing_factors"]) == 1
        assert rep["contributing_factors"][0]["vector"] == "NETWORK_SECURITY"
