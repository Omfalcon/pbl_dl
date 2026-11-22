"""Unit tests for Anti-SSRF URL validation and DNS resolution."""

import pytest
from app.scanner.anti_ssrf import (
    SSRFValidationError,
    is_ip_disallowed,
    validate_redirect_target,
    validate_url_and_resolve,
)
import ipaddress


class TestAntiSSRFValidation:
    """Test suite verifying strict Anti-SSRF boundaries and IP filtering."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "http://example.com/login",
            "https://example.com:8443/test?q=1#sec",
            "http://example.org:8080/path",
        ],
    )
    def test_valid_public_urls(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is True
        assert result.error_code is None
        assert len(result.resolved_ips) > 0
        assert result.pinned_ip is not None
        assert result.canonical_url.startswith(("http://", "https://"))

    @pytest.mark.parametrize(
        "url,expected_code",
        [
            ("ftp://example.com/file.txt", "INVALID_SCHEME"),
            ("file:///etc/passwd", "INVALID_SCHEME"),
            ("javascript:alert(1)", "INVALID_SCHEME"),
            ("data:text/html,<h1>Test</h1>", "INVALID_SCHEME"),
            ("gopher://127.0.0.1/", "INVALID_SCHEME"),
            ("dict://127.0.0.1:11211/stat", "INVALID_SCHEME"),
            ("", "INVALID_URL_EMPTY"),
            ("   ", "INVALID_URL_EMPTY"),
        ],
    )
    def test_invalid_scheme_rejection(self, url: str, expected_code: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == expected_code

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:80",
            "http://127.0.1.1",
            "http://[::1]",
            "http://ip6-localhost",
            "http://app.local",
            "http://service.internal",
            "http://admin.corp",
            "http://router.lan",
        ],
    )
    def test_localhost_rejection(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code in (
            "BLOCKED_HOSTNAME",
            "BLOCKED_DOMAIN_SUFFIX",
            "BLOCKED_IP_LITERAL",
            "SSRF_IP_BLOCKED",
            "DNS_RESOLUTION_FAILED",
        )

    @pytest.mark.parametrize(
        "ip_str",
        [
            "10.0.0.1",
            "10.254.254.254",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.1.254",
        ],
    )
    def test_private_ipv4_rejection(self, ip_str: str):
        url = f"http://{ip_str}/"
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == "BLOCKED_IP_LITERAL"

    @pytest.mark.parametrize(
        "ip_str",
        [
            "fc00::1",
            "fd00::1",
            "fd12:3456:789a:1::1",
        ],
    )
    def test_private_ipv6_rejection(self, ip_str: str):
        url = f"http://[{ip_str}]/"
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == "BLOCKED_IP_LITERAL"

    @pytest.mark.parametrize(
        "ip_str",
        [
            "169.254.1.1",
            "169.254.169.254",
            "fe80::1",
            "fe80::dead:beef",
        ],
    )
    def test_link_local_rejection(self, ip_str: str):
        url = f"http://[{ip_str}]/" if ":" in ip_str else f"http://{ip_str}/"
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == "BLOCKED_IP_LITERAL"

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://instance-data/latest/meta-data/",
        ],
    )
    def test_cloud_metadata_rejection(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code in ("BLOCKED_IP_LITERAL", "BLOCKED_HOSTNAME")

    @pytest.mark.parametrize(
        "port",
        [21, 22, 23, 25, 445, 1433, 3306, 3389, 5432, 6379, 27017],
    )
    def test_blocked_ports_rejection(self, port: int):
        url = f"http://example.com:{port}/"
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == "BLOCKED_PORT"

    def test_redirect_destination_validation(self):
        # Valid public redirect
        r1 = validate_redirect_target("/safe-page", base_url="https://example.com")
        assert r1.is_valid is True
        assert r1.canonical_url == "https://example.com/safe-page"

        # Invalid redirect to private LAN
        r2 = validate_redirect_target("http://192.168.1.1/admin", base_url="https://example.com")
        assert r2.is_valid is False
        assert r2.error_code == "BLOCKED_IP_LITERAL"

        # Invalid redirect to cloud metadata
        r3 = validate_redirect_target("http://169.254.169.254/", base_url="https://example.com")
        assert r3.is_valid is False
        assert r3.error_code == "BLOCKED_IP_LITERAL"
