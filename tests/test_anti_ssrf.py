"""Unit tests for Anti-SSRF URL validation, DNS resolution, and edge cases."""

from unittest.mock import patch
import pytest

from app.scanner.anti_ssrf import (
    SSRFValidationError,
    is_ip_disallowed,
    validate_redirect_target,
    validate_url_and_resolve,
)


class TestAntiSSRFValidation:
    """Test suite verifying strict Anti-SSRF boundaries, IP filtering, and URL parser hardening."""

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
            ("http://", "MISSING_HOSTNAME"),
            ("https://", "MISSING_HOSTNAME"),
        ],
    )
    def test_invalid_scheme_and_empty_rejection(self, url: str, expected_code: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == expected_code

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com:0/",
            "http://example.com:65536/",
            "http://example.com:99999/",
            "http://example.com:-1/",
            "http://example.com:port/",
        ],
    )
    def test_out_of_range_and_invalid_ports(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code in ("INVALID_PORT", "PORT_OUT_OF_RANGE", "PORT_NOT_PERMITTED")

    @pytest.mark.parametrize(
        "url",
        [
            "http://foo@bar@example.com/",
            "http://user:pass@evil@victim.com/",
            "http://example.com/\x00evil",
            "http://example.com/\r\nSet-Cookie:1",
            "http://[::1/",
            "http://::1]/",
        ],
    )
    def test_malformed_authorities_and_control_chars(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code in (
            "MALFORMED_AUTHORITY",
            "MALFORMED_URL",
            "MALFORMED_IPV6_BRACKETS",
            "INVALID_PORT",
            "INVALID_HOSTNAME",
            "URL_PARSE_ERROR",
        )

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
        "url",
        [
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:192.168.1.1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://[::ffff:169.254.169.254]/",
        ],
    )
    def test_ipv4_mapped_ipv6_rejection(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code == "BLOCKED_IP_LITERAL"

    @pytest.mark.parametrize(
        "url",
        [
            "http://2130706433/",            # DWORD integer for 127.0.0.1
            "http://0x7f000001/",            # Hex for 127.0.0.1
            "http://0177.0.0.1/",            # Dotted octal for 127.0.0.1
            "http://0x7f.0.0.1/",            # Mixed hex/octal for 127.0.0.1
            "http://%31%32%37%2e%30%2e%30%2e%31/", # Percent-encoded 127.0.0.1
        ],
    )
    def test_obfuscated_ip_literals_rejection(self, url: str):
        result = validate_url_and_resolve(url)
        assert result.is_valid is False
        assert result.error_code in ("BLOCKED_IP_LITERAL", "SSRF_IP_BLOCKED")

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

    def test_mixed_public_and_private_dns_answers_rejected(self):
        """Verify that if a domain resolves to multiple IPs and ANY is private, it is rejected."""
        with patch("app.scanner.anti_ssrf.resolve_hostname_ips") as mock_resolve:
            # Domain returns legitimate public IP and malicious LAN private IP
            mock_resolve.return_value = ["93.184.216.34", "192.168.1.50"]
            result = validate_url_and_resolve("https://dual-homed-target.com")
            assert result.is_valid is False
            assert result.error_code == "SSRF_IP_BLOCKED"
            assert "Private/LAN IP address rejected" in result.error_message

    def test_dns_rebinding_simulation(self):
        """Verify that DNS rebinding returning a loopback address is rejected."""
        with patch("app.scanner.anti_ssrf.resolve_hostname_ips") as mock_resolve:
            # Rebinding switch to loopback
            mock_resolve.return_value = ["127.0.0.1"]
            result = validate_url_and_resolve("https://rebind-attack.com")
            assert result.is_valid is False
            assert result.error_code == "SSRF_IP_BLOCKED"
            assert "Loopback IP address rejected" in result.error_message

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
