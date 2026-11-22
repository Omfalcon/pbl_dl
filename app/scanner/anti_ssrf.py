"""Anti-SSRF and DNS Validation Engine for PhishVision AI.

Provides comprehensive URL parsing, hostname verification, DNS resolution,
and IP address blacklisting to protect the scanning backend from Server-Side
Request Forgery (SSRF), DNS rebinding, and unauthorized LAN access.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from dataclasses import dataclass, field
from typing import Sequence


# Disallowed port list (common administrative/internal services)
BLOCKED_PORTS = {
    21,    # FTP
    22,    # SSH
    23,    # Telnet
    25,    # SMTP
    110,   # POP3
    135,   # RPC
    139,   # NetBIOS
    143,   # IMAP
    445,   # SMB
    1433,  # MSSQL
    1521,  # Oracle
    3306,  # MySQL
    3389,  # RDP
    5432,  # PostgreSQL
    6379,  # Redis
    8086,  # InfluxDB
    9200,  # Elasticsearch
    11211, # Memcached
    27017, # MongoDB
}

# Blocked hostnames (localhost, cloud metadata, common internal patterns)
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "instance-data",
    "metadata.google.internal",
    "metadata.packet.net",
    "169.254.169.254.xip.io",
    "169.254.169.254.nip.io",
}

BLOCKED_HOSTNAME_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".corp",
    ".lan",
    ".home.arpa",
)

# Custom blocked CIDR ranges that may not be covered by basic is_private
BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network (only valid as source)
    ipaddress.ip_network("10.0.0.0/8"),          # RFC 1918 Private-Use
    ipaddress.ip_network("100.64.0.0/10"),       # RFC 6598 Shared Address Space (Carrier-grade NAT)
    ipaddress.ip_network("127.0.0.0/8"),         # RFC 1122 Loopback
    ipaddress.ip_network("169.254.0.0/16"),      # RFC 3927 Link-Local (includes 169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),       # RFC 1918 Private-Use
    ipaddress.ip_network("192.0.0.0/24"),        # RFC 6890 IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),        # RFC 5737 Documentation (TEST-NET-1)
    ipaddress.ip_network("192.88.99.0/24"),      # RFC 7526 6to4 Relay Anycast
    ipaddress.ip_network("192.168.0.0/16"),      # RFC 1918 Private-Use
    ipaddress.ip_network("198.18.0.0/15"),       # RFC 2544 Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),     # RFC 5737 Documentation (TEST-NET-2)
    ipaddress.ip_network("203.0.113.0/24"),      # RFC 5737 Documentation (TEST-NET-3)
    ipaddress.ip_network("224.0.0.0/4"),         # RFC 5771 Multicast
    ipaddress.ip_network("240.0.0.0/4"),         # RFC 1112 Reserved for Future Use
    ipaddress.ip_network("255.255.255.255/32"),  # RFC 919 Limited Broadcast
    # IPv6 ranges
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("::/128"),              # Unspecified
    ipaddress.ip_network("::ffff:0:0/96"),       # IPv4-mapped IPv6
    ipaddress.ip_network("64:ff9b::/96"),        # IPv4/IPv6 translation
    ipaddress.ip_network("100::/64"),            # Discard prefix
    ipaddress.ip_network("2001:db8::/32"),       # Documentation
    ipaddress.ip_network("fc00::/7"),            # Unique Local Unicast (RFC 4193)
    ipaddress.ip_network("fe80::/10"),           # Link-Local Unicast (RFC 4291)
    ipaddress.ip_network("ff00::/8"),            # Multicast (RFC 4291)
]


class SSRFValidationError(ValueError):
    """Raised when a URL violates anti-SSRF or security validation policies."""
    def __init__(self, message: str, code: str = "SSRF_BLOCKED"):
        super().__init__(message)
        self.code = code


@dataclass
class ValidationResult:
    """Encapsulates the result of URL parsing and anti-SSRF validation."""
    is_valid: bool
    url: str
    canonical_url: str
    scheme: str
    hostname: str
    port: int
    resolved_ips: list[str] = field(default_factory=list)
    pinned_ip: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    def raise_if_invalid(self) -> None:
        """Helper to raise SSRFValidationError if validation failed."""
        if not self.is_valid:
            raise SSRFValidationError(
                self.error_message or "URL failed security validation",
                code=self.error_code or "VALIDATION_FAILED"
            )


def is_ip_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str]:
    """Inspect an IP address object and return whether it is blocked and the reason."""
    # Check standard Python library classifications
    if ip.is_loopback:
        return True, f"Loopback IP address rejected: {ip}"
    if ip.is_private:
        return True, f"Private/LAN IP address rejected (RFC 1918 / RFC 4193): {ip}"
    if ip.is_link_local:
        return True, f"Link-local / Cloud metadata IP address rejected: {ip}"
    if ip.is_multicast:
        return True, f"Multicast IP address rejected: {ip}"
    if ip.is_reserved:
        return True, f"Reserved IP address rejected: {ip}"
    if ip.is_unspecified:
        return True, f"Unspecified IP address rejected: {ip}"

    # Check against our explicit CIDR blacklist
    for network in BLOCKED_NETWORKS:
        if ip in network:
            return True, f"IP address belongs to restricted network {network}: {ip}"

    return False, ""


def resolve_hostname_ips(hostname: str, port: int) -> list[str]:
    """Resolve all IPv4 and IPv6 addresses for a hostname using socket.getaddrinfo."""
    try:
        addr_info = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise SSRFValidationError(f"DNS resolution failed for hostname '{hostname}': {e}", code="DNS_RESOLUTION_FAILED") from e
    except Exception as e:
        raise SSRFValidationError(f"Unexpected error resolving hostname '{hostname}': {e}", code="DNS_RESOLUTION_ERROR") from e

    resolved = set()
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        # Remove IPv6 scope ID if present (e.g. fe80::1%eth0)
        ip_str = ip_str.split("%")[0]
        resolved.add(ip_str)

    if not resolved:
        raise SSRFValidationError(f"No IP addresses resolved for hostname '{hostname}'", code="DNS_NO_RECORDS")

    return sorted(list(resolved))


def validate_url_and_resolve(
    url: str,
    allowed_schemes: Sequence[str] = ("http", "https"),
    allowed_ports: Sequence[int] | None = None,
    allow_custom_ports: bool = False
) -> ValidationResult:
    """Validate a user-submitted URL, resolve all DNS addresses, and enforce anti-SSRF rules.
    
    Args:
        url: The candidate URL string to evaluate.
        allowed_schemes: Permitted URL schemes (defaults to http, https).
        allowed_ports: Specific whitelist of allowed ports (defaults to 80, 443, 8080, 8443).
        allow_custom_ports: If True, permits any non-blocked port.
    
    Returns:
        ValidationResult detailing resolution, pinned IP, and security verdict.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return ValidationResult(
            is_valid=False,
            url=str(url),
            canonical_url="",
            scheme="",
            hostname="",
            port=0,
            error_code="INVALID_URL_EMPTY",
            error_message="URL must be a non-empty string"
        )

    url_clean = url.strip()

    # Parse URL
    try:
        parsed = urllib.parse.urlsplit(url_clean)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme="",
            hostname="",
            port=0,
            error_code="URL_PARSE_ERROR",
            error_message=f"Failed to parse URL: {e}"
        )

    # 1. Scheme Validation
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme=scheme,
            hostname="",
            port=0,
            error_code="INVALID_SCHEME",
            error_message=f"Scheme '{scheme}' is not permitted. Only {list(allowed_schemes)} are supported."
        )

    # 2. Hostname Validation
    hostname = parsed.hostname
    if not hostname:
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme=scheme,
            hostname="",
            port=0,
            error_code="MISSING_HOSTNAME",
            error_message="URL does not contain a valid hostname"
        )

    hostname_lower = hostname.lower().strip(".")

    # Reject known blocked hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme=scheme,
            hostname=hostname,
            port=0,
            error_code="BLOCKED_HOSTNAME",
            error_message=f"Access to blocked internal/metadata hostname '{hostname}' is forbidden"
        )

    for suffix in BLOCKED_HOSTNAME_SUFFIXES:
        if hostname_lower.endswith(suffix):
            return ValidationResult(
                is_valid=False,
                url=url_clean,
                canonical_url="",
                scheme=scheme,
                hostname=hostname,
                port=0,
                error_code="BLOCKED_DOMAIN_SUFFIX",
                error_message=f"Access to internal domain suffix '{suffix}' is forbidden"
            )

    # 3. Port Validation
    default_port = 443 if scheme == "https" else 80
    port = parsed.port if parsed.port is not None else default_port

    if port in BLOCKED_PORTS:
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme=scheme,
            hostname=hostname,
            port=port,
            error_code="BLOCKED_PORT",
            error_message=f"Access to administrative/internal service port {port} is blocked"
        )

    if allowed_ports is not None:
        if port not in allowed_ports:
            return ValidationResult(
                is_valid=False,
                url=url_clean,
                canonical_url="",
                scheme=scheme,
                hostname=hostname,
                port=port,
                error_code="PORT_NOT_PERMITTED",
                error_message=f"Port {port} is not in permitted list {allowed_ports}"
            )
    elif not allow_custom_ports and port not in (80, 443, 8080, 8443):
        return ValidationResult(
            is_valid=False,
            url=url_clean,
            canonical_url="",
            scheme=scheme,
            hostname=hostname,
            port=port,
            error_code="CUSTOM_PORT_BLOCKED",
            error_message=f"Non-standard web port {port} blocked in standard safe mode"
        )

    # 4. Check if Hostname is a Raw IP Literal
    is_literal_ip = False
    try:
        # Strip IPv6 brackets if present
        raw_ip_str = hostname_lower.strip("[]")
        ip_obj = ipaddress.ip_address(raw_ip_str)
        is_literal_ip = True
        blocked, reason = is_ip_disallowed(ip_obj)
        if blocked:
            return ValidationResult(
                is_valid=False,
                url=url_clean,
                canonical_url="",
                scheme=scheme,
                hostname=hostname,
                port=port,
                resolved_ips=[str(ip_obj)],
                error_code="BLOCKED_IP_LITERAL",
                error_message=reason
            )
        resolved_ips = [str(ip_obj)]
    except ValueError:
        # Not a literal IP, proceed with DNS resolution
        pass

    # 5. DNS Resolution & All-Address Inspection
    if not is_literal_ip:
        try:
            resolved_ips = resolve_hostname_ips(hostname_lower, port)
        except SSRFValidationError as e:
            return ValidationResult(
                is_valid=False,
                url=url_clean,
                canonical_url="",
                scheme=scheme,
                hostname=hostname,
                port=port,
                error_code=e.code,
                error_message=str(e)
            )

        # Inspect EVERY resolved IP address
        for ip_str in resolved_ips:
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                blocked, reason = is_ip_disallowed(ip_obj)
                if blocked:
                    return ValidationResult(
                        is_valid=False,
                        url=url_clean,
                        canonical_url="",
                        scheme=scheme,
                        hostname=hostname,
                        port=port,
                        resolved_ips=resolved_ips,
                        error_code="SSRF_IP_BLOCKED",
                        error_message=f"Hostname '{hostname}' resolved to blocked address: {reason}"
                    )
            except ValueError as e:
                return ValidationResult(
                    is_valid=False,
                    url=url_clean,
                    canonical_url="",
                    scheme=scheme,
                    hostname=hostname,
                    port=port,
                    error_code="INVALID_RESOLVED_IP",
                    error_message=f"Resolved invalid IP '{ip_str}': {e}"
                )

    # Canonicalize URL format
    canonical_port = f":{port}" if (scheme == "http" and port != 80) or (scheme == "https" and port != 443) else ""
    path = parsed.path if parsed.path else "/"
    query = f"?{parsed.query}" if parsed.query else ""
    canonical_url = f"{scheme}://{hostname_lower}{canonical_port}{path}{query}"

    pinned_ip = resolved_ips[0] if resolved_ips else None

    return ValidationResult(
        is_valid=True,
        url=url_clean,
        canonical_url=canonical_url,
        scheme=scheme,
        hostname=hostname_lower,
        port=port,
        resolved_ips=resolved_ips,
        pinned_ip=pinned_ip
    )


def validate_redirect_target(target_url: str, base_url: str | None = None) -> ValidationResult:
    """Validate a navigation or redirect target URL, resolving relative URLs if needed."""
    if not target_url:
        return ValidationResult(
            is_valid=False,
            url="",
            canonical_url="",
            scheme="",
            hostname="",
            port=0,
            error_code="EMPTY_REDIRECT",
            error_message="Redirect URL is empty"
        )

    # Resolve relative redirects against base URL
    if base_url:
        full_url = urllib.parse.urljoin(base_url, target_url)
    else:
        full_url = target_url

    return validate_url_and_resolve(full_url)
