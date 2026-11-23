"""TLS/SSL Certificate Inspection for PhishVision AI.

Collects structured certificate metadata (issuer, subject, SANs, validity,
fingerprint, expiry) using Python ssl module without disabling certificate
verification. Any field that cannot be safely obtained is reported as null
rather than fabricated.

Security constraints:
- TLS verification is NEVER disabled.
- Certificate metadata is obtained from already-established connections or
  from ssl.get_server_certificate() with timeout.
- Self-signed or invalid certificates are reported honestly; the scanner's
  TLS error field takes precedence for failed handshakes.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import ssl
import urllib.parse
from datetime import datetime, timezone
from typing import Any

from app.scanner.anti_ssrf import (
    SSRFValidationError,
    is_ip_disallowed,
    validate_url_and_resolve,
)

logger = logging.getLogger("phishvision.tls_inspector")

# Default connection timeout for certificate inspection (seconds)
_TLS_PROBE_TIMEOUT = 5


def _parse_x509_name(name_obj: Any) -> str:
    """Convert ssl subject/issuer tuple-of-tuples to a readable string."""
    if name_obj is None:
        return ""
    try:
        parts = []
        for rdn in name_obj:
            if isinstance(rdn, (list, tuple)):
                for attr in rdn:
                    if isinstance(attr, (list, tuple)) and len(attr) == 2:
                        parts.append(f"{attr[0]}={attr[1]}")
        return ", ".join(parts)
    except Exception:
        return str(name_obj)


def _extract_sans(cert_dict: dict[str, Any]) -> list[str]:
    """Extract Subject Alternative Names from a parsed certificate dictionary."""
    sans: list[str] = []
    try:
        san_list = cert_dict.get("subjectAltName", [])
        for san_type, san_value in san_list:
            sans.append(f"{san_type}:{san_value}")
    except Exception:
        pass
    return sans


def _parse_cert_date(date_str: str | None) -> str | None:
    """Convert ssl certificate date string to ISO 8601 UTC string."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return date_str


def inspect_tls_certificate(url: str, pinned_ip: str | None = None) -> dict[str, Any]:
    """Probe TLS certificate metadata for a given HTTPS URL.

    Anti-SSRF and DNS rebinding protections:
    1. The target URL is validated via `validate_url_and_resolve()` before any
       TCP socket is created. URLs resolving to private, loopback, multicast, or
       cloud metadata IP addresses (RFC 1918 / RFC 4193 / link-local) are blocked.
    2. Connection is established directly to a validated pinned IP address
       rather than re-resolving hostname, eliminating DNS rebinding windows.
    3. Python SSL context enforces strict certificate validation (`CERT_REQUIRED`),
       hostname verification (`check_hostname=True`), and sends the target
       `hostname` via Server Name Indication (SNI).

    Returns a structured dictionary with the following schema:
    {
        "tls_available": bool,          # True if certificate data was obtained
        "certificate_valid": bool,      # True if handshake succeeded
        "issuer": str | None,           # Certificate issuer DN
        "subject": str | None,          # Certificate subject DN
        "common_name": str | None,      # Subject CN field
        "san": list[str],               # Subject Alternative Names (DNS:..., IP:...)
        "not_before": str | None,       # ISO 8601 validity start
        "not_after": str | None,        # ISO 8601 validity end
        "is_expired": bool,             # True if certificate is currently expired
        "days_until_expiry": int | None,# Days until expiry (negative = expired)
        "sha256_fingerprint": str | None, # SHA-256 fingerprint of DER certificate
        "protocol_version": str | None, # TLS protocol negotiated
        "cipher_suite": str | None,     # Negotiated cipher suite
        "error": str | None,            # Handshake error description if applicable
    }

    Fields that cannot be safely obtained are set to null/None rather than
    fabricated. TLS verification is NEVER disabled.
    """
    result: dict[str, Any] = {
        "tls_available": False,
        "certificate_valid": False,
        "issuer": None,
        "subject": None,
        "common_name": None,
        "san": [],
        "not_before": None,
        "not_after": None,
        "is_expired": False,
        "days_until_expiry": None,
        "sha256_fingerprint": None,
        "protocol_version": None,
        "cipher_suite": None,
        "error": None,
    }

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        result["error"] = "TLS inspection only applies to HTTPS URLs."
        return result

    hostname = parsed.hostname
    if not hostname:
        result["error"] = "No hostname in URL."
        return result

    port = parsed.port or 443

    # Anti-SSRF Validation & DNS resolution pinning
    target_ip: str | None = None
    if pinned_ip:
        try:
            ip_obj = ipaddress.ip_address(pinned_ip)
            is_blocked, reason = is_ip_disallowed(ip_obj)
            if is_blocked:
                result["error"] = f"Anti-SSRF blocked TLS probe: {reason}"
                return result
            target_ip = pinned_ip
        except ValueError as e:
            result["error"] = f"Invalid pinned IP address: {e}"
            return result
    else:
        try:
            val_res = validate_url_and_resolve(url)
            if not val_res.is_valid:
                result["error"] = f"Anti-SSRF blocked TLS probe: {val_res.error_message}"
                return result
            if not val_res.resolved_ips:
                result["error"] = f"No resolved IP addresses for '{hostname}'"
                return result
            target_ip = val_res.resolved_ips[0]
        except SSRFValidationError as e:
            result["error"] = f"Anti-SSRF blocked TLS probe: {e}"
            return result
        except Exception as e:
            result["error"] = f"Validation failed during TLS probe: {e}"
            return result

    try:
        # Use default ssl context which enforces certificate verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED

        # Connect strictly to the validated pinned IP to prevent DNS rebinding
        with socket.create_connection((target_ip, port), timeout=_TLS_PROBE_TIMEOUT) as sock:
            # server_hostname=hostname preserves SNI and validates certificate against hostname
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert_der = ssock.getpeercert(binary_form=True)
                cert_dict = ssock.getpeercert()
                protocol = ssock.version()
                cipher_info = ssock.cipher()

                # Fingerprint
                if cert_der:
                    fingerprint = hashlib.sha256(cert_der).hexdigest()
                    result["sha256_fingerprint"] = fingerprint

                # Protocol and cipher
                result["protocol_version"] = protocol
                if cipher_info:
                    result["cipher_suite"] = cipher_info[0]

                # Subject
                subject_tuple = cert_dict.get("subject")
                subject_str = _parse_x509_name(subject_tuple)
                result["subject"] = subject_str or None

                # Common name
                try:
                    for rdn in (subject_tuple or []):
                        for attr in rdn:
                            if isinstance(attr, (list, tuple)) and len(attr) == 2 and attr[0] == "commonName":
                                result["common_name"] = attr[1]
                                break
                except Exception:
                    pass

                # Issuer
                issuer_tuple = cert_dict.get("issuer")
                result["issuer"] = _parse_x509_name(issuer_tuple) or None

                # Validity dates
                not_before_raw = cert_dict.get("notBefore")
                not_after_raw = cert_dict.get("notAfter")
                result["not_before"] = _parse_cert_date(not_before_raw)
                result["not_after"] = _parse_cert_date(not_after_raw)

                # Expiry check
                if not_after_raw:
                    try:
                        expiry_dt = datetime.strptime(not_after_raw, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        delta = expiry_dt - now
                        days_remaining = delta.days
                        result["is_expired"] = days_remaining < 0
                        result["days_until_expiry"] = days_remaining
                    except Exception:
                        pass

                # Subject Alternative Names
                result["san"] = _extract_sans(cert_dict)

                result["tls_available"] = True
                result["certificate_valid"] = True

    except ssl.SSLCertVerificationError as e:
        result["error"] = f"Certificate verification failed: {e.reason}"
        result["tls_available"] = True
        result["certificate_valid"] = False
    except ssl.SSLError as e:
        result["error"] = f"TLS/SSL error: {e}"
        result["tls_available"] = True
        result["certificate_valid"] = False
    except socket.timeout:
        result["error"] = f"TLS probe timed out after {_TLS_PROBE_TIMEOUT}s"
    except (socket.gaierror, OSError) as e:
        result["error"] = f"Network error during TLS probe: {e}"
    except Exception as e:
        result["error"] = f"Unexpected TLS inspection error: {e}"

    return result
