"""Public Suffix List (PSL) Domain Extraction Engine for PhishVision AI.

Provides robust, multi-label TLD parsing using tldextract with offline cache.
Accurately resolves multi-part public suffixes (.co.uk, .co.in, .gov.in), subdomains,
IPv4/IPv6 address literals, and Punycode/IDN internationalized hostnames.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tldextract


@dataclass(frozen=True)
class ParsedDomainInfo:
    """Structured representation of domain parsing results."""

    raw_url: str
    scheme: str
    original_hostname: str
    normalized_hostname: str
    unicode_hostname: str
    registrable_domain: str
    domain_name: str
    suffix: str
    subdomain: str
    port: int | None
    path: str
    query: str
    is_ip: bool
    is_punycode: bool


class PSLExtractor:
    """Reuses tldextract with local caching to parse URLs and hostnames accurately."""

    def __init__(self, cache_dir: str = "data/cache/tldextract") -> None:
        self.cache_dir = cache_dir
        # Ensure offline operation by passing empty suffix_list_urls
        self._extractor = tldextract.TLDExtract(
            cache_dir=self.cache_dir,
            suffix_list_urls=(),
        )

    @staticmethod
    def is_ip_address(host: str) -> bool:
        """Check whether the provided host string is an IPv4 or IPv6 address."""
        if not host:
            return False
        clean = host.strip("[]")
        try:
            ipaddress.ip_address(clean)
            return True
        except ValueError:
            return False

    def extract_from_url(self, url: str) -> ParsedDomainInfo:
        """Extract domain components and metadata from a full URL or hostname."""
        if not isinstance(url, str):
            url = str(url) if url is not None else ""
        raw_url = url.strip()

        if not raw_url:
            return ParsedDomainInfo(
                raw_url="",
                scheme="",
                original_hostname="",
                normalized_hostname="",
                unicode_hostname="",
                registrable_domain="",
                domain_name="",
                suffix="",
                subdomain="",
                port=None,
                path="",
                query="",
                is_ip=False,
                is_punycode=False,
            )

        # Ensure scheme is present for standard parsing
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw_url):
            url_to_parse = "http://" + raw_url
        else:
            url_to_parse = raw_url

        try:
            parsed = urllib.parse.urlsplit(url_to_parse)
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc
            path = parsed.path
            query = parsed.query
        except Exception:
            return ParsedDomainInfo(
                raw_url=raw_url,
                scheme="",
                original_hostname="",
                normalized_hostname="",
                unicode_hostname="",
                registrable_domain="",
                domain_name="",
                suffix="",
                subdomain="",
                port=None,
                path="",
                query="",
                is_ip=False,
                is_punycode=False,
            )

        # Handle userinfo if present in netloc
        if "@" in netloc:
            _, netloc = netloc.rsplit("@", 1)

        original_hostname = netloc
        port = None

        # Parse port and isolate hostname (handling bracketed IPv6)
        if netloc.startswith("["):
            bracket_end = netloc.find("]")
            if bracket_end != -1:
                original_hostname = netloc[: bracket_end + 1]
                rest = netloc[bracket_end + 1 :]
                if rest.startswith(":"):
                    try:
                        port = int(rest[1:])
                    except ValueError:
                        port = None
        elif ":" in netloc:
            parts = netloc.split(":")
            original_hostname = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                port = None

        clean_host = original_hostname.strip("[]")
        normalized_hostname = clean_host.lower()

        # Check for Punycode
        is_puny = normalized_hostname.startswith("xn--") or ".xn--" in normalized_hostname
        unicode_hostname = normalized_hostname
        if is_puny:
            try:
                # Decode each label through IDNA
                labels = normalized_hostname.split(".")
                decoded_labels = []
                for label in labels:
                    if label.startswith("xn--"):
                        decoded_labels.append(label.encode("ascii").decode("idna"))
                    else:
                        decoded_labels.append(label)
                unicode_hostname = ".".join(decoded_labels)
            except Exception:
                unicode_hostname = normalized_hostname

        # Check IP address
        if self.is_ip_address(normalized_hostname):
            return ParsedDomainInfo(
                raw_url=raw_url,
                scheme=scheme,
                original_hostname=original_hostname,
                normalized_hostname=normalized_hostname,
                unicode_hostname=unicode_hostname,
                registrable_domain=normalized_hostname,
                domain_name=normalized_hostname,
                suffix="",
                subdomain="",
                port=port,
                path=path,
                query=query,
                is_ip=True,
                is_punycode=is_puny,
            )

        # Multi-label PSL extraction via tldextract
        ext = self._extractor(normalized_hostname)
        top_domain = getattr(ext, "top_domain_under_public_suffix", "")
        domain_name = getattr(ext, "domain", "")
        suffix = getattr(ext, "suffix", "")
        subdomain = getattr(ext, "subdomain", "")

        if top_domain:
            registrable_domain = top_domain
        else:
            if domain_name and suffix:
                registrable_domain = f"{domain_name}.{suffix}"
            else:
                registrable_domain = domain_name or normalized_hostname

        return ParsedDomainInfo(
            raw_url=raw_url,
            scheme=scheme,
            original_hostname=original_hostname,
            normalized_hostname=normalized_hostname,
            unicode_hostname=unicode_hostname,
            registrable_domain=registrable_domain,
            domain_name=domain_name,
            suffix=suffix,
            subdomain=subdomain,
            port=port,
            path=path,
            query=query,
            is_ip=False,
            is_punycode=is_puny,
        )

    def extract_registrable_domain(self, url: str) -> str:
        """Convenience method returning strictly the registrable domain."""
        return self.extract_from_url(url).registrable_domain
