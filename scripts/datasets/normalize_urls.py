"""URL Normalization, Canonicalization & Registrable Domain Extraction Module."""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tldextract

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.normalize_urls")

# Query parameters commonly used for tracking that add noise to lexical models
MARKETING_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "zanpid",
    "msclkid",
    "_ga",
    "_gl",
    "ref",
    "mc_cid",
    "mc_eid",
}


@dataclass
class NormalizedURLRecord:
    """Standardized representation of a validated, canonicalized URL."""
    original_url: str
    canonical_url: str
    url_hash_sha256: str
    scheme: str
    hostname: str
    registrable_domain: str
    subdomain: str
    tld: str
    port: int | None
    path: str
    has_query: bool
    raw_label: str | None
    label: int | None  # 0 = Benign, 1 = Phishing, -1 = Unverified Candidate, None = Rejected
    verification_status: str  # "VERIFIED", "UNVERIFIED_CANDIDATE", "REJECTED_UNKNOWN"
    source: str
    target_brand: str | None = None
    collection_timestamp: str | None = None


class URLNormalizer:
    """Provides high-throughput URL canonicalization, tracking parameter stripping, and PSL parsing."""

    def __init__(self) -> None:
        # Cache tldextract to prevent repeated Public Suffix List lookups
        self._extractor = tldextract.TLDExtract(cache_dir="data/cache/tldextract", suffix_list_urls=())

    @staticmethod
    def parse_label(raw_label: Any) -> tuple[int | None, str]:
        """Map heterogeneous source labels into binary classification truth, candidate status, or rejection.

        Returns:
            (label_int, verification_status):
            - (0, "VERIFIED") for explicit legitimate/benign sources.
            - (1, "VERIFIED") for explicit verified phishing sources.
            - (-1, "UNVERIFIED_CANDIDATE") for candidate feeds (e.g. Tranco popular list).
            - (None, "REJECTED_UNKNOWN") for missing, ambiguous, or malformed labels.
        """
        if raw_label is None:
            return None, "REJECTED_UNKNOWN"

        if isinstance(raw_label, (int, float)):
            val = int(raw_label)
            if val == 0:
                return 0, "VERIFIED"
            elif val == 1:
                return 1, "VERIFIED"
            elif val == -1:
                return -1, "UNVERIFIED_CANDIDATE"
            else:
                return None, "REJECTED_UNKNOWN"

        lbl = str(raw_label).strip().lower()
        if not lbl or lbl in {"none", "nan", "null", "unknown", "pending", "unlabeled"}:
            return None, "REJECTED_UNKNOWN"

        if lbl in {"0", "benign", "legit", "legitimate", "not-phishing", "clean", "safe", "ham"}:
            return 0, "VERIFIED"
        if lbl in {"1", "phish", "phishing", "malicious", "bad", "spam"}:
            return 1, "VERIFIED"
        if lbl in {"candidate", "unverified", "tranco_candidate", "-1"}:
            return -1, "UNVERIFIED_CANDIDATE"

        return None, "REJECTED_UNKNOWN"

    def normalize(
        self,
        raw_url: str,
        label: Any = "unverified",
        source: str = "unknown",
        target_brand: str | None = None,
        timestamp: str | None = None,
    ) -> NormalizedURLRecord | None:
        """Canonicalize raw URL, remove noise parameters, parse userinfo/IPv6/IDN, and extract PSL domain."""
        if not raw_url or not isinstance(raw_url, str):
            return None

        url_str = raw_url.strip()
        if not url_str:
            return None

        # Parse label and verification status
        label_int, verification_status = self.parse_label(label)

        # Ensure scheme is present
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url_str):
            url_str = "http://" + url_str

        try:
            parsed = urllib.parse.urlsplit(url_str)
        except Exception:
            return None

        scheme = (parsed.scheme or "http").lower()
        if scheme not in {"http", "https"}:
            return None

        netloc = parsed.netloc.lower()
        if not netloc:
            return None

        # Handle userinfo (e.g. user:pass@host)
        userinfo = None
        if "@" in netloc:
            userinfo, netloc = netloc.rsplit("@", 1)

        # Handle IPv6 literals and port parsing
        port = None
        hostname = netloc
        if netloc.startswith("["):
            bracket_end = netloc.find("]")
            if bracket_end != -1:
                hostname = netloc[: bracket_end + 1]
                rest = netloc[bracket_end + 1 :]
                if rest.startswith(":"):
                    try:
                        port = int(rest[1:])
                    except ValueError:
                        return None
        else:
            if ":" in netloc:
                parts = netloc.split(":")
                hostname = parts[0]
                try:
                    port = int(parts[1])
                except ValueError:
                    return None

        # Drop standard default ports
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        if port is not None and not (1 <= port <= 65535):
            return None

        # Handle Internationalized Domain Names (IDN punycode normalization)
        hostname_normalized = hostname
        if not hostname.startswith("["):
            try:
                # Convert unicode hostnames to punycode
                hostname_normalized = hostname.encode("idna").decode("ascii").lower()
            except Exception:
                hostname_normalized = hostname.lower()

        # Clean query parameters
        cleaned_query = ""
        if parsed.query:
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            filtered_pairs = [
                (k, v) for k, v in query_pairs if k.lower() not in MARKETING_TRACKING_PARAMS
            ]
            if filtered_pairs:
                cleaned_query = urllib.parse.urlencode(filtered_pairs)

        # Normalize path
        path = parsed.path or "/"
        path = re.sub(r"/{2,}", "/", path)

        # Reconstruct canonical authority
        reconstructed_netloc = f"{hostname_normalized}:{port}" if port else hostname_normalized
        canonical = urllib.parse.urlunsplit(
            (scheme, reconstructed_netloc, path, cleaned_query, "")
        )

        # Domain extraction via PSL
        extracted = self._extractor(hostname_normalized)
        top_domain = getattr(extracted, "top_domain_under_public_suffix", "")
        if top_domain:
            reg_domain = top_domain
        else:
            domain = getattr(extracted, "domain", "")
            suffix = getattr(extracted, "suffix", "")
            reg_domain = f"{domain}.{suffix}" if domain and suffix else (domain or hostname_normalized)

        # SHA-256 hash of canonical URL
        url_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # Clean target brand string
        clean_brand = None
        if target_brand is not None:
            tb_str = str(target_brand).strip()
            if tb_str and tb_str.lower() not in {"none", "nan", ""}:
                clean_brand = tb_str

        return NormalizedURLRecord(
            original_url=raw_url.strip(),
            canonical_url=canonical,
            url_hash_sha256=url_hash,
            scheme=scheme,
            hostname=hostname_normalized,
            registrable_domain=reg_domain,
            subdomain=extracted.subdomain,
            tld=extracted.suffix,
            port=port,
            path=path,
            has_query=bool(cleaned_query),
            raw_label=str(label) if label is not None else None,
            label=label_int,
            verification_status=verification_status,
            source=source,
            target_brand=clean_brand,
            collection_timestamp=timestamp,
        )
