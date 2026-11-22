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
    label: int  # 0 = Benign/Legitimate, 1 = Phishing
    source: str
    target_brand: str | None = None
    collection_timestamp: str | None = None


class URLNormalizer:
    """Provides high-throughput URL canonicalization, tracking parameter stripping, and PSL parsing."""

    def __init__(self) -> None:
        # Cache tldextract to prevent repeated Public Suffix List lookups
        self._extractor = tldextract.TLDExtract(cache_dir="data/cache/tldextract", suffix_list_urls=())

    @staticmethod
    def parse_label(raw_label: Any) -> int:
        """Map heterogeneous source labels into binary classification truth (0=Benign, 1=Phishing)."""
        if raw_label is None:
            return 1  # Default to untrusted if ambiguous
        if isinstance(raw_label, (int, float)):
            return 1 if int(raw_label) >= 1 else 0

        lbl = str(raw_label).strip().lower()
        if lbl in {"0", "benign", "legit", "legitimate", "not-phishing", "clean", "safe", "ham"}:
            return 0
        if lbl in {"1", "phish", "phishing", "malicious", "bad", "spam"}:
            return 1
        return 1

    def normalize(
        self,
        raw_url: str,
        label: Any = 1,
        source: str = "unknown",
        target_brand: str | None = None,
        timestamp: str | None = None,
    ) -> NormalizedURLRecord | None:
        """Canonicalize raw URL, remove noise parameters, and extract registrable domain."""
        if not raw_url or not isinstance(raw_url, str):
            return None

        url_str = raw_url.strip()
        if not url_str:
            return None

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

        # Parse port
        port = None
        hostname = netloc
        if ":" in netloc and not netloc.startswith("["):
            parts = netloc.split(":")
            hostname = parts[0]
            try:
                port = int(parts[1])
                # Drop standard default ports
                if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
                    port = None
            except ValueError:
                return None

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

        # Build clean canonical URL
        reconstructed_netloc = f"{hostname}:{port}" if port else hostname
        canonical = urllib.parse.urlunsplit(
            (scheme, reconstructed_netloc, path, cleaned_query, "")
        )

        # Domain extraction via PSL
        extracted = self._extractor(hostname)
        reg_domain = (
            getattr(extracted, "top_domain_under_public_suffix", None)
            or getattr(extracted, "registered_domain", None)
            or hostname
        )

        # SHA-256 hash for fast indexing & deduplication
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
            hostname=hostname,
            registrable_domain=reg_domain,
            subdomain=extracted.subdomain,
            tld=extracted.suffix,
            port=port,
            path=path,
            has_query=bool(cleaned_query),
            label=self.parse_label(label),
            source=source,
            target_brand=clean_brand,
            collection_timestamp=timestamp,
        )
