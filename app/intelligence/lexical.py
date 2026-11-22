"""Lexical URL Feature Extraction Module for PhishVision AI.

Extracts over 30 discriminative lexical, structural, statistical, and brand-impersonation
features from raw or canonicalized URLs for baseline fast-path machine learning inference
and multimodal risk fusion.
"""

from __future__ import annotations

import ipaddress
import math
import re
import urllib.parse
from collections import Counter
from typing import Any
import numpy as np
import tldextract

from app.intelligence.url_domain.analyzer import DomainIntelligenceAnalyzer, DomainIntelligenceEvidence
from app.intelligence.url_domain.psl_extractor import PSLExtractor

# Top targeted brands for brand-in-subdomain/path impersonation detection
MONITORED_TARGET_BRANDS = {
    "paypal": ["paypal.com", "paypal-community.com", "paypal.co.uk", "paypal.me"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "office365.com"],
    "google": ["google.com", "accounts.google.com", "youtube.com", "gmail.com"],
    "netflix": ["netflix.com"],
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amazon.de"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "bankofamerica": ["bankofamerica.com", "bofa.com"],
    "dhl": ["dhl.com", "express.dhl"],
    "fedex": ["fedex.com"],
    "linkedin": ["linkedin.com"],
    "adobe": ["adobe.com"],
    "sbi": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com"],
    "hdfc": ["hdfcbank.com", "hdfc.com"],
}

# Suspicious keywords frequently observed in phishing lure paths/queries
SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "sign-in",
    "log-in",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "banking",
    "confirm",
    "confirmation",
    "wallet",
    "auth",
    "authenticate",
    "portal",
    "support",
    "recover",
    "recovery",
    "billing",
    "service",
    "ebayisapi",
    "webscr",
    "cmd",
    "password",
    "credential",
    "token",
    "unlock",
    "alert",
]

# TLDs with historically high abuse rates in phishing campaigns
SUSPICIOUS_TLDS = {
    "cfd", "top", "xyz", "tk", "ml", "ga", "cf", "gq", "buzz", "icu",
    "rest", "fit", "work", "cam", "sbs", "country", "stream", "gdn",
    "mom", "monster", "click", "link", "live", "club", "surf", "pw",
}

# Sensitive file extensions often dropped by phishing lures
SENSITIVE_EXTENSIONS = {
    ".exe", ".apk", ".zip", ".rar", ".7z", ".tar", ".gz", ".iso",
    ".php", ".asp", ".aspx", ".cgi", ".bin", ".vbs", ".bat", ".cmd", ".sh",
}


def compute_shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a given text string.
    
    H(X) = - sum_{x in X} p(x) * log2(p(x))
    Returns 0.0 for empty or single-character strings.
    """
    if not text or len(text) <= 1:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


class LexicalFeatureExtractor:
    """Extracts 40 discriminative numerical features from URLs."""

    # Explicit ordered registry of all 40 feature names
    FEATURE_NAMES: list[str] = [
        # 1. URL Length & Character Counts (10)
        "url_length",
        "num_dots_url",
        "num_hyphens_url",
        "num_underscores_url",
        "num_slashes_url",
        "num_question_marks_url",
        "num_equal_signs_url",
        "num_at_symbols_url",
        "num_percent_url",
        "num_ampersands_url",
        # 2. Hostname & Subdomain Structure (6)
        "hostname_length",
        "num_dots_hostname",
        "num_subdomains",
        "subdomain_length",
        "has_ip_hostname",
        "is_punycode",
        # 3. Domain & TLD Characteristics (6)
        "domain_length",
        "tld_length",
        "is_suspicious_tld",
        "num_hyphens_domain",
        "domain_digit_count",
        "vowel_ratio_domain",
        # 4. Path & Query Characteristics (7)
        "path_length",
        "path_depth",
        "query_length",
        "num_query_params",
        "has_fragment",
        "has_double_slash_redirect",
        "has_sensitive_file_extension",
        # 5. Suspicious Keywords & Security Tokens (5)
        "count_suspicious_keywords",
        "count_target_brand_keywords",
        "brand_in_subdomain_or_path",
        "num_encoded_chars",
        "has_non_standard_port",
        # 6. Information Entropy & Character Ratios (6)
        "url_entropy",
        "domain_entropy",
        "path_entropy",
        "digit_ratio_url",
        "symbol_ratio_url",
        "letter_ratio_url",
    ]

    def __init__(self, cache_dir: str = "data/cache/tldextract") -> None:
        self.feature_names = list(self.FEATURE_NAMES)
        self.num_features = len(self.feature_names)
        self.psl_extractor = PSLExtractor(cache_dir=cache_dir)
        self.domain_analyzer = DomainIntelligenceAnalyzer(psl_extractor=self.psl_extractor)
        self._tld_extractor = self.psl_extractor._extractor

    @staticmethod
    def _is_ip_address(host: str) -> bool:
        """Check if hostname is a valid IPv4 or IPv6 address."""
        return PSLExtractor.is_ip_address(host)

    def _parse_url(self, url: str) -> tuple[str, str, str, str, str, str, int | None]:
        """Safely parse URL into (scheme, hostname, domain, tld, subdomain, path, port).
        
        Reuses Public Suffix List parsing for accurate multi-label TLD handling
        (e.g. co.uk, co.in) while remaining robust against IP literals and malformed inputs.
        """
        info = self.psl_extractor.extract_from_url(url)
        return (
            info.scheme,
            info.normalized_hostname,
            info.registrable_domain,
            info.suffix,
            info.subdomain,
            info.path,
            info.port,
        )

    def analyze_domain(self, url: str) -> DomainIntelligenceEvidence:
        """Perform unified domain intelligence analysis (PSL, homoglyphs, typosquatting)."""
        return self.domain_analyzer.analyze(url)

    def extract_features(self, url: str) -> dict[str, float]:
        """Extract a dictionary of named lexical features from a URL."""
        if not isinstance(url, str):
            url = str(url) if url is not None else ""
        url = url.strip()

        url_lower = url.lower()
        url_len = len(url)

        if url_len == 0:
            return {name: 0.0 for name in self.feature_names}

        # 1. URL Length & Character Counts
        num_dots = float(url.count("."))
        num_hyphens = float(url.count("-"))
        num_underscores = float(url.count("_"))
        num_slashes = float(url.count("/"))
        num_question_marks = float(url.count("?"))
        num_equal_signs = float(url.count("="))
        num_at_symbols = float(url.count("@"))
        num_percent = float(url.count("%"))
        num_ampersands = float(url.count("&"))

        # Parsing via PSL
        scheme, hostname, domain, tld, subdomain, path, port = self._parse_url(url)
        host_len = float(len(hostname))
        num_dots_host = float(hostname.count("."))

        # Subdomain analysis from PSL parsed subdomain
        subdomain_labels = [s for s in subdomain.split(".") if s]
        num_subdomains = float(len(subdomain_labels))
        subdomain_len = float(len(subdomain))

        has_ip = 1.0 if self._is_ip_address(hostname) else 0.0
        is_punycode = 1.0 if "xn--" in hostname else 0.0

        # Domain & TLD characteristics
        domain_len = float(len(domain))
        tld_len = float(len(tld))
        is_susp_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
        num_hyphens_dom = float(domain.count("-"))
        dom_digits = float(sum(c.isdigit() for c in domain))
        dom_vowels = float(sum(c in "aeiou" for c in domain.lower()))
        vowel_ratio_dom = dom_vowels / max(domain_len, 1.0)

        # Path & Query characteristics
        path_len = float(len(path))
        path_segments = [seg for seg in path.split("/") if seg]
        path_depth = float(len(path_segments))

        query_idx = url.find("?")
        if query_idx != -1:
            query_str = url[query_idx + 1:]
            query_len = float(len(query_str))
            query_params = [q for q in query_str.split("&") if q]
            num_query_params = float(len(query_params))
        else:
            query_str = ""
            query_len = 0.0
            num_query_params = 0.0

        has_frag = 1.0 if "#" in url else 0.0

        # Double slash after scheme: e.g. http://evil.com//phish or http://legit.com/url=http://evil.com
        after_scheme = url.split("://", 1)[1] if "://" in url else url
        has_double_slash = 1.0 if "//" in after_scheme else 0.0

        # Sensitive file extension
        path_lower = path.lower()
        has_sens_ext = 0.0
        for ext in SENSITIVE_EXTENSIONS:
            if path_lower.endswith(ext):
                has_sens_ext = 1.0
                break

        # Suspicious keywords count
        count_susp_keywords = 0.0
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in url_lower:
                count_susp_keywords += 1.0

        # Brand keywords and brand-in-subdomain/path check
        count_brand_keywords = 0.0
        brand_impersonation_flag = 0.0
        for brand_name, official_domains in MONITORED_TARGET_BRANDS.items():
            if brand_name in url_lower:
                count_brand_keywords += 1.0
                # Check if brand appears in subdomain or path while domain is not official
                is_official = any(domain == off_d or domain.endswith("." + off_d) for off_d in official_domains)
                in_sub_or_path = (brand_name in subdomain.lower()) or (brand_name in path_lower)
                if in_sub_or_path and not is_official:
                    brand_impersonation_flag = 1.0

        # Encoded characters (%xx)
        num_encoded = float(len(re.findall(r"%[0-9a-fA-F]{2}", url)))

        # Non-standard port
        if port is not None and port not in (80, 443, 8080):
            has_non_std_port = 1.0
        else:
            has_non_std_port = 0.0

        # Information entropy & character ratios
        url_entropy = compute_shannon_entropy(url)
        domain_entropy = compute_shannon_entropy(domain)
        path_entropy = compute_shannon_entropy(path)

        num_digits = float(sum(c.isdigit() for c in url))
        num_letters = float(sum(c.isalpha() for c in url))
        num_symbols = float(url_len - num_digits - num_letters)

        digit_ratio = num_digits / max(float(url_len), 1.0)
        symbol_ratio = num_symbols / max(float(url_len), 1.0)
        letter_ratio = num_letters / max(float(url_len), 1.0)

        features: dict[str, float] = {
            "url_length": float(url_len),
            "num_dots_url": num_dots,
            "num_hyphens_url": num_hyphens,
            "num_underscores_url": num_underscores,
            "num_slashes_url": num_slashes,
            "num_question_marks_url": num_question_marks,
            "num_equal_signs_url": num_equal_signs,
            "num_at_symbols_url": num_at_symbols,
            "num_percent_url": num_percent,
            "num_ampersands_url": num_ampersands,
            "hostname_length": host_len,
            "num_dots_hostname": num_dots_host,
            "num_subdomains": num_subdomains,
            "subdomain_length": subdomain_len,
            "has_ip_hostname": has_ip,
            "is_punycode": is_punycode,
            "domain_length": domain_len,
            "tld_length": tld_len,
            "is_suspicious_tld": is_susp_tld,
            "num_hyphens_domain": num_hyphens_dom,
            "domain_digit_count": dom_digits,
            "vowel_ratio_domain": vowel_ratio_dom,
            "path_length": path_len,
            "path_depth": path_depth,
            "query_length": query_len,
            "num_query_params": num_query_params,
            "has_fragment": has_frag,
            "has_double_slash_redirect": has_double_slash,
            "has_sensitive_file_extension": has_sens_ext,
            "count_suspicious_keywords": count_susp_keywords,
            "count_target_brand_keywords": count_brand_keywords,
            "brand_in_subdomain_or_path": brand_impersonation_flag,
            "num_encoded_chars": num_encoded,
            "has_non_standard_port": has_non_std_port,
            "url_entropy": url_entropy,
            "domain_entropy": domain_entropy,
            "path_entropy": path_entropy,
            "digit_ratio_url": digit_ratio,
            "symbol_ratio_url": symbol_ratio,
            "letter_ratio_url": letter_ratio,
        }

        return features

    def extract_vector(self, url: str) -> np.ndarray:
        """Extract features as an ordered 1D NumPy float64 array of shape (40,)."""
        feats = self.extract_features(url)
        return np.array([feats[name] for name in self.feature_names], dtype=np.float64)

    def extract_batch(self, urls: list[str]) -> np.ndarray:
        """Extract features for a list of URLs as a 2D NumPy array of shape (N, 40)."""
        if not urls:
            return np.zeros((0, self.num_features), dtype=np.float64)
        matrix = np.empty((len(urls), self.num_features), dtype=np.float64)
        for i, url in enumerate(urls):
            matrix[i] = self.extract_vector(url)
        return matrix
