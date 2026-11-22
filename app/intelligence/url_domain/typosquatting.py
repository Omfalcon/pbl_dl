"""Typosquatting, Combosquatting, and Brand Impersonation Detection Module.

Calculates Levenshtein and Damerau-Levenshtein edit distances, normalized similarity scores,
leetspeak normalization, transposition detection, combosquatting, and subdomain impersonation
against protected brand identities while strictly distinguishing legitimate official domains.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.intelligence.url_domain.psl_extractor import PSLExtractor, ParsedDomainInfo

logger = logging.getLogger("phishvision.typosquatting")

# Built-in fallback catalog of protected brand targets and official domains
DEFAULT_PROTECTED_BRANDS: dict[str, dict[str, Any]] = {
    "sbi": {
        "display_name": "State Bank of India",
        "official_registrable_domains": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com"],
        "keywords": ["sbi", "onlinesbi", "yono"],
    },
    "paypal": {
        "display_name": "PayPal",
        "official_registrable_domains": ["paypal.com", "paypal.co.uk", "paypal-community.com", "paypal.me"],
        "keywords": ["paypal"],
    },
    "google": {
        "display_name": "Google",
        "official_registrable_domains": ["google.com", "youtube.com", "gmail.com", "google.co.in", "google.co.uk"],
        "keywords": ["google", "gmail", "youtube"],
    },
    "microsoft": {
        "display_name": "Microsoft",
        "official_registrable_domains": ["microsoft.com", "live.com", "office.com", "outlook.com", "microsoftonline.com"],
        "keywords": ["microsoft", "office365", "outlook", "onedrive", "azure"],
    },
    "apple": {
        "display_name": "Apple",
        "official_registrable_domains": ["apple.com", "icloud.com"],
        "keywords": ["apple", "icloud"],
    },
    "amazon": {
        "display_name": "Amazon",
        "official_registrable_domains": ["amazon.com", "amazon.in", "amazon.co.uk", "amazon.de"],
        "keywords": ["amazon", "primevideo", "aws"],
    },
    "netflix": {
        "display_name": "Netflix",
        "official_registrable_domains": ["netflix.com"],
        "keywords": ["netflix"],
    },
    "chase": {
        "display_name": "JPMorgan Chase",
        "official_registrable_domains": ["chase.com"],
        "keywords": ["chase"],
    },
    "wellsfargo": {
        "display_name": "Wells Fargo",
        "official_registrable_domains": ["wellsfargo.com"],
        "keywords": ["wellsfargo"],
    },
    "bofa": {
        "display_name": "Bank of America",
        "official_registrable_domains": ["bankofamerica.com", "bofa.com"],
        "keywords": ["bankofamerica", "bofa"],
    },
    "facebook": {
        "display_name": "Facebook",
        "official_registrable_domains": ["facebook.com", "fb.com", "meta.com"],
        "keywords": ["facebook", "meta"],
    },
    "instagram": {
        "display_name": "Instagram",
        "official_registrable_domains": ["instagram.com"],
        "keywords": ["instagram"],
    },
    "whatsapp": {
        "display_name": "WhatsApp",
        "official_registrable_domains": ["whatsapp.com"],
        "keywords": ["whatsapp"],
    },
    "linkedin": {
        "display_name": "LinkedIn",
        "official_registrable_domains": ["linkedin.com"],
        "keywords": ["linkedin"],
    },
    "adobe": {
        "display_name": "Adobe",
        "official_registrable_domains": ["adobe.com"],
        "keywords": ["adobe"],
    },
    "dhl": {
        "display_name": "DHL",
        "official_registrable_domains": ["dhl.com", "express.dhl"],
        "keywords": ["dhl"],
    },
    "fedex": {
        "display_name": "FedEx",
        "official_registrable_domains": ["fedex.com"],
        "keywords": ["fedex"],
    },
    "hdfc": {
        "display_name": "HDFC Bank",
        "official_registrable_domains": ["hdfcbank.com", "hdfc.com"],
        "keywords": ["hdfc", "hdfcbank"],
    },
}

LEETSPEAK_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "8": "b",
    "@": "a",
    "$": "s",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the standard Levenshtein distance between two strings."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i, c1 in enumerate(s1, 1):
        curr[0] = i
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev
    return prev[len2]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Damerau-Levenshtein distance between two strings.
    
    Permits insertion, deletion, substitution, and adjacent transposition.
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    d = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    max_dist = len1 + len2
    d[0][0] = max_dist

    for i in range(len1 + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(len2 + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    last_seen: dict[str, int] = {}

    for i in range(1, len1 + 1):
        db = 0
        c1 = s1[i - 1]
        for j in range(1, len2 + 1):
            c2 = s2[j - 1]
            k = last_seen.get(c2, 0)
            l = db
            cost = 0 if c1 == c2 else 1
            if cost == 0:
                db = j

            d[i + 1][j + 1] = min(
                d[i][j + 1] + 1,                        # deletion
                d[i + 1][j] + 1,                        # insertion
                d[i][j] + cost,                         # substitution
                d[k][l] + (i - k - 1) + 1 + (j - l - 1) # transposition
            )
        last_seen[c1] = i

    return d[len1 + 1][len2 + 1]


def compute_normalized_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity score in range [0.0, 1.0] using Damerau-Levenshtein."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = damerau_levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / float(max_len)))


@dataclass(frozen=True)
class TyposquattingEvidence:
    """Structured evidence returned by the TyposquattingDetector."""

    registrable_domain: str
    domain_name: str
    closest_brand: str | None
    closest_brand_display: str | None
    matched_official_domain: str | None
    is_official_brand_domain: bool
    levenshtein_distance: int | None
    damerau_levenshtein_distance: int | None
    similarity_score: float
    attack_type: str | None
    is_suspicious_typosquat: bool
    evidence_description: str


class TyposquattingDetector:
    """Detects typosquatting, combosquatting, and subdomain impersonation attacks."""

    def __init__(
        self,
        brands_manifest_path: str | Path = "data/brands/reference_gallery/manifest_brands.json",
        psl_extractor: PSLExtractor | None = None,
    ) -> None:
        self.psl_extractor = psl_extractor or PSLExtractor()
        self.brands: dict[str, dict[str, Any]] = dict(DEFAULT_PROTECTED_BRANDS)
        self._load_brand_manifest(brands_manifest_path)

    def _load_brand_manifest(self, path: str | Path) -> None:
        """Load and augment brand targets from JSON manifest if present on disk."""
        manifest_file = Path(path)
        if not manifest_file.exists():
            return
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for brand_entry in data.get("brands", []):
                bid = brand_entry.get("brand_id", "").lower()
                if not bid:
                    continue
                display = brand_entry.get("display_name", bid.title())
                official_doms = [d.lower() for d in brand_entry.get("official_registrable_domains", [])]
                synonyms = [s.lower() for s in brand_entry.get("synonyms", [])]
                keywords = list(set([bid] + synonyms))

                # Merge with existing defaults
                if bid in self.brands:
                    merged_doms = list(set(self.brands[bid]["official_registrable_domains"] + official_doms))
                    merged_kw = list(set(self.brands[bid]["keywords"] + keywords))
                    self.brands[bid]["official_registrable_domains"] = merged_doms
                    self.brands[bid]["keywords"] = merged_kw
                else:
                    self.brands[bid] = {
                        "display_name": display,
                        "official_registrable_domains": official_doms,
                        "keywords": keywords,
                    }
        except Exception as e:
            logger.warning(f"Failed to load brand manifest from {path}: {e}")

    @staticmethod
    def _normalize_leetspeak(text: str) -> str:
        """Replace common leetspeak substitutions with ASCII Latin letters."""
        text = text.replace("vv", "w")
        return "".join(LEETSPEAK_MAP.get(c, c) for c in text)

    def analyze(self, target: str | ParsedDomainInfo) -> TyposquattingEvidence:
        """Analyze a URL, hostname, or parsed domain info for typosquatting attacks."""
        if isinstance(target, ParsedDomainInfo):
            info = target
        else:
            info = self.psl_extractor.extract_from_url(target)

        reg_domain = info.registrable_domain.lower()
        dom_name = info.domain_name.lower()
        subdomain = info.subdomain.lower()

        if not reg_domain or info.is_ip:
            return TyposquattingEvidence(
                registrable_domain=reg_domain,
                domain_name=dom_name,
                closest_brand=None,
                closest_brand_display=None,
                matched_official_domain=None,
                is_official_brand_domain=False,
                levenshtein_distance=None,
                damerau_levenshtein_distance=None,
                similarity_score=0.0,
                attack_type=None,
                is_suspicious_typosquat=False,
                evidence_description="IP address or empty domain; no typosquatting evaluation.",
            )

        # 1. Check if the domain is an OFFICIAL brand domain
        for brand_id, bdata in self.brands.items():
            if reg_domain in bdata["official_registrable_domains"]:
                return TyposquattingEvidence(
                    registrable_domain=reg_domain,
                    domain_name=dom_name,
                    closest_brand=brand_id,
                    closest_brand_display=bdata["display_name"],
                    matched_official_domain=reg_domain,
                    is_official_brand_domain=True,
                    levenshtein_distance=0,
                    damerau_levenshtein_distance=0,
                    similarity_score=1.0,
                    attack_type="official_domain",
                    is_suspicious_typosquat=False,
                    evidence_description=f"Verified official domain for {bdata['display_name']} ({reg_domain}).",
                )

        # 2. Check for Subdomain Impersonation: Brand token appears in subdomain while registrable domain is third-party
        # e.g., sbi.co.in.attacker.com or paypal.security-login.com
        for brand_id, bdata in self.brands.items():
            for kw in bdata["keywords"]:
                if len(kw) >= 3 and (kw in subdomain):
                    return TyposquattingEvidence(
                        registrable_domain=reg_domain,
                        domain_name=dom_name,
                        closest_brand=brand_id,
                        closest_brand_display=bdata["display_name"],
                        matched_official_domain=None,
                        is_official_brand_domain=False,
                        levenshtein_distance=None,
                        damerau_levenshtein_distance=None,
                        similarity_score=0.95,
                        attack_type="subdomain_impersonation",
                        is_suspicious_typosquat=True,
                        evidence_description=(
                            f"Brand '{bdata['display_name']}' impersonated in subdomain ('{subdomain}') "
                            f"on third-party domain '{reg_domain}'."
                        ),
                    )

        # 3. Check for Combosquatting: Domain name contains brand name with hyphens or security keywords
        # e.g., sbi-secure-login-example.com or paypal-security-update.com
        for brand_id, bdata in self.brands.items():
            for kw in bdata["keywords"]:
                if len(kw) >= 3 and (f"{kw}-" in dom_name or f"-{kw}" in dom_name or (kw in dom_name and len(dom_name) > len(kw) + 3)):
                    return TyposquattingEvidence(
                        registrable_domain=reg_domain,
                        domain_name=dom_name,
                        closest_brand=brand_id,
                        closest_brand_display=bdata["display_name"],
                        matched_official_domain=None,
                        is_official_brand_domain=False,
                        levenshtein_distance=None,
                        damerau_levenshtein_distance=None,
                        similarity_score=0.90,
                        attack_type="combosquatting",
                        is_suspicious_typosquat=True,
                        evidence_description=(
                            f"Combosquatting detected: domain '{dom_name}' embeds protected brand token '{kw}' "
                            f"({bdata['display_name']}) with keywords/hyphens."
                        ),
                    )

        # 4. Check Edit Distance against protected brand names and official domain labels
        # Identify closest brand target using Damerau-Levenshtein
        best_brand: str | None = None
        best_brand_display: str | None = None
        min_damerau: int = 999
        min_lev: int = 999
        max_sim: float = 0.0
        detected_attack: str | None = None

        leet_dom_name = self._normalize_leetspeak(dom_name)

        for brand_id, bdata in self.brands.items():
            # Targets: brand ID, keywords, and domain labels of official domains
            targets = set([brand_id] + bdata["keywords"])
            for off_dom in bdata["official_registrable_domains"]:
                label = off_dom.split(".")[0]
                targets.add(label)

            for target_token in targets:
                target_token = target_token.lower()
                if not target_token or len(target_token) < 2:
                    continue

                d_dist = damerau_levenshtein_distance(dom_name, target_token)
                l_dist = levenshtein_distance(dom_name, target_token)
                sim = compute_normalized_similarity(dom_name, target_token)

                # Check leetspeak match (e.g. sb1 -> sbi)
                if d_dist > 0 and leet_dom_name == target_token:
                    d_dist = 1
                    l_dist = 1
                    sim = 0.95
                    attack_candidate = "substitution"
                elif d_dist == 1:
                    if len(dom_name) == len(target_token):
                        if l_dist == 2 and d_dist == 1:
                            attack_candidate = "transposition"
                        else:
                            attack_candidate = "substitution"
                    elif len(dom_name) == len(target_token) + 1:
                        attack_candidate = "insertion"
                    elif len(dom_name) == len(target_token) - 1:
                        attack_candidate = "omission"
                    else:
                        attack_candidate = "substitution"
                elif d_dist == 2 and len(dom_name) >= 5 and sim >= 0.70:
                    attack_candidate = "multi_edit_typosquat"
                else:
                    attack_candidate = None

                if d_dist < min_damerau or (d_dist == min_damerau and sim > max_sim):
                    min_damerau = d_dist
                    min_lev = l_dist
                    max_sim = sim
                    best_brand = brand_id
                    best_brand_display = bdata["display_name"]
                    detected_attack = attack_candidate

        # 5. Evaluate typosquatting decision threshold
        # For short brand names (len 2-4 like 'sbi', 'dhl'), distance 1 is high similarity
        # For longer brands (len >= 5 like 'paypal', 'google'), distance <= 2 is typosquatting
        is_typosquat = False
        description = ""

        if best_brand and (min_damerau == 1 or (min_damerau == 2 and max_sim >= 0.75)):
            is_typosquat = True
            description = (
                f"Typosquatting attack ({detected_attack or 'edit_distance'}) detected targeting '{best_brand_display}': "
                f"domain '{dom_name}' has edit distance {min_damerau} (similarity: {max_sim:.2f})."
            )
        elif best_brand:
            description = (
                f"No close typosquatting match. Closest brand '{best_brand_display}' with edit distance {min_damerau} "
                f"(similarity: {max_sim:.2f})."
            )
        else:
            description = "No protected brand similarity detected."

        return TyposquattingEvidence(
            registrable_domain=reg_domain,
            domain_name=dom_name,
            closest_brand=best_brand,
            closest_brand_display=best_brand_display,
            matched_official_domain=None,
            is_official_brand_domain=False,
            levenshtein_distance=min_lev if best_brand else None,
            damerau_levenshtein_distance=min_damerau if best_brand else None,
            similarity_score=float(max_sim),
            attack_type=detected_attack,
            is_suspicious_typosquat=is_typosquat,
            evidence_description=description,
        )
