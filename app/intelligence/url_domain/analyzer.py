"""Domain Intelligence Analyzer for PhishVision AI.

Combines Public Suffix List (PSL) domain parsing, Unicode homoglyph detection,
and typosquatting metric analysis into a unified, explainable structured evidence object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.intelligence.url_domain.homoglyph import HomoglyphDetector, HomoglyphEvidence
from app.intelligence.url_domain.psl_extractor import PSLExtractor, ParsedDomainInfo
from app.intelligence.url_domain.typosquatting import TyposquattingDetector, TyposquattingEvidence


@dataclass(frozen=True)
class DomainIntelligenceEvidence:
    """Consolidated structured domain intelligence evidence."""

    original_url: str
    original_hostname: str
    normalized_hostname: str
    registrable_domain: str
    domain_name: str
    suffix: str
    subdomain: str
    is_ip: bool
    is_punycode: bool
    # Homoglyph evidence
    detected_scripts: list[str]
    has_mixed_scripts: bool
    confusable_character_count: int
    confusable_character_mappings: list[dict[str, Any]]
    canonical_homoglyph_form: str
    is_suspicious_homoglyph: bool
    homoglyph_explanation: str
    # Typosquatting evidence
    closest_protected_brand: str | None
    closest_brand_display: str | None
    is_official_brand_domain: bool
    edit_distance: int | None
    similarity_score: float
    attack_type: str | None
    is_suspicious_typosquat: bool
    typosquat_explanation: str
    # High-level aggregate signals (for risk fusion consumption)
    has_suspicious_domain_signals: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence dataclass into a JSON-serializable dictionary."""
        return asdict(self)


class DomainIntelligenceAnalyzer:
    """Unified engine coordinating PSL parsing, homoglyph detection, and typosquatting analysis."""

    def __init__(
        self,
        psl_extractor: PSLExtractor | None = None,
        homoglyph_detector: HomoglyphDetector | None = None,
        typosquatting_detector: TyposquattingDetector | None = None,
    ) -> None:
        self.psl_extractor = psl_extractor or PSLExtractor()
        self.homoglyph_detector = homoglyph_detector or HomoglyphDetector(self.psl_extractor)
        self.typosquatting_detector = typosquatting_detector or TyposquattingDetector(psl_extractor=self.psl_extractor)

    def analyze(self, url: str) -> DomainIntelligenceEvidence:
        """Perform comprehensive domain intelligence analysis on a URL or hostname."""
        parsed: ParsedDomainInfo = self.psl_extractor.extract_from_url(url)
        homo_ev: HomoglyphEvidence = self.homoglyph_detector.analyze(parsed)
        typo_ev: TyposquattingEvidence = self.typosquatting_detector.analyze(parsed)

        # Combine signals into explainable summary
        suspicious_signals: list[str] = []
        if homo_ev.is_suspicious_homoglyph:
            suspicious_signals.append(f"Homoglyph: {homo_ev.evidence_description}")
        if typo_ev.is_suspicious_typosquat:
            suspicious_signals.append(f"Typosquatting: {typo_ev.evidence_description}")

        has_suspicious = len(suspicious_signals) > 0 and not typo_ev.is_official_brand_domain

        if typo_ev.is_official_brand_domain:
            summary = typo_ev.evidence_description
        elif suspicious_signals:
            summary = " | ".join(suspicious_signals)
        else:
            summary = "Domain appears neutral with no homoglyph or protected brand typosquatting signals detected."

        return DomainIntelligenceEvidence(
            original_url=parsed.raw_url,
            original_hostname=parsed.original_hostname,
            normalized_hostname=parsed.normalized_hostname,
            registrable_domain=parsed.registrable_domain,
            domain_name=parsed.domain_name,
            suffix=parsed.suffix,
            subdomain=parsed.subdomain,
            is_ip=parsed.is_ip,
            is_punycode=parsed.is_punycode,
            detected_scripts=homo_ev.detected_scripts,
            has_mixed_scripts=homo_ev.has_mixed_scripts,
            confusable_character_count=homo_ev.confusable_character_count,
            confusable_character_mappings=homo_ev.confusable_characters,
            canonical_homoglyph_form=homo_ev.canonical_form,
            is_suspicious_homoglyph=homo_ev.is_suspicious_homoglyph,
            homoglyph_explanation=homo_ev.evidence_description,
            closest_protected_brand=typo_ev.closest_brand,
            closest_brand_display=typo_ev.closest_brand_display,
            is_official_brand_domain=typo_ev.is_official_brand_domain,
            edit_distance=typo_ev.damerau_levenshtein_distance,
            similarity_score=typo_ev.similarity_score,
            attack_type=typo_ev.attack_type,
            is_suspicious_typosquat=typo_ev.is_suspicious_typosquat,
            typosquat_explanation=typo_ev.evidence_description,
            has_suspicious_domain_signals=has_suspicious,
            explanation=summary,
        )
