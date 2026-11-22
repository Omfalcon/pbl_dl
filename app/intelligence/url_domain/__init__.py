"""URL and Domain Intelligence Package for PhishVision AI.

Provides Public Suffix List (PSL) domain extraction, Unicode homoglyph analysis,
and typosquatting detection for baseline URL verification and multi-evidence risk fusion.
"""

from __future__ import annotations

from app.intelligence.url_domain.analyzer import (
    DomainIntelligenceAnalyzer,
    DomainIntelligenceEvidence,
)
from app.intelligence.url_domain.homoglyph import (
    CONFUSABLES_MAP,
    HomoglyphDetector,
    HomoglyphEvidence,
)
from app.intelligence.url_domain.psl_extractor import (
    ParsedDomainInfo,
    PSLExtractor,
)
from app.intelligence.url_domain.typosquatting import (
    TyposquattingDetector,
    TyposquattingEvidence,
    compute_normalized_similarity,
    damerau_levenshtein_distance,
    levenshtein_distance,
)

__all__ = [
    "PSLExtractor",
    "ParsedDomainInfo",
    "HomoglyphDetector",
    "HomoglyphEvidence",
    "CONFUSABLES_MAP",
    "TyposquattingDetector",
    "TyposquattingEvidence",
    "levenshtein_distance",
    "damerau_levenshtein_distance",
    "compute_normalized_similarity",
    "DomainIntelligenceAnalyzer",
    "DomainIntelligenceEvidence",
]
