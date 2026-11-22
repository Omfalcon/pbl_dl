"""Unicode Homoglyph and Mixed-Script Domain Detection Module for PhishVision AI.

Detects cross-script visual spoofing, IDN homograph attacks, and Unicode confusable
characters (e.g. Cyrillic/Greek characters masquerading as Latin characters).
Preserves the original hostname while producing explainable structured evidence.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.intelligence.url_domain.psl_extractor import PSLExtractor, ParsedDomainInfo


# Comprehensive mapping of Unicode lookalikes (confusables) to canonical Latin ASCII
CONFUSABLES_MAP: dict[str, str] = {
    # Cyrillic small letters
    "\u0430": "a",  # Cyrillic small letter a
    "\u0441": "c",  # Cyrillic small letter es
    "\u0435": "e",  # Cyrillic small letter ie
    "\u043e": "o",  # Cyrillic small letter o
    "\u0440": "p",  # Cyrillic small letter er
    "\u0455": "s",  # Cyrillic small letter dze
    "\u0456": "i",  # Cyrillic small letter byelorussian-ukrainian i
    "\u0458": "j",  # Cyrillic small letter je
    "\u0443": "y",  # Cyrillic small letter u
    "\u0445": "x",  # Cyrillic small letter ha
    "\u0501": "d",  # Cyrillic small letter komi de
    "\u051b": "q",  # Cyrillic small letter qa
    "\u051d": "w",  # Cyrillic small letter we
    "\u0432": "b",  # Cyrillic small letter ve
    "\u0442": "t",  # Cyrillic small letter te
    "\u043f": "n",  # Cyrillic small letter pe
    "\u043c": "m",  # Cyrillic small letter em
    "\u043d": "h",  # Cyrillic small letter en
    "\u043a": "k",  # Cyrillic small letter ka
    "\u0433": "r",  # Cyrillic small letter ghe
    "\u045d": "i",  # Cyrillic small letter i with grave
    "\u0450": "e",  # Cyrillic small letter ie with grave
    # Cyrillic capital letters
    "\u0410": "A",  # Cyrillic capital letter a
    "\u0412": "B",  # Cyrillic capital letter ve
    "\u0421": "C",  # Cyrillic capital letter es
    "\u0415": "E",  # Cyrillic capital letter ie
    "\u041d": "H",  # Cyrillic capital letter en
    "\u0406": "I",  # Cyrillic capital letter byelorussian-ukrainian i
    "\u0408": "J",  # Cyrillic capital letter je
    "\u041a": "K",  # Cyrillic capital letter ka
    "\u041c": "M",  # Cyrillic capital letter em
    "\u041e": "O",  # Cyrillic capital letter o
    "\u0420": "P",  # Cyrillic capital letter er
    "\u0405": "S",  # Cyrillic capital letter dze
    "\u0422": "T",  # Cyrillic capital letter te
    "\u0425": "X",  # Cyrillic capital letter ha
    "\u04AE": "Y",  # Cyrillic capital letter straight u
    # Greek small letters
    "\u03b1": "a",  # Greek small letter alpha
    "\u03b2": "b",  # Greek small letter beta
    "\u03b5": "e",  # Greek small letter epsilon
    "\u03b9": "i",  # Greek small letter iota
    "\u03ba": "k",  # Greek small letter kappa
    "\u03bd": "v",  # Greek small letter nu
    "\u03bf": "o",  # Greek small letter omicron
    "\u03c1": "p",  # Greek small letter rho
    "\u03c4": "t",  # Greek small letter tau
    "\u03c5": "u",  # Greek small letter upsilon
    "\u03c7": "x",  # Greek small letter chi
    "\u03c9": "w",  # Greek small letter omega
    # Greek capital letters
    "\u0391": "A",  # Greek capital letter alpha
    "\u0392": "B",  # Greek capital letter beta
    "\u0395": "E",  # Greek capital letter epsilon
    "\u0397": "H",  # Greek capital letter eta
    "\u0399": "I",  # Greek capital letter iota
    "\u039a": "K",  # Greek capital letter kappa
    "\u039c": "M",  # Greek capital letter mu
    "\u039d": "N",  # Greek capital letter nu
    "\u039f": "O",  # Greek capital letter omicron
    "\u03a1": "P",  # Greek capital letter rho
    "\u03a4": "T",  # Greek capital letter tau
    "\u03a5": "Y",  # Greek capital letter upsilon
    "\u03a7": "X",  # Greek capital letter chi
    "\u0396": "Z",  # Greek capital letter zeta
    # IPA / Phonetic extensions
    "\u0251": "a",  # Latin small letter alpha
    "\u0261": "g",  # Latin small letter script g
    "\u0269": "i",  # Latin small letter iota
    "\u028c": "v",  # Latin small letter turned v
    # Common fullwidth forms
    "\uff41": "a",
    "\uff42": "b",
    "\uff43": "c",
    "\uff44": "d",
    "\uff45": "e",
    "\uff4f": "o",
    "\uff50": "p",
}


def strip_diacritics(text: str) -> str:
    """Normalize Unicode text by stripping combining diacritical marks (e.g. café -> cafe).
    
    Separates standard Latin diacritic normalization from cross-script confusable detection.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@dataclass(frozen=True)
class HomoglyphEvidence:
    """Structured evidence returned by the HomoglyphDetector."""

    original_hostname: str
    normalized_hostname: str
    unicode_hostname: str
    is_idn: bool
    detected_scripts: list[str]
    has_mixed_scripts: bool
    confusable_character_count: int
    confusable_characters: list[dict[str, Any]]
    canonical_form: str
    diacritic_normalized_form: str
    is_suspicious_homoglyph: bool
    evidence_description: str


class HomoglyphDetector:
    """Analyzes URLs and hostnames for Unicode homoglyphs, confusable characters, and mixed scripts."""

    def __init__(self, psl_extractor: PSLExtractor | None = None) -> None:
        self.psl_extractor = psl_extractor or PSLExtractor()

    @staticmethod
    def get_character_script(char: str) -> str:
        """Classify a character into its primary script family."""
        if char in ".-_0123456789":
            return "COMMON"

        code = ord(char)
        # Fast ASCII check
        if (65 <= code <= 90) or (97 <= code <= 122):
            return "LATIN"

        try:
            name = unicodedata.name(char, "").upper()
        except ValueError:
            return "UNKNOWN"

        if name.startswith("LATIN"):
            return "LATIN"
        if name.startswith("CYRILLIC"):
            return "CYRILLIC"
        if name.startswith("GREEK"):
            return "GREEK"
        if name.startswith("ARABIC"):
            return "ARABIC"
        if name.startswith("DEVANAGARI"):
            return "DEVANAGARI"
        if name.startswith("HEBREW"):
            return "HEBREW"
        if any(name.startswith(p) for p in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            return "CJK"

        return "OTHER"

    def analyze(self, target: str | ParsedDomainInfo) -> HomoglyphEvidence:
        """Analyze a URL, hostname, or parsed domain info for homoglyphs."""
        if isinstance(target, ParsedDomainInfo):
            info = target
        else:
            info = self.psl_extractor.extract_from_url(target)

        orig_host = info.original_hostname
        norm_host = info.normalized_hostname
        unicode_host = info.unicode_hostname
        is_idn = info.is_punycode or (orig_host != unicode_host) or any(ord(c) > 127 for c in orig_host)

        if not unicode_host:
            return HomoglyphEvidence(
                original_hostname="",
                normalized_hostname="",
                unicode_hostname="",
                is_idn=False,
                detected_scripts=[],
                has_mixed_scripts=False,
                confusable_character_count=0,
                confusable_characters=[],
                canonical_form="",
                diacritic_normalized_form="",
                is_suspicious_homoglyph=False,
                evidence_description="Empty hostname provided.",
            )

        # Labels decomposition
        labels = unicode_host.split(".")
        all_scripts: set[str] = set()
        label_scripts: list[set[str]] = []
        confusables: list[dict[str, Any]] = []
        canonical_chars: list[str] = []

        has_mixed = False

        for label in labels:
            curr_label_scripts: set[str] = set()
            for pos, char in enumerate(label):
                script = self.get_character_script(char)
                if script != "COMMON":
                    curr_label_scripts.add(script)
                    all_scripts.add(script)

                if char in CONFUSABLES_MAP:
                    canonical_rep = CONFUSABLES_MAP[char]
                    confusables.append(
                        {
                            "char": char,
                            "codepoint": f"U+{ord(char):04X}",
                            "name": unicodedata.name(char, "UNKNOWN"),
                            "script": script,
                            "canonical_char": canonical_rep,
                            "position_in_label": pos,
                            "label": label,
                        }
                    )
            label_scripts.append(curr_label_scripts)
            # Mixed script detected if a single label mixes Latin with non-Latin scripts (e.g. Cyrillic/Greek)
            if "LATIN" in curr_label_scripts and len(curr_label_scripts) > 1:
                has_mixed = True

        # Build full canonical form (mapping confusables to canonical ASCII)
        for char in unicode_host:
            canonical_chars.append(CONFUSABLES_MAP.get(char, char))
        canonical_form = "".join(canonical_chars)
        diacritic_norm = strip_diacritics(unicode_host)

        sorted_scripts = sorted(list(all_scripts))

        # Determine suspicion:
        # 1. Mixed script in a label: HIGHLY SUSPICIOUS (e.g., Latin mixed with Cyrillic or Greek)
        # 2. Whole-script confusable spoof under ASCII TLD (all characters in non-Latin label map to Latin ASCII)
        # 3. Legitimate single-script IDN (e.g., pure Cyrillic президент.рф or pure Latin with diacritics café.com): NOT suspicious
        is_suspicious = False
        reasons: list[str] = []

        if has_mixed:
            is_suspicious = True
            reasons.append(
                f"Mixed-script label detected combining LATIN with {', '.join(s for s in sorted_scripts if s != 'LATIN')}."
            )

        if confusables and not has_mixed:
            # Check if domain uses a standard ASCII TLD (like .com, .org) with confusable characters
            tld = info.suffix.lower()
            ascii_tlds = {"com", "net", "org", "co.uk", "co.in", "io", "xyz", "info", "biz"}

            # Check if any label is a whole-script confusable spoof (all non-Latin characters map to ASCII)
            is_whole_script_spoof = False
            for label in labels:
                if label != tld and any(c in CONFUSABLES_MAP for c in label):
                    if all((c in CONFUSABLES_MAP) or (ord(c) <= 127 and (c.isalnum() or c in "-_")) for c in label):
                        is_whole_script_spoof = True
                        break

            if tld in ascii_tlds and is_whole_script_spoof:
                is_suspicious = True
                reasons.append(
                    f"Whole-script confusable spoof detected: non-Latin label maps to Latin ASCII under standard TLD '.{tld}'."
                )
            else:
                reasons.append(
                    f"Single-script Internationalized Domain ({', '.join(sorted_scripts)}) with {len(confusables)} native script characters."
                )

        if not confusables and not has_mixed:
            if is_idn:
                reasons.append(f"Legitimate Internationalized Domain Name using script(s): {', '.join(sorted_scripts) or 'COMMON'}.")
            else:
                reasons.append("Standard ASCII domain with zero confusable Unicode characters.")

        description = " ".join(reasons)

        return HomoglyphEvidence(
            original_hostname=orig_host,
            normalized_hostname=norm_host,
            unicode_hostname=unicode_host,
            is_idn=is_idn,
            detected_scripts=sorted_scripts,
            has_mixed_scripts=has_mixed,
            confusable_character_count=len(confusables),
            confusable_characters=confusables,
            canonical_form=canonical_form,
            diacritic_normalized_form=diacritic_norm,
            is_suspicious_homoglyph=is_suspicious,
            evidence_description=description,
        )
