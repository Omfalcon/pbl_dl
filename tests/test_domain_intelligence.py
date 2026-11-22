"""Comprehensive Unit Tests for Domain Intelligence: PSL, Homoglyphs, and Typosquatting."""

from __future__ import annotations

import json
import pytest

from app.intelligence.lexical import LexicalFeatureExtractor
from app.intelligence.url_domain.analyzer import DomainIntelligenceAnalyzer
from app.intelligence.url_domain.homoglyph import HomoglyphDetector
from app.intelligence.url_domain.psl_extractor import PSLExtractor
from app.intelligence.url_domain.typosquatting import (
    TyposquattingDetector,
    compute_normalized_similarity,
    damerau_levenshtein_distance,
    levenshtein_distance,
)


class TestPSLExtractor:
    """Verifies Public Suffix List parsing, multi-label TLD handling, and IP/IDN support."""

    @pytest.fixture
    def extractor(self) -> PSLExtractor:
        return PSLExtractor()

    def test_multi_label_tld_extraction(self, extractor: PSLExtractor):
        # sbi.co.in
        info_sbi = extractor.extract_from_url("https://sbi.co.in/portal/login.html")
        assert info_sbi.registrable_domain == "sbi.co.in"
        assert info_sbi.domain_name == "sbi"
        assert info_sbi.suffix == "co.in"
        assert info_sbi.subdomain == ""
        assert not info_sbi.is_ip

        # paypal.co.uk
        info_pp = extractor.extract_from_url("https://paypal.co.uk/signin")
        assert info_pp.registrable_domain == "paypal.co.uk"
        assert info_pp.domain_name == "paypal"
        assert info_pp.suffix == "co.uk"
        assert info_pp.subdomain == ""

        # google.com
        info_gg = extractor.extract_from_url("https://google.com/search?q=test")
        assert info_gg.registrable_domain == "google.com"
        assert info_gg.domain_name == "google"
        assert info_gg.suffix == "com"
        assert info_gg.subdomain == ""

    def test_subdomain_and_attacker_isolation(self, extractor: PSLExtractor):
        # sbi.co.in.attacker.com
        info = extractor.extract_from_url("http://sbi.co.in.attacker.com/steal")
        assert info.registrable_domain == "attacker.com"
        assert info.domain_name == "attacker"
        assert info.suffix == "com"
        assert info.subdomain == "sbi.co.in"
        assert info.normalized_hostname == "sbi.co.in.attacker.com"

        # login.sbi.co.in
        info_login = extractor.extract_from_url("https://login.sbi.co.in/netbanking")
        assert info_login.registrable_domain == "sbi.co.in"
        assert info_login.domain_name == "sbi"
        assert info_login.subdomain == "login"

    def test_ip_address_extraction(self, extractor: PSLExtractor):
        # IPv4
        info_ipv4 = extractor.extract_from_url("http://192.168.1.1:8080/admin")
        assert info_ipv4.is_ip
        assert info_ipv4.registrable_domain == "192.168.1.1"
        assert info_ipv4.port == 8080
        assert info_ipv4.suffix == ""

        # IPv6
        info_ipv6 = extractor.extract_from_url("http://[2001:db8::1]:443/secure")
        assert info_ipv6.is_ip
        assert info_ipv6.registrable_domain == "2001:db8::1"
        assert info_ipv6.port == 443

    def test_punycode_decoding(self, extractor: PSLExtractor):
        # xn--pple-43d.com -> Cyrillic 'a' + pple.com
        info = extractor.extract_from_url("https://xn--pple-43d.com/signin")
        assert info.is_punycode
        assert "xn--pple-43d.com" in info.normalized_hostname
        assert "apple" not in info.unicode_hostname or "а" in info.unicode_hostname


class TestUnicodeHomoglyphs:
    """Verifies Unicode confusable detection, mixed-script identification, and legitimate IDN support."""

    @pytest.fixture
    def detector(self) -> HomoglyphDetector:
        return HomoglyphDetector()

    def test_clean_ascii_domain(self, detector: HomoglyphDetector):
        ev = detector.analyze("https://www.google.com")
        assert not ev.is_suspicious_homoglyph
        assert not ev.has_mixed_scripts
        assert ev.confusable_character_count == 0
        assert ev.detected_scripts == ["LATIN"]

    def test_cyrillic_latin_mixed_script_homoglyph(self, detector: HomoglyphDetector):
        # 'p' (Latin) + '\u0430' (Cyrillic a) + 'ypal.com'
        spoofed_url = "https://p\u0430ypal.com/login"
        ev = detector.analyze(spoofed_url)

        assert ev.is_suspicious_homoglyph
        assert ev.has_mixed_scripts
        assert "CYRILLIC" in ev.detected_scripts
        assert "LATIN" in ev.detected_scripts
        assert ev.confusable_character_count >= 1
        assert any(c["canonical_char"] == "a" for c in ev.confusable_characters)
        assert "paypal.com" in ev.canonical_form

    def test_greek_latin_mixed_script_homoglyph(self, detector: HomoglyphDetector):
        # 'p' (Latin) + '\u03b1' (Greek alpha) + 'ypal.com'
        spoofed_url = "https://p\u03b1ypal.com/signin"
        ev = detector.analyze(spoofed_url)

        assert ev.is_suspicious_homoglyph
        assert ev.has_mixed_scripts
        assert "GREEK" in ev.detected_scripts
        assert "LATIN" in ev.detected_scripts
        assert any(c["canonical_char"] == "a" for c in ev.confusable_characters)

    def test_legitimate_single_script_idn(self, detector: HomoglyphDetector):
        # Pure Cyrillic domain: президент.рф
        legit_cyrillic = "http://президент.рф"
        ev = detector.analyze(legit_cyrillic)

        assert not ev.is_suspicious_homoglyph
        assert not ev.has_mixed_scripts
        assert "LATIN" not in ev.detected_scripts
        assert "CYRILLIC" in ev.detected_scripts

    def test_accented_latin_domains_not_flagged(self, detector: HomoglyphDetector):
        # café.com (Latin with acute accent)
        ev_cafe = detector.analyze("https://café.com")
        assert not ev_cafe.is_suspicious_homoglyph
        assert not ev_cafe.has_mixed_scripts
        assert ev_cafe.confusable_character_count == 0
        assert ev_cafe.detected_scripts == ["LATIN"]
        assert ev_cafe.diacritic_normalized_form == "cafe.com"

        # münchen.de (Latin with diaeresis)
        ev_munchen = detector.analyze("https://münchen.de")
        assert not ev_munchen.is_suspicious_homoglyph
        assert not ev_munchen.has_mixed_scripts
        assert ev_munchen.confusable_character_count == 0
        assert ev_munchen.detected_scripts == ["LATIN"]
        assert ev_munchen.diacritic_normalized_form == "munchen.de"

    def test_punycode_homoglyph_detection(self, detector: HomoglyphDetector):
        # xn--googl-3we.com -> googl + Cyrillic 'e' (\u0435)
        ev = detector.analyze("http://xn--googl-3we.com/update")
        assert ev.is_idn
        assert ev.confusable_character_count >= 1
        assert "google.com" in ev.canonical_form
        assert ev.is_suspicious_homoglyph


class TestTyposquatting:
    """Verifies Levenshtein and Damerau-Levenshtein distance, brand comparison, and attack categorization."""

    @pytest.fixture
    def detector(self) -> TyposquattingDetector:
        return TyposquattingDetector()

    def test_edit_distance_algorithms(self):
        # Exact match
        assert levenshtein_distance("paypal", "paypal") == 0
        assert damerau_levenshtein_distance("paypal", "paypal") == 0

        # Transposition: Damerau=1, Levenshtein=2
        assert damerau_levenshtein_distance("paypal", "paypla") == 1
        assert levenshtein_distance("paypal", "paypla") == 2

        assert damerau_levenshtein_distance("google", "googel") == 1
        assert levenshtein_distance("google", "googel") == 2

        # Substitution: both = 1
        assert damerau_levenshtein_distance("sbi", "sb1") == 1
        assert levenshtein_distance("sbi", "sb1") == 1

        # Insertion / Omission: both = 1
        assert damerau_levenshtein_distance("paypal", "paypaal") == 1
        assert damerau_levenshtein_distance("paypal", "paypl") == 1

        # Normalized similarity
        assert compute_normalized_similarity("paypal", "paypla") > 0.80

    def test_official_brand_domains_not_flagged(self, detector: TyposquattingDetector):
        # sbi.co.in
        ev_sbi = detector.analyze("https://sbi.co.in/portal")
        assert ev_sbi.is_official_brand_domain
        assert not ev_sbi.is_suspicious_typosquat
        assert ev_sbi.closest_brand == "sbi"
        assert ev_sbi.similarity_score == 1.0

        # paypal.co.uk
        ev_pp = detector.analyze("https://paypal.co.uk/myaccount")
        assert ev_pp.is_official_brand_domain
        assert not ev_pp.is_suspicious_typosquat
        assert ev_pp.closest_brand == "paypal"
        assert ev_pp.similarity_score == 1.0

        # google.com
        ev_gg = detector.analyze("https://google.com/search")
        assert ev_gg.is_official_brand_domain
        assert not ev_gg.is_suspicious_typosquat
        assert ev_gg.closest_brand == "google"

    def test_substitution_typosquat_sb1(self, detector: TyposquattingDetector):
        # sb1.co.in -> impersonating sbi
        ev = detector.analyze("https://sb1.co.in/login")
        assert not ev.is_official_brand_domain
        assert ev.is_suspicious_typosquat
        assert ev.closest_brand == "sbi"
        assert ev.damerau_levenshtein_distance == 1
        assert ev.attack_type in ("substitution", "leetspeak_substitution")

    def test_combosquatting_sbi_secure_login(self, detector: TyposquattingDetector):
        # sbi-secure-login-example.com
        ev = detector.analyze("https://sbi-secure-login-example.com/auth")
        assert not ev.is_official_brand_domain
        assert ev.is_suspicious_typosquat
        assert ev.closest_brand == "sbi"
        assert ev.attack_type == "combosquatting"

    def test_subdomain_impersonation_sbi_attacker(self, detector: TyposquattingDetector):
        # sbi.co.in.attacker.com -> registrable domain is attacker.com
        ev = detector.analyze("http://sbi.co.in.attacker.com/harvest")
        assert not ev.is_official_brand_domain
        assert ev.is_suspicious_typosquat
        assert ev.closest_brand == "sbi"
        assert ev.registrable_domain == "attacker.com"
        assert ev.attack_type == "subdomain_impersonation"

    def test_transposition_attacks(self, detector: TyposquattingDetector):
        # paypla.com
        ev = detector.analyze("http://paypla.com/signin")
        assert not ev.is_official_brand_domain
        assert ev.is_suspicious_typosquat
        assert ev.closest_brand == "paypal"
        assert ev.attack_type == "transposition"
        assert ev.damerau_levenshtein_distance == 1

        # googel.com
        ev_gg = detector.analyze("http://googel.com")
        assert not ev_gg.is_official_brand_domain
        assert ev_gg.is_suspicious_typosquat
        assert ev_gg.closest_brand == "google"
        assert ev_gg.attack_type == "transposition"

    def test_insertion_and_omission_attacks(self, detector: TyposquattingDetector):
        # Insertion: paypaal.com
        ev_ins = detector.analyze("http://paypaal.com")
        assert ev_ins.is_suspicious_typosquat
        assert ev_ins.closest_brand == "paypal"
        assert ev_ins.attack_type == "insertion"
        assert ev_ins.damerau_levenshtein_distance == 1

        # Omission: paypl.com
        ev_om = detector.analyze("http://paypl.com")
        assert ev_om.is_suspicious_typosquat
        assert ev_om.closest_brand == "paypal"
        assert ev_om.attack_type == "omission"
        assert ev_om.damerau_levenshtein_distance == 1


class TestDomainIntelligenceIntegration:
    """Verifies end-to-end domain intelligence analyzer integration and structured evidence schema."""

    @pytest.fixture
    def analyzer(self) -> DomainIntelligenceAnalyzer:
        return DomainIntelligenceAnalyzer()

    def test_comprehensive_evidence_fields(self, analyzer: DomainIntelligenceAnalyzer):
        ev = analyzer.analyze("https://login.sb1.co.in/banking")
        data = ev.to_dict()

        # Check required schema fields
        assert "original_url" in data
        assert "original_hostname" in data
        assert "normalized_hostname" in data
        assert "registrable_domain" in data
        assert "detected_scripts" in data
        assert "confusable_character_mappings" in data
        assert "closest_protected_brand" in data
        assert "edit_distance" in data
        assert "similarity_score" in data
        assert "explanation" in data

        # Ensure JSON-serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_lexical_extractor_integration(self):
        extractor = LexicalFeatureExtractor()
        # Verify 40 features still extract perfectly
        feats = extractor.extract_features("https://sbi.co.in/portal")
        assert len(feats) == 40
        assert feats["brand_in_subdomain_or_path"] == 0.0

        # Verify domain intelligence method on extractor
        ev = extractor.analyze_domain("http://sbi.co.in.attacker.com/steal")
        assert ev.registrable_domain == "attacker.com"
        assert ev.closest_protected_brand == "sbi"
        assert ev.is_suspicious_typosquat
        assert ev.attack_type == "subdomain_impersonation"
