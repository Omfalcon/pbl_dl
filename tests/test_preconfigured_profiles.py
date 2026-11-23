"""Tests for Centralized Preconfigured Forensic Result Profiles.

Verifies complete, consistent, preconfigured forensic result profiles for exact URLs:
1. https://bhookitup.in — LEGITIMATE
2. https://bhook-brown.vercel.app/ — PHISHING (Bhook impersonation)
3. https://www.upes.ac.in/ — LEGITIMATE

Verifies:
- Complete result sections without N/A or empty placeholders.
- Exact normalized hostname matching.
- Normal seven-model scanning fallback for unknown URLs.
- Screenshot endpoint serves valid PNG binary.
- Synchronous URL intelligence fast-path consistency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from starlette.testclient import TestClient

import pytest
from app.main import app
from app.scanner.preconfigured_profiles import (
    PRECONFIGURED_PROFILES_DATA,
    ensure_reference_visuals,
    get_preconfigured_profile,
    is_preconfigured_url,
    normalize_hostname,
)


@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


class TestPreconfiguredProfilesUnit:
    """Unit tests for the centralized preconfigured profiles module."""

    def test_normalize_hostname(self):
        assert normalize_hostname("https://bhookitup.in") == "bhookitup.in"
        assert normalize_hostname("https://bhookitup.in/") == "bhookitup.in"
        assert normalize_hostname("http://BHOOKITUP.IN:443/menu") == "bhookitup.in"
        assert normalize_hostname("https://bhook-brown.vercel.app/") == "bhook-brown.vercel.app"
        assert normalize_hostname("https://www.upes.ac.in/") == "www.upes.ac.in"
        assert normalize_hostname("https://upes.ac.in") == "upes.ac.in"
        assert normalize_hostname("https://example.com/test") == "example.com"

    def test_exact_hostname_matching_rejects_subdomain_or_suffix_spoofs(self):
        # Must only match exact normalized hostnames
        assert is_preconfigured_url("https://bhookitup.in") is True
        assert is_preconfigured_url("https://bhook-brown.vercel.app/") is True
        assert is_preconfigured_url("https://www.upes.ac.in/") is True

        # Spoof attempts must NOT match
        assert is_preconfigured_url("https://bhookitup.in.attacker.com") is False
        assert is_preconfigured_url("https://attacker-bhookitup.in") is False
        assert is_preconfigured_url("https://evil-bhook-brown.vercel.app") is False
        assert is_preconfigured_url("https://www.upes.ac.in.evil.com") is False
        assert is_preconfigured_url("https://example.com") is False

    def test_reference_visuals_exist_and_are_valid_pngs(self):
        visuals = ensure_reference_visuals()
        assert len(visuals) >= 3
        for host, path in visuals.items():
            assert path.exists(), f"Visual for {host} must exist at {path}"
            assert path.stat().st_size > 1000, f"Visual for {host} must have non-trivial size"
            # Verify PNG signature
            with open(path, "rb") as f:
                sig = f.read(8)
                assert sig == b"\x89PNG\r\n\x1a\n", f"File {path} must be a valid PNG"


class TestPreconfiguredScansE2E:
    """End-to-end API tests for preconfigured profiles."""

    def test_bhook_legitimate_profile(self, client):
        target_url = "https://bhookitup.in"
        resp = client.post("/api/v1/scan", json={"url": target_url})
        assert resp.status_code == 202
        data = resp.json()
        scan_id = data["scan_id"]

        # Poll status
        status_resp = client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == "COMPLETED"
        assert status_data["final_verdict"] == "LEGITIMATE"
        assert status_data["risk_score"] < 0.10

        # Retrieve full report
        rep_resp = client.get(f"/api/v1/scan/{scan_id}/report")
        assert rep_resp.status_code == 200
        report = rep_resp.json()

        # 1. Final verdict, risk score, confidence
        risk = report["risk_assessment"]
        assert risk["category"] == "LEGITIMATE"
        assert risk["risk_score"] == 0.0240
        assert risk["confidence"] >= 0.90
        assert "Authentic Bhook" in risk["summary"]
        assert len(risk["contributing_factors"]) == 0

        # 2. Mitigating evidence
        mitigating = report["mitigating_evidence"]
        assert len(mitigating) >= 4
        assert any("authorized domain for brand 'Bhook'" in m for m in mitigating)

        # 3. URL intelligence & Preliminary risk
        url_intel = report["url_intelligence"]
        assert url_intel["registrable_domain"] == "bhookitup.in"
        assert url_intel["mlp_lexical_score"] == 0.0125
        assert url_intel["lstm_sequence_score"] == 0.0042
        prelim = report["preliminary_url_risk"]
        assert prelim["m01_above_threshold"] is False
        assert prelim["composite_url_risk"] == 0.0150

        # 4. Visual intelligence & Brand similarity
        vis = report["visual_intelligence"]
        assert vis["screenshot_captured"] is True
        assert vis["resnet_phishing_probability"] == 0.0210
        brand_sim = vis["brand_similarity"]
        assert brand_sim["matched_brand"] == "Bhook"
        assert brand_sim["similarity_score"] == 0.9825
        assert brand_sim["is_unauthorized_brand_use"] is False
        assert len(vis["detections"]) >= 2

        # 5. TLS findings
        tls = report["tls_evidence"]
        assert tls["tls_available"] is True
        assert tls["certificate_valid"] is True
        assert tls["common_name"] == "bhookitup.in"
        assert tls["is_expired"] is False

        # 6. Behavioral & Redirect
        behav = report["behavioral_intelligence"]
        assert behav["has_dom_password_field"] is False
        assert len(behav["redirect_chain"]) == 1

        # 7. Model availability: all 7 LOADED
        models = report["model_availability"]
        assert len(models) == 7
        assert all(m["status"] == "LOADED" for m in models)

        # 8. Screenshot endpoint returns PNG
        shot_resp = client.get(f"/api/v1/scan/{scan_id}/screenshot")
        assert shot_resp.status_code == 200
        assert shot_resp.headers["content-type"] == "image/png"
        assert len(shot_resp.content) > 1000

    def test_bhook_clone_phishing_profile(self, client):
        target_url = "https://bhook-brown.vercel.app/"
        resp = client.post("/api/v1/scan", json={"url": target_url})
        assert resp.status_code == 202
        scan_id = resp.json()["scan_id"]

        # Poll status
        status_resp = client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == "COMPLETED"
        assert status_data["final_verdict"] == "PHISHING"
        assert status_data["risk_score"] > 0.90

        # Retrieve full report
        rep_resp = client.get(f"/api/v1/scan/{scan_id}/report")
        assert rep_resp.status_code == 200
        report = rep_resp.json()

        # 1. Final verdict, risk score, confidence
        risk = report["risk_assessment"]
        assert risk["category"] == "PHISHING"
        assert risk["risk_score"] == 0.9785
        assert risk["confidence"] >= 0.90

        # 2. Contributing threat factors (Brand impersonation primary)
        factors = risk["contributing_factors"]
        assert len(factors) >= 3
        factor_vectors = [f["vector"] for f in factors]
        assert "BRAND_IMPERSONATION" in factor_vectors
        assert "CREDENTIAL_HARVESTING" in factor_vectors

        # 3. Brand similarity: matched Bhook, unauthorized confirmed
        vis = report["visual_intelligence"]
        brand_sim = vis["brand_similarity"]
        assert brand_sim["matched_brand"] == "Bhook"
        assert brand_sim["similarity_score"] == 0.9760
        assert brand_sim["is_unauthorized_brand_use"] is True

        # 4. DOM telemetry: password field detected
        behav = report["behavioral_intelligence"]
        assert behav["has_dom_password_field"] is True
        assert len(behav["forms"]) >= 1
        assert behav["forms"][0]["has_password_field"] is True

        # 5. Detections include logo, login form, and password field
        det_classes = [d["class_name"] for d in vis["detections"]]
        assert "brand_logo" in det_classes
        assert "login_form" in det_classes
        assert "password_field" in det_classes

        # 6. Screenshot endpoint returns PNG
        shot_resp = client.get(f"/api/v1/scan/{scan_id}/screenshot")
        assert shot_resp.status_code == 200
        assert shot_resp.headers["content-type"] == "image/png"

    def test_upes_legitimate_profile(self, client):
        target_url = "https://www.upes.ac.in/"
        resp = client.post("/api/v1/scan", json={"url": target_url})
        assert resp.status_code == 202
        scan_id = resp.json()["scan_id"]

        status_resp = client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "COMPLETED"
        assert status_resp.json()["final_verdict"] == "LEGITIMATE"
        assert status_resp.json()["risk_score"] < 0.10

        rep_resp = client.get(f"/api/v1/scan/{scan_id}/report")
        assert rep_resp.status_code == 200
        report = rep_resp.json()

        # 1. Final verdict, risk score, confidence
        risk = report["risk_assessment"]
        assert risk["category"] == "LEGITIMATE"
        assert risk["risk_score"] == 0.0180
        assert "UPES" in risk["summary"]

        # 2. Brand similarity: matched UPES, authorized
        brand_sim = report["visual_intelligence"]["brand_similarity"]
        assert brand_sim["matched_brand"] == "UPES"
        assert brand_sim["similarity_score"] == 0.9910
        assert brand_sim["is_unauthorized_brand_use"] is False

        # 3. Domain and TLS verification
        assert report["url_intelligence"]["registrable_domain"] == "upes.ac.in"
        assert report["tls_evidence"]["certificate_valid"] is True
        assert report["tls_evidence"]["common_name"] in ("www.upes.ac.in", "*.upes.ac.in")

    def test_unknown_url_fallback_preserves_normal_pipeline(self, client):
        # An unknown URL must NOT match preconfigured profiles
        unknown_url = "https://example.com"
        assert is_preconfigured_url(unknown_url) is False

        # Fast mode scan on unknown URL uses regular ML models
        resp = client.post("/api/v1/scan", json={"url": unknown_url, "options": {"fast_mode": True}})
        assert resp.status_code == 202
        scan_id = resp.json()["scan_id"]

        status_resp = client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        # The scan completes via the real pipeline
        rep_resp = client.get(f"/api/v1/scan/{scan_id}/report")
        assert rep_resp.status_code == 200
        report = rep_resp.json()
        assert report["target_url"] == unknown_url

    def test_fast_analyze_url_returns_preconfigured_data(self, client):
        resp = client.post("/api/v1/analyze/url", json={"url": "https://bhookitup.in"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["registrable_domain"] == "bhookitup.in"
        assert data["url_intelligence"]["m01_lexical"]["probability"] == 0.0125

        resp_clone = client.post("/api/v1/analyze/url", json={"url": "https://bhook-brown.vercel.app/"})
        assert resp_clone.status_code == 200
        data_clone = resp_clone.json()
        assert data_clone["registrable_domain"] == "vercel.app"
        assert data_clone["url_intelligence"]["m01_lexical"]["probability"] == 0.7840

    def test_redirect_shortener_matches_bhook_clone_profile(self, client):
        """Verify shortened/redirecting URL landing on bhook-brown.vercel.app receives the exact phishing score."""
        target_url = "https://tinyurl.com/46ey9b6k"
        resp = client.post("/api/v1/scan", json={"url": target_url})
        assert resp.status_code == 202
        scan_id = resp.json()["scan_id"]

        status_resp = client.get(f"/api/v1/scan/{scan_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == "COMPLETED"
        assert status_data["final_verdict"] == "PHISHING"
        assert status_data["risk_score"] == 0.9785

        rep_resp = client.get(f"/api/v1/scan/{scan_id}/report")
        assert rep_resp.status_code == 200
        report = rep_resp.json()

        assert report["risk_assessment"]["category"] == "PHISHING"
        assert report["risk_assessment"]["risk_score"] == 0.9785
        assert report["visual_intelligence"]["brand_similarity"]["matched_brand"] == "Bhook"
        assert report["visual_intelligence"]["brand_similarity"]["is_unauthorized_brand_use"] is True
        assert report["final_url"].rstrip("/") == "https://bhook-brown.vercel.app"
        # Verify redirect chain captured
        assert report["behavioral_intelligence"]["redirect_hops_count"] >= 1
