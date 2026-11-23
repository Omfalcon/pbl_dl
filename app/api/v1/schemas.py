"""Pydantic Request & Response Schemas for PhishVision AI REST API.

Strictly aligned with API_CONTRACTS.md specifications (Phase 9).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, HttpUrl


class ScanCategoryEnum(str, Enum):
    LEGITIMATE = "LEGITIMATE"
    SUSPICIOUS = "SUSPICIOUS"
    PHISHING = "PHISHING"
    CRITICAL_PHISHING = "CRITICAL_PHISHING"
    UNKNOWN_INSUFFICIENT = "UNKNOWN / INSUFFICIENT EVIDENCE"
    BLOCKED_SSRF = "BLOCKED_SSRF"


class FactorSeverityEnum(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ViewportOptions(BaseModel):
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)


class ScanRequestOptions(BaseModel):
    fast_mode: bool = Field(default=False, description="If true, skips dynamic browser rendering and runs lexical URL models only")
    viewport: ViewportOptions = Field(default_factory=ViewportOptions)
    timeout_seconds: int = Field(default=25, ge=3, le=60)


class ScanRequest(BaseModel):
    url: str = Field(..., description="Full HTTP or HTTPS target URL to scan", examples=["https://secure-login.portal.net/auth"])
    options: ScanRequestOptions = Field(default_factory=ScanRequestOptions)


class ScanSubmitResponse(BaseModel):
    scan_id: str
    status: str
    target_url: str
    timestamp: str
    poll_url: str
    report_url: str


class ContributingFactor(BaseModel):
    vector: str
    severity: FactorSeverityEnum
    description: str
    evidence_ref: str | None = None


class RiskAssessment(BaseModel):
    category: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    contributing_factors: list[ContributingFactor] = Field(default_factory=list)


class URLIntelligenceReport(BaseModel):
    raw_url: str
    canonical_url: str | None = None
    hostname: str | None = None
    registrable_domain: str | None = None
    subdomain: str | None = None
    tld: str | None = None
    is_ip_address: bool = False
    has_punycode: bool = False
    url_entropy: float | None = None
    mlp_lexical_score: float | None = None
    lstm_sequence_score: float | None = None
    composite_url_risk: float | None = None


class DetectionBox(BaseModel):
    class_name: str
    confidence: float
    bbox: dict[str, float]
    consensus_detectors: list[str] = Field(default_factory=list)
    has_mask: bool = False


class BrandSimilarityReport(BaseModel):
    matched_brand: str | None = None
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_asset_id: str | None = None
    is_unauthorized_brand_use: bool = False


class VisualIntelligenceReport(BaseModel):
    screenshot_captured: bool = False
    screenshot_url: str | None = None
    resnet_phishing_probability: float | None = None
    detections: list[DetectionBox] = Field(default_factory=list)
    brand_similarity: BrandSimilarityReport = Field(default_factory=BrandSimilarityReport)


class RedirectHopReport(BaseModel):
    step: int
    url: str
    status_code: int
    domain: str


class FormReport(BaseModel):
    form_id: str
    action_url: str
    action_registrable_domain: str | None = None
    is_external_submission: bool = False
    has_password_field: bool = False


class BehavioralReport(BaseModel):
    redirect_hops_count: int = 0
    redirect_chain: list[RedirectHopReport] = Field(default_factory=list)
    cross_domain_redirects: bool = False
    lstm_sequence_score: float | None = None
    forms: list[FormReport] = Field(default_factory=list)
    has_dom_password_field: bool = False


# ---------------------------------------------------------------------------
# Phase 9: New schema types
# ---------------------------------------------------------------------------

class ScanStageProgress(BaseModel):
    """Real-time status of a single scan pipeline stage."""
    name: str
    status: str  # PENDING / IN_PROGRESS / COMPLETED / FAILED / SKIPPED / UNAVAILABLE
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None
    message: str = ""
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TLSEvidenceReport(BaseModel):
    """Structured TLS/SSL certificate inspection result."""
    tls_available: bool = False
    certificate_valid: bool = False
    issuer: str | None = None
    subject: str | None = None
    common_name: str | None = None
    san: list[str] = Field(default_factory=list)
    not_before: str | None = None
    not_after: str | None = None
    is_expired: bool = False
    days_until_expiry: int | None = None
    sha256_fingerprint: str | None = None
    protocol_version: str | None = None
    cipher_suite: str | None = None
    error: str | None = None


class RedirectIntelligenceReport(BaseModel):
    """Enriched redirect chain analysis."""
    hop_count: int = 0
    unique_domains: list[str] = Field(default_factory=list)
    cross_domain_hop_count: int = 0
    has_cross_domain_redirects: bool = False
    suspicious_redirect_patterns: list[str] = Field(default_factory=list)


class ModelAvailabilityReport(BaseModel):
    """Per-model checkpoint status."""
    model_id: str
    name: str
    status: str  # LOADED / MISSING / UNKNOWN


class PreliminaryURLRisk(BaseModel):
    """URL risk scored before browser rendering (Stage 1 output)."""
    m01_lexical_score: float | None = None
    m01_above_threshold: bool | None = None
    m07_lstm_score: float | None = None
    composite_url_risk: float | None = None
    analysis_stage: str = "pre_browser"
    note: str | None = None


class ScanStatusResponse(BaseModel):
    """Response schema for GET /api/v1/scan/{scan_id}."""
    scan_id: str
    status: str
    target_url: str
    timestamp: str
    current_stage: str | None = None
    progress_percent: float = 0.0
    stages: dict[str, Any] = Field(default_factory=dict)
    poll_url: str
    report_url: str | None = None
    final_verdict: str | None = None
    risk_score: float | None = None


class ScanReport(BaseModel):
    scan_id: str
    status: str
    target_url: str
    timestamp: str
    elapsed_seconds: float = 0.0
    risk_assessment: RiskAssessment
    url_intelligence: URLIntelligenceReport
    visual_intelligence: VisualIntelligenceReport
    behavioral_intelligence: BehavioralReport
    # Phase 9 extended fields
    preliminary_url_risk: dict[str, Any] | None = None
    tls_evidence: dict[str, Any] | None = None
    redirect_intelligence: dict[str, Any] | None = None
    model_availability: list[dict[str, Any]] = Field(default_factory=list)
    scan_stages: dict[str, Any] = Field(default_factory=dict)
    mitigating_evidence: list[str] = Field(default_factory=list)
    unavailable_evidence: list[str] = Field(default_factory=list)
    explanation: str | None = None
    disclaimer: str | None = None


class HealthStatusResponse(BaseModel):
    status: str  # OPERATIONAL, DEGRADED, SERVICE_UNAVAILABLE
    timestamp: str
    device: str
    models: dict[str, Any]
    active_scans_in_memory: int


class BrandAsset(BaseModel):
    brand_id: str
    brand_name: str
    primary_domain: str
    authorized_domains: list[str]
    logo_count: int

