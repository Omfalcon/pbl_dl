"""Pydantic Request & Response Schemas for PhishVision AI REST API.

Strictly aligned with API_CONTRACTS.md specifications.
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
    timeout_seconds: int = Field(default=15, ge=3, le=30)


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
