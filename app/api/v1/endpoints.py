"""FastAPI Router for PhishVision AI REST API (v1).

Implements the official API contracts specified in API_CONTRACTS.md:
- POST /api/v1/scan
- GET  /api/v1/scan/{scan_id}
- GET  /api/v1/scan/{scan_id}/report
- GET  /api/v1/scan/{scan_id}/screenshot
- GET  /api/v1/brands
- GET  /api/v1/health
- POST /api/v1/analyze/url
- POST /api/v1/analyze/brand-similarity
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from PIL import Image
import torch

from app.api.v1.schemas import (
    BrandAsset,
    BrandSimilarityReport,
    DetectionBox,
    FactorSeverityEnum,
    HealthStatusResponse,
    RiskAssessment,
    ScanReport,
    ScanRequest,
    ScanSubmitResponse,
    URLIntelligenceReport,
    VisualIntelligenceReport,
    BehavioralReport,
    ContributingFactor,
    RedirectHopReport,
    FormReport,
)
from app.ml.unified_pipeline import UnifiedPhishingPipeline
from app.scanner.adapter import ScannerMLAdapter
from app.scanner.anti_ssrf import SSRFValidationError, validate_url_and_resolve
from app.scanner.models import ScanOptions, ScanStatus
from app.scanner.playwright_scanner import PlaywrightScanner, _get_registrable_domain

logger = logging.getLogger("phishvision.api")

router = APIRouter(prefix="/api/v1", tags=["PhishVision V1"])

# Global singleton pipeline instance initialized on startup
_PIPELINE: UnifiedPhishingPipeline | None = None
_START_TIME = time.time()


def get_pipeline() -> UnifiedPhishingPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("Initializing global UnifiedPhishingPipeline...")
        _PIPELINE = UnifiedPhishingPipeline()
    return _PIPELINE


class ScanStore:
    """In-memory thread-safe store for ongoing and completed scan records.

    LIMITATION & HORIZONTAL SCALING NOTE:
    This store maintains scan state in the process-local memory of a single Python
    runtime. In a multi-worker server deployment (e.g., Uvicorn with multiple workers)
    or across multiple container instances / Kubernetes pods, scan records created
    in one process are not visible to other processes.

    PRODUCTION ARCHITECTURE RECOMMENDATION:
    For distributed or multi-worker production clusters:
    1. Scan Record Store: Replace ScanStore with an external ACID datastore or
       distributed cache (e.g., PostgreSQL or Redis with persistence).
    2. Asynchronous Job Processing: Decouple scans from FastAPI BackgroundTasks
       using a distributed queue (e.g., Celery, ARQ, or Redis Streams).
    3. Artifact & Screenshot Storage: Store screenshots and visual previews in
       shared object storage (e.g., AWS S3, Google Cloud Storage, or MinIO)
       instead of process-local temporary directories.
    """

    def __init__(self) -> None:
        self._scans: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, scan_id: str, requested_url: str) -> dict[str, Any]:
        record = {
            "scan_id": scan_id,
            "target_url": requested_url,
            "status": "IN_PROGRESS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "start_time": time.perf_counter(),
            "elapsed_seconds": 0.0,
            "screenshot_path": None,
            "report": None,
            "error": None,
        }
        async with self._lock:
            self._scans[scan_id] = record
        return record

    async def get(self, scan_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._scans.get(scan_id)

    async def update(self, scan_id: str, **kwargs: Any) -> None:
        async with self._lock:
            if scan_id in self._scans:
                self._scans[scan_id].update(kwargs)

    async def count(self) -> int:
        async with self._lock:
            return len(self._scans)


scan_store = ScanStore()


def _sanitize_report_paths(raw_report: dict[str, Any], scan_id: str) -> dict[str, Any]:
    """Sanitize internal server filesystem paths before presenting report to callers."""
    report = json.loads(json.dumps(raw_report))

    # Replace local file paths with API relative endpoints
    if "visual_intelligence" in report:
        vis = report["visual_intelligence"]
        if vis.get("screenshot_captured") or vis.get("screenshot_path"):
            vis["screenshot_url"] = f"/api/v1/scan/{scan_id}/screenshot"
        vis.pop("screenshot_path", None)
        vis.pop("preview_image_path", None)

    # Sanitize contributing factor evidence paths
    if "risk_assessment" in report:
        ra = report["risk_assessment"]
        if "contributing_factors" in ra:
            for factor in ra["contributing_factors"]:
                if factor.get("evidence_ref") and (":\\" in factor["evidence_ref"] or "/" in factor["evidence_ref"]):
                    factor["evidence_ref"] = Path(factor["evidence_ref"]).name

    return report


# Image safety and upload boundary constraints
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB default ceiling
MAX_IMAGE_DIMENSION = 4096  # Maximum width or height in pixels
MAX_IMAGE_PIXELS = 16_000_000  # Pillow decompression bomb protection threshold (~16 Megapixels)
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP"}


async def _run_scan_job(
    scan_id: str,
    target_url: str,
    fast_mode: bool,
    viewport_width: int,
    viewport_height: int,
    timeout_seconds: int,
) -> None:
    """Asynchronous background task to execute scanning and ML forensic analysis."""
    t0 = time.perf_counter()

    try:
        pipeline = get_pipeline()

        if fast_mode:
            # Fast lexical/sequence analysis path (no browser rendering)
            target_domain = _get_registrable_domain(target_url)
            analysis = await asyncio.to_thread(
                pipeline.analyze,
                url=target_url,
                screenshot=None,
                domain=target_domain,
                sample_id=scan_id,
                render_preview=False,
            )
            elapsed = round(time.perf_counter() - t0, 3)

            scan_report = _build_scan_report_schema(
                scan_id=scan_id,
                status="COMPLETED",
                target_url=target_url,
                elapsed_seconds=elapsed,
                analysis=analysis,
                screenshot_path=None,
            )
            await scan_store.update(
                scan_id,
                status="COMPLETED",
                elapsed_seconds=elapsed,
                report=scan_report,
            )
            return

        # Dynamic browser execution with Anti-SSRF enforcement
        scanner = PlaywrightScanner()
        scan_options = ScanOptions(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            timeout_seconds=timeout_seconds,
            capture_full_page=False,
        )

        scan_result = await scanner.scan(target_url, options=scan_options)

        if scan_result.status == ScanStatus.BLOCKED_SSRF:
            elapsed = round(time.perf_counter() - t0, 3)
            blocked_report = ScannerMLAdapter.build_blocked_or_failed_report(scan_result)
            await scan_store.update(
                scan_id,
                status="BLOCKED_SSRF",
                elapsed_seconds=elapsed,
                report=blocked_report,
            )
            return

        # Adapt scanner output to unified ML pipeline inputs
        pipe_input = ScannerMLAdapter.to_pipeline_input(scan_result)
        analysis = await asyncio.to_thread(
            pipeline.analyze,
            url=pipe_input["url"],
            screenshot=pipe_input["screenshot"],
            domain=pipe_input["domain"],
            redirect_chain=pipe_input["redirect_chain"],
            dom_forms=pipe_input["dom_forms"],
            sample_id=scan_id,
            render_preview=False,
        )

        elapsed = round(time.perf_counter() - t0, 3)
        final_status = "COMPLETED" if scan_result.status == ScanStatus.COMPLETED else "FAILED"

        scan_report = _build_scan_report_schema(
            scan_id=scan_id,
            status=final_status,
            target_url=scan_result.final_url or target_url,
            elapsed_seconds=elapsed,
            analysis=analysis,
            screenshot_path=pipe_input["screenshot"],
            scan_result=scan_result,
        )

        await scan_store.update(
            scan_id,
            status=final_status,
            elapsed_seconds=elapsed,
            screenshot_path=pipe_input["screenshot"],
            report=scan_report,
        )

    except Exception as e:
        logger.error(f"Error executing scan job for {scan_id}: {e}", exc_info=True)
        elapsed = round(time.perf_counter() - t0, 3)
        await scan_store.update(
            scan_id,
            status="FAILED",
            elapsed_seconds=elapsed,
            error=str(e),
        )


def _build_scan_report_schema(
    scan_id: str,
    status: str,
    target_url: str,
    elapsed_seconds: float,
    analysis: dict[str, Any],
    screenshot_path: str | None = None,
    scan_result: Any | None = None,
) -> dict[str, Any]:
    """Assemble final ScanReport conforming strictly to API_CONTRACTS.md."""
    fusion = analysis.get("fusion", {})
    url_res = analysis.get("url_analysis", {})
    vis_res = analysis.get("visual_analysis", {})
    dom_res = analysis.get("dom_evidence", {})

    m01 = url_res.get("m01_lexical", {})
    m07 = url_res.get("m07_lstm", {})
    brand_verif = vis_res.get("brand_verification", {})

    # Contributing factors
    factors = []
    if brand_verif.get("is_brand_impersonation"):
        factors.append({
            "vector": "BRAND_IMPERSONATION",
            "severity": "CRITICAL",
            "description": f"Visual logo match to protected brand '{brand_verif.get('predicted_brand')}' on unauthorized domain.",
            "evidence_ref": f"cosine_similarity:{brand_verif.get('max_similarity', 0.0):.4f}",
        })
    if dom_res.get("has_dom_password_field") or vis_res.get("ui_elements", {}).get("has_password_field"):
        factors.append({
            "vector": "CREDENTIAL_HARVESTING",
            "severity": "HIGH",
            "description": "Password input field detected in rendered UI.",
            "evidence_ref": "form_password_input",
        })
    if m01.get("is_above_calibrated_threshold"):
        factors.append({
            "vector": "URL_LEXICAL_ANOMALY",
            "severity": "MEDIUM",
            "description": f"Lexical MLP risk score ({m01.get('probability', 0.0):.3f}) exceeds calibrated threshold.",
            "evidence_ref": "m01_lexical_score",
        })

    # Detections list
    raw_elements = vis_res.get("ui_elements", {}).get("elements", [])
    detections = []
    for elem in raw_elements:
        box = elem.get("bbox_pixel", [0, 0, 0, 0])
        detections.append({
            "class_name": elem.get("class_name", "unknown"),
            "confidence": round(float(elem.get("confidence", 0.0)), 4),
            "bbox": {
                "x_min": float(box[0]),
                "y_min": float(box[1]),
                "x_max": float(box[2]),
                "y_max": float(box[3]),
            },
            "consensus_detectors": elem.get("detectors", []),
            "has_mask": elem.get("has_mask", False),
        })

    # Redirect chain
    redirect_chain = []
    if scan_result and scan_result.redirect_chain:
        for hop in scan_result.redirect_chain:
            redirect_chain.append({
                "step": hop.step,
                "url": hop.url,
                "status_code": hop.status_code,
                "domain": hop.domain,
            })

    # Forms
    forms = []
    if scan_result and scan_result.forms:
        for f in scan_result.forms:
            forms.append({
                "form_id": f.form_id,
                "action_url": f.action_url,
                "action_registrable_domain": f.action_domain,
                "is_external_submission": f.is_external_submission,
                "has_password_field": f.has_password_field,
            })

    risk_score = float(fusion.get("risk_score", 0.50))
    cat = fusion.get("threat_category", "UNKNOWN / INSUFFICIENT EVIDENCE")
    conf = float(np.abs(risk_score - 0.50) * 2.0) if cat != "UNKNOWN / INSUFFICIENT EVIDENCE" else 0.0

    report = {
        "scan_id": scan_id,
        "status": status,
        "target_url": target_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "risk_assessment": {
            "category": cat,
            "risk_score": round(risk_score, 4),
            "confidence": round(conf, 4),
            "summary": fusion.get("explanation", "Multi-model risk fusion completed."),
            "contributing_factors": factors,
        },
        "url_intelligence": {
            "raw_url": target_url,
            "registrable_domain": _get_registrable_domain(target_url),
            "mlp_lexical_score": m01.get("probability"),
            "lstm_sequence_score": m07.get("probability"),
            "composite_url_risk": url_res.get("composite_url_risk"),
        },
        "visual_intelligence": {
            "screenshot_captured": bool(screenshot_path and Path(screenshot_path).exists()),
            "screenshot_url": f"/api/v1/scan/{scan_id}/screenshot" if screenshot_path else None,
            "resnet_phishing_probability": vis_res.get("m02_phishing_probability"),
            "detections": detections,
            "brand_similarity": {
                "matched_brand": brand_verif.get("predicted_brand"),
                "similarity_score": round(float(brand_verif.get("max_similarity", 0.0)), 4),
                "is_unauthorized_brand_use": bool(brand_verif.get("is_brand_impersonation", False)),
            },
        },
        "behavioral_intelligence": {
            "redirect_hops_count": len(redirect_chain),
            "redirect_chain": redirect_chain,
            "cross_domain_redirects": len({h["domain"] for h in redirect_chain if h["domain"]}) > 1,
            "lstm_sequence_score": m07.get("probability"),
            "forms": forms,
            "has_dom_password_field": dom_res.get("has_dom_password_field", False),
        },
        "explanation": fusion.get("explanation"),
        "disclaimer": analysis.get("disclaimer"),
    }
    return report


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.post("/scan", response_model=ScanSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_scan(req: ScanRequest, background_tasks: BackgroundTasks) -> Any:
    """Submit a target URL for automated website scanning and seven-model forensic analysis."""
    url_str = req.url.strip()

    # 1. Scheme Validation
    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_URL_SCHEME",
                "message": "Only HTTP and HTTPS URLs are permitted for scanning.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # 2. Immediate Anti-SSRF Pre-flight Verification
    val_res = validate_url_and_resolve(url_str)
    if not val_res.is_valid:
        # In fast_mode, dynamic network requests are skipped, so unresolvable domains
        # are permitted for offline lexical/sequence classification unless blocked as private IPs.
        if req.options.fast_mode and val_res.error_code == "DNS_RESOLUTION_FAILED":
            logger.info(f"Fast mode requested for unresolvable domain '{url_str}'; proceeding with offline lexical analysis.")
        else:
            error_code_label = "DNS_RESOLUTION_FAILED" if val_res.error_code == "DNS_RESOLUTION_FAILED" else "SSRF_ATTEMPT_BLOCKED"
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error_code": error_code_label,
                    "message": f"Scan rejected: {val_res.error_message}",
                    "detail": {
                        "error_code": val_res.error_code,
                        "resolved_ips": val_res.resolved_ips,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    scan_id = str(uuid.uuid4())
    await scan_store.create(scan_id=scan_id, requested_url=url_str)

    # 3. Schedule Non-Blocking Background Scan Task via BackgroundTasks
    background_tasks.add_task(
        _run_scan_job,
        scan_id=scan_id,
        target_url=url_str,
        fast_mode=req.options.fast_mode,
        viewport_width=req.options.viewport.width,
        viewport_height=req.options.viewport.height,
        timeout_seconds=req.options.timeout_seconds,
    )

    return {
        "scan_id": scan_id,
        "status": "IN_PROGRESS",
        "target_url": url_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "poll_url": f"/api/v1/scan/{scan_id}",
        "report_url": f"/api/v1/scan/{scan_id}/report",
    }


@router.get("/scan/{scan_id}")
async def get_scan_status(scan_id: str) -> Any:
    """Poll current execution status of an ongoing or completed scan."""
    record = await scan_store.get(scan_id)
    if not record:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error_code": "RESOURCE_NOT_FOUND",
                "message": f"Scan with ID '{scan_id}' was not found.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return {
        "scan_id": record["scan_id"],
        "status": record["status"],
        "target_url": record["target_url"],
        "timestamp": record["timestamp"],
        "poll_url": f"/api/v1/scan/{scan_id}",
        "report_url": f"/api/v1/scan/{scan_id}/report" if record["status"] != "IN_PROGRESS" else None,
    }


@router.get("/scan/{scan_id}/report")
async def get_scan_report(scan_id: str) -> Any:
    """Retrieve full explainable forensic report for a completed scan."""
    record = await scan_store.get(scan_id)
    if not record:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error_code": "RESOURCE_NOT_FOUND",
                "message": f"Scan with ID '{scan_id}' was not found.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    if record["status"] == "IN_PROGRESS":
        return JSONResponse(
            status_code=status.HTTP_425_TOO_EARLY,
            content={
                "error_code": "SCAN_IN_PROGRESS",
                "message": "Scan is still executing. Poll /api/v1/scan/{scan_id} until completed.",
                "scan_id": scan_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    report = record.get("report")
    if not report:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_INFERENCE_ERROR",
                "message": record.get("error") or "Scan completed but report generation failed.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return _sanitize_report_paths(report, scan_id)


@router.get("/scan/{scan_id}/screenshot")
async def get_scan_screenshot(scan_id: str) -> Any:
    """Return captured PNG screenshot for a completed scan."""
    record = await scan_store.get(scan_id)
    if not record:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error_code": "RESOURCE_NOT_FOUND",
                "message": f"Scan with ID '{scan_id}' was not found.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    screenshot_path = record.get("screenshot_path")
    if not screenshot_path or not Path(screenshot_path).exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error_code": "RESOURCE_NOT_FOUND",
                "message": f"No screenshot available for scan ID '{scan_id}'.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return FileResponse(screenshot_path, media_type="image/png")


@router.get("/brands")
async def list_monitored_brands() -> Any:
    """List all monitored reference brands and protected domains."""
    manifest_path = Path("data/brands/reference_gallery/manifest_brands.json")
    if not manifest_path.exists():
        return {"total_brands": 0, "brands": []}

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        brands_list = []
        for b in data.get("brands", []):
            brands_list.append({
                "brand_id": b.get("brand_id"),
                "display_name": b.get("display_name"),
                "category": b.get("category"),
                "official_registrable_domains": b.get("official_registrable_domains", []),
                "authorized_subdomains": b.get("authorized_subdomains", []),
            })
        return {
            "total_brands": len(brands_list),
            "brands": brands_list,
        }
    except Exception as e:
        logger.error(f"Failed to read brand manifest: {e}")
        return {"total_brands": 0, "brands": []}


@router.get("/health", response_model=HealthStatusResponse)
async def health_check() -> Any:
    """Check system health, compute device, and explicit per-model checkpoint status."""
    try:
        pipeline = get_pipeline()
        checkpoint_status = pipeline.get_checkpoint_status()
        device_str = str(pipeline.device)
    except Exception as e:
        logger.error(f"Failed to inspect pipeline health: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "SERVICE_UNAVAILABLE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device": "unknown",
                "models": {},
                "active_scans_in_memory": await scan_store.count(),
                "error": str(e),
            },
        )

    total_models = len(checkpoint_status)
    loaded_count = sum(1 for m in checkpoint_status.values() if m["status"] == "LOADED")

    if total_models > 0 and loaded_count == total_models:
        overall_status = "OPERATIONAL"
    elif loaded_count > 0:
        overall_status = "DEGRADED"
    else:
        overall_status = "SERVICE_UNAVAILABLE"

    http_code = status.HTTP_200_OK if loaded_count > 0 else status.HTTP_503_SERVICE_UNAVAILABLE

    content = {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device_str,
        "models": checkpoint_status,
        "active_scans_in_memory": await scan_store.count(),
    }

    if http_code != status.HTTP_200_OK:
        return JSONResponse(status_code=http_code, content=content)
    return content


@router.post("/analyze/url")
async def analyze_url_fast(req: ScanRequest) -> Any:
    """Synchronous fast-path lexical and sequence URL intelligence."""
    pipeline = get_pipeline()
    url_str = req.url.strip()

    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_URL_SCHEME",
                "message": "Only HTTP and HTTPS URLs are permitted.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    analysis = await asyncio.to_thread(pipeline._analyze_urls, url_str)
    return {
        "target_url": url_str,
        "registrable_domain": _get_registrable_domain(url_str),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url_intelligence": analysis,
    }


@router.post("/analyze/brand-similarity")
async def analyze_brand_similarity(
    image: UploadFile = File(...),
    target_domain: str = "",
) -> Any:
    """Compare an uploaded image/logo against reference brands via Siamese CNN with DOS protection."""
    pipeline = get_pipeline()
    if not pipeline.m03_available or pipeline.siamese_matcher is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": "INTERNAL_INFERENCE_ERROR",
                "message": "M-03 Siamese Brand Matcher is not loaded or weights are missing.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    max_upload_bytes = int(os.getenv("PHISHVISION_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))

    # Stream upload in bounded chunks to prevent unbounded memory allocation
    chunk_size = 64 * 1024
    raw_bytes = bytearray()
    while chunk := await image.read(chunk_size):
        raw_bytes.extend(chunk)
        if len(raw_bytes) > max_upload_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error_code": "PAYLOAD_TOO_LARGE",
                    "message": f"Uploaded image exceeds maximum permitted size of {max_upload_bytes} bytes ({max_upload_bytes // (1024 * 1024)} MB).",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    if len(raw_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "EMPTY_PAYLOAD",
                "message": "Uploaded file is empty (0 bytes).",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Decode and validate image safely against decompression bombs and corrupted payloads
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(io.BytesIO(raw_bytes))
        img_format = (img.format or "").upper()
        if img_format not in ALLOWED_IMAGE_FORMATS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error_code": "UNSUPPORTED_IMAGE_FORMAT",
                    "message": f"Unsupported image format '{img_format}'. Allowed formats: {sorted(list(ALLOWED_IMAGE_FORMATS))}.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or (width * height) > MAX_IMAGE_PIXELS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error_code": "IMAGE_DIMENSIONS_TOO_LARGE",
                    "message": f"Image dimensions {width}x{height} ({width * height} pixels) exceed safety limits (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}, {MAX_IMAGE_PIXELS} pixels).",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        img = img.convert("RGB")
    except Image.DecompressionBombError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "DECOMPRESSION_BOMB_DETECTED",
                "message": f"Image decompression bomb protection triggered: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_IMAGE_PAYLOAD",
                "message": f"Failed to decode image payload: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    try:
        res = await asyncio.to_thread(
            pipeline.siamese_matcher.verify_brand_authorization,
            query_image=img,
            target_registrable_domain=target_domain.strip(),
        )
        return {
            "target_domain": target_domain,
            "filename": image.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "brand_similarity": res,
        }
    except Exception as e:
        logger.error(f"Brand similarity analysis failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_INFERENCE_ERROR",
                "message": f"Inference execution failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
