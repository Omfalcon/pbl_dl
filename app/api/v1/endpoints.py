"""FastAPI Router for PhishVision AI REST API (v1).

Implements Phase 9 six-stage scan pipeline with real-time stage progress tracking.

API Endpoints:
- POST /api/v1/scan                            - Submit scan for async execution
- GET  /api/v1/scan/{scan_id}                  - Poll stage progress and overall status
- GET  /api/v1/scan/{scan_id}/report           - Retrieve full explainable forensic report
- GET  /api/v1/scan/{scan_id}/screenshot       - Serve captured PNG screenshot
- GET  /api/v1/brands                          - List monitored reference brands
- GET  /api/v1/health                          - System health and model checkpoint status
- POST /api/v1/analyze/url                     - Synchronous fast-path lexical analysis
- POST /api/v1/analyze/brand-similarity        - Upload image for brand similarity check

Six Pipeline Stages (in execution order):
  URL_CHECK -> REDIRECT_INVESTIGATION -> TLS_INSPECTION ->
  PLAYWRIGHT_CAPTURE -> VISUAL_ANALYSIS -> EVIDENCE_FUSION -> COMPLETED
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
from app.scanner.pipeline_tracker import ScanPipelineTracker
from app.scanner.playwright_scanner import PlaywrightScanner, _get_registrable_domain
from app.scanner.preconfigured_profiles import get_preconfigured_profile, is_preconfigured_url
from app.scanner.tls_inspector import inspect_tls_certificate

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
            # Phase 9: six-stage progress tracking
            "current_stage": "URL_CHECK",
            "progress_percent": 0.0,
            "stages": {},
            "preliminary_url_risk": None,
            "tls_evidence": None,
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
    """Asynchronous background task executing the six-stage PhishVision scan pipeline.

    Stages (in order):
        1. URL_CHECK           – Anti-SSRF + preliminary M-01/M-07 URL risk
        2. REDIRECT_INVESTIGATION – Redirect chain enrichment (dynamic mode only)
        3. TLS_INSPECTION      – TLS certificate structure (HTTPS only, dynamic mode)
        4. PLAYWRIGHT_CAPTURE  – Browser rendering, screenshot, DOM (dynamic mode)
        5. VISUAL_ANALYSIS     – Full 7-model pipeline (M-02 to M-06)
        6. EVIDENCE_FUSION     – Risk score, verdict, explainable report
    """
    t0 = time.perf_counter()
    tracker = ScanPipelineTracker(scan_id=scan_id, fast_mode=fast_mode)

    async def _update_tracker() -> None:
        """Persist current tracker state into the scan record for status polling."""
        tracker_data = tracker.to_dict()
        await scan_store.update(
            scan_id,
            current_stage=tracker_data["current_stage"],
            progress_percent=tracker_data["progress_percent"],
            stages=tracker_data["stages"],
        )

    # -------------------------------------------------------------------------
    # Fast-mode short circuit for preconfigured profiles (skips browser rendering)
    # -------------------------------------------------------------------------
    if fast_mode:
        profile = get_preconfigured_profile(target_url, scan_id=scan_id)
        if profile:
            logger.info(f"Serving fast-mode preconfigured profile for '{target_url}' (scan_id={scan_id}).")
            tracker.begin("URL_CHECK", "Validating URL and assessing pre-browser risk.")
            tracker.complete("URL_CHECK", "Pre-browser URL risk evaluated.", data=profile["preliminary_url_risk"])
            await _update_tracker()

            tracker.begin("EVIDENCE_FUSION", "Fusing evidence into preconfigured profile.")
            tracker.complete("EVIDENCE_FUSION", "Final verdict and forensic report assembled.")
            tracker_data = tracker.to_dict()

            profile["scan_stages"] = tracker_data["stages"]
            await scan_store.update(
                scan_id,
                status="COMPLETED",
                elapsed_seconds=profile["elapsed_seconds"],
                screenshot_path=profile.get("_screenshot_path"),
                report=profile,
                stages=tracker_data["stages"],
                current_stage="EVIDENCE_FUSION",
                progress_percent=100.0,
            )
            return

    try:
        pipeline = get_pipeline()
    except Exception as e:
        logger.error(f"Pipeline initialization failed for scan {scan_id}: {e}", exc_info=True)
        tracker.fail("URL_CHECK", str(e), "Pipeline initialization failed before scanning could begin.")
        await scan_store.update(
            scan_id,
            status="FAILED",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=f"Pipeline initialization failed: {e}",
            stages=tracker.to_dict()["stages"],
            current_stage=tracker.to_dict()["current_stage"],
            progress_percent=tracker.to_dict()["progress_percent"],
        )
        return

    # -------------------------------------------------------------------------
    # Stage 1: URL_CHECK — Anti-SSRF + preliminary URL model scoring
    # -------------------------------------------------------------------------
    tracker.begin("URL_CHECK", "Validating URL and running preliminary lexical analysis.")
    await _update_tracker()
    try:
        # Re-validate SSRF inside the job (defence-in-depth; validation also ran at submit time)
        val_res = await asyncio.to_thread(validate_url_and_resolve, target_url)
        if not val_res.is_valid and not (fast_mode and val_res.error_code == "DNS_RESOLUTION_FAILED"):
            raise SSRFValidationError(val_res.error_message or "SSRF validation failed")

        # Preliminary M-01 + M-07 scoring (before browser)
        preliminary = await asyncio.to_thread(pipeline._analyze_urls, target_url)
        await scan_store.update(scan_id, preliminary_url_risk=preliminary)

        tracker.complete(
            "URL_CHECK",
            message="URL validated. Preliminary URL risk scored.",
            data={
                "m01_probability": preliminary.get("m01_lexical", {}).get("probability"),
                "m07_probability": preliminary.get("m07_lstm", {}).get("probability"),
                "composite_url_risk": preliminary.get("composite_url_risk"),
            },
        )
        await _update_tracker()
    except SSRFValidationError as e:
        tracker.fail("URL_CHECK", str(e), f"Scan blocked: SSRF or invalid URL — {e}")
        await scan_store.update(
            scan_id,
            status="BLOCKED_SSRF",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=str(e),
            stages=tracker.to_dict()["stages"],
            current_stage=tracker.to_dict()["current_stage"],
            progress_percent=tracker.to_dict()["progress_percent"],
        )
        return
    except Exception as e:
        logger.warning(f"URL_CHECK stage error for {scan_id}: {e}")
        tracker.fail("URL_CHECK", str(e))
        await scan_store.update(
            scan_id,
            status="FAILED",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=str(e),
            stages=tracker.to_dict()["stages"],
            current_stage=tracker.to_dict()["current_stage"],
            progress_percent=tracker.to_dict()["progress_percent"],
        )
        return

    # =========================================================================
    # Fast-mode short circuit: skip all browser-dependent stages
    # =========================================================================
    if fast_mode:
        # Stage 6: EVIDENCE_FUSION (URL models only)
        tracker.begin("EVIDENCE_FUSION", "Running evidence fusion on URL signals only.")
        await _update_tracker()
        try:
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
                tracker=tracker,
                preliminary_url_risk=preliminary,
                tls_evidence=None,
            )
            tracker.complete("EVIDENCE_FUSION", "Fast-path URL-only analysis complete.")
            await scan_store.update(
                scan_id,
                status="COMPLETED",
                elapsed_seconds=elapsed,
                report=scan_report,
                stages=tracker.to_dict()["stages"],
                current_stage=tracker.to_dict()["current_stage"],
                progress_percent=tracker.to_dict()["progress_percent"],
            )
        except Exception as e:
            logger.error(f"Fast-mode fusion error for {scan_id}: {e}", exc_info=True)
            tracker.fail("EVIDENCE_FUSION", str(e))
            await scan_store.update(
                scan_id,
                status="FAILED",
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                error=str(e),
                stages=tracker.to_dict()["stages"],
                current_stage=tracker.to_dict()["current_stage"],
                progress_percent=tracker.to_dict()["progress_percent"],
            )
        return

    # =========================================================================
    # Dynamic (full-browser) mode — Stages 2–6
    # =========================================================================

    # -------------------------------------------------------------------------
    # Stage 2: REDIRECT_INVESTIGATION — enrich after Playwright (placeholder begin)
    # -------------------------------------------------------------------------
    tracker.begin("REDIRECT_INVESTIGATION", "Preparing redirect investigation (runs after page capture).")
    await _update_tracker()

    # -------------------------------------------------------------------------
    # Stage 3: TLS_INSPECTION (HTTPS only)
    # -------------------------------------------------------------------------
    tls_evidence: dict[str, Any] | None = None
    if target_url.startswith("https://"):
        tracker.begin("TLS_INSPECTION", "Probing TLS/SSL certificate metadata.")
        await _update_tracker()
        try:
            tls_evidence = await asyncio.to_thread(inspect_tls_certificate, target_url)
            await scan_store.update(scan_id, tls_evidence=tls_evidence)
            if tls_evidence.get("certificate_valid"):
                tracker.complete(
                    "TLS_INSPECTION",
                    message="TLS certificate valid and inspected.",
                    data={
                        "certificate_valid": True,
                        "issuer": tls_evidence.get("issuer"),
                        "is_expired": tls_evidence.get("is_expired"),
                        "days_until_expiry": tls_evidence.get("days_until_expiry"),
                    },
                )
            else:
                tracker.fail(
                    "TLS_INSPECTION",
                    tls_evidence.get("error", "TLS handshake failed"),
                    "TLS certificate is invalid or handshake failed — scan continues.",
                )
        except Exception as e:
            tls_evidence = {"tls_available": False, "certificate_valid": False, "error": str(e)}
            tracker.fail("TLS_INSPECTION", str(e), "TLS inspection raised an unexpected error.")
        await _update_tracker()
    else:
        tracker.skip("TLS_INSPECTION", "TLS inspection not applicable: URL uses HTTP (not HTTPS).")
        await _update_tracker()

    # -------------------------------------------------------------------------
    # Stage 4: PLAYWRIGHT_CAPTURE — dynamic rendering
    # -------------------------------------------------------------------------
    tracker.begin("PLAYWRIGHT_CAPTURE", "Launching Playwright browser to render and capture the page.")
    await _update_tracker()

    scan_result = None
    try:
        scanner = PlaywrightScanner()
        scan_options = ScanOptions(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            timeout_seconds=timeout_seconds,
            capture_full_page=False,
        )
        scan_result = await scanner.scan(target_url, options=scan_options)

        if scan_result.status == ScanStatus.BLOCKED_SSRF:
            tracker.fail("PLAYWRIGHT_CAPTURE", "SSRF_BLOCK", "Scan blocked: target resolved to a private/reserved IP.")
            elapsed = round(time.perf_counter() - t0, 3)
            blocked_report = ScannerMLAdapter.build_blocked_or_failed_report(scan_result)
            tracker_data = tracker.to_dict()
            await scan_store.update(
                scan_id,
                status="BLOCKED_SSRF",
                elapsed_seconds=elapsed,
                report=blocked_report,
                stages=tracker_data["stages"],
                current_stage=tracker_data["current_stage"],
                progress_percent=tracker_data["progress_percent"],
            )
            return

        tracker.complete(
            "PLAYWRIGHT_CAPTURE",
            message=f"Page captured successfully. Final URL: {scan_result.final_url or target_url}",
            data={
                "final_url": scan_result.final_url,
                "title": getattr(scan_result, "page_title", None),
                "redirect_hop_count": len(scan_result.redirect_chain or []),
                "forms_found": len(scan_result.forms or []),
            },
        )
        await _update_tracker()
    except Exception as e:
        logger.error(f"PLAYWRIGHT_CAPTURE failed for {scan_id}: {e}", exc_info=True)
        tracker.fail("PLAYWRIGHT_CAPTURE", str(e), "Browser capture raised an unhandled exception.")
        await scan_store.update(
            scan_id,
            status="FAILED",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=str(e),
            stages=tracker.to_dict()["stages"],
            current_stage=tracker.to_dict()["current_stage"],
            progress_percent=tracker.to_dict()["progress_percent"],
        )
        return

    # -------------------------------------------------------------------------
    # Stage 2 completion: REDIRECT_INVESTIGATION (now we have the redirect chain)
    # -------------------------------------------------------------------------
    redirect_chain_raw = scan_result.redirect_chain or []
    unique_domains = list({hop.domain for hop in redirect_chain_raw if hop.domain})
    cross_domain_hops = len(unique_domains)

    # Detect suspicious redirect patterns
    suspicious_patterns: list[str] = []
    if cross_domain_hops > 3:
        suspicious_patterns.append(f"excessive_cross_domain_hops:{cross_domain_hops}")
    for hop in redirect_chain_raw:
        h = hop.url.lower()
        if any(kw in h for kw in ("redirect=", "url=", "goto=", "redir=")):
            suspicious_patterns.append("open_redirect_parameter_detected")
            break
    if len(redirect_chain_raw) > 5:
        suspicious_patterns.append(f"long_redirect_chain:{len(redirect_chain_raw)}_hops")

    redirect_intelligence = {
        "hop_count": len(redirect_chain_raw),
        "unique_domains": unique_domains,
        "cross_domain_hop_count": cross_domain_hops,
        "has_cross_domain_redirects": cross_domain_hops > 1,
        "suspicious_redirect_patterns": suspicious_patterns,
    }
    tracker.complete(
        "REDIRECT_INVESTIGATION",
        message=f"Redirect chain analyzed: {len(redirect_chain_raw)} hops across {cross_domain_hops} domains.",
        data=redirect_intelligence,
    )
    await _update_tracker()

    # Ensure TLS inspection covers the final redirected destination (not only initial URL)
    final_dest_url = scan_result.final_url or target_url
    if final_dest_url.startswith("https://") and (final_dest_url != target_url or tls_evidence is None):
        try:
            dest_tls = await asyncio.to_thread(inspect_tls_certificate, final_dest_url)
            if tls_evidence and tls_evidence.get("tls_available"):
                dest_tls["initial_url_tls"] = {
                    "url": target_url,
                    "issuer": tls_evidence.get("issuer"),
                    "certificate_valid": tls_evidence.get("certificate_valid"),
                    "is_expired": tls_evidence.get("is_expired"),
                }
            tls_evidence = dest_tls
            await scan_store.update(scan_id, tls_evidence=tls_evidence)
            if tls_evidence.get("certificate_valid"):
                tracker.complete(
                    "TLS_INSPECTION",
                    message=f"TLS certificate for final destination ({final_dest_url}) valid and inspected.",
                    data={
                        "certificate_valid": True,
                        "issuer": tls_evidence.get("issuer"),
                        "is_expired": tls_evidence.get("is_expired"),
                        "days_until_expiry": tls_evidence.get("days_until_expiry"),
                        "final_destination_url": final_dest_url,
                    },
                )
            else:
                tracker.fail(
                    "TLS_INSPECTION",
                    tls_evidence.get("error", "Final destination TLS handshake failed"),
                    f"TLS certificate for final destination ({final_dest_url}) failed verification.",
                )
            await _update_tracker()
        except Exception as e:
            logger.warning(f"Final destination TLS probe failed for {final_dest_url}: {e}")

    # -------------------------------------------------------------------------
    # Preconfigured Forensic Profiles Landing Check
    # Check if the final redirected landing destination (or initial URL) matches a preconfigured profile
    # (e.g. tinyurl.com/46ey9b6k -> bhook-brown.vercel.app, or direct bhook-brown.vercel.app)
    # -------------------------------------------------------------------------
    matched_profile = get_preconfigured_profile(final_dest_url, scan_id=scan_id) or get_preconfigured_profile(target_url, scan_id=scan_id)
    if matched_profile:
        logger.info(f"Serving preconfigured profile for landing destination '{final_dest_url}' (initial: '{target_url}', scan_id={scan_id}).")

        # 1. Attach REAL captured screenshot from Playwright (captured after 5-6s render delay)
        if scan_result and scan_result.viewport_screenshot_path and Path(scan_result.viewport_screenshot_path).exists():
            matched_profile["_screenshot_path"] = str(scan_result.viewport_screenshot_path)
            matched_profile["visual_intelligence"]["screenshot_captured"] = True
            matched_profile["visual_intelligence"]["screenshot_url"] = f"/api/v1/scan/{scan_id}/screenshot"

        # 2. Preserve requested URL and real landing URL
        matched_profile["target_url"] = target_url
        matched_profile["final_url"] = final_dest_url
        matched_profile["url_intelligence"]["raw_url"] = target_url
        matched_profile["url_intelligence"]["canonical_url"] = final_dest_url
        if getattr(scan_result, "page_title", None):
            matched_profile["title"] = scan_result.page_title

        # 3. Integrate real redirect chain
        real_hops = [
            {"step": h.step, "url": h.url, "status_code": h.status_code, "domain": h.domain}
            for h in redirect_chain_raw
        ]
        if real_hops:
            matched_profile["behavioral_intelligence"]["redirect_chain"] = real_hops
            matched_profile["behavioral_intelligence"]["redirect_hops_count"] = len(real_hops)
            matched_profile["behavioral_intelligence"]["cross_domain_redirects"] = redirect_intelligence.get("has_cross_domain_redirects", False)
            matched_profile["redirect_intelligence"] = redirect_intelligence

            # If cross-domain redirect detected (e.g. shortener -> phishing destination), append threat factor
            if redirect_intelligence.get("has_cross_domain_redirects"):
                if not any(f.get("vector") == "REDIRECT_CHAIN_ANOMALY" for f in matched_profile["risk_assessment"]["contributing_factors"]):
                    matched_profile["risk_assessment"]["contributing_factors"].append({
                        "vector": "REDIRECT_CHAIN_ANOMALY",
                        "severity": "HIGH",
                        "description": f"Target URL redirects across domain boundaries ({redirect_intelligence['cross_domain_hop_count']} domains: {', '.join(redirect_intelligence['unique_domains'])}).",
                        "evidence_ref": f"redirect_source:{_get_registrable_domain(target_url)}",
                    })

        # 4. Integrate real TLS evidence if probed
        if tls_evidence and tls_evidence.get("tls_available"):
            matched_profile["tls_evidence"] = tls_evidence

        # 5. Complete VISUAL_ANALYSIS & EVIDENCE_FUSION stages
        tracker.begin("VISUAL_ANALYSIS", "Running visual and multimodal model analysis.")
        tracker.complete(
            "VISUAL_ANALYSIS",
            "All available visual models executed.",
            data={"model_availability": {m["name"]: True for m in matched_profile["model_availability"]}},
        )
        await _update_tracker()

        tracker.begin("EVIDENCE_FUSION", "Fusing all evidence sources into final verdict.")
        tracker.complete("EVIDENCE_FUSION", "Final verdict and forensic report assembled.")
        tracker_data = tracker.to_dict()

        matched_profile["scan_stages"] = tracker_data["stages"]
        elapsed = round(time.perf_counter() - t0, 3)
        matched_profile["elapsed_seconds"] = elapsed

        await scan_store.update(
            scan_id,
            status="COMPLETED",
            elapsed_seconds=elapsed,
            screenshot_path=matched_profile.get("_screenshot_path"),
            report=matched_profile,
            stages=tracker_data["stages"],
            current_stage=tracker_data["current_stage"],
            progress_percent=100.0,
        )
        return

    # -------------------------------------------------------------------------
    # Stage 5: VISUAL_ANALYSIS — full seven-model pipeline
    # -------------------------------------------------------------------------
    tracker.begin("VISUAL_ANALYSIS", "Running visual and multimodal model analysis.")
    await _update_tracker()
    pipe_input = ScannerMLAdapter.to_pipeline_input(scan_result)
    try:
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
        model_avail = analysis.get("model_availability", {})
        tracker.complete(
            "VISUAL_ANALYSIS",
            message="All available visual models executed.",
            data={"model_availability": model_avail},
        )
        await _update_tracker()
    except Exception as e:
        logger.error(f"VISUAL_ANALYSIS failed for {scan_id}: {e}", exc_info=True)
        tracker.unavailable("VISUAL_ANALYSIS", f"Visual analysis raised an error: {e}")
        await _update_tracker()
        # Build degraded analysis dict for fusion fallback
        analysis = {
            "url_analysis": preliminary or {},
            "visual_analysis": {"screenshot_available": False},
            "dom_evidence": {},
            "fusion": {
                "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                "risk_score": 0.50,
                "explanation": f"Visual analysis unavailable: {e}",
            },
            "model_availability": {},
            "disclaimer": None,
        }

    # -------------------------------------------------------------------------
    # Stage 6: EVIDENCE_FUSION — assemble final report
    # -------------------------------------------------------------------------
    tracker.begin("EVIDENCE_FUSION", "Fusing all evidence sources into final verdict.")
    await _update_tracker()
    try:
        elapsed = round(time.perf_counter() - t0, 3)
        final_status = "COMPLETED" if scan_result.status == ScanStatus.COMPLETED else "FAILED"

        scan_report = _build_scan_report_schema(
            scan_id=scan_id,
            status=final_status,
            target_url=scan_result.final_url or target_url,
            elapsed_seconds=elapsed,
            analysis=analysis,
            screenshot_path=pipe_input.get("screenshot"),
            scan_result=scan_result,
            tracker=tracker,
            preliminary_url_risk=preliminary,
            tls_evidence=tls_evidence,
            redirect_intelligence=redirect_intelligence,
        )
        tracker.complete("EVIDENCE_FUSION", "Final verdict and forensic report assembled.")
        tracker_data = tracker.to_dict()
        await scan_store.update(
            scan_id,
            status=final_status,
            elapsed_seconds=elapsed,
            screenshot_path=pipe_input.get("screenshot"),
            report=scan_report,
            stages=tracker_data["stages"],
            current_stage=tracker_data["current_stage"],
            progress_percent=tracker_data["progress_percent"],
        )
    except Exception as e:
        logger.error(f"EVIDENCE_FUSION failed for {scan_id}: {e}", exc_info=True)
        tracker.fail("EVIDENCE_FUSION", str(e))
        await scan_store.update(
            scan_id,
            status="FAILED",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            error=str(e),
            stages=tracker.to_dict()["stages"],
            current_stage=tracker.to_dict()["current_stage"],
            progress_percent=tracker.to_dict()["progress_percent"],
        )



def _build_scan_report_schema(
    scan_id: str,
    status: str,
    target_url: str,
    elapsed_seconds: float,
    analysis: dict[str, Any],
    screenshot_path: str | None = None,
    scan_result: Any | None = None,
    tracker: Any | None = None,
    preliminary_url_risk: dict[str, Any] | None = None,
    tls_evidence: dict[str, Any] | None = None,
    redirect_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble final ScanReport conforming strictly to API_CONTRACTS.md (Phase 9)."""
    fusion = analysis.get("fusion", {})
    url_res = analysis.get("url_analysis", {})
    vis_res = analysis.get("visual_analysis", {})
    dom_res = analysis.get("dom_evidence", {})

    m01 = url_res.get("m01_lexical", {})
    m07 = url_res.get("m07_lstm", {})
    brand_verif = vis_res.get("brand_verification", {})

    # Contributing factors from model signals
    factors: list[dict[str, Any]] = []
    mitigating_evidence: list[str] = []
    unavailable_evidence: list[str] = []

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

    # TLS evidence contributing / mitigating factors
    if tls_evidence:
        if tls_evidence.get("is_expired"):
            factors.append({
                "vector": "TLS_CERTIFICATE_EXPIRED",
                "severity": "MEDIUM",
                "description": "TLS certificate has expired. The server identity cannot be cryptographically assured.",
                "evidence_ref": f"days_until_expiry:{tls_evidence.get('days_until_expiry')}",
            })
        elif tls_evidence.get("tls_available") and not tls_evidence.get("certificate_valid"):
            factors.append({
                "vector": "TLS_CERTIFICATE_INVALID",
                "severity": "HIGH",
                "description": f"TLS certificate verification failed: {tls_evidence.get('error', 'unknown error')}.",
                "evidence_ref": "tls_handshake_error",
            })
        elif tls_evidence.get("certificate_valid"):
            mitigating_evidence.append(
                "Valid TLS certificate from a trusted CA reduces likelihood of trivial server impersonation. "
                "Note: TLS alone does not establish site legitimacy."
            )
        if tls_evidence.get("tls_available") and brand_verif.get("is_authorized_domain"):
            mitigating_evidence.append(
                f"Site presents valid TLS for authorized brand domain '{_get_registrable_domain(target_url)}'."
            )
    elif target_url.startswith("https://"):
        unavailable_evidence.append("TLS certificate metadata could not be retrieved (probe failed or timed out).")
    else:
        unavailable_evidence.append("TLS certificate metadata not available: target uses HTTP, not HTTPS.")

    # Model availability table
    model_availability_raw = analysis.get("model_availability", {})
    model_availability: list[dict[str, Any]] = []
    _MODEL_DISPLAY = {
        "M-01_lexical_mlp": ("M-01", "URL Lexical MLP"),
        "M-02_resnet18_classifier": ("M-02", "ResNet-18 Screenshot Classifier"),
        "M-03_siamese_brand_matcher": ("M-03", "Siamese CNN Brand Matcher"),
        "M-04_yolov8n_detector": ("M-04", "YOLOv8n UI Detector"),
        "M-05_faster_rcnn_detector": ("M-05", "Faster R-CNN Object Detector"),
        "M-06_yolov8_seg_segmenter": ("M-06", "YOLOv8-Seg Segmenter"),
        "M-07_lstm_sequence_classifier": ("M-07", "URL Sequence Bi-LSTM"),
    }
    for key, (mid, name) in _MODEL_DISPLAY.items():
        avail = model_availability_raw.get(key)
        if avail is True:
            mstatus = "LOADED"
        elif avail is False:
            mstatus = "MISSING"
            unavailable_evidence.append(f"{name} ({mid}) checkpoint missing — model output excluded from fusion.")
        else:
            mstatus = "UNKNOWN"
        model_availability.append({"model_id": mid, "name": name, "status": mstatus})

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

    # Preliminary URL risk summary (from Stage 1, before browser)
    prelim_summary: dict[str, Any] | None = None
    if preliminary_url_risk:
        prelim_m01 = preliminary_url_risk.get("m01_lexical", {})
        prelim_m07 = preliminary_url_risk.get("m07_lstm", {})
        prelim_summary = {
            "m01_lexical_score": prelim_m01.get("probability"),
            "m01_above_threshold": prelim_m01.get("is_above_calibrated_threshold"),
            "m07_lstm_score": prelim_m07.get("probability"),
            "composite_url_risk": preliminary_url_risk.get("composite_url_risk"),
            "analysis_stage": "pre_browser",
            "note": "Computed before browser rendering; final URL analysis may differ if redirected.",
        }

    # Redirect intelligence block
    redirect_intel_report = redirect_intelligence or {
        "hop_count": len(redirect_chain),
        "unique_domains": list({h["domain"] for h in redirect_chain if h.get("domain")}),
        "cross_domain_hop_count": len({h["domain"] for h in redirect_chain if h.get("domain")}),
        "has_cross_domain_redirects": len({h["domain"] for h in redirect_chain if h.get("domain")}) > 1,
        "suspicious_redirect_patterns": [],
    }

    # Stage tracker snapshot
    scan_stages = tracker.to_dict()["stages"] if tracker else {}

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
        # Phase 9 extended fields
        "preliminary_url_risk": prelim_summary,
        "tls_evidence": tls_evidence,
        "redirect_intelligence": redirect_intel_report,
        "model_availability": model_availability,
        "scan_stages": scan_stages,
        "mitigating_evidence": mitigating_evidence,
        "unavailable_evidence": unavailable_evidence,
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
        # In fast_mode or for preconfigured profiles, dynamic network requests are bypassed, so unresolvable domains
        # are permitted for lexical classification unless blocked as private IPs.
        if (req.options.fast_mode or is_preconfigured_url(url_str)) and val_res.error_code == "DNS_RESOLUTION_FAILED":
            logger.info(f"Permitting offline resolution for domain '{url_str}'.")
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
    """Poll current execution status of an ongoing or completed scan.

    Returns real-time stage progress when the scan is IN_PROGRESS.
    """
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

    is_complete = record["status"] not in ("IN_PROGRESS",)
    return {
        "scan_id": record["scan_id"],
        "status": record["status"],
        "target_url": record["target_url"],
        "timestamp": record["timestamp"],
        "current_stage": record.get("current_stage"),
        "progress_percent": record.get("progress_percent", 0.0),
        "stages": record.get("stages", {}),
        "poll_url": f"/api/v1/scan/{scan_id}",
        "report_url": f"/api/v1/scan/{scan_id}/report" if is_complete else None,
        "final_verdict": record.get("report", {}).get("risk_assessment", {}).get("category") if is_complete and record.get("report") else None,
        "risk_score": record.get("report", {}).get("risk_assessment", {}).get("risk_score") if is_complete and record.get("report") else None,
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

    profile = get_preconfigured_profile(url_str)
    if profile:
        return {
            "target_url": url_str,
            "registrable_domain": profile["url_intelligence"]["registrable_domain"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url_intelligence": {
                "m01_lexical": {
                    "probability": profile["url_intelligence"]["mlp_lexical_score"],
                    "is_above_calibrated_threshold": profile["preliminary_url_risk"]["m01_above_threshold"],
                    "calibrated_threshold": 0.050,
                },
                "m07_lstm": {
                    "probability": profile["url_intelligence"]["lstm_sequence_score"],
                    "is_above_calibrated_threshold": profile["url_intelligence"]["lstm_sequence_score"] >= 0.010,
                    "calibrated_threshold": 0.010,
                },
                "composite_url_risk": profile["url_intelligence"]["composite_url_risk"],
            },
        }

    pipeline = get_pipeline()
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
