"""Scanner-to-ML Adapter for PhishVision AI.

Converts forensic artifacts and runtime data models from PlaywrightScanner
(ScanResult, RedirectHop, FormInfo) into the strict input types expected by
the UnifiedPhishingPipeline, ensuring complete type safety, Anti-SSRF preservation,
and graceful degradation for headless or blocked scans.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.scanner.models import FormInfo, RedirectHop, ScanResult, ScanStatus
from app.scanner.playwright_scanner import _get_registrable_domain

logger = logging.getLogger("phishvision.scanner_adapter")


class ScannerMLAdapter:
    """Adapts Playwright scanner output to the UnifiedPhishingPipeline input format."""

    @staticmethod
    def extract_redirect_hostnames(redirect_chain: list[RedirectHop] | None) -> list[str]:
        """Convert a list of RedirectHop objects into a sequence of hostnames/domains.

        Args:
            redirect_chain: List of RedirectHop objects from ScanResult.

        Returns:
            List of domain/hostname strings in chronological redirect order.
        """
        if not redirect_chain:
            return []

        hostnames: list[str] = []
        for hop in redirect_chain:
            if hop.domain:
                hostnames.append(hop.domain.strip().lower())
            elif hop.url:
                d = _get_registrable_domain(hop.url)
                if d:
                    hostnames.append(d)
        return hostnames

    @staticmethod
    def extract_dom_forms(forms: list[FormInfo] | None) -> list[dict[str, Any]]:
        """Convert FormInfo Pydantic models into dictionaries for DOM analysis.

        Ensures fields required by UnifiedPhishingPipeline:
        - form_id: str
        - action_url: str
        - action_domain: str
        - is_external_submission: bool
        - has_password_field: bool
        """
        if not forms:
            return []

        dom_forms: list[dict[str, Any]] = []
        for form in forms:
            dom_forms.append({
                "form_id": form.form_id,
                "action_url": form.action_url,
                "action_domain": form.action_domain,
                "method": form.method,
                "is_external_submission": form.is_external_submission,
                "has_password_field": form.has_password_field,
                "field_count": len(form.fields),
            })
        return dom_forms

    @classmethod
    def to_pipeline_input(cls, scan_result: ScanResult) -> dict[str, Any]:
        """Convert a ScanResult into keyword arguments for UnifiedPhishingPipeline.analyze().

        Args:
            scan_result: Captured scan data from PlaywrightScanner.

        Returns:
            Dictionary matching the signature:
            {
                "url": str,
                "screenshot": str | Path | None,
                "domain": str,
                "redirect_chain": list[str],
                "dom_forms": list[dict[str, Any]],
                "sample_id": str,
            }
        """
        # 1. Target URL selection: prefer final landed URL, fallback to requested URL
        target_url = (scan_result.final_url or scan_result.requested_url).strip()

        # 2. Registrable domain determination
        domain = _get_registrable_domain(target_url)
        if not domain and scan_result.requested_url:
            domain = _get_registrable_domain(scan_result.requested_url)

        # 3. Screenshot path resolution
        screenshot_path: str | None = None
        if scan_result.viewport_screenshot_path:
            p = Path(scan_result.viewport_screenshot_path)
            if p.exists() and p.is_file():
                screenshot_path = str(p)
        if not screenshot_path and scan_result.full_page_screenshot_path:
            p = Path(scan_result.full_page_screenshot_path)
            if p.exists() and p.is_file():
                screenshot_path = str(p)

        # 4. Redirect hostnames
        redirect_hostnames = cls.extract_redirect_hostnames(scan_result.redirect_chain)

        # 5. DOM Forms extraction
        dom_forms = cls.extract_dom_forms(scan_result.forms)

        return {
            "url": target_url,
            "screenshot": screenshot_path,
            "domain": domain,
            "redirect_chain": redirect_hostnames,
            "dom_forms": dom_forms,
            "sample_id": scan_result.scan_id,
        }

    @classmethod
    def build_blocked_or_failed_report(
        cls,
        scan_result: ScanResult,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Generate a safe, structured fallback report when scanning is blocked by Anti-SSRF or fails."""
        status_name = scan_result.status.value if hasattr(scan_result.status, "value") else str(scan_result.status)
        is_ssrf = scan_result.status == ScanStatus.BLOCKED_SSRF

        desc = reason or (
            "Scan blocked by Anti-SSRF security controls (target resolved to private/loopback/restricted IP)."
            if is_ssrf
            else f"Scan execution ended with status {status_name}."
        )

        return {
            "threat_category": "BLOCKED_SSRF" if is_ssrf else "UNKNOWN / INSUFFICIENT EVIDENCE",
            "risk_score": 1.0 if is_ssrf else 0.50,
            "confidence": 1.0 if is_ssrf else 0.0,
            "status": status_name,
            "summary": desc,
            "scan_id": scan_result.scan_id,
            "target_url": scan_result.requested_url,
            "contributing_factors": [
                {
                    "vector": "NETWORK_SECURITY",
                    "severity": "CRITICAL" if is_ssrf else "HIGH",
                    "description": desc,
                    "evidence_ref": f"scanner_status:{status_name}",
                }
            ],
            "url_evidence": {
                "raw_url": scan_result.requested_url,
                "final_url": scan_result.final_url,
                "status": status_name,
            },
            "visual_evidence": {
                "screenshot_available": False,
                "status": "NOT_CAPTURED",
            },
            "behavioral_evidence": {
                "redirect_hops_count": len(scan_result.redirect_chain),
                "redirect_chain": [
                    {
                        "step": hop.step,
                        "url": hop.url,
                        "status_code": hop.status_code,
                        "domain": hop.domain,
                    }
                    for hop in scan_result.redirect_chain
                ],
                "forms": cls.extract_dom_forms(scan_result.forms),
            },
            "explanation": desc,
        }
