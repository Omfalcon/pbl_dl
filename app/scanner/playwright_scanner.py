"""Secure Playwright-based Website Scanner for PhishVision AI.

Executes isolated browser sessions with pre-connection anti-SSRF enforcement,
redirect tracing, viewport/full-page screenshot rendering, DOM extraction,
form analysis, and script metadata collection.
"""

from __future__ import annotations

import asyncio
import os
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import tldextract
from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from app.scanner.anti_ssrf import (
    SSRFValidationError,
    validate_redirect_target,
    validate_url_and_resolve,
)
from app.scanner.models import (
    FormField,
    FormInfo,
    RedirectHop,
    ScanOptions,
    ScanResult,
    ScanStatus,
    ScriptInfo,
)


# Default directories for scan evidence
DEFAULT_SCREENSHOT_DIR = Path("data/scans/screenshots")
DEFAULT_DOM_DIR = Path("data/scans/dom")


def _get_registrable_domain(url_or_host: str) -> str:
    """Extract the registrable domain (e.g. example.com, sbi.co.in) using Public Suffix List."""
    try:
        ext = tldextract.extract(url_or_host)
        if hasattr(ext, "top_domain_under_public_suffix") and ext.top_domain_under_public_suffix:
            return ext.top_domain_under_public_suffix
        return ext.registered_domain if ext.registered_domain else ext.domain
    except Exception:
        return ""


def parse_dom_evidence(
    html_content: str,
    base_url: str
) -> tuple[str | None, list[FormInfo], list[ScriptInfo], dict[str, str]]:
    """Parse rendered HTML content using BeautifulSoup to extract forms, inputs, scripts, and meta tags."""
    if not html_content:
        return None, [], [], {}

    soup = BeautifulSoup(html_content, "html.parser")
    base_domain = _get_registrable_domain(base_url)

    # 1. Page Title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # 2. Meta Tags
    meta_tags: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        key = meta.get("name") or meta.get("property") or meta.get("http-equiv")
        content = meta.get("content")
        if key and content:
            meta_tags[str(key).strip().lower()] = str(content).strip()

    # 3. Forms & Input Fields
    forms: list[FormInfo] = []
    for idx, form in enumerate(soup.find_all("form")):
        form_id = form.get("id") or form.get("name") or f"form_{idx + 1}"
        action_raw = form.get("action", "")
        method = (form.get("method") or "GET").upper()

        # Resolve action URL relative to base URL
        action_full = urllib.parse.urljoin(base_url, action_raw) if action_raw else base_url
        action_domain = _get_registrable_domain(action_full)
        is_external = bool(action_domain and base_domain and action_domain != base_domain)

        # Parse Fields
        fields: list[FormField] = []
        has_password = False

        # Input elements
        for inp in form.find_all(["input", "button", "textarea"]):
            field_type = (inp.get("type") or "text").lower()
            name = inp.get("name") or ""
            fid = inp.get("id") or ""
            placeholder = inp.get("placeholder") or ""
            autocomplete = inp.get("autocomplete") or ""
            is_req = inp.has_attr("required")
            is_pwd = field_type == "password" or "password" in name.lower() or "passwd" in name.lower()

            if is_pwd:
                has_password = True

            fields.append(
                FormField(
                    name=str(name),
                    field_id=str(fid),
                    field_type=str(field_type),
                    placeholder=str(placeholder),
                    autocomplete=str(autocomplete),
                    is_password=is_pwd,
                    is_required=is_req,
                )
            )

        forms.append(
            FormInfo(
                form_id=str(form_id),
                action_url=action_full,
                action_domain=action_domain,
                method=method,
                is_external_submission=is_external,
                fields=fields,
                has_password_field=has_password,
            )
        )

    # 4. Script Metadata
    scripts: list[ScriptInfo] = []
    for s in soup.find_all("script"):
        src = s.get("src")
        if src:
            src_full = urllib.parse.urljoin(base_url, src)
            src_domain = _get_registrable_domain(src_full)
            is_ext = bool(src_domain and base_domain and src_domain != base_domain)
            scripts.append(
                ScriptInfo(
                    src=src_full,
                    domain=src_domain,
                    is_external=is_ext,
                    is_inline=False,
                    async_attr=s.has_attr("async"),
                    defer_attr=s.has_attr("defer"),
                )
            )
        else:
            scripts.append(
                ScriptInfo(
                    src="",
                    domain=base_domain,
                    is_external=False,
                    is_inline=True,
                    async_attr=False,
                    defer_attr=False,
                )
            )

    return title, forms, scripts, meta_tags


class PlaywrightScanner:
    """High-assurance secure website scanner utilizing Playwright headless browser."""

    def __init__(
        self,
        screenshot_dir: Path | str = DEFAULT_SCREENSHOT_DIR,
        dom_dir: Path | str = DEFAULT_DOM_DIR,
    ):
        self.screenshot_dir = Path(screenshot_dir)
        self.dom_dir = Path(dom_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.dom_dir.mkdir(parents=True, exist_ok=True)

    async def scan(self, url: str, options: ScanOptions | None = None) -> ScanResult:
        """Asynchronously scan a target URL in an isolated headless browser context.
        
        Args:
            url: Target URL to inspect.
            options: Scan configuration options.
            
        Returns:
            ScanResult containing full forensic artifacts and telemetry.
        """
        scan_id = str(uuid.uuid4())
        opts = options or ScanOptions()
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # ---------------------------------------------------------------------
        # 1. Pre-Connection Anti-SSRF Validation
        # ---------------------------------------------------------------------
        val_result = validate_url_and_resolve(url)
        if not val_result.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ScanResult(
                scan_id=scan_id,
                requested_url=url,
                final_url=url,
                status=ScanStatus.BLOCKED_SSRF,
                timestamp=timestamp,
                scan_duration_ms=round(elapsed_ms, 2),
                errors=[val_result.error_message or "Blocked by Anti-SSRF policy"],
                security_metadata={
                    "validation_code": val_result.error_code,
                    "resolved_ips": val_result.resolved_ips,
                    "blocked_reason": val_result.error_message,
                },
            )

        canonical_url = val_result.canonical_url
        pinned_ip = val_result.pinned_ip
        errors: list[str] = []
        warnings: list[str] = []
        redirect_chain: list[RedirectHop] = []
        domain_transitions: list[str] = [val_result.hostname]

        # ---------------------------------------------------------------------
        # 2. Ephemeral Browser Context Initialization
        # ---------------------------------------------------------------------
        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-extensions",
                    "--disable-gpu",
                ],
            )

            context: BrowserContext = await browser.new_context(
                viewport={"width": opts.viewport_width, "height": opts.viewport_height},
                user_agent=opts.user_agent
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 PhishVision/0.1",
                ignore_https_errors=True,
                accept_downloads=False,
            )

            page: Page = await context.new_page()

            # Prevent automatic file downloads
            page.on("download", lambda download: asyncio.create_task(download.cancel()))

            # Track request route for runtime Anti-SSRF re-validation on redirects
            ssrf_aborted = False

            async def route_interceptor(route):
                nonlocal ssrf_aborted
                request_url = route.request.url
                # Only strictly validate http/https requests
                if request_url.startswith(("http://", "https://")):
                    step_val = validate_redirect_target(request_url, base_url=canonical_url)
                    if not step_val.is_valid:
                        ssrf_aborted = True
                        warnings.append(f"Sub-request or redirect blocked by Anti-SSRF: {request_url} ({step_val.error_message})")
                        await route.abort("blockedbyclient")
                        return
                await route.continue_()

            await page.route("**/*", route_interceptor)

            # Record Redirect Responses
            def handle_response(response: Response):
                try:
                    req = response.request
                    if req.is_navigation_request():
                        status = response.status
                        hop_url = response.url
                        hop_domain = _get_registrable_domain(hop_url)
                        if hop_domain and hop_domain not in domain_transitions:
                            domain_transitions.append(hop_domain)

                        headers_dict = dict(list(response.headers.items())[:15])  # Cap header count
                        redirect_chain.append(
                            RedirectHop(
                                step=len(redirect_chain) + 1,
                                url=hop_url,
                                status_code=status,
                                domain=hop_domain or urllib.parse.urlsplit(hop_url).hostname or "",
                                headers=headers_dict,
                            )
                        )
                except Exception as ex:
                    warnings.append(f"Telemetry logging error on response: {ex}")

            page.on("response", handle_response)

            # -----------------------------------------------------------------
            # 3. Dynamic Page Navigation with Graceful Timeouts
            # -----------------------------------------------------------------
            status = ScanStatus.COMPLETED
            final_status_code: int | None = None
            final_url = canonical_url

            try:
                response = await page.goto(
                    canonical_url,
                    timeout=opts.timeout_seconds * 1000,
                    wait_until="domcontentloaded",
                )
                if response:
                    final_status_code = response.status
                    final_url = response.url
                else:
                    final_url = page.url
            except PlaywrightTimeoutError:
                status = ScanStatus.TIMED_OUT
                errors.append(f"Navigation timed out after {opts.timeout_seconds} seconds")
                final_url = page.url or canonical_url
            except PlaywrightError as pe:
                status = ScanStatus.FAILED
                errors.append(f"Browser navigation error: {pe.message}")
                final_url = page.url or canonical_url
            except Exception as e:
                status = ScanStatus.FAILED
                errors.append(f"Unexpected browser error: {e}")
                final_url = page.url or canonical_url

            if ssrf_aborted and status == ScanStatus.FAILED:
                status = ScanStatus.BLOCKED_SSRF

            # -----------------------------------------------------------------
            # 4. Forensic Evidence Collection (Screenshots & DOM)
            # -----------------------------------------------------------------
            viewport_shot_path: str | None = None
            fullpage_shot_path: str | None = None
            dom_path: str | None = None
            html_content = ""

            try:
                # Capture rendered HTML content
                html_content = await page.content()
                if html_content:
                    dom_file = self.dom_dir / f"{scan_id}_dom.html"
                    dom_file.write_text(html_content, encoding="utf-8", errors="replace")
                    dom_path = str(dom_file)
            except Exception as e:
                warnings.append(f"Failed to capture DOM: {e}")

            try:
                # Viewport Screenshot (Standard 1280x720)
                v_file = self.screenshot_dir / f"{scan_id}_viewport.png"
                await page.screenshot(path=str(v_file), full_page=False)
                viewport_shot_path = str(v_file)
            except Exception as e:
                warnings.append(f"Failed to capture viewport screenshot: {e}")

            if opts.capture_full_page and status == ScanStatus.COMPLETED:
                try:
                    # Full-page Screenshot
                    f_file = self.screenshot_dir / f"{scan_id}_full.png"
                    await page.screenshot(path=str(f_file), full_page=True, timeout=5000)
                    fullpage_shot_path = str(f_file)
                except Exception as e:
                    warnings.append(f"Failed to capture full-page screenshot: {e}")

            # -----------------------------------------------------------------
            # 5. DOM Information Extraction (Forms, Inputs, Scripts, Metadata)
            # -----------------------------------------------------------------
            page_title, forms, scripts, meta_tags = parse_dom_evidence(html_content, final_url)

            # Cleanup browser resources
            await context.close()
            await browser.close()

        # Compute elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ScanResult(
            scan_id=scan_id,
            requested_url=url,
            final_url=final_url,
            status=status,
            status_code=final_status_code,
            timestamp=timestamp,
            page_title=page_title,
            viewport_screenshot_path=viewport_shot_path,
            full_page_screenshot_path=fullpage_shot_path,
            dom_snapshot_path=dom_path,
            redirect_chain=redirect_chain,
            domain_transitions=domain_transitions,
            forms=forms,
            scripts=scripts,
            meta_tags=meta_tags,
            scan_duration_ms=round(elapsed_ms, 2),
            errors=errors,
            warnings=warnings,
            security_metadata={
                "pinned_ip": pinned_ip,
                "resolved_ips": val_result.resolved_ips,
                "pre_validation_valid": val_result.is_valid,
            },
        )

    def scan_sync(self, url: str, options: ScanOptions | None = None) -> ScanResult:
        """Synchronous wrapper around the async scan implementation."""
        try:
            return asyncio.run(self.scan(url, options))
        except RuntimeError:
            # If an event loop is already running (e.g. in test runner or async caller)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.scan(url, options)).result()
