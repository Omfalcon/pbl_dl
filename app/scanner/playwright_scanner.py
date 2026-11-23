"""Secure Playwright-based Website Scanner for PhishVision AI.

Executes isolated browser sessions with connection-level Anti-SSRF enforcement,
validating egress proxy tunneling, context-level route interception, service worker
blocking, popup containment, secure TLS certificate validation, and comprehensive
forensic DOM and screenshot extraction.
"""

from __future__ import annotations

import asyncio
import os
import sys
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
from app.scanner.proxy import ValidatingEgressProxy


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

        action_full = urllib.parse.urljoin(base_url, action_raw) if action_raw else base_url
        action_domain = _get_registrable_domain(action_full)
        is_external = bool(action_domain and base_domain and action_domain != base_domain)

        fields: list[FormField] = []
        has_password = False

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


def _get_browser_launch_args() -> list[str]:
    """Return hardened Chromium launch flags appropriate for the current platform."""
    base_args = [
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-translate",
        "--metrics-recording-only",
        "--no-first-run",
        "--safebrowsing-disable-auto-update",
    ]

    # Only pass --no-sandbox if explicitly mandated by unprivileged Linux container execution
    is_linux = sys.platform.startswith("linux")
    container_mode = os.environ.get("PHISHVISION_CONTAINER_MODE", "0") == "1"

    if is_linux and container_mode:
        base_args.extend(["--no-sandbox", "--disable-setuid-sandbox"])

    return base_args


def _is_tls_certificate_error(error_message: str) -> bool:
    """Detect whether a Playwright navigation failure stems from an invalid TLS certificate."""
    tls_error_tokens = [
        "net::err_cert_authority_invalid",
        "net::err_cert_common_name_invalid",
        "net::err_cert_date_invalid",
        "net::err_cert_invalid",
        "net::err_cert_revoked",
        "net::err_cert_weak_signature_algorithm",
        "net::err_ssl_pinned_key_not_in_cert_chain",
        "certificate_verify_failed",
        "ssl_error",
        "err_ssl_protocol_error",
    ]
    msg_lower = error_message.lower()
    return any(token in msg_lower for token in tls_error_tokens)


class PlaywrightScanner:
    """High-assurance secure website scanner with connection-level egress enforcement."""

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
        """Asynchronously scan a target URL with Anti-SSRF proxy enforcement and secure TLS checking."""
        if sys.platform == "win32":
            try:
                loop = asyncio.get_running_loop()
                # On Windows, async subprocess creation (required by Playwright) requires ProactorEventLoop.
                # Under reloaders (like uvicorn --reload), a SelectorEventLoop is installed, which lacks
                # subprocess support and raises NotImplementedError.
                is_proactor = isinstance(loop, getattr(asyncio, "ProactorEventLoop", ()))
                if not is_proactor:
                    def _runner() -> ScanResult:
                        p_loop = asyncio.ProactorEventLoop()
                        asyncio.set_event_loop(p_loop)
                        try:
                            return p_loop.run_until_complete(self._scan_impl(url, options))
                        finally:
                            p_loop.close()

                    return await asyncio.to_thread(_runner)
            except Exception:
                pass
        return await self._scan_impl(url, options)

    async def _scan_impl(self, url: str, options: ScanOptions | None = None) -> ScanResult:
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
        tls_error_detail: str | None = None
        popups_closed_count = 0

        # ---------------------------------------------------------------------
        # 2. Start Validating Egress Proxy (Network-Level Connection Pinning)
        # ---------------------------------------------------------------------
        proxy: ValidatingEgressProxy | None = None
        browser: Browser | None = None
        context: BrowserContext | None = None
        page: Page | None = None

        try:
            proxy_config = None
            if opts.enable_egress_proxy:
                proxy = ValidatingEgressProxy()
                await proxy.start()
                proxy_config = {"server": proxy.proxy_url}

            # -----------------------------------------------------------------
            # 3. Isolated Browser Context Setup
            # -----------------------------------------------------------------
            async with async_playwright() as p:
                launch_args = _get_browser_launch_args()
                browser = await p.chromium.launch(
                    headless=opts.headless,
                    args=launch_args,
                    proxy=proxy_config,
                )

                # Configure context with service worker blocking & TLS verification
                context_kwargs: dict[str, Any] = {
                    "viewport": {"width": opts.viewport_width, "height": opts.viewport_height},
                    "user_agent": opts.user_agent
                    or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 PhishVision/0.1",
                    "ignore_https_errors": opts.ignore_https_errors,
                    "accept_downloads": False,
                }

                if opts.block_service_workers:
                    context_kwargs["service_workers"] = "block"

                context = await browser.new_context(**context_kwargs)

                # Create primary scanning page
                page = await context.new_page()
                page.on("download", lambda d: asyncio.create_task(d.cancel()))

                # Popup containment: close any secondary / unsolicited popup pages
                if opts.block_popups:
                    def handle_popup(new_page: Page):
                        nonlocal popups_closed_count
                        if new_page != page:
                            popups_closed_count += 1
                            warnings.append(f"Unsolicited popup blocked and closed: {new_page.url or 'about:blank'}")
                            asyncio.create_task(new_page.close())

                    context.on("page", handle_popup)

                # Context-level request routing & revalidation (covers all pages, frames, workers)
                route_abort_count = 0

                async def context_route_interceptor(route):
                    nonlocal route_abort_count
                    req_url = route.request.url
                    if req_url.startswith(("http://", "https://")):
                        target_val = validate_redirect_target(req_url, base_url=canonical_url)
                        if not target_val.is_valid:
                            route_abort_count += 1
                            warnings.append(
                                f"Subresource or navigation blocked by route policy: {req_url} ({target_val.error_message})"
                            )
                            await route.abort("blockedbyclient")
                            return
                    await route.continue_()

                await context.route("**/*", context_route_interceptor)

                # Telemetry listener for redirects
                def handle_response(response: Response):
                    try:
                        req = response.request
                        if req.is_navigation_request():
                            status = response.status
                            hop_url = response.url
                            hop_domain = _get_registrable_domain(hop_url)
                            if hop_domain and hop_domain not in domain_transitions:
                                domain_transitions.append(hop_domain)

                            headers_dict = dict(list(response.headers.items())[:15])
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
                        warnings.append(f"Telemetry logging error: {ex}")

                page.on("response", handle_response)

                # -------------------------------------------------------------
                # 4. Execute Page Navigation
                # -------------------------------------------------------------
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
                    final_url = str(page.url) if isinstance(getattr(page, "url", None), str) and page.url else canonical_url
                except PlaywrightError as pe:
                    if _is_tls_certificate_error(pe.message):
                        status = ScanStatus.CERT_ERROR
                        tls_error_detail = pe.message
                        errors.append(f"TLS certificate validation failed: {pe.message}")
                        warnings.append("Security warning: Target domain served an invalid or untrusted TLS certificate")
                    else:
                        status = ScanStatus.FAILED
                        errors.append(f"Browser navigation error: {pe.message}")
                    final_url = str(page.url) if isinstance(getattr(page, "url", None), str) and page.url else canonical_url
                except Exception as e:
                    status = ScanStatus.FAILED
                    errors.append(f"Unexpected browser failure: {e}")
                    final_url = str(page.url) if isinstance(getattr(page, "url", None), str) and page.url else canonical_url

                # If proxy blocked the primary connection
                if proxy and proxy.blocked_requests and status == ScanStatus.FAILED:
                    status = ScanStatus.BLOCKED_SSRF
                    errors.append(f"Egress connection blocked by Anti-SSRF proxy: {proxy.blocked_requests[0]['reason']}")

                # -------------------------------------------------------------
                # 5. Capture Screenshots & Rendered DOM Evidence
                # -------------------------------------------------------------
                # Allow dynamic client-side JS / SPAs (Vercel, React, Next.js) 5-6 seconds
                # to complete client-side hydration, fetch assets, and render full UI before capture
                if status in (ScanStatus.COMPLETED, ScanStatus.TIMED_OUT):
                    try:
                        await page.wait_for_timeout(5500)
                    except Exception:
                        await asyncio.sleep(5.5)

                viewport_shot_path: str | None = None
                fullpage_shot_path: str | None = None
                dom_path: str | None = None
                html_content = ""

                try:
                    content_val = await page.content()
                    if isinstance(content_val, str) and content_val:
                        html_content = content_val
                        dom_file = self.dom_dir / f"{scan_id}_dom.html"
                        dom_file.write_text(html_content, encoding="utf-8", errors="replace")
                        dom_path = str(dom_file)
                except Exception as e:
                    warnings.append(f"Failed to capture DOM: {e}")

                if status in (ScanStatus.COMPLETED, ScanStatus.TIMED_OUT) and html_content:
                    try:
                        v_file = self.screenshot_dir / f"{scan_id}_viewport.png"
                        await page.screenshot(path=str(v_file), full_page=False, timeout=5000)
                        viewport_shot_path = str(v_file)
                    except Exception as e:
                        warnings.append(f"Failed to capture viewport screenshot: {e}")

                    if opts.capture_full_page and status == ScanStatus.COMPLETED:
                        try:
                            f_file = self.screenshot_dir / f"{scan_id}_full.png"
                            await page.screenshot(path=str(f_file), full_page=True, timeout=5000)
                            fullpage_shot_path = str(f_file)
                        except Exception as e:
                            warnings.append(f"Failed to capture full-page screenshot: {e}")

                # -------------------------------------------------------------
                # 6. Parse DOM Intelligence
                # -------------------------------------------------------------
                page_title, forms, scripts, meta_tags = parse_dom_evidence(html_content, final_url)

                # Ensure context and browser are explicitly closed
                await context.close()
                await browser.close()
                context = None
                browser = None

        finally:
            # -----------------------------------------------------------------
            # 7. Guaranteed Resource & Proxy Cleanup
            # -----------------------------------------------------------------
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass
            if proxy:
                try:
                    await proxy.stop()
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        proxy_blocked = len(proxy.blocked_requests) if proxy else 0

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
            tls_error=tls_error_detail,
            pinned_ip=pinned_ip,
            proxy_blocked_count=proxy_blocked,
            scan_duration_ms=round(elapsed_ms, 2),
            errors=errors,
            warnings=warnings,
            security_metadata={
                "pinned_ip": pinned_ip,
                "resolved_ips": val_result.resolved_ips,
                "pre_validation_valid": val_result.is_valid,
                "proxy_intercepted_count": len(proxy.intercepted_requests) if proxy else 0,
                "popups_closed_count": popups_closed_count,
            },
        )

    def scan_sync(self, url: str, options: ScanOptions | None = None) -> ScanResult:
        """Synchronous wrapper around the async scan implementation."""
        if sys.platform == "win32":
            def _runner() -> ScanResult:
                p_loop = asyncio.ProactorEventLoop()
                asyncio.set_event_loop(p_loop)
                try:
                    return p_loop.run_until_complete(self._scan_impl(url, options))
                finally:
                    p_loop.close()

            try:
                return asyncio.run(self.scan(url, options))
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(_runner).result()

        try:
            return asyncio.run(self.scan(url, options))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.scan(url, options)).result()
