"""PhishVision AI — Manual Scanner Testing CLI Tool.

Quickly and safely test the Phase 1 Secure Website Scanner against any URL.

Usage:
    # 1. Interactive mode:
    python scripts/manual_scan.py

    # 2. Direct URL scan:
    python scripts/manual_scan.py https://example.com

    # 3. Watch the browser open in real-time (headful):
    python scripts/manual_scan.py https://example.com --headful

    # 4. Test Anti-SSRF blocking:
    python scripts/manual_scan.py http://127.0.0.1:8080
    python scripts/manual_scan.py http://169.254.169.254
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path so imports work cleanly from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.scanner.models import ScanOptions, ScanStatus
from app.scanner.playwright_scanner import PlaywrightScanner


def print_banner() -> None:
    print("=" * 70)
    print("   🛡️   PhishVision AI — Secure Scanner (Phase 1 Manual Test)   🛡️")
    print("=" * 70)


def print_result(result) -> None:
    print("\n" + "-" * 70)
    print("📊 SCAN RESULTS")
    print("-" * 70)
    print(f"• Scan ID         : {result.scan_id}")
    print(f"• Requested URL   : {result.requested_url}")
    print(f"• Final URL       : {result.final_url}")
    print(f"• Status          : {result.status.value}")
    print(f"• HTTP Code       : {result.status_code if result.status_code else 'N/A'}")
    print(f"• Page Title      : {result.page_title or 'N/A'}")
    print(f"• Pinned Outbound : {result.pinned_ip or 'None'}")
    print(f"• Scan Duration   : {result.scan_duration_ms:.2f} ms")

    if result.status == ScanStatus.BLOCKED_SSRF:
        print("\n🚫 ANTI-SSRF PROTECTION TRIGGERED:")
        print(f"  - Block Reason  : {result.security_metadata.get('blocked_reason')}")
        print(f"  - Error Code    : {result.security_metadata.get('validation_code')}")
        print(f"  - Resolved IPs  : {result.security_metadata.get('resolved_ips')}")
        print("-" * 70)
        return

    if result.status == ScanStatus.CERT_ERROR:
        print("\n⚠️ TLS/SSL CERTIFICATE ERROR:")
        print(f"  - Error Detail  : {result.tls_error}")
        print("-" * 70)
        return

    # Redirects
    if result.redirect_chain:
        print(f"\n🔄 REDIRECT CHAIN ({len(result.redirect_chain)} hops):")
        for hop in result.redirect_chain:
            print(f"  [{hop.step}] {hop.status_code} -> {hop.url} ({hop.domain})")

    # Forms & Login Fields
    print(f"\n📝 FORMS DETECTED: {len(result.forms)}")
    for idx, f in enumerate(result.forms, 1):
        ext_flag = "⚠️ EXTERNAL ACTION" if f.is_external_submission else "Internal"
        pwd_flag = "🔑 CONTAINS PASSWORD FIELD" if f.has_password_field else "No password field"
        print(f"  Form #{idx} [{f.method.upper()}] -> action: '{f.action_url or '(empty)'}' ({ext_flag})")
        print(f"    - Fields ({len(f.fields)}): {[f.field_type + ':' + (f.name or f.field_id or 'anon') for f in f.fields]}")
        if f.has_password_field:
            print(f"    - {pwd_flag}")

    # Scripts
    print(f"\n📜 SCRIPTS DETECTED: {len(result.scripts)}")
    ext_scripts = [s.src for s in result.scripts if s.is_external]
    print(f"  - External Scripts : {len(ext_scripts)}")
    if ext_scripts[:3]:
        for s in ext_scripts[:3]:
            print(f"    * {s}")
        if len(ext_scripts) > 3:
            print(f"    * ... and {len(ext_scripts) - 3} more")

    # Artifacts saved on disk
    print("\n📁 FORENSIC EVIDENCE ARTIFACTS:")
    if result.viewport_screenshot_path and os.path.exists(result.viewport_screenshot_path):
        print(f"  📸 Viewport Screenshot : {result.viewport_screenshot_path}")
    if result.full_page_screenshot_path and os.path.exists(result.full_page_screenshot_path):
        print(f"  📸 Full-page Screenshot: {result.full_page_screenshot_path}")
    if result.dom_snapshot_path and os.path.exists(result.dom_snapshot_path):
        size_kb = os.path.getsize(result.dom_snapshot_path) / 1024
        print(f"  📄 DOM Snapshot (HTML) : {result.dom_snapshot_path} ({size_kb:.1f} KB)")

    if result.errors:
        print(f"\n⚠️ Errors encountered: {result.errors}")
    print("-" * 70)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishVision AI — Manual Website Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", default=None, help="Website URL to scan (e.g. https://example.com)")
    parser.add_argument("--headful", action="store_true", help="Launch visible browser window (watch it scan)")
    parser.add_argument("--timeout", type=int, default=15, help="Scan timeout in seconds (default: 15)")
    parser.add_argument("--ignore-tls", action="store_true", help="Ignore TLS certificate errors")
    parser.add_argument("--disable-proxy", action="store_true", help="Disable the validating egress proxy")

    args = parser.parse_args()

    print_banner()

    target_url = args.url
    if not target_url:
        print("\nChoose an option or enter a custom URL:")
        print("  [1] Safe benign site       : https://example.com")
        print("  [2] Wikipedia (rich DOM)   : https://en.wikipedia.org")
        print("  [3] Anti-SSRF test         : http://127.0.0.1:8080 (Loopback blocked)")
        print("  [4] Anti-SSRF test         : http://169.254.169.254 (Cloud metadata blocked)")
        print("  [5] Custom URL entry")

        choice = input("\nEnter choice [1-5] or paste URL directly: ").strip()
        if choice == "1":
            target_url = "https://example.com"
        elif choice == "2":
            target_url = "https://en.wikipedia.org"
        elif choice == "3":
            target_url = "http://127.0.0.1:8080"
        elif choice == "4":
            target_url = "http://169.254.169.254"
        elif choice == "5":
            target_url = input("Enter target URL: ").strip()
        elif choice.startswith("http://") or choice.startswith("https://"):
            target_url = choice
        else:
            if choice:
                target_url = "https://" + choice
            else:
                target_url = "https://example.com"

    options = ScanOptions(
        headless=not args.headful,
        timeout_seconds=args.timeout,
        ignore_https_errors=args.ignore_tls,
        enable_egress_proxy=not args.disable_proxy,
    )

    print(f"\n🚀 Initiating scan for: {target_url}")
    print(f"⚙️  Options: Headless={options.headless}, Timeout={options.timeout_seconds}s, EgressProxy={options.enable_egress_proxy}")

    scanner = PlaywrightScanner()
    result = await scanner.scan(target_url, options=options)

    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
