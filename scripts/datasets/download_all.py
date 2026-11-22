"""PhishVision AI — Central Automated Dataset Acquisition Orchestrator.

Usage:
    # 1. Download all verified public sources:
    python scripts/datasets/download_all.py

    # 2. Download specific sources only:
    python scripts/datasets/download_all.py --sources openphish,tranco

    # 3. Limit Tranco to top 10,000 domains:
    python scripts/datasets/download_all.py --sources tranco --top-n 10000

    # 4. Force re-download:
    python scripts/datasets/download_all.py --no-skip-existing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.download_sources import (
    DownloadResult,
    OpenPhishDownloader,
    PhishTankDownloader,
    PhreshPhishDownloader,
    TrancoDownloader,
    URLhausDownloader,
    ZenodoDownloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phishvision.download_all")


def print_banner() -> None:
    print("=" * 80)
    print("   🌐  PhishVision AI — Automated Dataset Acquisition (Phase 2)  🌐")
    print("=" * 80)


def print_summary_table(results: list[DownloadResult]) -> None:
    print("\n" + "=" * 80)
    print("📋 DATASET ACQUISITION SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Source':<18} | {'Status':<10} | {'Records':<10} | {'Size':<12} | {'Time (s)':<8} | {'Hash (SHA-256)':<12}")
    print("-" * 80)

    for r in results:
        status_str = "✅ SUCCESS" if r.success else "❌ FAILED"
        if not r.success and "AUTH" in (r.error_message or "").upper():
            status_str = "⚠️ AUTH REQ"

        records_str = str(r.records_count) if r.records_count is not None else "N/A"
        size_str = f"{r.file_size_bytes / 1024:.1f} KB" if r.file_size_bytes < 1048576 else f"{r.file_size_bytes / 1048576:.2f} MB"
        if not r.success:
            size_str = "0 B"
        hash_str = r.sha256[:10] + "..." if r.sha256 else "N/A"

        print(f"{r.source_name:<18} | {status_str:<10} | {records_str:<10} | {size_str:<12} | {r.download_duration_sec:<8.2f} | {hash_str:<12}")

    print("-" * 80)
    failed = [r for r in results if not r.success and not ("AUTH" in (r.error_message or "").upper())]
    auth_gated = [r for r in results if not r.success and ("AUTH" in (r.error_message or "").upper())]

    if auth_gated:
        print("\n⚠️ Note on Auth-Gated Feeds:")
        for r in auth_gated:
            print(f"  • {r.source_name}: {r.error_message}")
    if failed:
        print("\n❌ Errors encountered:")
        for r in failed:
            print(f"  • {r.source_name}: {r.error_message}")
    print("\nProvenance logs saved in: data/metadata/provenance.json")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishVision AI — Central Automated Dataset Acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated list of sources (openphish, phishtank, tranco, zenodo, urlhaus, or all)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Target base directory for raw data (default: data/raw)",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
        help="Target directory for provenance metadata (default: data/metadata)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50000,
        help="Number of top Tranco domains to filter (default: 50000)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force re-download even if files already exist on disk",
    )

    args = parser.parse_args()
    print_banner()

    skip_existing = not args.no_skip_existing
    sources = [s.strip().lower() for s in args.sources.split(",")]
    run_all = "all" in sources

    results: list[DownloadResult] = []

    # 1. OpenPhish
    if run_all or "openphish" in sources:
        logger.info("\n--- [1/5] Acquiring OpenPhish Community Feed ---")
        op = OpenPhishDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(op.fetch(skip_existing=skip_existing))

    # 2. PhishTank
    if run_all or "phishtank" in sources:
        logger.info("\n--- [2/5] Acquiring PhishTank Verified Feed ---")
        pt = PhishTankDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(pt.fetch(skip_existing=skip_existing))

    # 3. Tranco
    if run_all or "tranco" in sources:
        logger.info("\n--- [3/5] Acquiring Tranco Research Domain Benchmark ---")
        tr = TrancoDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(tr.fetch(top_n=args.top_n, skip_existing=skip_existing))

    # 4. Zenodo Brand Metadata
    if run_all or "zenodo" in sources:
        logger.info("\n--- [4/5] Acquiring Zenodo Benchmark Brand Mapping ---")
        zen = ZenodoDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(zen.fetch_brand_mapping(skip_existing=skip_existing))

    # 5. PhreshPhish (Hugging Face)
    if run_all or "phreshphish" in sources:
        logger.info("\n--- [5/6] Acquiring PhreshPhish Benchmark Shard ---")
        pp = PhreshPhishDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(pp.fetch(shard_name="train-000.parquet", skip_existing=skip_existing))

    # 6. URLhaus (Auth-gated)
    if run_all or "urlhaus" in sources:
        logger.info("\n--- [6/6] Checking abuse.ch URLhaus Feed ---")
        uh = URLhausDownloader(raw_dir=args.raw_dir, metadata_dir=args.metadata_dir)
        results.append(uh.fetch(skip_existing=skip_existing))

    print_summary_table(results)


if __name__ == "__main__":
    main()
