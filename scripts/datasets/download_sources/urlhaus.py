"""Automated Downloader for abuse.ch URLhaus Feed."""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.urlhaus")


class URLhausDownloader(BaseDownloader):
    """Downloads malware and phishing distribution URLs from abuse.ch URLhaus.

    Note: abuse.ch retired keyless public CSV exports (HTTP 410 Gone).
    A free personal Auth-Key from https://auth.abuse.ch/ is required and
    must be passed via URLHAUS_AUTH_KEY environment variable.
    """

    BASE_EXPORT_URL = "https://urlhaus-api.abuse.ch/v2/files/exports/{auth_key}/recent.csv"

    def fetch(self, skip_existing: bool = False) -> DownloadResult:
        start_time = time.time()
        output_dir = self.raw_dir / "urls"
        output_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = output_dir / f"urlhaus_recent_{today_str}.csv"

        auth_key = os.environ.get("URLHAUS_AUTH_KEY")
        if not auth_key:
            msg = (
                "URLHAUS_AUTH_KEY is not set. abuse.ch keyless public exports are retired (HTTP 410 Gone). "
                "Please register a free personal key at https://auth.abuse.ch/ and set URLHAUS_AUTH_KEY in .env"
            )
            logger.warning(msg)
            res = DownloadResult(
                source_name="URLhaus",
                success=False,
                error_message=msg,
                download_duration_sec=round(time.time() - start_time, 2),
                extra_metadata={"auth_required": True, "auth_portal": "https://auth.abuse.ch/"},
            )
            self.record_provenance(res)
            return res

        export_url = self.BASE_EXPORT_URL.format(auth_key=auth_key)
        logger.info("Fetching authenticated URLhaus recent export")
        success, err = self.download_file(
            url=export_url,
            target_path=target_file,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not target_file.exists():
            res = DownloadResult(
                source_name="URLhaus",
                success=False,
                error_message=err or "Export download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Count CSV records
        records_count = 0
        try:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                records_count = sum(1 for line in f if line.strip() and not line.startswith("#"))
        except Exception as e:
            logger.warning(f"Error parsing URLhaus export lines: {e}")

        sha256, md5 = self.compute_hashes(target_file)
        file_size = target_file.stat().st_size

        res = DownloadResult(
            source_name="URLhaus",
            success=True,
            file_path=str(target_file),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            records_count=records_count,
            download_duration_sec=round(duration, 2),
        )
        self.record_provenance(res)
        logger.info(f"URLhaus download complete: {records_count} records, {file_size / 1024:.1f} KB")
        return res
