"""Automated Downloader for PhishTank Verified Feeds."""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.phishtank")


class PhishTankDownloader(BaseDownloader):
    """Downloads active and verified phishing URL dumps from PhishTank."""

    FEED_URL = "http://data.phishtank.com/data/online-valid.csv"

    def fetch(self, skip_existing: bool = False) -> DownloadResult:
        start_time = time.time()
        output_dir = self.raw_dir / "urls"
        output_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = output_dir / f"phishtank_online_valid_{today_str}.csv"

        api_key = os.environ.get("PHISHTANK_API_KEY")
        headers = {
            "User-Agent": "phishtank/PhishVision-Academic-Research-1.0",
        }
        url = self.FEED_URL
        if api_key:
            url = f"{self.FEED_URL}?app_key={api_key}"
            logger.info("Using configured PhishTank developer app key from environment")

        logger.info(f"Fetching PhishTank verified dump from {self.FEED_URL}")
        success, err = self.download_file(
            url=url,
            target_path=target_file,
            headers=headers,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not target_file.exists():
            res = DownloadResult(
                source_name="PhishTank",
                success=False,
                error_message=err or "File download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Validate CSV and count records
        records_count = 0
        try:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                records_count = sum(1 for _ in reader)
        except Exception as e:
            logger.warning(f"Error parsing PhishTank CSV header: {e}")

        sha256, md5 = self.compute_hashes(target_file)
        file_size = target_file.stat().st_size

        res = DownloadResult(
            source_name="PhishTank",
            success=True,
            file_path=str(target_file),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            records_count=records_count,
            download_duration_sec=round(duration, 2),
            extra_metadata={"feed_url": self.FEED_URL, "active_date": today_str},
        )
        self.record_provenance(res)
        logger.info(f"PhishTank download complete: {records_count} verified phishing records, {file_size / 1024 / 1024:.2f} MB")
        return res
