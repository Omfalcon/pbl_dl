"""Automated Downloader for OpenPhish Community Feed."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.openphish")


class OpenPhishDownloader(BaseDownloader):
    """Downloads and records the active OpenPhish community feed."""

    FEED_URL = "https://openphish.com/feed.txt"

    def fetch(self, skip_existing: bool = False) -> DownloadResult:
        start_time = time.time()
        output_dir = self.raw_dir / "urls"
        output_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = output_dir / f"openphish_{today_str}.txt"

        logger.info(f"Fetching OpenPhish community feed from {self.FEED_URL}")
        success, err = self.download_file(
            url=self.FEED_URL,
            target_path=target_file,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not target_file.exists():
            res = DownloadResult(
                source_name="OpenPhish",
                success=False,
                error_message=err or "File download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Count lines & calculate hashes
        urls_count = 0
        with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
            urls_count = sum(1 for line in f if line.strip() and not line.startswith("#"))

        sha256, md5 = self.compute_hashes(target_file)
        file_size = target_file.stat().st_size

        res = DownloadResult(
            source_name="OpenPhish",
            success=True,
            file_path=str(target_file),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            records_count=urls_count,
            download_duration_sec=round(duration, 2),
            extra_metadata={"feed_url": self.FEED_URL, "active_date": today_str},
        )
        self.record_provenance(res)
        logger.info(f"OpenPhish download complete: {urls_count} URLs, {file_size / 1024:.1f} KB")
        return res
