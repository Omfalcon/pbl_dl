"""Automated Downloader for Zenodo Research Datasets."""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.zenodo")


class ZenodoDownloader(BaseDownloader):
    """Downloads curated benchmark datasets from Zenodo Open Access repositories."""

    API_RECORD_URL = "https://zenodo.org/api/records/{record_id}"

    # Default monitored Zenodo benchmark records
    RECORD_BRAND_DATASET = "8041387"      # Phishing Website Dataset (Brand Impersonation, 86 brands)
    RECORD_MULTIMODAL = "23477065"       # Phishing & Legitimate Websites Benchmark (Phish-Iris)

    def get_record_metadata(self, record_id: str) -> dict[str, Any]:
        """Fetch record metadata, DOI, files list, and checksums from Zenodo REST API."""
        url = self.API_RECORD_URL.format(record_id=record_id)
        token = os.environ.get("ZENODO_ACCESS_TOKEN")
        headers = {"User-Agent": self.DEFAULT_USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def download_file_by_name(
        self,
        record_id: str,
        file_key: str,
        output_subdir: str = "metadata",
        skip_existing: bool = True,
    ) -> DownloadResult:
        """Download a specific file from a Zenodo record with official checksum verification."""
        start_time = time.time()
        output_dir = self.raw_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / file_key

        try:
            meta = self.get_record_metadata(record_id)
        except Exception as e:
            res = DownloadResult(
                source_name=f"Zenodo:{record_id}:{file_key}",
                success=False,
                error_message=f"Failed to fetch Zenodo record metadata: {e}",
                download_duration_sec=round(time.time() - start_time, 2),
            )
            self.record_provenance(res)
            return res

        # Locate file inside files list
        matched_file = None
        for f in meta.get("files", []):
            if f.get("key") == file_key:
                matched_file = f
                break

        if not matched_file:
            res = DownloadResult(
                source_name=f"Zenodo:{record_id}:{file_key}",
                success=False,
                error_message=f"File '{file_key}' not found in Zenodo record {record_id}",
                download_duration_sec=round(time.time() - start_time, 2),
            )
            self.record_provenance(res)
            return res

        download_url = matched_file.get("links", {}).get("self")
        expected_checksum = matched_file.get("checksum", "")  # e.g., "md5:abcdef..."
        expected_md5 = None
        if expected_checksum.startswith("md5:"):
            expected_md5 = expected_checksum.split("md5:")[1].strip()

        logger.info(f"Downloading Zenodo file {file_key} ({matched_file.get('size', 0)} bytes) from record {record_id}")
        success, err = self.download_file(
            url=download_url,
            target_path=target_path,
            expected_md5=expected_md5,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not target_path.exists():
            res = DownloadResult(
                source_name=f"Zenodo:{record_id}:{file_key}",
                success=False,
                error_message=err or "Download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        sha256, md5 = self.compute_hashes(target_path)
        file_size = target_path.stat().st_size

        res = DownloadResult(
            source_name=f"Zenodo:{record_id}:{file_key}",
            success=True,
            file_path=str(target_path),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            download_duration_sec=round(duration, 2),
            extra_metadata={
                "zenodo_record_id": record_id,
                "doi": meta.get("doi"),
                "title": meta.get("metadata", {}).get("title"),
            },
        )
        self.record_provenance(res)
        return res

    def fetch_brand_mapping(self, skip_existing: bool = True) -> DownloadResult:
        """Download brands.csv mapping file from Zenodo Record 8041387."""
        return self.download_file_by_name(
            record_id=self.RECORD_BRAND_DATASET,
            file_key="brands.csv",
            output_subdir="metadata",
            skip_existing=skip_existing,
        )
