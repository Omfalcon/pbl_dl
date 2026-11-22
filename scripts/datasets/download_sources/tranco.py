"""Automated Downloader for Tranco Top Domains Benchmark."""

from __future__ import annotations

import csv
import logging
import time
import zipfile
from pathlib import Path

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.tranco")


class TrancoDownloader(BaseDownloader):
    """Downloads and unpacks the Tranco Top 1 Million research domain list."""

    FEED_URL = "https://tranco-list.eu/top-1m.csv.zip"

    def fetch(self, top_n: int = 50000, skip_existing: bool = True) -> DownloadResult:
        start_time = time.time()
        output_dir = self.raw_dir / "urls"
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_target = output_dir / "tranco_top1m.csv.zip"
        csv_target = output_dir / "tranco_top1m.csv"
        filtered_target = output_dir / f"tranco_top_{top_n}.csv" if top_n else csv_target

        logger.info(f"Downloading Tranco archive from {self.FEED_URL}")
        success, err = self.download_file(
            url=self.FEED_URL,
            target_path=zip_target,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not zip_target.exists():
            res = DownloadResult(
                source_name="Tranco",
                success=False,
                error_message=err or "Archive download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Extract CSV from ZIP
        try:
            with zipfile.ZipFile(zip_target, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV file found inside Tranco ZIP archive")
                extracted_name = csv_names[0]
                zf.extract(extracted_name, path=output_dir)
                raw_extracted = output_dir / extracted_name
                if raw_extracted != csv_target:
                    if csv_target.exists():
                        csv_target.unlink()
                    raw_extracted.rename(csv_target)
            logger.info(f"Extracted Tranco master list to {csv_target}")
        except Exception as e:
            res = DownloadResult(
                source_name="Tranco",
                success=False,
                error_message=f"ZIP extraction error: {e}",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Filter top N domains if specified
        total_records = 0
        if top_n and top_n > 0:
            with open(csv_target, "r", encoding="utf-8", errors="ignore") as f_in, \
                 open(filtered_target, "w", encoding="utf-8", newline="") as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                writer.writerow(["rank", "domain"])
                for idx, row in enumerate(reader):
                    if idx >= top_n:
                        break
                    if len(row) >= 2:
                        writer.writerow([row[0].strip(), row[1].strip()])
                        total_records += 1
            logger.info(f"Filtered top {total_records} domains into {filtered_target}")
        else:
            with open(csv_target, "r", encoding="utf-8", errors="ignore") as f:
                total_records = sum(1 for _ in f)

        sha256, md5 = self.compute_hashes(filtered_target)
        file_size = filtered_target.stat().st_size

        res = DownloadResult(
            source_name="Tranco",
            success=True,
            file_path=str(filtered_target),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            records_count=total_records,
            download_duration_sec=round(duration, 2),
            extra_metadata={"feed_url": self.FEED_URL, "top_n": top_n},
        )
        self.record_provenance(res)
        return res
