"""Automated Downloader for Hugging Face PhreshPhish Dataset."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .base import BaseDownloader, DownloadResult

logger = logging.getLogger("phishvision.datasets.phreshphish")


class PhreshPhishDownloader(BaseDownloader):
    """Downloads content-rich Parquet slices from Hugging Face phreshphish/phreshphish."""

    BASE_URL = "https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/data/{filename}"

    def fetch(self, shard_name: str = "train-000.parquet", skip_existing: bool = True) -> DownloadResult:
        start_time = time.time()
        output_dir = self.raw_dir / "urls"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_file = output_dir / f"phreshphish_{shard_name}"

        url = self.BASE_URL.format(filename=shard_name)
        logger.info(f"Downloading PhreshPhish shard '{shard_name}' from {url}")

        success, err = self.download_file(
            url=url,
            target_path=target_file,
            skip_existing=skip_existing,
        )

        duration = time.time() - start_time
        if not success or not target_file.exists():
            res = DownloadResult(
                source_name=f"PhreshPhish:{shard_name}",
                success=False,
                error_message=err or "Download failed",
                download_duration_sec=round(duration, 2),
            )
            self.record_provenance(res)
            return res

        # Read row count using pyarrow / pandas
        records_count = None
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(target_file)
            records_count = parquet_file.metadata.num_rows
        except Exception as e:
            logger.warning(f"Failed to inspect parquet row count: {e}")

        sha256, md5 = self.compute_hashes(target_file)
        file_size = target_file.stat().st_size

        res = DownloadResult(
            source_name=f"PhreshPhish:{shard_name}",
            success=True,
            file_path=str(target_file),
            file_size_bytes=file_size,
            sha256=sha256,
            md5=md5,
            records_count=records_count,
            download_duration_sec=round(duration, 2),
            extra_metadata={"shard": shard_name, "dataset": "phreshphish/phreshphish"},
        )
        self.record_provenance(res)
        logger.info(f"PhreshPhish shard complete: {records_count} rows, {file_size / 1024 / 1024:.2f} MB")
        return res
