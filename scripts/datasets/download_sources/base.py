"""Base Downloader with Streaming, Checksum Verification & Provenance Tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phishvision.datasets")


@dataclass
class DownloadResult:
    """Metadata detailing the execution and provenance of a dataset acquisition job."""
    source_name: str
    success: bool
    file_path: str | None = None
    file_size_bytes: int = 0
    sha256: str | None = None
    md5: str | None = None
    records_count: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    download_duration_sec: float = 0.0
    error_message: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class BaseDownloader:
    """Base class for all automated dataset acquisition sources in PhishVision AI."""

    DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhishVision-Research/1.0"

    def __init__(
        self,
        raw_dir: Path | str = "data/raw",
        metadata_dir: Path | str = "data/metadata",
        timeout: int = 30,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.metadata_dir = Path(metadata_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._load_env()

    def _load_env(self) -> None:
        """Read .env file if present in the workspace root."""
        env_file = Path(".env")
        if not env_file.exists():
            return
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

    @staticmethod
    def compute_hashes(file_path: Path) -> tuple[str, str]:
        """Compute both SHA-256 and MD5 hashes of a file in streaming chunks."""
        sha256_h = hashlib.sha256()
        md5_h = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256_h.update(chunk)
                md5_h.update(chunk)
        return sha256_h.hexdigest(), md5_h.hexdigest()

    def download_file(
        self,
        url: str,
        target_path: Path,
        expected_md5: str | None = None,
        expected_sha256: str | None = None,
        headers: dict[str, str] | None = None,
        skip_existing: bool = True,
        chunk_size: int = 65536,
    ) -> tuple[bool, str | None]:
        """Stream download file from URL to target_path with checksum validation and resume support."""
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Check existing file
        if skip_existing and target_path.exists() and target_path.stat().st_size > 0:
            if expected_md5 or expected_sha256:
                cur_sha256, cur_md5 = self.compute_hashes(target_path)
                if expected_md5 and cur_md5.lower() == expected_md5.lower():
                    logger.info(f"Skipping existing verified file: {target_path} (MD5 match: {cur_md5})")
                    return True, None
                if expected_sha256 and cur_sha256.lower() == expected_sha256.lower():
                    logger.info(f"Skipping existing verified file: {target_path} (SHA-256 match)")
                    return True, None
            else:
                logger.info(f"Skipping existing file: {target_path} ({target_path.stat().st_size} bytes)")
                return True, None

        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        req_headers = {"User-Agent": self.DEFAULT_USER_AGENT}
        if headers:
            req_headers.update(headers)

        # Attempt resume if partial file exists
        initial_bytes = 0
        if temp_path.exists():
            initial_bytes = temp_path.stat().st_size
            if initial_bytes > 0:
                req_headers["Range"] = f"bytes={initial_bytes}-"
                logger.info(f"Resuming download from byte {initial_bytes} for {target_path.name}")

        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=req_headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as response:
                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size else None

                mode = "ab" if initial_bytes > 0 and response.status == 206 else "wb"
                if mode == "wb":
                    initial_bytes = 0

                downloaded = initial_bytes
                start_time = time.time()
                last_log_time = start_time

                with open(temp_path, mode) as f:
                    while chunk := response.read(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_log_time > 3.0:
                            if total_bytes:
                                pct = (downloaded / (initial_bytes + total_bytes)) * 100
                                logger.info(f"Downloading {target_path.name}: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)")
                            else:
                                logger.info(f"Downloading {target_path.name}: {downloaded / 1024 / 1024:.1f} MB")
                            last_log_time = now

            # Verify integrity
            calc_sha256, calc_md5 = self.compute_hashes(temp_path)
            if expected_md5 and calc_md5.lower() != expected_md5.lower():
                temp_path.unlink(missing_ok=True)
                return False, f"MD5 mismatch: expected {expected_md5}, got {calc_md5}"
            if expected_sha256 and calc_sha256.lower() != expected_sha256.lower():
                temp_path.unlink(missing_ok=True)
                return False, f"SHA-256 mismatch: expected {expected_sha256}, got {calc_sha256}"

            # Rename temp to target
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            logger.info(f"Successfully saved {target_path.name} ({target_path.stat().st_size} bytes)")
            return True, None

        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range Not Satisfiable (file already fully downloaded)
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    if target_path.exists():
                        target_path.unlink()
                    temp_path.rename(target_path)
                    return True, None
            temp_path.unlink(missing_ok=True)
            return False, f"HTTP Error {e.code}: {e.reason}"
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            return False, f"Download failed: {type(e).__name__} - {e}"

    def record_provenance(self, result: DownloadResult) -> None:
        """Append or update dataset provenance in data/metadata/provenance.json."""
        prov_file = self.metadata_dir / "provenance.json"
        history: list[dict[str, Any]] = []
        if prov_file.exists():
            try:
                with open(prov_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(asdict(result))
        with open(prov_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Recorded provenance for {result.source_name} into {prov_file}")
