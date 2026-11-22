"""Unit Tests for Automated Dataset Downloaders and Provenance Verification."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.datasets.download_sources.base import BaseDownloader, DownloadResult
from scripts.datasets.download_sources.openphish import OpenPhishDownloader
from scripts.datasets.download_sources.phishtank import PhishTankDownloader
from scripts.datasets.download_sources.phreshphish import PhreshPhishDownloader
from scripts.datasets.download_sources.tranco import TrancoDownloader
from scripts.datasets.download_sources.urlhaus import URLhausDownloader
from scripts.datasets.download_sources.zenodo import ZenodoDownloader


@pytest.fixture
def temp_dataset_dirs(tmp_path: Path):
    raw = tmp_path / "raw"
    meta = tmp_path / "metadata"
    raw.mkdir(parents=True)
    meta.mkdir(parents=True)
    return raw, meta


class TestBaseDownloader:
    """Tests for BaseDownloader streaming, checksum verification, and provenance."""

    def test_compute_hashes(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello PhishVision Dataset!", encoding="utf-8")
        sha256, md5 = BaseDownloader.compute_hashes(test_file)
        assert len(sha256) == 64
        assert len(md5) == 32
        # Check reproducibility
        sha256_2, md5_2 = BaseDownloader.compute_hashes(test_file)
        assert sha256 == sha256_2
        assert md5 == md5_2

    def test_record_provenance(self, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = BaseDownloader(raw_dir=raw, metadata_dir=meta)

        res = DownloadResult(
            source_name="TestSource",
            success=True,
            file_path=str(raw / "test.txt"),
            file_size_bytes=100,
            sha256="fake_sha256",
            md5="fake_md5",
            records_count=10,
        )
        downloader.record_provenance(res)

        prov_file = meta / "provenance.json"
        assert prov_file.exists()
        with open(prov_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["source_name"] == "TestSource"
        assert data[0]["sha256"] == "fake_sha256"

    def test_skip_existing_verified_file(self, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = BaseDownloader(raw_dir=raw, metadata_dir=meta)
        target = raw / "exists.txt"
        target.write_text("already here", encoding="utf-8")

        success, err = downloader.download_file(
            url="http://fake.url/file.txt",
            target_path=target,
            skip_existing=True,
        )
        assert success is True
        assert err is None


class TestOpenPhishDownloader:
    """Tests for OpenPhish community feed parser."""

    @patch("scripts.datasets.download_sources.base.BaseDownloader.download_file")
    def test_fetch_openphish_success(self, mock_download, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = OpenPhishDownloader(raw_dir=raw, metadata_dir=meta)

        # Simulate written file
        def fake_download(*args, **kwargs):
            target = kwargs["target_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "http://phish1.com/login\nhttp://phish2.com/secure\n# comment\n\nhttp://phish3.com\n",
                encoding="utf-8",
            )
            return True, None

        mock_download.side_effect = fake_download
        res = downloader.fetch(skip_existing=False)

        assert res.success is True
        assert res.source_name == "OpenPhish"
        assert res.records_count == 3
        assert res.sha256 is not None


class TestPhishTankDownloader:
    """Tests for PhishTank verified CSV feed parser."""

    @patch("scripts.datasets.download_sources.base.BaseDownloader.download_file")
    def test_fetch_phishtank_csv(self, mock_download, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = PhishTankDownloader(raw_dir=raw, metadata_dir=meta)

        def fake_download(*args, **kwargs):
            target = kwargs["target_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["phish_id", "url", "phish_detail_url", "submission_time", "verified", "online", "target"])
                writer.writerow(["1", "http://fake-paypal.com", "http://detail/1", "2026-11-22", "yes", "yes", "PayPal"])
                writer.writerow(["2", "http://fake-microsoft.com", "http://detail/2", "2026-11-22", "yes", "yes", "Microsoft"])
            return True, None

        mock_download.side_effect = fake_download
        res = downloader.fetch(skip_existing=False)

        assert res.success is True
        assert res.source_name == "PhishTank"
        assert res.records_count == 2


class TestPhreshPhishDownloader:
    """Tests for PhreshPhish Hugging Face Parquet downloader."""

    @patch("scripts.datasets.download_sources.base.BaseDownloader.download_file")
    def test_fetch_phreshphish_shard(self, mock_download, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = PhreshPhishDownloader(raw_dir=raw, metadata_dir=meta)

        def fake_download(*args, **kwargs):
            target = kwargs["target_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fake parquet content", encoding="utf-8")
            return True, None

        mock_download.side_effect = fake_download
        res = downloader.fetch(shard_name="test-000.parquet", skip_existing=False)

        assert res.success is True
        assert "PhreshPhish" in res.source_name
        assert res.file_path is not None


class TestTrancoDownloader:
    """Tests for Tranco ZIP extraction and top-N domain filtering."""

    @patch("scripts.datasets.download_sources.base.BaseDownloader.download_file")
    def test_fetch_tranco_filtering(self, mock_download, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = TrancoDownloader(raw_dir=raw, metadata_dir=meta)

        def fake_download(*args, **kwargs):
            zip_target = kwargs["target_path"]
            zip_target.parent.mkdir(parents=True, exist_ok=True)
            # Create a mock zip containing top-1m.csv
            inner_csv = raw / "temp_top.csv"
            with open(inner_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for i in range(1, 21):
                    writer.writerow([str(i), f"domain{i}.com"])

            with zipfile.ZipFile(zip_target, "w") as zf:
                zf.write(inner_csv, arcname="top-1m.csv")
            inner_csv.unlink()
            return True, None

        mock_download.side_effect = fake_download
        res = downloader.fetch(top_n=5, skip_existing=False)

        assert res.success is True
        assert res.source_name == "Tranco"
        assert res.records_count == 5

        # Verify output CSV content
        filtered_path = Path(res.file_path)
        assert filtered_path.exists()
        with open(filtered_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 6  # header + 5 rows


class TestURLhausDownloader:
    """Tests for abuse.ch URLhaus Auth-Key handling."""

    def test_missing_auth_key_graceful_exit(self, temp_dataset_dirs, monkeypatch):
        raw, meta = temp_dataset_dirs
        monkeypatch.delenv("URLHAUS_AUTH_KEY", raising=False)
        downloader = URLhausDownloader(raw_dir=raw, metadata_dir=meta)

        res = downloader.fetch()
        assert res.success is False
        assert "URLHAUS_AUTH_KEY is not set" in res.error_message


class TestZenodoDownloader:
    """Tests for Zenodo metadata fetching and file downloading."""

    @patch("scripts.datasets.download_sources.zenodo.ZenodoDownloader.get_record_metadata")
    @patch("scripts.datasets.download_sources.base.BaseDownloader.download_file")
    def test_download_file_by_name(self, mock_download, mock_meta, temp_dataset_dirs):
        raw, meta = temp_dataset_dirs
        downloader = ZenodoDownloader(raw_dir=raw, metadata_dir=meta)

        mock_meta.return_value = {
            "doi": "10.5281/zenodo.8041387",
            "metadata": {"title": "Phishing Website Dataset"},
            "files": [
                {
                    "key": "brands.csv",
                    "size": 16951,
                    "checksum": "md5:c1a2b3c4d5e6f7",
                    "links": {"self": "https://zenodo.org/api/files/brands.csv"},
                }
            ],
        }

        def fake_download(*args, **kwargs):
            target = kwargs["target_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("brand,count\npaypal,100\n", encoding="utf-8")
            return True, None

        mock_download.side_effect = fake_download
        res = downloader.download_file_by_name(
            record_id="8041387",
            file_key="brands.csv",
            skip_existing=False,
        )

        assert res.success is True
        assert res.source_name == "Zenodo:8041387:brands.csv"
        assert res.extra_metadata["doi"] == "10.5281/zenodo.8041387"
