"""Unit Tests for External Visual Dataset Ingestion, Path Traversal Safety & Validation."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from scripts.datasets.generate_splits import DomainAwareSplitter
from scripts.datasets.ingest_external_dataset import safe_extract_zip


class TestSafeZipExtraction:
    """Verifies that zip extraction strictly enforces path traversal and binary execution guards."""

    def test_safe_extraction_blocks_zip_slip(self, tmp_path: Path):
        zip_file = tmp_path / "malicious.zip"
        extract_to = tmp_path / "extracted"

        # Create a synthetic zip with path traversal and forbidden executable
        with zipfile.ZipFile(zip_file, "w") as z:
            # Safe member
            z.writestr("safe_folder/image.png", b"VALID_IMAGE_BYTES")
            # Zip-slip traversal member
            z.writestr("../evil.txt", b"EVIL_PAYLOAD")
            # Forbidden executable
            z.writestr("safe_folder/run.exe", b"MZ_EXECUTABLE_HEADER")

        extracted_count, skipped = safe_extract_zip(zip_file, extract_to)

        # Only the safe image should be extracted
        assert extracted_count == 1
        assert (extract_to / "safe_folder" / "image.png").exists()
        assert not (tmp_path / "evil.txt").exists()
        assert not (extract_to / "safe_folder" / "run.exe").exists()
        assert len(skipped) == 2


class TestVisualCorpusIntegrity:
    """Verifies the catalogued visual dataset and domain-aware visual splits."""

    @pytest.fixture
    def visual_corpus_df(self):
        corpus_path = Path("data/processed/screenshots/visual_corpus.parquet")
        if not corpus_path.exists():
            pytest.skip("visual_corpus.parquet not yet generated")
        return pd.read_parquet(corpus_path)

    def test_visual_corpus_counts_and_labels(self, visual_corpus_df):
        assert len(visual_corpus_df) == 1000
        # 800 Phishing (1), 200 Legitimate (0)
        assert (visual_corpus_df["label"] == 1).sum() == 800
        assert (visual_corpus_df["label"] == 0).sum() == 200

        # Mandatory columns
        expected_cols = {
            "sample_id",
            "original_url",
            "canonical_url",
            "registrable_domain",
            "screenshot_relative_path",
            "screenshot_absolute_path",
            "label",
            "verification_status",
            "source",
            "width",
            "height",
            "channels",
            "sha256",
            "phash",
        }
        assert expected_cols.issubset(set(visual_corpus_df.columns))

    def test_visual_files_exist_on_disk(self, visual_corpus_df):
        # Sample 10 items to verify actual on-disk existence without slow disk thrashing
        sample_rows = visual_corpus_df.sample(10, random_state=42)
        for _, row in sample_rows.iterrows():
            abs_p = Path(row["screenshot_absolute_path"])
            assert abs_p.exists()
            assert abs_p.stat().st_size > 0

    def test_visual_splits_zero_domain_and_duplicate_leakage(self):
        import imagehash

        splits_dir = Path("data/splits/visual")
        if not (splits_dir / "train.parquet").exists():
            pytest.skip("Visual splits not yet generated")

        df_tr = pd.read_parquet(splits_dir / "train.parquet")
        df_val = pd.read_parquet(splits_dir / "val.parquet")
        df_test = pd.read_parquet(splits_dir / "test.parquet")

        # Total conservation
        assert len(df_tr) + len(df_val) + len(df_test) == 1000

        # Zero domain leakage
        tr_doms = set(df_tr["registrable_domain"])
        val_doms = set(df_val["registrable_domain"])
        test_doms = set(df_test["registrable_domain"])

        no_leak, overlaps = DomainAwareSplitter.verify_no_domain_leakage(tr_doms, val_doms, test_doms)
        assert no_leak is True
        assert len(overlaps) == 0

        # Zero SHA-256 exact duplicate leakage
        sha_tr = set(df_tr["sha256"])
        sha_val = set(df_val["sha256"])
        sha_test = set(df_test["sha256"])
        assert len(sha_tr.intersection(sha_val)) == 0
        assert len(sha_tr.intersection(sha_test)) == 0
        assert len(sha_val.intersection(sha_test)) == 0

        # Zero pHash perceptual duplicate leakage (Hamming distance <= 4)
        ph_tr = [imagehash.hex_to_hash(h) for h in df_tr["phash"].dropna()]
        ph_val = [imagehash.hex_to_hash(h) for h in df_val["phash"].dropna()]
        ph_test = [imagehash.hex_to_hash(h) for h in df_test["phash"].dropna()]

        for h_tr in ph_tr:
            assert all((h_tr - h_v) > 4 for h_v in ph_val)
            assert all((h_tr - h_t) > 4 for h_t in ph_test)

        for h_v in ph_val:
            assert all((h_v - h_t) > 4 for h_t in ph_test)
