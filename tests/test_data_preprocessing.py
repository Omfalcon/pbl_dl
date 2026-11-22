"""Unit Tests for Milestone 2B Data Cleaning, Preprocessing & Perceptual Hashing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.datasets.deduplicate import DatasetDeduplicator
from scripts.datasets.normalize_urls import NormalizedURLRecord, URLNormalizer
from scripts.datasets.validate_images import ImageValidator


@pytest.fixture
def sample_images(tmp_path: Path):
    """Creates synthetic images for validation and deduplication testing."""
    # 1. Valid rich image (random high variance noise/gradient)
    valid_path = tmp_path / "valid.png"
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(valid_path)

    # 2. Near-duplicate of valid image (slight noise variation)
    near_dup_path = tmp_path / "near_dup.png"
    arr_near = np.clip(arr.astype(int) + np.random.randint(-2, 3, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr_near).save(near_dup_path)

    # 3. Exact duplicate of valid image
    exact_dup_path = tmp_path / "exact_dup.png"
    exact_dup_path.write_bytes(valid_path.read_bytes())

    # 4. Blank solid white screenshot
    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(blank_path)

    # 5. Empty 0-byte file
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")

    # 6. Corrupt non-image file
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_PNG_HEADER")

    # 7. Too small image
    small_path = tmp_path / "small.png"
    Image.new("RGB", (20, 20), color=(100, 150, 200)).save(small_path)

    return {
        "valid": valid_path,
        "near_dup": near_dup_path,
        "exact_dup": exact_dup_path,
        "blank": blank_path,
        "empty": empty_path,
        "corrupt": corrupt_path,
        "small": small_path,
    }


class TestImageValidator:
    """Verifies image integrity, blank detection, and corrupt rejection."""

    def test_valid_image(self, sample_images):
        v = ImageValidator()
        res = v.validate_file(sample_images["valid"])
        assert res.is_valid is True
        assert res.is_blank is False
        assert res.width == 200
        assert res.height == 200
        assert res.channels == 3
        assert res.std_dev > 10.0

    def test_blank_image_rejected(self, sample_images):
        v = ImageValidator(blank_std_threshold=3.0)
        res = v.validate_file(sample_images["blank"])
        assert res.is_valid is False
        assert res.is_blank is True
        assert res.error_reason == "BLANK_UNIFORM_SCREENSHOT"
        assert res.std_dev < 1.0

    def test_empty_zero_byte_file(self, sample_images):
        v = ImageValidator()
        res = v.validate_file(sample_images["empty"])
        assert res.is_valid is False
        assert res.error_reason == "EMPTY_FILE_ZERO_BYTES"

    def test_corrupt_file(self, sample_images):
        v = ImageValidator()
        res = v.validate_file(sample_images["corrupt"])
        assert res.is_valid is False
        assert "CORRUPT" in res.error_reason

    def test_too_small_image(self, sample_images):
        v = ImageValidator(min_width=100, min_height=100)
        res = v.validate_file(sample_images["small"])
        assert res.is_valid is False
        assert "TOO_SMALL" in res.error_reason

    def test_standardize_image(self, sample_images, tmp_path):
        out = tmp_path / "resized.png"
        ok, err = ImageValidator.standardize_image(
            sample_images["valid"],
            out,
            target_size=(224, 224),
        )
        assert ok is True
        assert err is None
        with Image.open(out) as img:
            assert img.size == (224, 224)


class TestURLNormalizer:
    """Verifies URL canonicalization, parameter stripping, and domain extraction."""

    def test_tracking_parameter_stripping(self):
        norm = URLNormalizer()
        raw = "https://example.com/login?utm_source=twitter&utm_medium=cpc&session=123&fbclid=abc"
        rec = norm.normalize(raw, label=1)
        assert rec is not None
        assert "utm_source" not in rec.canonical_url
        assert "fbclid" not in rec.canonical_url
        assert "session=123" in rec.canonical_url

    def test_default_ports_stripped(self):
        norm = URLNormalizer()
        rec_http = norm.normalize("http://example.com:80/path")
        rec_https = norm.normalize("https://example.com:443/path")
        assert rec_http.canonical_url == "http://example.com/path"
        assert rec_https.canonical_url == "https://example.com/path"

    def test_registrable_domain_extraction(self):
        norm = URLNormalizer()
        rec = norm.normalize("https://sub.login.paypal.co.uk/signin")
        assert rec.registrable_domain == "paypal.co.uk"
        assert rec.tld == "co.uk"
        assert rec.subdomain == "sub.login"

    def test_label_parsing(self):
        norm = URLNormalizer()
        assert norm.parse_label("benign") == 0
        assert norm.parse_label("legitimate") == 0
        assert norm.parse_label("not-phishing") == 0
        assert norm.parse_label(0) == 0

        assert norm.parse_label("phish") == 1
        assert norm.parse_label("phishing") == 1
        assert norm.parse_label(1) == 1


class TestDatasetDeduplicator:
    """Verifies exact URL deduplication, exact image deduplication, and pHash clustering."""

    def test_url_deduplication(self):
        norm = URLNormalizer()
        r1 = norm.normalize("https://example.com/login?a=1&utm_source=ad")
        r2 = norm.normalize("https://example.com/login?a=1&utm_medium=email")
        r3 = norm.normalize("https://other.org/")

        # r1 and r2 resolve to the exact same canonical URL after tracking stripping!
        recs, rep = DatasetDeduplicator.deduplicate_urls([r1, r2, r3])
        assert rep.total_input == 3
        assert rep.unique_retained == 2
        assert rep.exact_duplicates_removed == 1
        assert rep.unique_domains == 2

    def test_exact_and_near_duplicate_images(self, sample_images):
        dedup = DatasetDeduplicator(phash_distance_threshold=5)
        paths = [
            sample_images["valid"],
            sample_images["exact_dup"],
            sample_images["near_dup"],
        ]
        unique_paths, rep = dedup.deduplicate_images(paths)

        # exact_dup removed by sha256; near_dup removed by pHash threshold
        assert rep.total_images_scanned == 3
        assert rep.exact_duplicates_removed == 1
        assert rep.near_duplicates_removed == 1
        assert rep.unique_retained == 1
        assert unique_paths[0] == sample_images["valid"]
