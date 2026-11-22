"""Deduplication Engine for URLs, Exact Images & Perceptual Hashing (pHash)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import imagehash
from PIL import Image

from .normalize_urls import NormalizedURLRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.deduplicate")


@dataclass
class URLDeduplicationReport:
    """Statistics from deduplicating a corpus of normalized URLs."""
    total_input: int = 0
    unique_retained: int = 0
    exact_duplicates_removed: int = 0
    unique_domains: int = 0


@dataclass
class ImageDeduplicationReport:
    """Statistics from deduplicating screenshot image files."""
    total_images_scanned: int = 0
    unique_retained: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    duplicate_clusters: dict[str, list[str]] = field(default_factory=dict)


class DatasetDeduplicator:
    """Deduplicates URLs and screenshot images with cryptographic and perceptual algorithms."""

    def __init__(self, phash_distance_threshold: int = 4) -> None:
        self.phash_distance_threshold = phash_distance_threshold

    @staticmethod
    def deduplicate_urls(
        records: Iterable[NormalizedURLRecord],
    ) -> tuple[list[NormalizedURLRecord], URLDeduplicationReport]:
        """Drop duplicate URLs using SHA-256 canonical hash matching."""
        seen_hashes: set[str] = set()
        seen_domains: set[str] = set()
        unique_records: list[NormalizedURLRecord] = []
        total = 0
        duplicates = 0

        for r in records:
            total += 1
            if r.url_hash_sha256 in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(r.url_hash_sha256)
            if r.registrable_domain:
                seen_domains.add(r.registrable_domain)
            unique_records.append(r)

        report = URLDeduplicationReport(
            total_input=total,
            unique_retained=len(unique_records),
            exact_duplicates_removed=duplicates,
            unique_domains=len(seen_domains),
        )
        logger.info(
            f"URL Deduplication: {report.unique_retained}/{report.total_input} retained "
            f"({report.exact_duplicates_removed} duplicates removed across {report.unique_domains} domains)"
        )
        return unique_records, report

    @staticmethod
    def compute_image_sha256(file_path: Path | str) -> str:
        """Compute cryptographic SHA-256 of image bytes."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_phash(file_path: Path | str) -> imagehash.ImageHash:
        """Compute 64-bit perceptual hash (DCT-based) for an image."""
        with Image.open(file_path) as img:
            return imagehash.phash(img)

    def deduplicate_images(
        self,
        image_paths: list[Path | str],
    ) -> tuple[list[Path], ImageDeduplicationReport]:
        """Eliminate exact and near-duplicate screenshots using pHash Hamming distance clustering."""
        report = ImageDeduplicationReport(total_images_scanned=len(image_paths))
        if not image_paths:
            return [], report

        # Step 1: Exact byte deduplication via SHA-256
        seen_sha256: dict[str, Path] = {}
        exact_unique_paths: list[Path] = []

        for p in image_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            file_sha = self.compute_image_sha256(path_obj)
            if file_sha in seen_sha256:
                report.exact_duplicates_removed += 1
                orig = seen_sha256[file_sha]
                report.duplicate_clusters.setdefault(str(orig), []).append(str(path_obj))
            else:
                seen_sha256[file_sha] = path_obj
                exact_unique_paths.append(path_obj)

        # Step 2: Perceptual near-duplicate clustering via pHash
        retained_paths: list[Path] = []
        retained_hashes: list[imagehash.ImageHash] = []

        for p in exact_unique_paths:
            try:
                curr_phash = self.compute_phash(p)
            except Exception as e:
                logger.warning(f"Could not compute pHash for {p}: {e}")
                continue

            # Compare against existing retained images
            is_near_dup = False
            for idx, existing_hash in enumerate(retained_hashes):
                distance = curr_phash - existing_hash  # Hamming distance
                if distance <= self.phash_distance_threshold:
                    is_near_dup = True
                    report.near_duplicates_removed += 1
                    rep_path = str(retained_paths[idx])
                    report.duplicate_clusters.setdefault(rep_path, []).append(str(p))
                    break

            if not is_near_dup:
                retained_paths.append(p)
                retained_hashes.append(curr_phash)

        report.unique_retained = len(retained_paths)
        logger.info(
            f"Image Deduplication: {report.unique_retained}/{report.total_images_scanned} retained "
            f"({report.exact_duplicates_removed} exact duplicates, {report.near_duplicates_removed} near-duplicates removed)"
        )
        return retained_paths, report
