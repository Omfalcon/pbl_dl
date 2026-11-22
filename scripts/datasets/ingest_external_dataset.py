"""Secure Extraction, Validation, and Pipeline Integration for External Visual Dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import zipfile

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.deduplicate import DatasetDeduplicator
from scripts.datasets.generate_splits import DomainAwareSplitter
from scripts.datasets.normalize_urls import URLNormalizer
from scripts.datasets.validate_images import ImageValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.ingest_external")


def safe_extract_zip(
    zip_path: Path | str,
    target_dir: Path | str,
) -> tuple[int, list[str]]:
    """Safely extract archive with Zip-Slip path traversal protection and forbidden extension blocking."""
    z_path = Path(zip_path)
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not z_path.exists():
        raise FileNotFoundError(f"ZIP archive not found: {z_path}")

    extracted_count = 0
    skipped: list[str] = []

    FORBIDDEN_EXTS = {".exe", ".bat", ".cmd", ".sh", ".dll", ".so", ".vbs", ".ps1", ".bin"}

    with zipfile.ZipFile(z_path, "r") as z:
        for member in z.infolist():
            filename = member.filename

            # Skip directory entries
            if member.is_dir() or filename.endswith("/"):
                continue

            # Path traversal check
            target_path = (out_dir / filename).resolve()
            if not str(target_path).startswith(str(out_dir)):
                logger.warning(f"Zip-Slip attempt detected and blocked: {filename}")
                skipped.append(filename)
                continue

            # Disallow executable / script extensions
            if target_path.suffix.lower() in FORBIDDEN_EXTS:
                logger.warning(f"Forbidden extension blocked: {filename}")
                skipped.append(filename)
                continue

            # Extract safely
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

            extracted_count += 1

    logger.info(f"Safely extracted {extracted_count} files into {out_dir} (Skipped/Blocked: {len(skipped)})")
    return extracted_count, skipped


def validate_and_catalog_dataset(
    extracted_dir: Path | str = "data/raw/external_dataset",
    output_dir: Path | str = "data/processed/screenshots",
) -> dict:
    """Validate extracted screenshots, cross-reference metadata/labels, and create catalog."""
    ext_dir = Path(extracted_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify master CSV
    csv_candidates = list(ext_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No metadata CSV found in {ext_dir}")

    master_csv = csv_candidates[0]
    logger.info(f"Loading master visual dataset CSV: {master_csv}")

    img_validator = ImageValidator(min_width=100, min_height=100, blank_std_threshold=3.0)
    url_normalizer = URLNormalizer()

    records = []
    sha256_seen: dict[str, str] = {}
    phashes_seen: list[tuple[str, str]] = []  # (phash_str, file_rel_path)

    stats = {
        "total_catalogued": 0,
        "valid_images": 0,
        "corrupt_images": 0,
        "blank_images": 0,
        "missing_files": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "class_0_legitimate": 0,
        "class_1_phishing": 0,
        "rejected_unknown_label": 0,
        "dimensions": {},
    }

    with open(master_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_catalogued"] += 1
            raw_url = row.get("url", "").strip()
            rel_screenshot = row.get("screenshot", "").strip()
            raw_lbl = row.get("label", "").strip()

            # Screenshot file path (relative to CSV directory or extracted_dir)
            screenshot_path = master_csv.parent / rel_screenshot
            if not screenshot_path.exists():
                screenshot_path = ext_dir / rel_screenshot
            if not screenshot_path.exists():
                # Try finding basename
                matches = list(ext_dir.rglob(Path(rel_screenshot).name))
                if matches:
                    screenshot_path = matches[0]

            if not screenshot_path.exists():
                stats["missing_files"] += 1
                continue

            # Validate Image Integrity
            img_res = img_validator.validate_file(screenshot_path)
            if not img_res.is_valid:
                if img_res.is_blank:
                    stats["blank_images"] += 1
                else:
                    stats["corrupt_images"] += 1
                continue

            stats["valid_images"] += 1
            dim_key = f"{img_res.width}x{img_res.height}"
            stats["dimensions"][dim_key] = stats["dimensions"].get(dim_key, 0) + 1

            # Parse and Validate Label
            lbl_int, ver_status = url_normalizer.parse_label(raw_lbl)
            if ver_status == "REJECTED_UNKNOWN" or lbl_int not in (0, 1):
                stats["rejected_unknown_label"] += 1
                continue

            if lbl_int == 0:
                stats["class_0_legitimate"] += 1
            else:
                stats["class_1_phishing"] += 1

            # Check cross-reference metadata.txt if present
            meta_txt = screenshot_path.parent / "metadata.txt"
            cross_verified_label = True
            if meta_txt.exists():
                try:
                    txt_content = meta_txt.read_text(encoding="utf-8", errors="ignore")
                    for line in txt_content.splitlines():
                        if line.startswith("Label:"):
                            txt_lbl = line.split(":", 1)[1].strip()
                            if txt_lbl != raw_lbl:
                                cross_verified_label = False
                except Exception:
                    pass

            # Compute SHA-256 & Perceptual Hash
            file_bytes = screenshot_path.read_bytes()
            sha256_hash = hashlib.sha256(file_bytes).hexdigest()

            is_exact_dup = False
            if sha256_hash in sha256_seen:
                stats["exact_duplicates"] += 1
                is_exact_dup = True
            else:
                sha256_seen[sha256_hash] = str(screenshot_path)

            phash_obj = None
            try:
                phash_obj = DatasetDeduplicator.compute_phash(screenshot_path)
            except Exception:
                pass

            is_near_dup = False
            if phash_obj is not None:
                for existing_phash, _ in phashes_seen:
                    if (phash_obj - existing_phash) <= 4:
                        is_near_dup = True
                        stats["near_duplicates"] += 1
                        break
                if not is_exact_dup:
                    phashes_seen.append((phash_obj, str(screenshot_path)))

            phash_str = str(phash_obj) if phash_obj is not None else None

            # Normalize URL & extract registrable domain
            norm_rec = url_normalizer.normalize(raw_url, label=lbl_int, source="external_visual_dataset")
            reg_dom = norm_rec.registrable_domain if norm_rec else f"__domain_{stats['total_catalogued']}__"
            canon_url = norm_rec.canonical_url if norm_rec else raw_url

            records.append({
                "sample_id": f"ext_vis_{len(records):05d}",
                "original_url": raw_url,
                "canonical_url": canon_url,
                "registrable_domain": reg_dom,
                "screenshot_relative_path": str(screenshot_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
                "screenshot_absolute_path": str(screenshot_path.resolve()),
                "raw_label": raw_lbl,
                "label": lbl_int,
                "verification_status": "VERIFIED",
                "label_cross_verified": cross_verified_label,
                "source": "DatasetV1_external",
                "width": img_res.width,
                "height": img_res.height,
                "channels": img_res.channels,
                "std_dev": round(img_res.std_dev, 2),
                "sha256": sha256_hash,
                "phash": phash_str,
                "is_exact_duplicate": is_exact_dup,
                "is_near_duplicate": is_near_dup,
            })

    # Save to Parquet and CSV
    df = pd.DataFrame(records)

    # Compute disjoint connected components across Domain, SHA-256, and pHash (Hamming <= 4)
    # to guarantee ZERO domain, exact duplicate, and perceptual duplicate leakage across splits
    parent = {i: i for i in range(len(df))}

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Union by domain
    dom_map = defaultdict(list)
    for idx, dom in enumerate(df["registrable_domain"]):
        dom_map[dom].append(idx)
    for indices in dom_map.values():
        for idx in indices[1:]:
            union(indices[0], idx)

    # Union by exact SHA-256
    sha_map = defaultdict(list)
    for idx, sha in enumerate(df["sha256"]):
        sha_map[sha].append(idx)
    for indices in sha_map.values():
        for idx in indices[1:]:
            union(indices[0], idx)

    # Union by perceptual hash (Hamming distance <= 4)
    phash_objs = [imagehash.hex_to_hash(h) if (pd.notna(h) and h) else None for h in df["phash"]]
    for i in range(len(df)):
        h1 = phash_objs[i]
        if h1 is None:
            continue
        for j in range(i + 1, len(df)):
            h2 = phash_objs[j]
            if h2 is None:
                continue
            if (h1 - h2) <= 4:
                union(i, j)

    df["visual_cluster_id"] = [f"cluster_{find(i)}" for i in range(len(df))]

    parquet_path = out_dir / "visual_corpus.parquet"
    csv_path = out_dir / "visual_corpus.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    stats["total_records_saved"] = len(df)
    stats["unique_domains_count"] = df["registrable_domain"].nunique() if not df.empty else 0
    stats["unique_visual_clusters_count"] = df["visual_cluster_id"].nunique() if not df.empty else 0

    report_path = out_dir / "visual_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Visual dataset catalogued successfully: {len(df)} records "
        f"({stats['class_1_phishing']} Phish, {stats['class_0_legitimate']} Legit) "
        f"across {stats['unique_domains_count']} domains and {stats['unique_visual_clusters_count']} visual clusters. "
        f"Saved to {parquet_path}"
    )

    # Perform class-aware cluster-level split for Visual Intelligence models
    if len(df) > 0:
        splitter = DomainAwareSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        visual_splits_dir = Path("data/splits/visual")
        visual_splits_dir.mkdir(parents=True, exist_ok=True)

        df_tr, df_val, df_test, split_rep = splitter.split_dataframe(
            df, domain_column="visual_cluster_id", label_column="label"
        )

        df_tr.to_parquet(visual_splits_dir / "train.parquet", index=False)
        df_tr.to_csv(visual_splits_dir / "train.csv", index=False)

        df_val.to_parquet(visual_splits_dir / "val.parquet", index=False)
        df_val.to_csv(visual_splits_dir / "val.csv", index=False)

        df_test.to_parquet(visual_splits_dir / "test.parquet", index=False)
        df_test.to_csv(visual_splits_dir / "test.csv", index=False)

        # Check visual duplication metrics across splits
        sha_tr = set(df_tr["sha256"])
        sha_val = set(df_val["sha256"])
        sha_test = set(df_test["sha256"])
        sha_leaks = len(sha_tr.intersection(sha_val)) + len(sha_tr.intersection(sha_test)) + len(sha_val.intersection(sha_test))

        with open(visual_splits_dir / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": len(df),
                "train_samples": len(df_tr),
                "val_samples": len(df_val),
                "test_samples": len(df_test),
                "domain_leakage_detected": split_rep.domain_leakage_detected,
                "overlapping_domains_count": split_rep.overlapping_domains_count,
                "overlapping_sha256_count": sha_leaks,
                "record_conservation_verified": split_rep.record_conservation_verified,
            }, f, indent=2)

        logger.info(
            f"Visual splits created: Train={len(df_tr)}, Val={len(df_val)}, Test={len(df_test)} "
            f"(Domain Leakage={split_rep.overlapping_domains_count}, SHA Leakage={sha_leaks})"
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, validate and split external visual dataset.")
    parser.add_argument("--zip-path", default="data/raw/dataset.zip", help="Path to raw zip file")
    parser.add_argument("--extract-dir", default="data/raw/external_dataset", help="Target extraction directory")
    parser.add_argument("--output-dir", default="data/processed/screenshots", help="Processed visual corpus output")
    args = parser.parse_args()

    zip_p = Path(args.zip_path)
    extract_p = Path(args.extract_dir)

    if not (extract_p / "DatasetV1").exists():
        logger.info(f"Extracting {zip_p} into {extract_p}...")
        safe_extract_zip(zip_p, extract_p)
    else:
        logger.info(f"Directory {extract_p / 'DatasetV1'} already extracted; proceeding directly to validation.")

    validate_and_catalog_dataset(extract_p, args.output_dir)


if __name__ == "__main__":
    main()
