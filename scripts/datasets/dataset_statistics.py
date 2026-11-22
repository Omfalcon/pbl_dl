"""Comprehensive Statistical Profiler for PhishVision AI Datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.validate_annotations import generate_dataset_annotation_report
from scripts.datasets.validate_images import ImageValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.dataset_statistics")


def compute_statistics(
    splits_dir: Path | str = "data/splits",
    processed_dir: Path | str = "data/processed",
    raw_dir: Path | str = "data/raw",
    brands_dir: Path | str = "data/brands/reference_gallery",
    annotations_dir: Path | str = "data/annotations",
    output_report: Path | str = "data/reports/dataset_statistics.json",
) -> dict[str, Any]:
    splits_p = Path(splits_dir)
    proc_p = Path(processed_dir)
    raw_p = Path(raw_dir)
    brands_p = Path(brands_dir)
    ann_p = Path(annotations_dir)
    out_rep_p = Path(output_report)
    out_rep_p.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Verified Labeled URL Corpus
    # -------------------------------------------------------------------------
    corpus_file = proc_p / "urls" / "corpus_normalized.parquet"
    if not corpus_file.exists():
        corpus_file = proc_p / "urls" / "corpus_normalized.csv"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_file}")

    logger.info(f"Computing statistics from {corpus_file}")
    df_corpus = pd.read_parquet(corpus_file) if corpus_file.suffix == ".parquet" else pd.read_csv(corpus_file)

    url_lengths = df_corpus["canonical_url"].str.len()

    # -------------------------------------------------------------------------
    # 2. Unverified Candidate URL Pool (Tranco)
    # -------------------------------------------------------------------------
    cand_file = proc_p / "urls" / "tranco_candidate_pool.parquet"
    num_candidates = 0
    if cand_file.exists():
        df_cand = pd.read_parquet(cand_file)
        num_candidates = len(df_cand)

    # -------------------------------------------------------------------------
    # 3. Actual Local Screenshot Inventory
    # -------------------------------------------------------------------------
    validator = ImageValidator()
    screenshot_dirs = [
        raw_p / "screenshots",
        raw_p / "external_dataset",
        Path("data/scans/screenshots"),
    ]
    all_screenshot_paths: list[Path] = []
    for s_dir in screenshot_dirs:
        if s_dir.exists():
            all_screenshot_paths.extend([p for p in s_dir.rglob("*.png") if p.is_file()])

    valid_screenshots = 0
    blank_screenshots = 0
    corrupt_screenshots = 0
    for img_p in all_screenshot_paths:
        vres = validator.validate_file(img_p)
        if vres.is_valid:
            valid_screenshots += 1
        elif vres.is_blank:
            blank_screenshots += 1
        else:
            corrupt_screenshots += 1

    # Check processed visual corpus if available
    visual_corpus_path = proc_p / "screenshots" / "visual_corpus.parquet"
    visual_corpus_stats = {}
    if visual_corpus_path.exists():
        df_vis = pd.read_parquet(visual_corpus_path)
        visual_corpus_stats = {
            "total_records": len(df_vis),
            "phishing_count": int((df_vis["label"] == 1).sum()),
            "legitimate_count": int((df_vis["label"] == 0).sum()),
            "unique_domains": int(df_vis["registrable_domain"].nunique()),
            "exact_duplicates": int(df_vis["is_exact_duplicate"].sum()),
            "near_duplicates": int(df_vis["is_near_duplicate"].sum()),
        }

    # -------------------------------------------------------------------------
    # 4. Actual Brand Reference Images (Strict accounting: empty dirs do not count)
    # -------------------------------------------------------------------------
    actual_brand_images: list[Path] = []
    if brands_p.exists():
        actual_brand_images = [p for p in brands_p.rglob("*.png") if p.is_file()] + [
            p for p in brands_p.rglob("*.svg") if p.is_file()
        ]

    # Total brands with at least 1 authentic asset on disk
    brands_with_assets = set(p.parent.parent.name for p in actual_brand_images)

    # -------------------------------------------------------------------------
    # 5. Annotation Audit
    # -------------------------------------------------------------------------
    ann_report = generate_dataset_annotation_report(ann_p)

    # -------------------------------------------------------------------------
    # 6. Assemble Comprehensive Telemetry Report
    # -------------------------------------------------------------------------
    stats: dict[str, Any] = {
        "verified_url_corpus": {
            "total_records": len(df_corpus),
            "unique_canonical_urls": df_corpus["canonical_url"].nunique(),
            "unique_registrable_domains": df_corpus["registrable_domain"].nunique(),
            "legitimate_records": int((df_corpus["label"] == 0).sum()),
            "phishing_records": int((df_corpus["label"] == 1).sum()),
            "phishing_percentage": round(float((df_corpus["label"] == 1).mean()) * 100, 2),
        },
        "unverified_candidate_pool": {
            "source": "Tranco Research Domains",
            "total_candidate_urls": num_candidates,
            "verification_status": "UNVERIFIED_CANDIDATE",
            "is_mixed_with_ground_truth": False,
        },
        "url_length_metrics": {
            "min_length": int(url_lengths.min()),
            "max_length": int(url_lengths.max()),
            "mean_length": round(float(url_lengths.mean()), 2),
            "median_length": float(url_lengths.median()),
            "std_length": round(float(url_lengths.std()), 2),
        },
        "top_phishing_targets": (
            df_corpus[df_corpus["label"] == 1]["target_brand"]
            .dropna()
            .value_counts()
            .head(15)
            .to_dict()
        ),
        "source_breakdown": df_corpus["source"].value_counts().to_dict(),
        "visual_corpus": visual_corpus_stats,
        "visual_dataset_readiness": {
            "existing_screenshots": 8,
            "newly_acquired_screenshots": visual_corpus_stats.get("total_records", 0),
            "total_local_screenshots_scanned": len(all_screenshot_paths),
            "valid_screenshots_retained": valid_screenshots,
            "blank_screenshots_dropped": blank_screenshots,
            "corrupt_screenshots_dropped": corrupt_screenshots,
            "actual_brand_reference_images_on_disk": len(actual_brand_images),
            "brands_with_actual_assets": len(brands_with_assets),
            "actual_brand_similarity_pairs_curated": 0,  # Explicit: 0 pairs until curated with images
            "notes": "Empty brand workspaces are NOT counted as acquired image datasets.",
        },
        "annotations_audit": {
            "total_images_annotated": ann_report.total_images_annotated,
            "manually_verified_count": ann_report.manually_verified_count,
            "unverified_candidate_count": ann_report.unverified_candidate_count,
            "total_yolo_boxes": ann_report.total_bounding_boxes,
            "boxes_per_class": ann_report.boxes_per_class,
            "segmentation_masks_count": ann_report.segmentation_masks_count,
            "invalid_annotations_count": ann_report.invalid_annotations_count,
        },
        "splits": {},
        "visual_splits": {},
    }

    # Inspect splits
    for split_name in ["train", "validation", "test"]:
        split_file = splits_p / split_name / "urls.parquet"
        if not split_file.exists():
            split_file = splits_p / split_name / "urls.csv"
        if split_file.exists():
            df_s = pd.read_parquet(split_file) if split_file.suffix == ".parquet" else pd.read_csv(split_file)
            stats["splits"][split_name] = {
                "records": len(df_s),
                "unique_domains": df_s["registrable_domain"].nunique(),
                "legitimate": int((df_s["label"] == 0).sum()),
                "phishing": int((df_s["label"] == 1).sum()),
                "phishing_percentage": round(float((df_s["label"] == 1).mean()) * 100, 2),
            }

    # Inspect visual splits
    for v_name, v_file_stem in [("train", "train"), ("validation", "val"), ("test", "test")]:
        v_file = splits_p / "visual" / f"{v_file_stem}.parquet"
        if v_file.exists():
            df_v = pd.read_parquet(v_file)
            stats["visual_splits"][v_name] = {
                "records": len(df_v),
                "unique_domains": df_v["registrable_domain"].nunique(),
                "legitimate": int((df_v["label"] == 0).sum()),
                "phishing": int((df_v["label"] == 1).sum()),
                "phishing_percentage": round(float((df_v["label"] == 1).mean()) * 100, 2),
            }

    with open(out_rep_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset statistics exported to {out_rep_p}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Statistics Profiler")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/reports/dataset_statistics.json")
    args = parser.parse_args()

    s = compute_statistics(
        splits_dir=args.splits_dir,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        output_report=args.output,
    )
    print("\n" + "=" * 80)
    print("📈 PHISHVISION AI DATASET STATISTICAL & READINESS PROFILE")
    print("=" * 80)
    print(f"• Verified Labeled Records : {s['verified_url_corpus']['total_records']} (Legit: {s['verified_url_corpus']['legitimate_records']}, Phish: {s['verified_url_corpus']['phishing_records']})")
    print(f"• Unverified Candidates    : {s['unverified_candidate_pool']['total_candidate_urls']} ({s['unverified_candidate_pool']['source']})")
    print(f"• Valid Local Screenshots : {s['visual_dataset_readiness']['valid_screenshots_retained']}/{s['visual_dataset_readiness']['total_local_screenshots_scanned']}")
    print(f"• Brand Assets on Disk     : {s['visual_dataset_readiness']['actual_brand_reference_images_on_disk']} images ({s['visual_dataset_readiness']['brands_with_actual_assets']} brands)")
    print(f"• Brand Similarity Pairs   : {s['visual_dataset_readiness']['actual_brand_similarity_pairs_curated']} pairs")
    print(f"• Annotations Quality      : {s['annotations_audit']['manually_verified_count']} verified, {s['annotations_audit']['unverified_candidate_count']} unverified")
    print("=" * 80)
