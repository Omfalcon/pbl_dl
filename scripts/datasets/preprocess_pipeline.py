"""PhishVision AI — Master Dataset Preprocessing, Cleaning & Deduplication Pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.deduplicate import DatasetDeduplicator
from scripts.datasets.normalize_urls import NormalizedURLRecord, URLNormalizer
from scripts.datasets.validate_images import ImageValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.preprocess_pipeline")


def run_pipeline(
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = "data/processed",
    reports_dir: Path | str = "data/reports",
    tranco_limit: int = 50000,
) -> dict:
    raw_p = Path(raw_dir)
    proc_p = Path(processed_dir)
    rep_p = Path(reports_dir)
    proc_p.mkdir(parents=True, exist_ok=True)
    rep_p.mkdir(parents=True, exist_ok=True)
    (proc_p / "urls").mkdir(parents=True, exist_ok=True)
    (proc_p / "screenshots").mkdir(parents=True, exist_ok=True)

    normalizer = URLNormalizer()
    verified_records: list[NormalizedURLRecord] = []
    candidate_records: list[NormalizedURLRecord] = []
    rejected_records: list[NormalizedURLRecord] = []
    start_time = time.time()

    # -------------------------------------------------------------------------
    # 1. Ingest PhishTank Feeds (Verified Phishing)
    # -------------------------------------------------------------------------
    pt_files = list((raw_p / "urls").glob("phishtank_*.csv"))
    if pt_files:
        pt_path = sorted(pt_files)[-1]
        logger.info(f"Ingesting PhishTank dump: {pt_path.name}")
        with open(pt_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "")
                target = row.get("target", "")
                ts = row.get("verification_time") or row.get("submission_time")
                rec = normalizer.normalize(
                    raw_url=url,
                    label=1,
                    source="PhishTank",
                    target_brand=target,
                    timestamp=ts,
                )
                if rec:
                    if rec.verification_status == "VERIFIED":
                        verified_records.append(rec)
                    else:
                        rejected_records.append(rec)

    # -------------------------------------------------------------------------
    # 2. Ingest OpenPhish Feeds (Verified Phishing)
    # -------------------------------------------------------------------------
    op_files = list((raw_p / "urls").glob("openphish_*.txt"))
    if op_files:
        op_path = sorted(op_files)[-1]
        logger.info(f"Ingesting OpenPhish feed: {op_path.name}")
        with open(op_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    rec = normalizer.normalize(raw_url=line, label=1, source="OpenPhish")
                    if rec:
                        if rec.verification_status == "VERIFIED":
                            verified_records.append(rec)
                        else:
                            rejected_records.append(rec)

    # -------------------------------------------------------------------------
    # 3. Ingest PhreshPhish Feeds (Verified Ground Truth: Benign / Phish)
    # -------------------------------------------------------------------------
    pp_files = list((raw_p / "urls").glob("phreshphish_*.parquet"))
    if pp_files:
        for pp_p in pp_files:
            logger.info(f"Ingesting PhreshPhish parquet: {pp_p.name}")
            try:
                df_pp = pd.read_parquet(pp_p)
                for _, row in df_pp.iterrows():
                    raw_lbl = row.get("label")
                    rec = normalizer.normalize(
                        raw_url=row.get("url", ""),
                        label=raw_lbl,
                        source="PhreshPhish",
                        target_brand=row.get("target"),
                        timestamp=str(row.get("date", "")),
                    )
                    if rec:
                        if rec.verification_status == "VERIFIED":
                            verified_records.append(rec)
                        else:
                            rejected_records.append(rec)
            except Exception as e:
                logger.warning(f"Error reading PhreshPhish parquet: {e}")

    # -------------------------------------------------------------------------
    # 4. Ingest Tranco Benign Research Domains as Unverified Candidates
    #    (Crucial: Never automatically classify every Tranco domain as verified legitimate)
    # -------------------------------------------------------------------------
    tr_files = list((raw_p / "urls").glob("tranco_top_*.csv"))
    if not tr_files:
        tr_files = list((raw_p / "urls").glob("tranco_top1m.csv"))
    if tr_files:
        tr_path = sorted(tr_files)[0]
        logger.info(f"Ingesting Tranco top domains as unverified candidates: {tr_path.name}")
        count = 0
        with open(tr_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    domain = row[1].strip()
                    rec = normalizer.normalize(
                        raw_url=f"https://{domain}",
                        label="candidate",  # Explicit candidate label
                        source="Tranco",
                    )
                    if rec:
                        candidate_records.append(rec)
                        count += 1
                        if tranco_limit and count >= tranco_limit:
                            break

    logger.info(
        f"Raw URLs Ingested: {len(verified_records)} verified, "
        f"{len(candidate_records)} unverified candidates, {len(rejected_records)} rejected"
    )

    # -------------------------------------------------------------------------
    # 5. Deduplicate Verified Corpus & Candidate Pool
    # -------------------------------------------------------------------------
    dedup = DatasetDeduplicator()
    unique_verified, ver_report = dedup.deduplicate_urls(verified_records)
    unique_candidates, cand_report = dedup.deduplicate_urls(candidate_records)

    # Save Verified Corpus
    df_verified = pd.DataFrame([asdict(r) for r in unique_verified])
    out_ver_parquet = proc_p / "urls" / "corpus_normalized.parquet"
    out_ver_csv = proc_p / "urls" / "corpus_normalized.csv"
    df_verified.to_parquet(out_ver_parquet, index=False)
    df_verified.to_csv(out_ver_csv, index=False)
    logger.info(f"Saved {len(df_verified)} verified URLs to {out_ver_parquet}")

    # Save Unverified Candidate Pool (Tranco) separately
    df_candidates = pd.DataFrame([asdict(r) for r in unique_candidates])
    out_cand_parquet = proc_p / "urls" / "tranco_candidate_pool.parquet"
    out_cand_csv = proc_p / "urls" / "tranco_candidate_pool.csv"
    df_candidates.to_parquet(out_cand_parquet, index=False)
    df_candidates.to_csv(out_cand_csv, index=False)
    logger.info(f"Saved {len(df_candidates)} unverified candidates to {out_cand_parquet}")

    # Label distribution of verified corpus
    class_dist = df_verified["label"].value_counts().to_dict()
    label_map = {0: "Legitimate", 1: "Phishing"}
    readable_dist = {label_map.get(k, str(k)): v for k, v in class_dist.items()}

    # -------------------------------------------------------------------------
    # 6. Validate Existing Screenshot Images
    # -------------------------------------------------------------------------
    validator = ImageValidator()
    screenshot_dirs = [
        raw_p / "screenshots",
        PROJECT_ROOT / "data" / "scans" / "screenshots",
    ]

    all_image_paths: list[Path] = []
    for s_dir in screenshot_dirs:
        if s_dir.exists():
            all_image_paths.extend([p for p in s_dir.rglob("*.png") if p.is_file()])

    valid_images: list[Path] = []
    corrupt_images = 0
    blank_images = 0

    for img_p in all_image_paths:
        v_res = validator.validate_file(img_p)
        if v_res.is_valid:
            valid_images.append(img_p)
        elif v_res.is_blank:
            blank_images += 1
        else:
            corrupt_images += 1

    img_unique_paths, img_dedup_report = dedup.deduplicate_images(valid_images)

    # -------------------------------------------------------------------------
    # 7. Compile Detailed Preprocessing Report
    # -------------------------------------------------------------------------
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - start_time, 2),
        "urls": {
            "verified_raw_ingested": ver_report.total_input,
            "verified_unique_retained": ver_report.unique_retained,
            "verified_duplicates_removed": ver_report.exact_duplicates_removed,
            "verified_unique_registrable_domains": ver_report.unique_domains,
            "verified_class_distribution": readable_dist,
            "unverified_candidate_pool_size": cand_report.unique_retained,
            "rejected_unknown_count": len(rejected_records),
            "output_verified_parquet": str(out_ver_parquet),
            "output_candidates_parquet": str(out_cand_parquet),
        },
        "images": {
            "total_scanned": len(all_image_paths),
            "valid_retained": len(img_unique_paths),
            "corrupt_dropped": corrupt_images,
            "blank_dropped": blank_images,
            "exact_duplicate_images_removed": img_dedup_report.exact_duplicates_removed,
            "near_duplicate_images_removed": img_dedup_report.near_duplicates_removed,
        },
    }

    report_file = rep_p / "preprocessing_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Preprocessing report written to {report_file}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Preprocessing Pipeline")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="data/reports")
    parser.add_argument("--tranco-limit", type=int, default=50000)
    args = parser.parse_args()

    rep = run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        tranco_limit=args.tranco_limit,
    )
    print("\n" + "=" * 75)
    print("📊 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"• Verified Unique Cleaned URLs    : {rep['urls']['verified_unique_retained']}")
    print(f"• Verified Unique Domains         : {rep['urls']['verified_unique_registrable_domains']}")
    print(f"• Verified Class Distribution     : {rep['urls']['verified_class_distribution']}")
    print(f"• Unverified Tranco Candidates    : {rep['urls']['unverified_candidate_pool_size']}")
    print(f"• Rejected/Unknown Labels Dropped : {rep['urls']['rejected_unknown_count']}")
    print(f"• Images Valid & Retained         : {rep['images']['valid_retained']}/{rep['images']['total_scanned']}")
    print(f"• Processing Time                 : {rep['duration_seconds']}s")
    print("=" * 75)
