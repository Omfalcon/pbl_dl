"""Comprehensive Statistical Profiler for PhishVision AI Datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.dataset_statistics")


def compute_statistics(
    splits_dir: Path | str = "data/splits",
    processed_dir: Path | str = "data/processed",
    output_report: Path | str = "data/reports/dataset_statistics.json",
) -> dict[str, Any]:
    splits_p = Path(splits_dir)
    proc_p = Path(processed_dir)
    out_rep_p = Path(output_report)
    out_rep_p.parent.mkdir(parents=True, exist_ok=True)

    corpus_file = proc_p / "urls" / "corpus_normalized.parquet"
    if not corpus_file.exists():
        corpus_file = proc_p / "urls" / "corpus_normalized.csv"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_file}")

    logger.info(f"Computing statistics from {corpus_file}")
    df_corpus = pd.read_parquet(corpus_file) if corpus_file.suffix == ".parquet" else pd.read_csv(corpus_file)

    url_lengths = df_corpus["canonical_url"].str.len()
    stats: dict[str, Any] = {
        "corpus_overview": {
            "total_records": len(df_corpus),
            "unique_canonical_urls": df_corpus["canonical_url"].nunique(),
            "unique_registrable_domains": df_corpus["registrable_domain"].nunique(),
            "legitimate_records": int((df_corpus["label"] == 0).sum()),
            "phishing_records": int((df_corpus["label"] == 1).sum()),
            "phishing_percentage": round(float((df_corpus["label"] == 1).mean()) * 100, 2),
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
        "splits": {},
    }

    # Inspect splits if generated
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

    with open(out_rep_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset statistics exported to {out_rep_p}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Statistics Profiler")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/reports/dataset_statistics.json")
    args = parser.parse_args()

    s = compute_statistics(args.splits_dir, args.processed_dir, args.output)
    print("\n" + "=" * 75)
    print("📈 PHISHVISION AI DATASET STATISTICAL PROFILE")
    print("=" * 75)
    print(f"• Total Validated Records : {s['corpus_overview']['total_records']}")
    print(f"• Unique Domains Covered  : {s['corpus_overview']['unique_registrable_domains']}")
    print(f"• Class Breakdown         : {s['corpus_overview']['legitimate_records']} Legit vs {s['corpus_overview']['phishing_records']} Phish ({s['corpus_overview']['phishing_percentage']}%)")
    print(f"• Average URL Length      : {s['url_length_metrics']['mean_length']} chars (Median: {s['url_length_metrics']['median_length']})")
    print(f"• Top Phishing Targets    : {list(s['top_phishing_targets'].keys())[:5]}")
    print("=" * 75)
