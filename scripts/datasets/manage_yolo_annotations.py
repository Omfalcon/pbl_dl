"""CLI Script for YOLO Annotation Manifest Initialization, Validation & Statistics (Phase 7.1)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is first in sys.path to avoid site-packages shadowing
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.vision.yolo_annotations import (
    YOLO_DETECTION_CLASSES,
    YOLOAnnotationManifestManager,
    YOLOAnnotationValidator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.manage_yolo_annotations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage YOLO Annotation Pipeline & Manifest")
    parser.add_argument("--action", choices=["init-manifest", "validate", "stats"], default="init-manifest",
                        help="Action to perform: init-manifest, validate, or stats")
    parser.add_argument("--manifest-csv", default="data/annotations/detection/annotation_manifest.csv",
                        help="Path to manifest CSV")
    parser.add_argument("--manifest-json", default="data/annotations/detection/annotation_manifest.json",
                        help="Path to manifest JSON")
    parser.add_argument("--corpus-parquet", default="data/processed/screenshots/visual_corpus.parquet",
                        help="Path to visual corpus parquet")
    parser.add_argument("--output-stats", default="data/reports/yolo_annotation_audit.json",
                        help="Path to write statistics report")
    args = parser.parse_args()

    mgr = YOLOAnnotationManifestManager(
        manifest_csv_path=args.manifest_csv,
        manifest_json_path=args.manifest_json,
    )

    if args.action == "init-manifest":
        logger.info(f"Initializing annotation manifest from corpus: {args.corpus_parquet}")
        manifest_df = mgr.build_initial_manifest_from_corpus(
            corpus_parquet=args.corpus_parquet,
            save=True,
        )
        logger.info(f"Successfully generated manifest with {len(manifest_df)} total records.")
        stats = mgr.compute_dataset_statistics(manifest_df)
        logger.info(f"Dataset summary: {json.dumps(stats, indent=2)}")

        # Save stats
        out_stats_p = Path(args.output_stats)
        out_stats_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_stats_p, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset audit statistics to {out_stats_p}")

    elif args.action == "validate":
        validator = YOLOAnnotationValidator()
        det_dir = Path("data/annotations/detection")
        txt_files = list(det_dir.glob("*.txt"))
        logger.info(f"Validating {len(txt_files)} YOLO annotation files in {det_dir}...")
        valid_count = 0
        invalid_count = 0
        empty_count = 0
        for tf in txt_files:
            res = validator.validate_yolo_file(tf)
            if res.is_valid:
                valid_count += 1
                if res.is_empty:
                    empty_count += 1
            else:
                invalid_count += 1
                logger.warning(f"Invalid annotation {tf.name}: {res.errors}")
        logger.info(f"Validation complete: {valid_count} valid ({empty_count} empty), {invalid_count} invalid.")

    elif args.action == "stats":
        stats = mgr.compute_dataset_statistics()
        logger.info(f"Dataset statistics:\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
