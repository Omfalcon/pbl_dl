"""CLI runner for Full Validation Multi-Region Aggregation Experiment (Phase 6.1.3)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is first in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.vision.aggregation_experiment import run_full_validation_aggregation_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishvision.run_aggregation")


def main() -> None:
    parser = argparse.ArgumentParser(description="M-02 Multi-Region Aggregation Experiment on Full Validation Split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/resnet18/best_model.pt",
        help="Path to trained ResNet-18 checkpoint",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="data/splits/visual/val.parquet",
        help="Path to validation parquet split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports/resnet18_error_analysis/aggregation_experiment",
        help="Output directory for aggregation reports",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    args = parser.parse_args()

    logger.info("Executing Full Validation Multi-Region Aggregation Experiment...")
    res = run_full_validation_aggregation_experiment(
        checkpoint_path=args.checkpoint,
        val_parquet_path=args.val_split,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    logger.info("Aggregation experiment completed successfully!")
    logger.info(f"Comparison CSV: {res['csv_path']}")
    logger.info(f"Results JSON: {res['json_path']}")
    logger.info(f"Markdown Report: {res['markdown_path']}")


if __name__ == "__main__":
    main()
