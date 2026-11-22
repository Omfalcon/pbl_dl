"""CLI entry point to execute validation error analysis on M-02 ResNet-18."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is first in sys.path to avoid collisions with site-packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.vision.error_analysis import run_validation_error_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishvision.run_error_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="M-02 ResNet-18 Validation Error Analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/resnet18/best_model.pt",
        help="Path to trained checkpoint file (default: models/checkpoints/resnet18/best_model.pt)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="data/splits/visual/val.parquet",
        help="Path to validation parquet split (default: data/splits/visual/val.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports/resnet18_error_analysis",
        help="Output directory for error analysis artifacts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for phishing classification",
    )
    args = parser.parse_args()

    logger.info("Executing M-02 ResNet-18 Validation Error Analysis...")
    result = run_validation_error_analysis(
        checkpoint_path=args.checkpoint,
        val_split_path=args.val_split,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    logger.info("Validation Error Analysis successfully completed!")
    logger.info(f"Summary JSON: {result['summary_json_path']}")
    logger.info(f"Error CSV: {result['csv_path']}")
    logger.info(f"Markdown Report: {result['markdown_path']}")


if __name__ == "__main__":
    main()
