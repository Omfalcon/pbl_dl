"""CLI entry point to execute multi-region screenshot coverage analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is first in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.vision.multi_region import run_multi_region_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishvision.run_multi_region")


def main() -> None:
    parser = argparse.ArgumentParser(description="M-02 ResNet-18 Multi-Region Coverage Analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/resnet18/best_model.pt",
        help="Path to trained checkpoint file",
    )
    parser.add_argument(
        "--error-csv",
        type=str,
        default="data/reports/resnet18_error_analysis/error_analysis.csv",
        help="Path to error_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports/resnet18_error_analysis/multi_region",
        help="Output directory for multi-region artifacts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold",
    )
    args = parser.parse_args()

    logger.info("Executing M-02 ResNet-18 Multi-Region Coverage Analysis...")
    res = run_multi_region_analysis(
        checkpoint_path=args.checkpoint,
        error_csv_path=args.error_csv,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    logger.info(f"Predictions CSV: {res['predictions_csv']}")
    logger.info(f"Summary JSON: {res['summary_json']}")
    logger.info(f"Contact Sheet: {res['contact_sheet']}")
    logger.info(f"Markdown Report: {res['markdown_report']}")


if __name__ == "__main__":
    main()
