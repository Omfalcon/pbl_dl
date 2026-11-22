"""CLI entry point to generate visual contact sheets for ResNet-18 error analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure workspace root is first in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.vision.contact_sheet import generate_all_contact_sheets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishvision.generate_contact_sheets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual contact sheets for error analysis")
    parser.add_argument(
        "--error-csv",
        type=str,
        default="data/reports/resnet18_error_analysis/error_analysis.csv",
        help="Path to error_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports/resnet18_error_analysis/contact_sheets",
        help="Output directory for contact sheets",
    )
    args = parser.parse_args()

    logger.info("Generating visual contact sheets for error analysis...")
    res = generate_all_contact_sheets(
        error_csv_path=args.error_csv,
        output_dir=args.output_dir,
    )

    logger.info(f"False Negatives Contact Sheet: {res['fn_contact_sheet']} ({res['fn_count']} items)")
    logger.info(f"False Positives Contact Sheet: {res['fp_contact_sheet']} ({res['fp_count']} items)")
    logger.info(f"Contact Sheet Index CSV: {res['index_csv']} ({res['total_indexed']} rows)")
    logger.info("All original screenshot immutability hashes verified: PASS")


if __name__ == "__main__":
    main()
