"""CLI Script to Generate Synthetic YOLO Dataset (Phase 7.2)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.vision.synthetic_generator import SyntheticWebpageGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.generate_synthetic_yolo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Synthetic YOLOv8n Dataset")
    parser.add_argument("--output-dir", default="data/synthetic_yolo", help="Output directory")
    parser.add_argument("--train-count", type=int, default=560, help="Number of training images")
    parser.add_argument("--val-count", type=int, default=120, help="Number of validation images")
    parser.add_argument("--test-count", type=int, default=120, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }

    logger.info(f"Generating synthetic YOLO dataset: {counts} (seed={args.seed})")
    generator = SyntheticWebpageGenerator(seed=args.seed)
    stats = generator.generate_dataset(output_dir=args.output_dir, counts=counts)
    logger.info(f"Synthetic dataset generation complete! Summary:\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
