"""Curated Brand Reference Gallery Builder.

Initializes directory structure and authentic assets for the 18 monitored brands
defined in data/brands/reference_gallery/manifest_brands.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishvision.brand_gallery")


def initialize_gallery_structure() -> None:
    manifest_path = PROJECT_ROOT / "data" / "brands" / "reference_gallery" / "manifest_brands.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gallery_root = manifest_path.parent
    brands = manifest.get("brands", [])
    logger.info(f"Initializing curated reference folders for {len(brands)} target brands...")

    for b in brands:
        bid = b["brand_id"]
        brand_dir = gallery_root / bid
        (brand_dir / "logos").mkdir(parents=True, exist_ok=True)
        (brand_dir / "canonical_screenshots").mkdir(parents=True, exist_ok=True)
        (brand_dir / "embeddings").mkdir(parents=True, exist_ok=True)

        meta_file = brand_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(b, f, indent=2)

    logger.info(f"Successfully initialized {len(brands)} brand workspaces in {gallery_root}")


if __name__ == "__main__":
    initialize_gallery_structure()
