"""CLI Pipeline for Acquiring and Registering Authentic Brand Reference Assets.

Audits reference gallery status and provides utilities to download or import official
brand logos and canonical login screenshots with cryptographic SHA-256 provenance tracking.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.vision.brand_acquisition import (
    BrandReferenceAcquisitionPipeline,
    audit_brand_reference_gallery,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phishvision.acquire_brand_references")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curated Brand Reference Gallery Acquisition Tool")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit current status of the 18 brand reference workspaces on disk",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="data/brands/reference_gallery",
        help="Path to brand reference gallery directory",
    )
    parser.add_argument(
        "--register-asset",
        action="store_true",
        help="Register a downloaded asset for a target brand",
    )
    parser.add_argument("--brand-id", type=str, help="Target brand ID (e.g. google, paypal, sbi)")
    parser.add_argument("--asset-type", type=str, help="Asset type (logo_primary, logo_icon, canonical_login)")
    parser.add_argument("--image-path", type=str, help="Path to local image file to register")
    parser.add_argument("--source-url", type=str, help="Official source URL where asset was captured")
    parser.add_argument("--notes", type=str, default="", help="Optional notes on capture environment")
    args = parser.parse_args()

    audit_res = audit_brand_reference_gallery(gallery_dir=args.gallery_dir)

    logger.info("================ BRAND REFERENCE GALLERY AUDIT ================")
    logger.info(f"Target Brands: {audit_res['target_brands_count']}")
    logger.info(f"Brands with Workspaces: {audit_res['brands_with_workspaces']}")
    logger.info(f"Brands with Reference Images on Disk: {audit_res['brands_with_reference_images']}")
    logger.info(f"Total Reference Images on Disk: {audit_res['total_reference_images_on_disk']}")
    logger.info(f"Status: {audit_res['blocker_status']}")
    logger.info("================================================================")

    if args.register_asset:
        if not all([args.brand_id, args.asset_type, args.image_path, args.source_url]):
            logger.error("Error: --register-asset requires --brand-id, --asset-type, --image-path, and --source-url.")
            sys.exit(1)

        pipeline = BrandReferenceAcquisitionPipeline(gallery_dir=args.gallery_dir)
        rec = pipeline.register_brand_asset(
            brand_id=args.brand_id,
            asset_type=args.asset_type,
            image_input=args.image_path,
            source_url=args.source_url,
            notes=args.notes,
        )
        logger.info(f"Successfully registered asset: {rec}")


if __name__ == "__main__":
    main()
