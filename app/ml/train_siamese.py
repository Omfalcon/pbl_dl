"""PhishVision AI — Siamese ResNet-18 Training CLI (Phase 6.2).

Entry point for training the M-03 Siamese CNN Brand Impersonation Matcher.
Enforces strict checks on brand reference gallery and pair manifest availability.
If assets are missing or empty on disk, reports the exact blocker and exits
gracefully without fabricating data or training on fake inputs.

Usage:
    python -m app.ml.train_siamese --config configs/siamese.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is at index 0 of sys.path to prevent module collisions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from app.ml.vision.brand_acquisition import audit_brand_reference_gallery
from app.ml.vision.siamese import (
    ContrastiveLoss,
    CosineContrastiveLoss,
    SiameseResNet18,
    TripletLoss,
)
from app.ml.vision.siamese_dataset import (
    SiamesePairDataset,
    SiameseTripletDataset,
)
from app.ml.vision.siamese_trainer import SiameseTrainer
from app.ml.vision.preprocessing import build_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] phishvision.train_siamese: %(message)s",
)
logger = logging.getLogger("phishvision.train_siamese")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="PhishVision AI — Siamese ResNet-18 Training Entry Point",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/siamese.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform configuration and dataset validation without executing training.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration dictionary from YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main execution workflow for Siamese ResNet-18 training."""
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    logger.info("Loading Siamese configuration from: %s", config_path)
    config = load_yaml_config(config_path)

    # 1. Audit Brand Reference Gallery on disk
    data_cfg = config.get("data", config.get("dataset", {}))
    gallery_dir = PROJECT_ROOT / data_cfg.get(
        "gallery_dir", "data/brands/reference_gallery"
    )
    audit = audit_brand_reference_gallery(gallery_dir)

    total_reference_images = audit["total_reference_images_on_disk"]
    logger.info(
        "Physical gallery audit: %d target brands checked, %d reference images found on disk.",
        audit["target_brands_count"],
        total_reference_images,
    )

    manifest_rel = data_cfg.get(
        "pairs_manifest", data_cfg.get("pairs_csv", "data/brands/manifest_siamese_pairs.csv")
    )
    pairs_csv = PROJECT_ROOT / manifest_rel
    if not pairs_csv.is_file():
        fallback_csv = PROJECT_ROOT / "data" / "brands" / "manifest_siamese_pairs.csv"
        if fallback_csv.is_file():
            pairs_csv = fallback_csv

    if total_reference_images == 0 or not pairs_csv.is_file():
        logger.warning(
            "================================================================================"
        )
        logger.warning("TRAINING BLOCKED: INSUFFICIENT BRAND REFERENCE DATA")
        logger.warning(
            "Found %d images in gallery (%s).", total_reference_images, gallery_dir
        )
        logger.warning(
            "Pairs manifest exists: %s (%s)",
            pairs_csv.is_file(),
            pairs_csv,
        )
        logger.warning(
            "Model training cannot proceed without authentic reference assets and a verified"
        )
        logger.warning(
            "pair manifest. To acquire assets, use scripts/datasets/acquire_brand_references.py"
        )
        logger.warning(
            "================================================================================"
        )
        if not args.dry_run:
            logger.info("Exiting gracefully. No model weights or synthetic data were altered.")
            sys.exit(0)

    # 2. Build Datasets & DataLoaders if assets exist
    logger.info("Loading training and validation pair datasets...")
    train_transform = build_transforms(crop_strategy="aspect_ratio_pad", image_size=224, is_training=True)
    val_transform = build_transforms(crop_strategy="aspect_ratio_pad", image_size=224, is_training=False)

    train_dataset = SiamesePairDataset(
        manifest_csv=pairs_csv,
        split="train",
        transform=train_transform,
    )
    val_dataset = SiamesePairDataset(
        manifest_csv=pairs_csv,
        split="val",
        transform=val_transform,
    )

    logger.info("Loaded datasets: %d train pairs, %d val pairs.", len(train_dataset), len(val_dataset))

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Initialize Model
    model_cfg = config.get("model", {})
    embedding_dim = model_cfg.get("embedding_dim", 256)
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", False)

    model = SiameseResNet18(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    # 4. Initialize Loss
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cosine_contrastive")
    margin = loss_cfg.get("margin", 0.5)

    if loss_type == "cosine_contrastive":
        criterion = CosineContrastiveLoss(margin=margin)
    elif loss_type == "contrastive":
        criterion = ContrastiveLoss(margin=margin)
    elif loss_type == "triplet":
        criterion = TripletLoss(margin=margin)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # 5. Execute Trainer
    output_dir = PROJECT_ROOT / config.get("output", {}).get(
        "checkpoint_dir", "models/checkpoints/siamese"
    )
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        output_dir=output_dir,
    )

    # Dry-run validation check: full pipeline initialized without executing epochs
    if args.dry_run:
        logger.info(
            "Dry-run verification completed successfully: datasets (%d train, %d val), loaders, model (%s), and trainer verified. Training was not executed.",
            len(train_dataset),
            len(val_dataset),
            model.__class__.__name__,
        )
        return

    logger.info("Starting Siamese ResNet-18 training...")
    history = trainer.train()
    logger.info(
        "Training finished. Best Val F1: %.4f at Epoch %d",
        history.get("best_val_f1", 0.0),
        history.get("best_epoch", -1),
    )


if __name__ == "__main__":
    main()
