"""CLI Entry Point for Training ResNet-18 Binary Screenshot Classifier (M-02).

Usage:
    python -m app.ml.train_resnet18 --config configs/resnet18.yaml
    python -m app.ml.train_resnet18 --config configs/resnet18.yaml --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root is at position 0 of sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from app.ml.vision.dataset import ScreenshotDataset
from app.ml.vision.evaluation import save_evaluation_report
from app.ml.vision.models import ResNet18ScreenshotClassifier
from app.ml.vision.trainer import ResNet18Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishvision.train_resnet18")


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and configuration overrides."""
    parser = argparse.ArgumentParser(description="Train ResNet-18 Website Screenshot Classifier (M-02)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu').")
    parser.add_argument(
        "--crop-strategy",
        type=str,
        default=None,
        choices=["top_viewport", "aspect_ratio_pad"],
        help="Override cropping strategy.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train ResNet-18 from scratch.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Evaluate the best restored checkpoint on the holdout test set after training.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training execution orchestrator."""
    args = parse_args()
    logger.info(f"Loading configuration from {args.config}...")
    config = load_yaml_config(args.config)

    # Apply CLI argument overrides
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.crop_strategy is not None:
        prep_cfg["crop_strategy"] = args.crop_strategy
    if args.pretrained is not None:
        model_cfg["pretrained"] = args.pretrained

    # Seed reproducibility
    seed = train_cfg.get("random_seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Determine device
    device_pref = train_cfg.get("device", "cuda")
    if device_pref == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_pref if torch.cuda.is_available() else "cpu")
    logger.info(f"Active training device: {device}")

    # 1. Instantiate Training and Validation Datasets
    logger.info(f"Loading training split from {data_cfg['train_split']}...")
    train_dataset = ScreenshotDataset(
        data_source=data_cfg["train_split"],
        repo_root=data_cfg.get("repo_root", "."),
        crop_strategy=prep_cfg.get("crop_strategy", "top_viewport"),
        image_size=prep_cfg.get("image_size", 224),
        viewport_ratio=prep_cfg.get("viewport_ratio", 1.0),
        is_training=True,
        horizontal_flip=prep_cfg.get("horizontal_flip", False),
        color_jitter=prep_cfg.get("color_jitter", False),
    )
    logger.info(f"Train Dataset Summary: {train_dataset.get_summary()}")

    logger.info(f"Loading validation split from {data_cfg['val_split']}...")
    val_dataset = ScreenshotDataset(
        data_source=data_cfg["val_split"],
        repo_root=data_cfg.get("repo_root", "."),
        crop_strategy=prep_cfg.get("crop_strategy", "top_viewport"),
        image_size=prep_cfg.get("image_size", 224),
        viewport_ratio=prep_cfg.get("viewport_ratio", 1.0),
        is_training=False,
        horizontal_flip=False,
    )
    logger.info(f"Val Dataset Summary: {val_dataset.get_summary()}")

    # 2. Build DataLoaders
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", True) and (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 3. Dynamic Class Weights from Training Split
    if train_cfg.get("use_class_weights", True):
        class_weights = train_dataset.compute_class_weights()
        logger.info(f"Dynamic Training Class Weights: [Legit={class_weights[0]:.4f}, Phish={class_weights[1]:.4f}]")
    else:
        class_weights = None

    # 4. Instantiate Model
    pretrained = model_cfg.get("pretrained", True)
    dropout_p = model_cfg.get("dropout_p", 0.3)
    logger.info(f"Initializing ResNet-18 (pretrained={pretrained}, dropout={dropout_p})...")
    model = ResNet18ScreenshotClassifier(
        pretrained=pretrained,
        dropout_p=dropout_p,
        num_classes=model_cfg.get("num_classes", 2),
    )

    # 5. Instantiate Trainer & Run Training
    trainer = ResNet18Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=train_cfg.get("epochs", 15),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        scheduler_type=train_cfg.get("scheduler", "cosine"),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
        checkpoint_dir=paths_cfg.get("checkpoint_dir", "models/checkpoints/resnet18"),
        report_dir=paths_cfg.get("report_dir", "data/reports"),
        device=device,
        preprocessing_config=prep_cfg,
        random_seed=seed,
    )

    train_result = trainer.train()
    logger.info(
        f"Training completed successfully! Best epoch: {train_result['best_epoch']} "
        f"(Val Macro F1: {train_result['best_val_macro_f1']:.4f})"
    )

    # 6. Optional Holdout Test Set Evaluation
    if args.evaluate_test:
        test_split_path = data_cfg.get("test_split")
        if test_split_path and Path(test_split_path).exists():
            logger.info(f"Evaluating best restored model on holdout test split ({test_split_path})...")
            test_dataset = ScreenshotDataset(
                data_source=test_split_path,
                repo_root=data_cfg.get("repo_root", "."),
                crop_strategy=prep_cfg.get("crop_strategy", "top_viewport"),
                image_size=prep_cfg.get("image_size", 224),
                viewport_ratio=prep_cfg.get("viewport_ratio", 1.0),
                is_training=False,
                horizontal_flip=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            test_loss, test_metrics = trainer.evaluate(test_loader)
            logger.info("=== Holdout Test Partition Evaluation ===")
            logger.info(f"  Accuracy:         {test_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1:         {test_metrics['macro_avg']['f1']:.4f}")
            logger.info(f"  Phishing F1:      {test_metrics['class_1_phishing']['f1']:.4f}")
            logger.info(f"  Legitimate F1:    {test_metrics['class_0_legitimate']['f1']:.4f}")
            logger.info(f"  ROC-AUC:          {test_metrics.get('roc_auc')}")
            logger.info(f"  PR-AUC:           {test_metrics.get('pr_auc')}")
            logger.info(f"  Confusion Matrix: {test_metrics['confusion_matrix']['raw']}")

            eval_report_path = paths_cfg.get("evaluation_report", "data/reports/resnet18_evaluation.json")
            save_evaluation_report(
                metrics=test_metrics,
                output_path=eval_report_path,
                metadata={
                    "model_id": "M-02",
                    "architecture": "ResNet-18",
                    "best_epoch": train_result["best_epoch"],
                    "dataset_summary": test_dataset.get_summary(),
                    "preprocessing_config": prep_cfg,
                },
            )


if __name__ == "__main__":
    main()
