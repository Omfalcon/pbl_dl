"""YOLOv8n Training Pipeline for Synthetic UI Elements (Phase 7.3).

Executes reproducible training of Ultralytics YOLOv8n on the synthetic website
element dataset using conservative, hardware-aware hyperparameters for RTX 2050 (4GB VRAM).
"""

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

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.train_yolov8")


def load_config(config_path: str | Path) -> dict:
    """Load YAML training configuration."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on Synthetic Web UI Elements")
    parser.add_argument("--config", default="configs/yolov8n_synthetic.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image resolution")
    parser.add_argument("--device", default=None, help="Override device (e.g. 0 or cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    # Verify dataset yaml
    data_yaml = Path(cfg["data"]["dataset_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found at {data_yaml}. Run generate_synthetic_yolo.py first!"
        )

    # Determine device
    device_req = args.device or cfg["training"].get("device", 0)
    if device_req != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA is requested but unavailable. Falling back to CPU.")
        device_req = "cpu"
    elif torch.cuda.is_available() and device_req != "cpu":
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Active training device: CUDA ({device_name})")

    # Hyperparameters
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    imgsz = args.imgsz or cfg["data"]["imgsz"]
    seed = cfg["training"]["random_seed"]
    workers = cfg["data"]["workers"]
    patience = cfg["training"]["patience"]
    half_precision = cfg["training"].get("half", True) and (device_req != "cpu")
    project = cfg["paths"]["project"]
    run_name = cfg["paths"]["name"]

    logger.info(
        f"Training Setup: model=YOLOv8n, imgsz={imgsz}, batch={batch_size}, epochs={epochs}, "
        f"patience={patience}, device={device_req}, seed={seed}, half={half_precision}"
    )

    from ultralytics import YOLO

    model_weights = cfg["model"]["weights"]
    logger.info(f"Initializing YOLO model with weights: {model_weights}")
    model = YOLO(model_weights)

    # Train model
    results = model.train(
        data=str(data_yaml.resolve().as_posix()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        device=device_req,
        seed=seed,
        patience=patience,
        half=half_precision,
        project=project,
        name=run_name,
        save=True,
        plots=True,
        verbose=True,
    )

    logger.info("YOLOv8n training run completed successfully!")
    logger.info(f"Checkpoint artifacts saved under: {Path(project) / run_name}")


if __name__ == "__main__":
    main()
