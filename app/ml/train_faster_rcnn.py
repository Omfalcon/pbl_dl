"""Faster R-CNN Training & Evaluation Pipeline for Synthetic UI Elements (Phase 7.6).

Executes reproducible training and evaluation of torchvision Faster R-CNN (ResNet-50-FPN)
on the synthetic website element dataset using conservative, memory-efficient settings
for NVIDIA RTX 2050 (4GB VRAM). Generates full mAP metrics and side-by-side comparison
reports against M-04 YOLOv8n.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from app.ml.vision.faster_rcnn import (
    FASTER_RCNN_CLASSES,
    FASTER_RCNN_TO_YOLO,
    SyntheticFasterRCNNDataset,
    create_faster_rcnn_model,
    detection_collate_fn,
    evaluate_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.train_faster_rcnn")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML training and evaluation configuration."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Enforce fixed random seeds across all engines for strict reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train Faster R-CNN for one epoch with mixed precision and gradient scaling."""
    model.train()
    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_box = 0.0
    running_loss_rpn = 0.0

    num_batches = len(data_loader)
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        loss_val = float(losses.item())
        running_loss += loss_val
        running_loss_cls += float(loss_dict.get("loss_classifier", 0.0))
        running_loss_box += float(loss_dict.get("loss_box_reg", 0.0))
        running_loss_rpn += float(loss_dict.get("loss_objectness", 0.0)) + float(loss_dict.get("loss_rpn_box_reg", 0.0))

        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or (batch_idx + 1) == num_batches:
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{num_batches}] - "
                f"Total Loss: {loss_val:.4f} (Cls: {loss_dict.get('loss_classifier', 0):.4f}, "
                f"Box: {loss_dict.get('loss_box_reg', 0):.4f})"
            )

    return {
        "loss": running_loss / max(1, num_batches),
        "loss_classifier": running_loss_cls / max(1, num_batches),
        "loss_box_reg": running_loss_box / max(1, num_batches),
        "loss_rpn": running_loss_rpn / max(1, num_batches),
    }


def compute_validation_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute validation loss by keeping model in train mode with no gradient updates."""
    # Faster R-CNN only computes losses when model.training is True
    model.train()
    total_val_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += float(losses.item())

    return total_val_loss / max(1, num_batches)


def generate_loss_curve(history: list[dict[str, Any]], output_path: Path) -> None:
    """Generate and save multi-component training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    cls_loss = [h["train_loss_classifier"] for h in history]
    box_loss = [h["train_loss_box_reg"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Total Loss", color="#2563eb", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Total Loss", color="#dc2626", linewidth=2, linestyle="--")
    plt.plot(epochs, cls_loss, label="Train Classifier Loss", color="#10b981", linewidth=1.5, alpha=0.8)
    plt.plot(epochs, box_loss, label="Train Box Reg Loss", color="#f59e0b", linewidth=1.5, alpha=0.8)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN (ResNet-50-FPN) Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved loss curve visualization to: {output_path}")


def generate_comparison_report(
    faster_rcnn_metrics: dict[str, Any],
    yolo_report_dir: Path | str,
    output_dir: Path | str,
    chk_size_mb: float = 0.0,
) -> dict[str, Any]:
    """Generate side-by-side comparison report between M-04 YOLOv8n and M-05 Faster R-CNN."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir = Path(yolo_report_dir)

    # Load YOLO metrics
    yolo_metrics_path = yolo_dir / "metrics.json"
    yolo_metrics: dict[str, Any] = {}
    if yolo_metrics_path.exists():
        with open(yolo_metrics_path, "r", encoding="utf-8") as f:
            yolo_metrics = json.load(f)

    # Load YOLO class metrics
    yolo_class_csv = yolo_dir / "class_metrics.csv"
    yolo_class_df = None
    if yolo_class_csv.exists():
        yolo_class_df = pd.read_csv(yolo_class_csv)

    yolo_map50 = yolo_metrics.get("mAP50", "N/A")
    yolo_map50_95 = yolo_metrics.get("mAP50_95", "N/A")
    yolo_p = yolo_metrics.get("precision", "N/A")
    yolo_r = yolo_metrics.get("recall", "N/A")

    fr_map50 = faster_rcnn_metrics.get("mAP50", "N/A")
    fr_map50_95 = faster_rcnn_metrics.get("mAP50_95", "N/A")
    fr_p = faster_rcnn_metrics.get("mean_precision", "N/A")
    fr_r = faster_rcnn_metrics.get("mean_recall", "N/A")
    fr_latency = faster_rcnn_metrics.get("mean_latency_ms", "N/A")
    fr_vram = faster_rcnn_metrics.get("peak_vram_mb", "N/A")

    # Build comparison dictionary
    comparison = {
        "task": "Synthetic Website Element Object Detection",
        "dataset_split": "test (120 images)",
        "models": {
            "M-04_YOLOv8n": {
                "architecture": "YOLOv8n (Anchor-Free One-Stage)",
                "checkpoint_size_mb": 6.2,
                "mAP50": yolo_map50,
                "mAP50_95": yolo_map50_95,
                "precision": yolo_p,
                "recall": yolo_r,
                "inference_latency_ms": 9.2,
                "hardware_profile": "NVIDIA RTX 2050 (4GB VRAM)",
            },
            "M-05_FasterRCNN": {
                "architecture": "Faster R-CNN ResNet-50-FPN (Two-Stage)",
                "checkpoint_size_mb": round(chk_size_mb, 2),
                "mAP50": fr_map50,
                "mAP50_95": fr_map50_95,
                "precision": fr_p,
                "recall": fr_r,
                "inference_latency_ms": fr_latency,
                "peak_vram_mb": fr_vram,
                "hardware_profile": "NVIDIA RTX 2050 (4GB VRAM)",
            },
        },
    }

    # Save comparison json
    with open(out_dir / "comparison_with_yolo.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save class metrics CSV
    class_rows = []
    for cid, cinfo in faster_rcnn_metrics.get("class_metrics", {}).items():
        class_rows.append({
            "class_id": cinfo["class_id"],
            "class_name": cinfo["class_name"],
            "total_gt": cinfo["total_ground_truth"],
            "precision": cinfo["precision"],
            "recall": cinfo["recall"],
            "f1_score": cinfo["f1_score"],
            "mAP50": cinfo["mAP50"],
            "mAP50_95": cinfo["mAP50_95"],
        })
    if class_rows:
        df_classes = pd.DataFrame(class_rows)
        df_classes.to_csv(out_dir / "class_metrics.csv", index=False)

    # Markdown report
    md_content = f"""# M-05 Faster R-CNN vs M-04 YOLOv8n — Detection Comparison Report

**Phase:** Phase 7.6 Object Detection Benchmark  
**Evaluation Split:** `test` (120 synthetic images, untouched holdout)  
**Hardware Platform:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA 12.4)  
**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Performance Comparison

| Metric | M-04 YOLOv8n (One-Stage) | M-05 Faster R-CNN (Two-Stage) | Delta / Winner |
| :--- | :---: | :---: | :---: |
| **Architecture** | YOLOv8 Nano (Ultralytics) | Faster R-CNN (ResNet-50-FPN) | Different Backbones |
| **mAP@50** | **{yolo_map50}** | **{fr_map50}** | Comparison on test split |
| **mAP@50-95** | **{yolo_map50_95}** | **{fr_map50_95}** | COCO localization standard |
| **Mean Precision** | {yolo_p} | {fr_p} | True Positives / Detections |
| **Mean Recall** | {yolo_r} | {fr_r} | Detected Objects / Ground Truth |
| **Inference Latency** | ~9.2 ms / image | {fr_latency} ms / image | Latency profile |
| **Checkpoint Size** | 6.2 MB | {chk_size_mb:.1f} MB | Storage footprint |
| **Peak VRAM** | < 1.0 GB | {fr_vram} MB | Memory efficiency |

---

## 2. Per-Class Average Precision (Faster R-CNN)

| Class ID | Class Name | Total GT | Precision | Recall | F1 Score | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in class_rows:
        md_content += f"| {r['class_id']} | `{r['class_name']}` | {r['total_gt']} | {r['precision']} | {r['recall']} | {r['f1_score']} | {r['mAP50']} | {r['mAP50_95']} |\n"

    md_content += """
---

## 3. Architectural Analysis: One-Stage vs. Two-Stage

1. **YOLOv8n (Anchor-Free One-Stage):**
   - Directly predicts bounding boxes and class logits from feature pyramid maps in a single pass.
   - Extremely lightweight (3.0M parameters, 6.2 MB checkpoint).
   - Low latency (~9ms) making it ideal for high-throughput scanning pipelines.

2. **Faster R-CNN (ResNet-50-FPN Two-Stage):**
   - Stage 1: Region Proposal Network (RPN) proposes candidate regions of interest.
   - Stage 2: RoIAlign extracts fixed-size feature vectors per proposal, followed by classification and bounding-box refinement.
   - ResNet-50 backbone (~41.5M parameters) provides richer multi-scale semantic representations.
   - Higher memory requirement and inference latency due to proposal generation and RoI pooling stages.

---

## 4. Scientific Integrity & Generalization Notice

> [!WARNING]
> **Controlled Synthetic Dataset Experiment:**  
> This benchmark evaluates bounding box localization on programmatically rendered synthetic webpage templates. While it establishes how Faster R-CNN and YOLOv8n learn UI layout hierarchies (login cards, input fields, submit buttons, and logos), **these benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations. In Phase 7.7+, human-annotated bounding boxes from the 1,000 catalogued screenshots will be used to evaluate true real-world generalization.
"""

    report_path = out_dir / "FASTER_RCNN_EXPERIMENT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved experiment comparison report to: {report_path}")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and Evaluate Faster R-CNN on Synthetic Web UI Elements")
    parser.add_argument("--config", default="configs/faster_rcnn.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Override device (0 or cpu)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only using existing checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    # Set seed
    seed = cfg["training"]["random_seed"]
    set_seed(seed)

    # Determine device
    device_req = args.device or cfg["training"].get("device", 0)
    if device_req != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA is requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    elif torch.cuda.is_available() and device_req != "cpu":
        dev_id = int(device_req) if str(device_req).isdigit() else 0
        device = torch.device(f"cuda:{dev_id}")
        logger.info(f"Active device: CUDA ({torch.cuda.get_device_name(dev_id)})")
    else:
        device = torch.device("cpu")

    # Paths
    chk_dir = Path(cfg["paths"]["checkpoint_dir"])
    chk_dir.mkdir(parents=True, exist_ok=True)
    best_chk_path = Path(args.checkpoint) if args.checkpoint else chk_dir / cfg["paths"]["checkpoint_name"]
    report_dir = Path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = Path(cfg["data"]["manifest_csv"])
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}. Run synthetic generator first!")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    workers = cfg["data"]["workers"]
    amp_enabled = cfg["training"].get("amp", True) and (device.type == "cuda")

    # Handle Evaluation-Only Mode
    if args.eval_only:
        logger.info(f"Running in EVALUATION-ONLY mode on checkpoint: {best_chk_path}")
        if not best_chk_path.exists():
            logger.warning(f"Checkpoint not found at: {best_chk_path}")
            logger.warning("Model training has not yet been executed. Generating pre-evaluation manifest.")

            pre_eval_report = {
                "status": "PENDING_TRAINING_EXECUTION",
                "model_id": "M-05",
                "architecture": "FasterRCNN_ResNet50_FPN",
                "checkpoint_expected": str(best_chk_path),
                "test_split": f"{cfg['data']['dataset_dir']}/images/test",
                "note": "Awaiting user execution of training command. Run app.ml.train_faster_rcnn first.",
            }
            with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(pre_eval_report, f, indent=2)

            pre_md = f"""# M-05 Faster R-CNN — Experiment Report (Pre-Evaluation)

**Status:** PENDING TRAINING EXECUTION  
**Target Checkpoint:** `{best_chk_path}`  
**Dataset Manifest:** `{manifest_csv}`  
**Target Split:** `test`  

Training has not yet been executed. Execute training manually via:

```powershell
& "C:\\Users\\hp\\PycharmProjects\\blansole_footscan\\.venv\\Scripts\\python.exe" -m app.ml.train_faster_rcnn --config {args.config}
```
"""
            with open(report_dir / "FASTER_RCNN_EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
                f.write(pre_md)
            return

        # Checkpoint exists: load and evaluate
        test_dataset = SyntheticFasterRCNNDataset(manifest_csv=manifest_csv, split="test", imgsz=cfg["data"]["imgsz"])
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=detection_collate_fn,
        )

        model = create_faster_rcnn_model(
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
            min_size=cfg["model"]["min_size"],
            max_size=cfg["model"]["max_size"],
        )
        chk_data = torch.load(best_chk_path, map_location=device, weights_only=False)
        state_dict = chk_data["model_state_dict"] if "model_state_dict" in chk_data else chk_data
        model.load_state_dict(state_dict)
        model.to(device)

        eval_metrics = evaluate_dataset(model, test_loader, device=device)
        chk_size_mb = best_chk_path.stat().st_size / (1024 * 1024)

        with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

        generate_comparison_report(
            faster_rcnn_metrics=eval_metrics,
            yolo_report_dir=cfg["paths"]["yolo_report_dir"],
            output_dir=report_dir,
            chk_size_mb=chk_size_mb,
        )
        logger.info(f"Evaluation complete. mAP50: {eval_metrics['mAP50']}, mAP50-95: {eval_metrics['mAP50_95']}")
        return

    # Training Setup
    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    momentum = cfg["training"]["momentum"]
    weight_decay = cfg["training"]["weight_decay"]
    patience = cfg["training"]["patience"]

    logger.info(f"Initializing datasets from: {manifest_csv}")
    train_dataset = SyntheticFasterRCNNDataset(manifest_csv=manifest_csv, split="train", imgsz=cfg["data"]["imgsz"])
    val_dataset = SyntheticFasterRCNNDataset(manifest_csv=manifest_csv, split="val", imgsz=cfg["data"]["imgsz"])
    test_dataset = SyntheticFasterRCNNDataset(manifest_csv=manifest_csv, split="test", imgsz=cfg["data"]["imgsz"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"Instantiating Faster R-CNN: num_classes={cfg['model']['num_classes']}, "
        f"min_size={cfg['model']['min_size']}, trainable_layers={cfg['model']['trainable_backbone_layers']}"
    )
    model = create_faster_rcnn_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=True,
        weights=cfg["model"]["weights"],
        trainable_backbone_layers=cfg["model"]["trainable_backbone_layers"],
        min_size=cfg["model"]["min_size"],
        max_size=cfg["model"]["max_size"],
    )
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    logger.info(
        f"Beginning Faster R-CNN Training: epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, amp={amp_enabled}, device={device}"
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )
        lr_scheduler.step()

        # Validation loss
        val_loss = compute_validation_loss(model, val_loader, device=device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{epochs}] Summary ({elapsed:.1f}s) - "
            f"Train Loss: {train_stats['loss']:.4f}, Val Loss: {val_loss:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 4),
            "val_loss": round(val_loss, 4),
            "train_loss_classifier": round(train_stats["loss_classifier"], 4),
            "train_loss_box_reg": round(train_stats["loss_box_reg"], 4),
            "train_loss_rpn": round(train_stats["loss_rpn"], 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, best_chk_path)
            logger.info(f"-> Saved new best model checkpoint to: {best_chk_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without validation improvement.")
                break

        # Free cached GPU memory between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save training history and loss curves
    with open(report_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    generate_loss_curve(history, report_dir / "loss_curves.png")

    # Evaluate Best Checkpoint on Test Split
    logger.info("Evaluating best checkpoint on untouched test split...")
    chk_data = torch.load(best_chk_path, map_location=device, weights_only=False)
    model.load_state_dict(chk_data["model_state_dict"])
    model.to(device)

    eval_metrics = evaluate_dataset(model, test_loader, device=device)
    chk_size_mb = best_chk_path.stat().st_size / (1024 * 1024)

    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    generate_comparison_report(
        faster_rcnn_metrics=eval_metrics,
        yolo_report_dir=cfg["paths"]["yolo_report_dir"],
        output_dir=report_dir,
        chk_size_mb=chk_size_mb,
    )
    logger.info(
        f"Phase 7.6 Faster R-CNN Training & Evaluation Finished! "
        f"mAP50: {eval_metrics['mAP50']}, mAP50-95: {eval_metrics['mAP50_95']}"
    )


if __name__ == "__main__":
    main()
