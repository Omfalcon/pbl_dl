"""YOLOv8-Seg Training & Evaluation Pipeline for Synthetic UI Elements (Phase 7.7).

Executes reproducible training and evaluation of Ultralytics YOLOv8n-seg on the
synthetic webpage element dataset with exact instance segmentation polygon labels.
Saves comprehensive Box and Mask mAP metrics to data/reports/yolo_seg/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from app.ml.vision.yolo_seg import (
    YOLO_SEG_CLASSES,
    SyntheticSegmentationGenerator,
    YOLOSegInferencer,
    generate_segmentation_evaluation_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.train_yolo_seg")


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset(dataset_yaml_path: str | Path, seed: int = 42) -> Path:
    """Ensure that the synthetic segmentation dataset exists; generate if missing."""
    p = Path(dataset_yaml_path)
    if p.exists():
        return p

    logger.info(f"Dataset YAML not found at {p}. Generating synthetic segmentation dataset...")
    generator = SyntheticSegmentationGenerator(seed=seed)
    generator.generate_dataset(output_dir=p.parent, counts={"train": 560, "val": 120, "test": 120})
    return p


def extract_evaluation_metrics(
    val_results: Any,
    checkpoint_path: Path | str,
    split_name: str = "test",
) -> dict[str, Any]:
    """Extract dual Box and Mask evaluation metrics including complete per-class metrics."""
    chk_p = Path(checkpoint_path)
    chk_size_mb = chk_p.stat().st_size / (1024 * 1024) if chk_p.exists() else 0.0

    box_indices = (
        {int(cid): idx for idx, cid in enumerate(val_results.box.ap_class_index)}
        if hasattr(val_results.box, "ap_class_index") and len(val_results.box.ap_class_index)
        else {}
    )
    seg_indices = (
        {int(cid): idx for idx, cid in enumerate(val_results.seg.ap_class_index)}
        if hasattr(val_results.seg, "ap_class_index") and len(val_results.seg.ap_class_index)
        else {}
    )

    class_metrics = {}
    for cid, cname in YOLO_SEG_CLASSES.items():
        box_p, box_r, box_ap50, box_ap = 0.0, 0.0, 0.0, 0.0
        if cid in box_indices:
            res_box = val_results.box.class_result(box_indices[cid])
            box_p, box_r, box_ap50, box_ap = float(res_box[0]), float(res_box[1]), float(res_box[2]), float(res_box[3])

        seg_p, seg_r, seg_ap50, seg_ap = 0.0, 0.0, 0.0, 0.0
        if cid in seg_indices:
            res_seg = val_results.seg.class_result(seg_indices[cid])
            seg_p, seg_r, seg_ap50, seg_ap = float(res_seg[0]), float(res_seg[1]), float(res_seg[2]), float(res_seg[3])

        class_metrics[str(cid)] = {
            "class_id": cid,
            "class_name": cname,
            "box_precision": round(box_p, 4),
            "box_recall": round(box_r, 4),
            "box_mAP50": round(box_ap50, 4),
            "box_mAP50_95": round(box_ap, 4),
            "mask_precision": round(seg_p, 4),
            "mask_recall": round(seg_r, 4),
            "mask_mAP50": round(seg_ap50, 4),
            "mask_mAP50_95": round(seg_ap, 4),
        }

    return {
        "status": "EVALUATION_COMPLETE",
        "model_id": "M-06",
        "architecture": "YOLOv8n-seg",
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "box_mAP50": round(float(val_results.box.map50), 4),
        "box_mAP50_95": round(float(val_results.box.map), 4),
        "box_precision": round(float(val_results.box.mp), 4),
        "box_recall": round(float(val_results.box.mr), 4),
        "mask_mAP50": round(float(val_results.seg.map50), 4),
        "mask_mAP50_95": round(float(val_results.seg.map), 4),
        "mask_precision": round(float(val_results.seg.mp), 4),
        "mask_recall": round(float(val_results.seg.mr), 4),
        "checkpoint_size_mb": round(chk_size_mb, 2),
        "class_metrics": class_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and Evaluate YOLOv8n-Seg on Synthetic UI Elements")
    parser.add_argument("--config", default="configs/yolo_seg.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image resolution")
    parser.add_argument("--device", default=None, help="Override device (0 or cpu)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only using existing checkpoint")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic segmentation dataset only")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    seed = cfg["training"]["random_seed"]
    set_seed(seed)

    # Handle dataset generation
    data_yaml = ensure_dataset(cfg["data"]["dataset_yaml"], seed=seed)
    if args.generate_data:
        logger.info(f"Synthetic segmentation dataset verified at: {data_yaml}")
        return

    # Determine device
    device_req = args.device or cfg["training"].get("device", 0)
    if device_req != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA is requested but unavailable. Falling back to CPU.")
        device_req = "cpu"
    elif torch.cuda.is_available() and device_req != "cpu":
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Active training device: CUDA ({device_name})")

    # Paths
    report_dir = Path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    chk_project = Path(cfg["paths"]["project"])
    chk_name = cfg["paths"]["name"]
    expected_chk = Path(args.checkpoint) if args.checkpoint else chk_project / chk_name / "weights" / "best.pt"

    # Evaluation-Only Mode
    if args.eval_only:
        logger.info(f"Running in EVALUATION-ONLY mode on checkpoint: {expected_chk}")
        if not expected_chk.exists():
            # Check Ultralytics default runs/segment location
            alt_chk1 = Path("runs/segment") / chk_project / chk_name / "weights" / "best.pt"
            alt_chk2 = Path("runs/segment") / chk_project.name / chk_name / "weights" / "best.pt"
            if alt_chk1.exists():
                logger.info(f"Found checkpoint at run location: {alt_chk1}")
                expected_chk = alt_chk1
            elif alt_chk2.exists():
                logger.info(f"Found checkpoint at run location: {alt_chk2}")
                expected_chk = alt_chk2

        if not expected_chk.exists():
            logger.warning(f"Checkpoint not found at: {expected_chk}")
            logger.warning("Training has not yet been executed. Generating pre-evaluation manifest.")

            pre_eval_report = {
                "status": "PENDING_TRAINING_EXECUTION",
                "model_id": "M-06",
                "architecture": "YOLOv8n-seg",
                "checkpoint_expected": str(expected_chk),
                "test_split": f"{cfg['data']['dataset_dir']}/images/test",
                "note": "Awaiting user execution of training command. Run app.ml.train_yolo_seg first.",
            }
            with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(pre_eval_report, f, indent=2)

            pre_md = f"""# M-06 YOLOv8n-Seg Synthetic UI Elements — Experiment Report (Pre-Evaluation)

**Status:** PENDING TRAINING EXECUTION  
**Target Checkpoint:** `{expected_chk}`  
**Dataset YAML:** `{data_yaml}`  
**Target Split:** `test`  

Training has not yet been executed. Execute training manually via:

```powershell
& "C:\\Users\\hp\\PycharmProjects\\blansole_footscan\\.venv\\Scripts\\python.exe" -m app.ml.train_yolo_seg --config {args.config}
```
"""
            with open(report_dir / "YOLO_SEG_EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
                f.write(pre_md)
            logger.info(f"Pre-evaluation report written to: {report_dir / 'YOLO_SEG_EXPERIMENT_REPORT.md'}")
            return

        # Checkpoint exists: Evaluate using Ultralytics
        from ultralytics import YOLO

        model = YOLO(str(expected_chk))
        logger.info(f"Evaluating {expected_chk} on test split...")
        val_batch = cfg["training"].get("batch_size", 4)
        val_results = model.val(
            data=str(data_yaml.resolve().as_posix()),
            split="test",
            batch=val_batch,
            workers=0,
            imgsz=cfg["data"]["imgsz"],
            device=device_req,
            plots=True,
            save=True,
        )

        eval_data = extract_evaluation_metrics(
            val_results=val_results,
            checkpoint_path=expected_chk,
            split_name="test",
        )

        generate_segmentation_evaluation_report(
            eval_metrics=eval_data,
            output_dir=report_dir,
            checkpoint_path=expected_chk,
        )

        # Copy any training plots and confusion matrices from run dir if present
        source_run_dirs = [
            expected_chk.parents[1],
            Path("runs/segment") / chk_project / chk_name,
            Path("runs/segment") / chk_project.name / chk_name,
        ]
        for sdir in source_run_dirs:
            if sdir.exists():
                for plot_file in sdir.glob("*.png"):
                    try:
                        shutil.copy(plot_file, report_dir / plot_file.name)
                    except Exception:
                        pass
                for csv_file in sdir.glob("*.csv"):
                    try:
                        shutil.copy(csv_file, report_dir / csv_file.name)
                    except Exception:
                        pass
                break

        logger.info(
            f"Evaluation complete. Box mAP50: {eval_data['box_mAP50']:.4f}, Mask mAP50: {eval_data['mask_mAP50']:.4f}"
        )
        return

    # Training Setup
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    imgsz = args.imgsz or cfg["data"]["imgsz"]
    workers = cfg["data"]["workers"]
    patience = cfg["training"]["patience"]
    half_precision = cfg["training"].get("half", True) and (device_req != "cpu")
    weights = cfg["model"]["weights"]

    logger.info(
        f"Training Setup: model=YOLOv8n-seg, imgsz={imgsz}, batch={batch_size}, epochs={epochs}, "
        f"patience={patience}, device={device_req}, half={half_precision}, seed={seed}"
    )

    from ultralytics import YOLO

    model = YOLO(weights)
    train_results = model.train(
        data=str(data_yaml.resolve().as_posix()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        device=device_req,
        seed=seed,
        patience=patience,
        half=half_precision,
        project=str(chk_project),
        name=chk_name,
        exist_ok=True,
        save=True,
        plots=True,
    )

    # Post-training evaluation on untouched test split
    best_chk = chk_project / chk_name / "weights" / "best.pt"
    if not best_chk.exists():
        # Fallback to run dir
        best_chk = Path(train_results.save_dir) / "weights" / "best.pt"

    logger.info(f"Evaluating best checkpoint ({best_chk}) on test split...")
    best_model = YOLO(str(best_chk))
    val_results = best_model.val(
        data=str(data_yaml.resolve().as_posix()),
        split="test",
        batch=batch_size,
        workers=0,
        imgsz=imgsz,
        device=device_req,
        plots=True,
        save=True,
    )

    eval_data = extract_evaluation_metrics(
        val_results=val_results,
        checkpoint_path=best_chk,
        split_name="test",
    )

    generate_segmentation_evaluation_report(
        eval_metrics=eval_data,
        output_dir=report_dir,
        checkpoint_path=best_chk,
    )

    # Copy plots and confusion matrix to report_dir
    save_dir = Path(train_results.save_dir)
    for plot_file in save_dir.glob("*.png"):
        try:
            shutil.copy(plot_file, report_dir / plot_file.name)
        except Exception:
            pass

    logger.info(f"Phase 7.7 YOLO-Seg Training & Evaluation Complete! Results saved to: {report_dir}")


if __name__ == "__main__":
    main()
