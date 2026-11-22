"""YOLOv8n Evaluation & Reporting Engine for Synthetic UI Elements (Phase 7.4).

Evaluates trained YOLOv8n checkpoints on the untouched synthetic test split,
generates performance metrics, confusion matrices, loss & PR curves, sample visual overlays,
and writes comprehensive reports to data/reports/yolo/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml

from app.ml.vision.yolo_annotations import YOLO_DETECTION_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.evaluate_yolov8")


def load_config(config_path: str | Path) -> dict:
    """Load YAML training/evaluation configuration."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluation(
    checkpoint_path: Path | str = "models/checkpoints/yolov8n_synthetic/run1/weights/best.pt",
    config_path: Path | str = "configs/yolov8n_synthetic.yaml",
    split: str = "test",
    output_dir: Path | str = "data/reports/yolo",
    device: str | int | None = None,
) -> dict:
    """Run evaluation on test split and generate all artifacts and reports."""
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of experiment config
    exp_cfg_path = out_dir / "experiment_config.yaml"
    with open(exp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, indent=2)

    chk_p = Path(checkpoint_path)
    if not chk_p.exists():
        # Check Ultralytics default runs/detect location
        alt_chk = Path("runs/detect") / checkpoint_path
        if alt_chk.exists():
            logger.info(f"Resolved checkpoint at Ultralytics run location: {alt_chk}")
            chk_p = alt_chk
            # Sync to canonical project location
            canon_p = Path("models/checkpoints/yolov8n_synthetic/run1")
            canon_p.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(alt_chk.parents[1], canon_p, dirs_exist_ok=True)
                logger.info(f"Synchronized checkpoint artifacts to canonical path: {canon_p}")
            except Exception as e:
                logger.warning(f"Could not sync artifacts to {canon_p}: {e}")

    if not chk_p.exists():
        logger.warning(f"Checkpoint not found at: {chk_p}")
        logger.warning("Training has not yet been executed by user. Generating pre-evaluation manifest.")

        pre_eval_report = {
            "status": "PENDING_TRAINING_EXECUTION",
            "model_id": "M-04",
            "architecture": "YOLOv8n",
            "checkpoint_expected": str(chk_p),
            "test_split": f"data/synthetic_yolo/images/{split}",
            "note": "Awaiting user execution of training command. Run app.ml.train_yolov8 first.",
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(pre_eval_report, f, indent=2)

        # Write preliminary report document
        report_md = f"""# YOLOv8n Synthetic UI Elements — Experiment Report (Pre-Evaluation)

**Status:** PENDING TRAINING EXECUTION  
**Target Checkpoint:** `{chk_p}`  
**Dataset YAML:** `{cfg['data']['dataset_yaml']}`  
**Target Split:** `{split}`  

Training has not yet been executed. Once the training command completes, re-run this evaluation script:

```powershell
& "C:\\Users\\hp\\PycharmProjects\\blansole_footscan\\.venv\\Scripts\\python.exe" -m app.ml.evaluate_yolov8 --checkpoint {chk_p}
```
"""
        with open(out_dir / "YOLO_EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        return pre_eval_report

    # Checkpoint exists: proceed with Ultralytics validation
    logger.info(f"Loading trained YOLOv8n checkpoint from: {chk_p}")
    from ultralytics import YOLO

    model = YOLO(str(chk_p))

    # Resolve device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    data_yaml = Path(cfg["data"]["dataset_yaml"]).resolve().as_posix()
    imgsz = cfg["data"]["imgsz"]

    logger.info(f"Running validation on '{split}' split (imgsz={imgsz}, device={device})...")
    val_results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=4,
        device=device,
        plots=True,
        save_json=True,
        project=str(out_dir),
        name="val_artifacts",
    )

    # Extract metrics
    metrics_box = val_results.box
    overall_metrics = {
        "status": "EVALUATION_COMPLETE",
        "model_id": "M-04",
        "architecture": "YOLOv8n",
        "checkpoint": str(chk_p),
        "split": split,
        "mAP50": float(metrics_box.map50),
        "mAP50_95": float(metrics_box.map),
        "precision": float(metrics_box.mp),
        "recall": float(metrics_box.mr),
    }

    # Per-class metrics
    class_rows = []
    for cid, cname in YOLO_DETECTION_CLASSES.items():
        if cid < len(metrics_box.maps):
            ap50_95 = float(metrics_box.maps[cid])
        else:
            ap50_95 = 0.0

        p = float(metrics_box.p[cid]) if cid < len(metrics_box.p) else 0.0
        r = float(metrics_box.r[cid]) if cid < len(metrics_box.r) else 0.0
        f1 = float(metrics_box.f1[cid]) if cid < len(metrics_box.f1) else 0.0

        class_rows.append({
            "class_id": cid,
            "class_name": cname,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "mAP50_95": round(ap50_95, 4),
        })

    class_df = pd.DataFrame(class_rows)
    class_csv_path = out_dir / "class_metrics.csv"
    class_df.to_csv(class_csv_path, index=False)
    logger.info(f"Saved class metrics to {class_csv_path}")

    # Copy plots and confusion matrices from run or val_artifacts
    artifact_src = out_dir / "val_artifacts"
    if artifact_src.exists():
        for plot_file in artifact_src.glob("*.png"):
            dest_file = out_dir / plot_file.name
            shutil.copy2(plot_file, dest_file)
            logger.info(f"Copied metric plot: {dest_file.name}")

    # Also copy training plots if available from run directory
    run_dir = chk_p.parents[1]
    for tr_plot in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxP_curve.png", "BoxR_curve.png", "BoxPR_curve.png"]:
        src_tr = run_dir / tr_plot
        if src_tr.exists():
            shutil.copy2(src_tr, out_dir / tr_plot)

    # Save metrics JSON
    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_json_path}")

    # Generate Markdown Report
    report_md = f"""# M-04 YOLOv8n Synthetic UI Elements — Test Evaluation Report

**Evaluation Date:** {pd.Timestamp.now().isoformat()}  
**Model Architecture:** Ultralytics YOLOv8n (Nano)  
**Evaluated Checkpoint:** `{chk_p}`  
**Test Dataset Split:** `{split}` (`data/synthetic_yolo/images/{split}`)  
**Pretrained Prior:** Pre-trained COCO initialization + Synthetic Web UI fine-tuning  

---

## 1. Executive Performance Summary

| Metric | Overall Test Score | Description |
| :--- | :---: | :--- |
| **mAP@50** | **{overall_metrics['mAP50']:.4f}** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@50-95** | **{overall_metrics['mAP50_95']:.4f}** | Primary COCO-standard localization & classification accuracy |
| **Precision (Mean)** | **{overall_metrics['precision']:.4f}** | Ratio of true positive detections over all predicted boxes |
| **Recall (Mean)** | **{overall_metrics['recall']:.4f}** | Ratio of detected ground-truth objects over all existing objects |

---

## 2. Per-Class Detection Performance

{class_df.to_markdown(index=False)}

---

## 3. Metric Visualizations on Disk

The following visual curves and confusion matrices have been generated under `{out_dir}`:
- `results.png` (Training & Validation loss curves for Box loss, Class loss, DFL loss, and mAP)
- `BoxPR_curve.png` (Precision-Recall curve across all 6 classes)
- `confusion_matrix.png` & `confusion_matrix_normalized.png` (Element misclassification matrix)
- `BoxF1_curve.png` (F1 score vs confidence threshold)
- `val_batch0_labels.jpg` vs `val_batch0_pred.jpg` (Visual ground-truth vs model prediction comparison)

---

## 4. Synthetic Experiment Disclaimer & Real-World Roadmap

> [!WARNING]
> **Controlled Synthetic Experiment Notice:**  
> This experiment measures bounding box localization on programmatically rendered synthetic webpage templates. While it establishes that YOLOv8n effectively learns UI layout hierarchies (login cards, input fields, submit buttons, and logos), **these synthetic benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations. In Phase 7.5+, human-verified bounding boxes from the 1,000 catalogued screenshots will be incorporated to measure true real-world generalization.
"""
    with open(out_dir / "YOLO_EXPERIMENT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Saved experiment report to {out_dir / 'YOLO_EXPERIMENT_REPORT.md'}")

    return overall_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on Synthetic Test Split")
    parser.add_argument("--checkpoint", default="models/checkpoints/yolov8n_synthetic/run1/weights/best.pt",
                        help="Path to trained YOLOv8n best.pt checkpoint")
    parser.add_argument("--config", default="configs/yolov8n_synthetic.yaml", help="Path to config file")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--output-dir", default="data/reports/yolo", help="Directory for evaluation artifacts")
    parser.add_argument("--device", default=None, help="Device to run on (e.g. 0 or cpu)")
    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
    )
    logger.info(f"Evaluation process complete: status={results.get('status')}")


if __name__ == "__main__":
    main()
