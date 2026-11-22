"""Faster R-CNN with ResNet-50-FPN Detection Architecture & Evaluation (Phase 7.6).

Implements M-05 Faster R-CNN for comparative evaluation against M-04 YOLOv8n
on the synthetic website element dataset. Supports hardware-aware memory limits
for NVIDIA RTX 2050 (4GB VRAM), robust empty-image handling, and full COCO/Pascal VOC
mAP metric calculation in pure Python/PyTorch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from app.ml.vision.yolo_annotations import YOLO_DETECTION_CLASSES, YOLOBoundingBox

logger = logging.getLogger("phishvision.faster_rcnn")

# Faster R-CNN Detection Classes (1-indexed for foreground; 0 is background)
FASTER_RCNN_CLASSES: dict[int, str] = {
    0: "__background__",
    1: "brand_logo",
    2: "login_form",
    3: "username_field",
    4: "password_field",
    5: "submit_button",
    6: "payment_element",
}

# Bidirectional Class Mapping between YOLO (0..5) and Faster R-CNN (1..6)
YOLO_TO_FASTER_RCNN: dict[int, int] = {cid: cid + 1 for cid in range(6)}
FASTER_RCNN_TO_YOLO: dict[int, int] = {cid + 1: cid for cid in range(6)}

# UI Palette for Visual Overlays (matching Phase 7 YOLO standards)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "brand_logo": (37, 99, 235),       # Vibrant Blue
    "login_form": (16, 185, 129),      # Emerald Green
    "username_field": (245, 158, 11),   # Amber/Orange
    "password_field": (239, 68, 68),    # Crimson Red
    "submit_button": (139, 92, 246),    # Indigo/Purple
    "payment_element": (236, 72, 153),  # Magenta/Pink
}


def create_faster_rcnn_model(
    num_classes: int = 7,
    pretrained: bool = True,
    weights: str | None = "COCO_V1",
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 512,
) -> torchvision.models.detection.FasterRCNN:
    """Instantiate Faster R-CNN with ResNet-50-FPN backbone and customized box predictor.

    Args:
        num_classes: Total classes including background (7 for 6 UI classes + 1 background).
        pretrained: Whether to load pre-trained backbone weights.
        weights: Torchvision weights identifier (e.g. 'COCO_V1' or None).
        trainable_backbone_layers: Number of trainable ResNet-50 stages (from 0 to 5).
        min_size: Minimum image dimension for internal model transforms (512 saves VRAM).
        max_size: Maximum image dimension for internal model transforms.

    Returns:
        Configured FasterRCNN PyTorch model.
    """
    weights_enum = None
    if pretrained and weights:
        try:
            weights_enum = FasterRCNN_ResNet50_FPN_Weights[weights]
        except (KeyError, AttributeError):
            weights_enum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights_enum,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    # Replace FastRCNN box predictor head with our customized num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class SyntheticFasterRCNNDataset(Dataset):
    """PyTorch Dataset for Faster R-CNN loading from synthetic dataset manifest and YOLO labels."""

    def __init__(
        self,
        manifest_csv: str | Path = "data/synthetic_yolo/manifest.csv",
        split: str = "train",
        dataset_root: str | Path | None = None,
        imgsz: int = 512,
        transform: Callable | None = None,
    ) -> None:
        """Initialize dataset from manifest.

        Args:
            manifest_csv: Path to manifest.csv generated during Phase 7.2.
            split: Dataset partition ('train', 'val', or 'test').
            dataset_root: Optional override for dataset root directory.
            imgsz: Target square dimension.
            transform: Optional transform applied to PIL image.
        """
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        self.split = split
        self.imgsz = imgsz
        self.transform = transform
        self.root_dir = Path(dataset_root) if dataset_root else self.manifest_path.parent

        df = pd.read_csv(self.manifest_path)
        if "split" not in df.columns:
            raise ValueError(f"Manifest missing required 'split' column: {self.manifest_path}")

        self.df = df[df["split"] == self.split].reset_index(drop=True)
        logger.info(f"Loaded SyntheticFasterRCNNDataset split='{self.split}' with {len(self.df)} samples.")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_or_abs: str) -> Path:
        """Resolve a path relative to workspace root or dataset root."""
        p = Path(rel_or_abs)
        if p.exists():
            return p.resolve()
        # Try relative to workspace
        candidate = Path.cwd() / rel_or_abs
        if candidate.exists():
            return candidate.resolve()
        # Try relative to root_dir
        candidate = self.root_dir / p.name
        if candidate.exists():
            return candidate.resolve()
        return p.resolve()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["image_path"])
        label_path = self._resolve_path(row["label_path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # 1. Load image
        with Image.open(img_path) as img:
            image_rgb = img.convert("RGB")
            orig_w, orig_h = image_rgb.size

        # 2. Parse YOLO label file and convert to Faster R-CNN coordinates [x1, y1, x2, y2]
        boxes_list: list[list[float]] = []
        labels_list: list[int] = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    yolo_cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Convert normalized YOLO to absolute pixel coords
                    bx1 = (xc - w / 2.0) * orig_w
                    by1 = (yc - h / 2.0) * orig_h
                    bx2 = (xc + w / 2.0) * orig_w
                    by2 = (yc + h / 2.0) * orig_h

                    # Boundary clamping and non-zero dimension assertions
                    x1 = max(0.0, min(float(orig_w), bx1))
                    y1 = max(0.0, min(float(orig_h), by1))
                    x2 = max(0.0, min(float(orig_w), bx2))
                    y2 = max(0.0, min(float(orig_h), by2))

                    if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                        boxes_list.append([x1, y1, x2, y2])
                        labels_list.append(YOLO_TO_FASTER_RCNN[yolo_cid])

        # 3. Build target dictionary
        if boxes_list:
            boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd_tensor = torch.zeros((len(boxes_list),), dtype=torch.int64)
        else:
            # Proper zero-box handling for empty images / 404 pages
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target: dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        # 4. Apply image transform
        if self.transform is not None:
            image_tensor = self.transform(image_rgb)
        else:
            # Default ToTensor conversion
            image_tensor = torchvision.transforms.functional.to_tensor(image_rgb)

        return image_tensor, target


def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Custom collate function for variable-length detection targets."""
    return tuple(zip(*batch))


# =========================================================================
# Pure Python/PyTorch mAP & Evaluation Metrics Engine
# =========================================================================

def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Intersection-over-Union between two sets of boxes [N, 4] and [M, 4]."""
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.size(0), box2.size(0)), dtype=torch.float32, device=box1.device)

    # Box format: [x1, y1, x2, y2]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    lt = torch.max(box1[:, None, :2], box2[None, :, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-8)
    return iou


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute 101-point COCO-standard interpolated Average Precision."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point interpolation: 0.00, 0.01, ..., 1.00
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    interp_precisions = mpre[inds]
    return float(np.mean(interp_precisions))


def evaluate_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    iou_thresholds: list[float] | None = None,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate Faster R-CNN on detection dataset and calculate exact mAP50, mAP50-95, per-class AP, P, R, and confusion matrix.

    Args:
        model: Trained FasterRCNN model in eval mode.
        data_loader: Test or validation DataLoader.
        device: CUDA or CPU execution device.
        iou_thresholds: List of IoU thresholds (defaults to 0.50:0.05:0.95).
        score_threshold: Minimum confidence score to consider detection.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05).tolist()]

    model.eval()
    all_predictions: list[dict[str, Any]] = []
    all_ground_truths: list[dict[str, Any]] = []

    inference_latencies_ms: list[float] = []

    # Reset peak memory stat
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    global_sample_idx = 0
    with torch.no_grad():
        for img_idx, (images, targets) in enumerate(data_loader):
            images_device = [img.to(device) for img in images]

            t_start = time.perf_counter()
            outputs = model(images_device)
            t_end = time.perf_counter()

            latency_ms = (t_end - t_start) * 1000.0 / len(images)
            inference_latencies_ms.append(latency_ms)

            for i in range(len(images)):
                pred = outputs[i]
                tgt = targets[i]

                # Filter by score threshold
                pred_scores = pred["scores"].detach().cpu()
                pred_labels = pred["labels"].detach().cpu()
                pred_boxes = pred["boxes"].detach().cpu()

                mask = pred_scores >= score_threshold
                filtered_boxes = pred_boxes[mask]
                filtered_scores = pred_scores[mask]
                filtered_labels = pred_labels[mask]

                # Use a monotonic global counter to avoid image_id collisions
                # when the last batch has fewer samples than batch_size
                sample_id = global_sample_idx
                global_sample_idx += 1

                all_predictions.append({
                    "image_id": sample_id,
                    "boxes": filtered_boxes,
                    "scores": filtered_scores,
                    "labels": filtered_labels,  # 1..6
                })

                all_ground_truths.append({
                    "image_id": sample_id,
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),  # 1..6
                })

    # Memory usage
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Calculate per-class metrics across IoU thresholds
    # Foreground classes: 1..6 (mapped to YOLO 0..5)
    class_results: dict[int, dict[str, Any]] = {}
    ap_per_iou: dict[float, list[float]] = {iou: [] for iou in iou_thresholds}

    # Confusion matrix: rows = ground truth (0..6, 0=bg), cols = predicted (0..6, 0=bg)
    confusion_mat = np.zeros((7, 7), dtype=np.int32)

    for fr_cls in range(1, 7):
        yolo_cid = FASTER_RCNN_TO_YOLO[fr_cls]
        class_name = FASTER_RCNN_CLASSES[fr_cls]

        # Gather all ground-truth boxes for this class across all images
        gt_by_img: dict[int, torch.Tensor] = {}
        total_gt = 0
        for sample in all_ground_truths:
            img_id = sample["image_id"]
            gt_mask = sample["labels"] == fr_cls
            gt_boxes = sample["boxes"][gt_mask]
            gt_by_img[img_id] = gt_boxes
            total_gt += len(gt_boxes)

        # Gather all detections for this class across all images
        det_list: list[tuple[float, int, torch.Tensor]] = []
        for sample in all_predictions:
            img_id = sample["image_id"]
            det_mask = sample["labels"] == fr_cls
            det_boxes = sample["boxes"][det_mask]
            det_scores = sample["scores"][det_mask]
            for s, b in zip(det_scores, det_boxes):
                det_list.append((float(s), img_id, b))

        # Sort detections descending by score
        det_list.sort(key=lambda x: x[0], reverse=True)

        class_ap_across_ious: list[float] = []
        p_at_50 = 0.0
        r_at_50 = 0.0
        f1_at_50 = 0.0

        for iou_thresh in iou_thresholds:
            if total_gt == 0:
                class_ap_across_ious.append(0.0)
                ap_per_iou[iou_thresh].append(0.0)
                continue

            if not det_list:
                class_ap_across_ious.append(0.0)
                ap_per_iou[iou_thresh].append(0.0)
                continue

            # Track which GT boxes have been matched
            matched_gt: dict[int, set[int]] = {img_id: set() for img_id in gt_by_img}

            tp = np.zeros(len(det_list), dtype=np.float32)
            fp = np.zeros(len(det_list), dtype=np.float32)

            for det_idx, (score, img_id, det_box) in enumerate(det_list):
                gt_boxes = gt_by_img.get(img_id, torch.zeros((0, 4)))
                if len(gt_boxes) == 0:
                    fp[det_idx] = 1.0
                    continue

                ious = compute_box_iou(det_box.unsqueeze(0), gt_boxes).squeeze(0)
                max_iou, max_idx = ious.max(dim=0)
                max_iou_val = float(max_iou)
                gt_box_idx = int(max_idx)

                if max_iou_val >= iou_thresh and gt_box_idx not in matched_gt[img_id]:
                    tp[det_idx] = 1.0
                    matched_gt[img_id].add(gt_box_idx)
                else:
                    fp[det_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / total_gt
            precisions = cum_tp / (cum_tp + cum_fp)

            ap_val = compute_ap(recalls, precisions)
            class_ap_across_ious.append(ap_val)
            ap_per_iou[iou_thresh].append(ap_val)

            if abs(iou_thresh - 0.50) < 1e-4:
                p_at_50 = float(precisions[-1]) if len(precisions) > 0 else 0.0
                r_at_50 = float(recalls[-1]) if len(recalls) > 0 else 0.0
                f1_at_50 = (2 * p_at_50 * r_at_50 / (p_at_50 + r_at_50)) if (p_at_50 + r_at_50) > 0 else 0.0

        ap50 = class_ap_across_ious[0] if class_ap_across_ious else 0.0
        ap50_95 = float(np.mean(class_ap_across_ious)) if class_ap_across_ious else 0.0

        class_results[yolo_cid] = {
            "class_id": yolo_cid,
            "faster_rcnn_class_id": fr_cls,
            "class_name": class_name,
            "total_ground_truth": total_gt,
            "total_detections": len(det_list),
            "precision": round(p_at_50, 4),
            "recall": round(r_at_50, 4),
            "f1_score": round(f1_at_50, 4),
            "mAP50": round(ap50, 4),
            "mAP50_95": round(ap50_95, 4),
        }

    # Confusion matrix computation at IoU 0.50
    for sample_pred, sample_gt in zip(all_predictions, all_ground_truths):
        p_boxes = sample_pred["boxes"]
        p_labels = sample_pred["labels"]
        g_boxes = sample_gt["boxes"]
        g_labels = sample_gt["labels"]

        matched_gt_set = set()

        if len(p_boxes) > 0 and len(g_boxes) > 0:
            iou_mat = compute_box_iou(p_boxes, g_boxes)
            for p_i in range(len(p_boxes)):
                pred_cls = int(p_labels[p_i])
                best_iou, best_g_i = iou_mat[p_i].max(dim=0)
                best_iou_val = float(best_iou)
                best_g_idx = int(best_g_i)

                if best_iou_val >= 0.50 and best_g_idx not in matched_gt_set:
                    matched_gt_set.add(best_g_idx)
                    true_cls = int(g_labels[best_g_idx])
                    confusion_mat[true_cls, pred_cls] += 1
                else:
                    # Predicted foreground on background or un-matched
                    confusion_mat[0, pred_cls] += 1

            # Any unmatched ground truth counts as false negative (predicted as background)
            for g_i in range(len(g_boxes)):
                if g_i not in matched_gt_set:
                    true_cls = int(g_labels[g_i])
                    confusion_mat[true_cls, 0] += 1
        elif len(p_boxes) > 0:
            # All predicted on background
            for p_i in range(len(p_boxes)):
                pred_cls = int(p_labels[p_i])
                confusion_mat[0, pred_cls] += 1
        elif len(g_boxes) > 0:
            # All missed
            for g_i in range(len(g_boxes)):
                true_cls = int(g_labels[g_i])
                confusion_mat[true_cls, 0] += 1

    overall_map50 = float(np.mean([class_results[cid]["mAP50"] for cid in class_results]))
    overall_map50_95 = float(np.mean([class_results[cid]["mAP50_95"] for cid in class_results]))
    mean_precision = float(np.mean([class_results[cid]["precision"] for cid in class_results]))
    mean_recall = float(np.mean([class_results[cid]["recall"] for cid in class_results]))
    mean_latency_ms = float(np.mean(inference_latencies_ms)) if inference_latencies_ms else 0.0

    return {
        "status": "EVALUATION_COMPLETE",
        "model_id": "M-05",
        "architecture": "FasterRCNN_ResNet50_FPN",
        "mAP50": round(overall_map50, 4),
        "mAP50_95": round(overall_map50_95, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "class_metrics": class_results,
        "confusion_matrix": confusion_mat.tolist(),
    }


# =========================================================================
# Faster R-CNN Inference & Visual Preview Helper
# =========================================================================

class FasterRCNNInferencer:
    """Production and diagnostic inference interface for M-05 Faster R-CNN."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | int | None = None,
        score_threshold: float = 0.50,
    ) -> None:
        if device is None or (isinstance(device, str) and device != "cpu" and not torch.cuda.is_available()):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.score_threshold = score_threshold
        self.model = create_faster_rcnn_model(num_classes=7, pretrained=False)

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            chk = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in chk:
                self.model.load_state_dict(chk["model_state_dict"])
            else:
                self.model.load_state_dict(chk)
            logger.info(f"Loaded Faster R-CNN checkpoint from: {checkpoint_path}")
        else:
            logger.warning("Faster R-CNN initialized without trained weights.")

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image: Image.Image | Path | str,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform object detection on an input image."""
        thresh = score_threshold if score_threshold is not None else self.score_threshold

        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        else:
            img_pil = image.convert("RGB")

        orig_w, orig_h = img_pil.size
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])

        out = outputs[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()

        keep = scores >= thresh
        filt_boxes = boxes[keep]
        filt_scores = scores[keep]
        filt_labels = labels[keep]

        detections = []
        for b, s, l in zip(filt_boxes, filt_scores, filt_labels):
            fr_cid = int(l)
            yolo_cid = FASTER_RCNN_TO_YOLO.get(fr_cid, fr_cid - 1)
            cname = FASTER_RCNN_CLASSES.get(fr_cid, f"class_{fr_cid}")

            x1, y1, x2, y2 = b
            detections.append({
                "class_id": yolo_cid,
                "faster_rcnn_class_id": fr_cid,
                "class_name": cname,
                "confidence": round(float(s), 4),
                "box_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            })

        return {
            "image_size": [orig_w, orig_h],
            "num_detections": len(detections),
            "detections": detections,
        }

    def render_annotated_preview(
        self,
        image: Image.Image | Path | str,
        output_path: Path | str,
        score_threshold: float | None = None,
    ) -> Path:
        """Render and save an annotated preview of the detections."""
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        else:
            img_pil = image.convert("RGB").copy()

        pred = self.predict(img_pil, score_threshold=score_threshold)
        draw = ImageDraw.Draw(img_pil)

        for det in pred["detections"]:
            cname = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["box_xyxy"]
            color = CLASS_COLORS.get(cname, (255, 0, 0))

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label banner
            label_text = f"{cname} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 7, y1], fill=color)
            draw.text((x1 + 2, max(0, y1 - 16)), label_text, fill=(255, 255, 255))

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(out_p, format="PNG")
        return out_p
