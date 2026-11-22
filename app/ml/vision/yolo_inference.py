"""YOLOv8n UI Element & Logo Detection Inference Engine (M-04 Phase 7.5).

Loads trained YOLOv8n weights, detects the 6 supported UI classes, extracts
brand_logo crops with customizable padding (default 10%), and renders visual overlays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from app.ml.vision.preprocessing import safe_load_image
from app.ml.vision.yolo_annotations import (
    CLASS_NAME_TO_ID,
    YOLO_DETECTION_CLASSES,
    YOLOBoundingBox,
)

logger = logging.getLogger("phishvision.ml.vision.yolo_inference")

# Class color palette for visualization overlays (RGB)
UI_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (59, 130, 246),   # brand_logo: Blue
    1: (168, 85, 247),  # login_form: Purple
    2: (16, 185, 129),  # username_field: Green
    3: (239, 68, 68),   # password_field: Red
    4: (245, 158, 11),  # submit_button: Amber
    5: (236, 72, 153),  # payment_element: Pink
}


class YOLOUIElementDetector:
    """Production and diagnostic inference interface for M-04 YOLO UI Component Detector."""

    def __init__(
        self,
        checkpoint_path: str | Path = "models/checkpoints/yolov8n_synthetic/run1/weights/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 512,
        device: str | int | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # Resolve device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.is_available = False
        self.classes = YOLO_DETECTION_CLASSES

        # Locate checkpoint
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            alt_ckpt = Path("runs/detect") / checkpoint_path
            if alt_ckpt.exists():
                ckpt_p = alt_ckpt

        if ckpt_p.exists():
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLOv8n detector from {ckpt_p} on device {self.device}...")
                self.model = YOLO(str(ckpt_p))
                self.is_available = True
                self.checkpoint_path = str(ckpt_p)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.is_available = False
                self.checkpoint_path = None
        else:
            logger.warning(f"YOLO checkpoint not found at: {checkpoint_path}")
            self.checkpoint_path = None

        # Load fonts for preview rendering
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

    def detect(
        self,
        image_input: str | Path | Image.Image,
        conf_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Runs UI object detection on a website screenshot.

        Returns:
            Dict containing detected boxes, counts per class, and element summary.
        """
        if not self.is_available or self.model is None:
            return {
                "status": "MODEL_UNAVAILABLE",
                "is_available": False,
                "total_detections": 0,
                "detections": [],
                "class_counts": {c: 0 for c in self.classes.values()},
            }

        conf = conf_threshold or self.conf_threshold

        # Load PIL image to verify dimensions
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = safe_load_image(image_input)

        img_w, img_h = pil_img.size

        # Run Ultralytics inference
        results = self.model.predict(
            source=pil_img,
            conf=conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        class_counts = {c: 0 for c in self.classes.values()}

        if results and len(results) > 0:
            boxes_obj = results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                for box_data in boxes_obj:
                    cid = int(box_data.cls[0].item())
                    score = float(box_data.conf[0].item())
                    cname = self.classes.get(cid, f"class_{cid}")

                    # xyxy coordinates in pixel space
                    xyxy = box_data.xyxy[0].cpu().numpy().tolist()
                    x1 = max(0, int(round(xyxy[0])))
                    y1 = max(0, int(round(xyxy[1])))
                    x2 = min(img_w - 1, int(round(xyxy[2])))
                    y2 = min(img_h - 1, int(round(xyxy[3])))

                    # normalized yolo box [xc, yc, w, h]
                    xc = round(((x1 + x2) / 2.0) / img_w, 6)
                    yc = round(((y1 + y2) / 2.0) / img_h, 6)
                    bw = round((x2 - x1) / img_w, 6)
                    bh = round((y2 - y1) / img_h, 6)

                    det = {
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": round(score, 4),
                        "bbox_pixel": [x1, y1, x2, y2],
                        "bbox_normalized": [xc, yc, bw, bh],
                    }
                    detections.append(det)
                    if cname in class_counts:
                        class_counts[cname] += 1

        return {
            "status": "SUCCESS",
            "is_available": True,
            "image_size": [img_w, img_h],
            "total_detections": len(detections),
            "class_counts": class_counts,
            "has_login_form": class_counts["login_form"] > 0,
            "has_credential_inputs": (class_counts["username_field"] > 0 or class_counts["password_field"] > 0),
            "has_submit_button": class_counts["submit_button"] > 0,
            "has_payment_element": class_counts["payment_element"] > 0,
            "detections": detections,
        }

    def extract_logo_crops(
        self,
        image_input: str | Path | Image.Image,
        padding_pct: float = 0.10,
        conf_threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Extracts brand_logo bounding boxes with approximately 10% context padding.

        Returns:
            List of dicts with 'crop_image' (PIL.Image), confidence, and coordinates.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = safe_load_image(image_input)

        img_w, img_h = pil_img.size
        det_result = self.detect(pil_img, conf_threshold=conf_threshold)

        logo_crops: list[dict[str, Any]] = []
        for det in det_result.get("detections", []):
            if det["class_name"] == "brand_logo":
                x1, y1, x2, y2 = det["bbox_pixel"]
                bw = x2 - x1
                bh = y2 - y1

                # Apply 10% padding
                pad_x = int(round(bw * padding_pct))
                pad_y = int(round(bh * padding_pct))

                padded_x1 = max(0, x1 - pad_x)
                padded_y1 = max(0, y1 - pad_y)
                padded_x2 = min(img_w - 1, x2 + pad_x)
                padded_y2 = min(img_h - 1, y2 + pad_y)

                # Ensure non-empty crop
                if padded_x2 > padded_x1 and padded_y2 > padded_y1:
                    crop = pil_img.crop((padded_x1, padded_y1, padded_x2, padded_y2))
                    logo_crops.append({
                        "crop_image": crop,
                        "confidence": det["confidence"],
                        "original_bbox": [x1, y1, x2, y2],
                        "padded_bbox": [padded_x1, padded_y1, padded_x2, padded_y2],
                        "crop_dimensions": [crop.width, crop.height],
                    })

        return logo_crops

    def render_annotated_preview(
        self,
        image_input: str | Path | Image.Image,
        detections: list[dict[str, Any]] | None = None,
        output_path: str | Path | None = None,
        conf_threshold: float | None = None,
    ) -> Image.Image:
        """Renders bounding box overlays with class pills on image."""
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB").copy()
        else:
            img = safe_load_image(image_input).copy()

        if detections is None:
            det_result = self.detect(img, conf_threshold=conf_threshold)
            target_dets = det_result.get("detections", [])
        else:
            target_dets = detections

        draw = ImageDraw.Draw(img)

        for det in target_dets:
            x1, y1, x2, y2 = det["bbox_pixel"]
            cid = det["class_id"]
            color = UI_CLASS_COLORS.get(cid, (255, 255, 0))

            # Draw outer rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw text pill
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            pill_h = 18
            pill_w = len(label) * 8 + 8
            pill_y1 = max(0, y1 - pill_h)
            draw.rectangle([x1, pill_y1, x1 + pill_w, y1], fill=color)
            draw.text((x1 + 4, pill_y1 + 2), label, fill=(255, 255, 255), font=self.font)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_p, format="PNG")
            logger.info(f"Saved annotated preview to: {out_p}")

        return img
