"""Annotation Generator and Format Converter for Detection & Segmentation Tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .validate_annotations import (
    DETECTION_CLASSES,
    SEGMENTATION_CLASSES,
    AnnotationValidator,
    BoundingBox,
)
from .visualize_annotations import AnnotationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.generate_annotations")


class AnnotationManager:
    """Manages annotation files, generation of candidate boxes, validation, and previews."""

    def __init__(
        self,
        annotations_dir: Path | str = "data/annotations",
    ) -> None:
        self.annotations_dir = Path(annotations_dir)
        self.detection_dir = self.annotations_dir / "detection"
        self.segmentation_dir = self.annotations_dir / "segmentation"
        self.previews_dir = self.annotations_dir / "previews"

        self.detection_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)

        self.validator = AnnotationValidator()
        self.visualizer = AnnotationVisualizer()

    def save_yolo_annotation(
        self,
        image_name: str,
        boxes: list[BoundingBox],
        is_verified: bool = False,
    ) -> Path:
        """Save bounding boxes to a standardized YOLO .txt file alongside verification status."""
        stem = Path(image_name).stem
        txt_path = self.detection_dir / f"{stem}.txt"
        meta_path = self.detection_dir / f"{stem}_meta.json"

        # Write YOLO lines
        with open(txt_path, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}\n")

        # Record verification status
        meta = {
            "image_name": image_name,
            "num_boxes": len(boxes),
            "is_verified": is_verified,
            "classes": [b.class_name for b in boxes],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return txt_path

    def generate_preview(
        self,
        image_path: Path | str,
        annotation_path: Path | str,
    ) -> Path | None:
        """Render and save visualization preview of annotation on image."""
        img_p = Path(image_path)
        val_res = self.validator.validate_yolo_file(annotation_path)
        if not val_res.is_valid:
            logger.warning(f"Cannot render preview for invalid annotation: {val_res.errors}")
            return None

        preview_path = self.previews_dir / f"{img_p.stem}_annotated.png"
        success = self.visualizer.render_detection_overlay(img_p, val_res.boxes, preview_path)
        if success:
            logger.info(f"Saved annotation preview to {preview_path}")
            return preview_path
        return None
