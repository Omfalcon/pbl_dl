"""Validation Suite for YOLO Object Detection Bounding Boxes & Semantic Segmentation Masks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.validate_annotations")

# Standardized class hierarchies for PhishVision AI Visual Intelligence
DETECTION_CLASSES: dict[int, str] = {
    0: "logo",
    1: "username_input",
    2: "password_input",
    3: "submit_button",
    4: "qr_code",
}

SEGMENTATION_CLASSES: dict[int, str] = {
    0: "background",
    1: "header",
    2: "navigation",
    3: "credential_form",
    4: "main_content",
    5: "footer",
}


@dataclass
class BoundingBox:
    """Represents a normalized YOLO format bounding box: class_id x_center y_center width height."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    class_name: str = ""

    def __post_init__(self) -> None:
        if not self.class_name and self.class_id in DETECTION_CLASSES:
            self.class_name = DETECTION_CLASSES[self.class_id]


@dataclass
class AnnotationValidationResult:
    """Validation report for a single image's annotation file."""
    annotation_path: str
    is_valid: bool
    num_boxes: int = 0
    errors: list[str] = field(default_factory=list)
    boxes: list[BoundingBox] = field(default_factory=list)


class AnnotationValidator:
    """Validates YOLO detection text files and U-Net semantic segmentation mask PNGs."""

    def __init__(
        self,
        num_detection_classes: int = len(DETECTION_CLASSES),
        num_segmentation_classes: int = len(SEGMENTATION_CLASSES),
    ) -> None:
        self.num_detection_classes = num_detection_classes
        self.num_segmentation_classes = num_segmentation_classes

    def validate_yolo_file(self, txt_path: Path | str) -> AnnotationValidationResult:
        """Inspect YOLO .txt annotation file for syntax, normalized ranges [0, 1], and valid class IDs."""
        p = Path(txt_path)
        if not p.exists():
            return AnnotationValidationResult(
                annotation_path=str(p),
                is_valid=False,
                errors=["FILE_NOT_FOUND"],
            )

        errors: list[str] = []
        valid_boxes: list[BoundingBox] = []

        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return AnnotationValidationResult(
                annotation_path=str(p),
                is_valid=False,
                errors=[f"READ_ERROR_{type(e).__name__}"],
            )

        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            parts = line_str.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Expected 5 tokens, got {len(parts)} ('{line_str}')")
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"Line {line_num}: Failed to parse numeric tokens: '{line_str}'")
                continue

            # Class ID boundary check
            if class_id < 0 or class_id >= self.num_detection_classes:
                errors.append(
                    f"Line {line_num}: Invalid class_id {class_id}. Allowed range [0, {self.num_detection_classes - 1}]"
                )

            # Coordinates boundary check [0.0, 1.0]
            for name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Line {line_num}: {name} out of bounds: {val} (must be in [0, 1])")

            # Positive dimension check
            if w <= 0.0 or h <= 0.0:
                errors.append(f"Line {line_num}: Non-positive dimension w={w}, h={h}")

            # Check if bbox extends outside image canvas boundaries
            if xc - w / 2.0 < -0.01 or xc + w / 2.0 > 1.01 or yc - h / 2.0 < -0.01 or yc + h / 2.0 > 1.01:
                errors.append(f"Line {line_num}: Box bounds exceed [0, 1] canvas: xc={xc}, yc={yc}, w={w}, h={h}")

            if not errors:
                valid_boxes.append(BoundingBox(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h))

        is_valid = len(errors) == 0
        return AnnotationValidationResult(
            annotation_path=str(p),
            is_valid=is_valid,
            num_boxes=len(valid_boxes),
            errors=errors,
            boxes=valid_boxes,
        )

    def validate_segmentation_mask(
        self,
        mask_path: Path | str,
        expected_shape: tuple[int, int] | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate that a semantic segmentation mask PNG contains valid class pixel indices."""
        p = Path(mask_path)
        if not p.exists():
            return False, ["FILE_NOT_FOUND"]

        errors: list[str] = []
        try:
            with Image.open(p) as img:
                arr = np.array(img)

                # Check shape
                if expected_shape and arr.shape[:2] != expected_shape:
                    errors.append(f"Mask shape {arr.shape[:2]} does not match expected {expected_shape}")

                # Mask must be single channel integer indices
                if arr.ndim > 2:
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        arr = arr[:, :, 0]
                    else:
                        errors.append(f"Mask must be 2D single channel; got {arr.ndim}D shape {arr.shape}")

                # Validate pixel values are within class range
                unique_vals = np.unique(arr)
                invalid_vals = [int(v) for v in unique_vals if v < 0 or v >= self.num_segmentation_classes]
                if invalid_vals:
                    errors.append(
                        f"Found invalid class pixel values: {invalid_vals}. Allowed [0, {self.num_segmentation_classes - 1}]"
                    )
        except Exception as e:
            return False, [f"MASK_DECODE_ERROR_{type(e).__name__}: {e}"]

        return len(errors) == 0, errors
