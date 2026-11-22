"""Validation Suite for YOLO Object Detection Bounding Boxes & Semantic Segmentation Masks."""

from __future__ import annotations

import json
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


@dataclass
class DatasetAnnotationQualityReport:
    """Dataset-level quality audit report separating human-verified vs unverified annotations."""
    total_images_annotated: int = 0
    manually_verified_count: int = 0
    unverified_candidate_count: int = 0
    total_bounding_boxes: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    segmentation_masks_count: int = 0
    invalid_annotations_count: int = 0


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
        """Inspect YOLO .txt annotation file. Each bounding box is validated independently."""
        p = Path(txt_path)
        if not p.exists():
            return AnnotationValidationResult(
                annotation_path=str(p),
                is_valid=False,
                errors=["FILE_NOT_FOUND"],
            )

        all_errors: list[str] = []
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
                all_errors.append(f"Line {line_num}: Expected 5 tokens, got {len(parts)} ('{line_str}')")
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                all_errors.append(f"Line {line_num}: Failed to parse numeric tokens: '{line_str}'")
                continue

            # Validate each bounding box independently
            box_errors: list[str] = []

            # Class ID boundary check
            if class_id < 0 or class_id >= self.num_detection_classes:
                box_errors.append(
                    f"Line {line_num}: Invalid class_id {class_id}. Allowed range [0, {self.num_detection_classes - 1}]"
                )

            # Coordinates boundary check [0.0, 1.0]
            for name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
                if not (0.0 <= val <= 1.0):
                    box_errors.append(f"Line {line_num}: {name} out of bounds: {val} (must be in [0, 1])")

            # Positive dimension check
            if w <= 0.0 or h <= 0.0:
                box_errors.append(f"Line {line_num}: Non-positive dimension w={w}, h={h}")

            # Check if bbox extends outside image canvas boundaries
            if xc - w / 2.0 < -0.01 or xc + w / 2.0 > 1.01 or yc - h / 2.0 < -0.01 or yc + h / 2.0 > 1.01:
                box_errors.append(f"Line {line_num}: Box bounds exceed [0, 1] canvas: xc={xc}, yc={yc}, w={w}, h={h}")

            if not box_errors:
                valid_boxes.append(BoundingBox(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h))
            else:
                all_errors.extend(box_errors)

        is_valid = len(all_errors) == 0
        return AnnotationValidationResult(
            annotation_path=str(p),
            is_valid=is_valid,
            num_boxes=len(valid_boxes),
            errors=all_errors,
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


class SegmentationMaskManager:
    """Handles saving, loading, and cross-modality pairing verification of semantic masks."""

    @staticmethod
    def save_mask(mask: np.ndarray, mask_path: Path | str) -> Path:
        """Save 2D uint8 numpy array mask as an uncompressed 8-bit grayscale PNG."""
        p = Path(mask_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        assert mask.ndim == 2, f"Mask array must be 2D; got {mask.ndim}D"
        mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8, mode="L")
        img.save(p, format="PNG")
        return p

    @staticmethod
    def load_mask(mask_path: Path | str) -> np.ndarray:
        """Load 2D segmentation mask as uint8 numpy array."""
        p = Path(mask_path)
        if not p.exists():
            raise FileNotFoundError(f"Mask file not found: {p}")
        with Image.open(p) as img:
            return np.array(img, dtype=np.uint8)

    @classmethod
    def validate_image_mask_pair(
        cls,
        image_path: Path | str,
        mask_path: Path | str,
    ) -> tuple[bool, list[str]]:
        """Validate that an image and its corresponding segmentation mask exist and match in dimensions."""
        img_p = Path(image_path)
        mask_p = Path(mask_path)

        if not img_p.exists():
            return False, [f"Image file not found: {img_p}"]
        if not mask_p.exists():
            return False, [f"Mask file not found: {mask_p}"]

        errors: list[str] = []
        try:
            with Image.open(img_p) as img, Image.open(mask_p) as mask:
                if img.size != mask.size:
                    errors.append(
                        f"Dimension mismatch: image is {img.size[0]}x{img.size[1]}, mask is {mask.size[0]}x{mask.size[1]}"
                    )
        except Exception as e:
            errors.append(f"Pair inspection failed: {e}")

        # Also validate mask pixel ranges
        validator = AnnotationValidator()
        mask_valid, mask_errors = validator.validate_segmentation_mask(mask_p)
        errors.extend(mask_errors)

        return len(errors) == 0, errors


def generate_dataset_annotation_report(
    annotations_dir: Path | str = "data/annotations",
) -> DatasetAnnotationQualityReport:
    """Generate aggregate quality audit report strictly distinguishing manually verified annotations."""
    ann_p = Path(annotations_dir)
    rep = DatasetAnnotationQualityReport()
    validator = AnnotationValidator()

    for txt_f in (ann_p / "detection").glob("*.txt"):
        rep.total_images_annotated += 1
        res = validator.validate_yolo_file(txt_f)
        if not res.is_valid:
            rep.invalid_annotations_count += 1
        else:
            rep.total_bounding_boxes += len(res.boxes)
            for b in res.boxes:
                rep.boxes_per_class[b.class_name] = rep.boxes_per_class.get(b.class_name, 0) + 1

        # Check metadata JSON for human verification flag
        meta_f = txt_f.with_name(f"{txt_f.stem}_meta.json")
        is_verified = False
        if meta_f.exists():
            try:
                with open(meta_f, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    is_verified = bool(meta.get("is_verified", False))
            except Exception:
                pass

        if is_verified:
            rep.manually_verified_count += 1
        else:
            rep.unverified_candidate_count += 1

    rep.segmentation_masks_count = len(list((ann_p / "segmentation").glob("*.png")))
    return rep
