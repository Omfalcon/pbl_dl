"""YOLO Object Detection Annotation & Manifest Management Pipeline (Phase 7.1).

Defines standardized detection classes, bounding box data structures, strict
validation routines, empty-image handling, and manifest tracking for Phase 7 M-04.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.yolo_annotations")

# Standardized Phase 7 Detection Classes (0-indexed)
YOLO_DETECTION_CLASSES: dict[int, str] = {
    0: "brand_logo",
    1: "login_form",
    2: "username_field",
    3: "password_field",
    4: "submit_button",
    5: "payment_element",
}

CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in YOLO_DETECTION_CLASSES.items()}


@dataclass
class YOLOBoundingBox:
    """Represents a normalized YOLO bounding box: class_id, x_center, y_center, width, height."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None
    class_name: str = ""

    def __post_init__(self) -> None:
        if not self.class_name and self.class_id in YOLO_DETECTION_CLASSES:
            self.class_name = YOLO_DETECTION_CLASSES[self.class_id]

    def to_yolo_line(self) -> str:
        """Format as standard YOLO space-delimited text line."""
        if self.confidence is not None:
            return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f} {self.confidence:.4f}"
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_xyxy(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert normalized YOLO coordinates to absolute pixel box (xmin, ymin, xmax, ymax)."""
        bx = (self.x_center - self.width / 2.0) * img_width
        by = (self.y_center - self.height / 2.0) * img_height
        bw = self.width * img_width
        bh = self.height * img_height

        xmin = max(0, int(round(bx)))
        ymin = max(0, int(round(by)))
        xmax = min(img_width - 1, int(round(bx + bw)))
        ymax = min(img_height - 1, int(round(by + bh)))
        return xmin, ymin, xmax, ymax

    def compute_iou(self, other: YOLOBoundingBox) -> float:
        """Compute Intersection over Union (IoU) with another bounding box in normalized coordinates."""
        b1_x1 = self.x_center - self.width / 2.0
        b1_y1 = self.y_center - self.height / 2.0
        b1_x2 = self.x_center + self.width / 2.0
        b1_y2 = self.y_center + self.height / 2.0

        b2_x1 = other.x_center - other.width / 2.0
        b2_y1 = other.y_center - other.height / 2.0
        b2_x2 = other.x_center + other.width / 2.0
        b2_y2 = other.y_center + other.height / 2.0

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        b1_area = self.width * self.height
        b2_area = other.width * other.height
        union_area = b1_area + b2_area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


@dataclass
class YOLOValidationResult:
    """Result of validating a YOLO annotation file."""
    annotation_path: str
    is_valid: bool
    is_empty: bool = False
    num_boxes: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    boxes: list[YOLOBoundingBox] = field(default_factory=list)


class YOLOAnnotationValidator:
    """Validates YOLO-format text annotations, bounding-box geometry, and image pairings."""

    def __init__(
        self,
        class_mapping: dict[int, str] | None = None,
        canvas_margin_tolerance: float = 0.02,
    ) -> None:
        self.classes = class_mapping or YOLO_DETECTION_CLASSES
        self.canvas_margin = canvas_margin_tolerance

    def validate_yolo_file(
        self,
        txt_path: Path | str,
        image_path: Path | str | None = None,
    ) -> YOLOValidationResult:
        """Validate a single YOLO annotation text file.

        Enforces:
        - File existence.
        - Empty file validness (empty image with 0 objects).
        - 5 numeric tokens per line (or 6 if confidence included).
        - Class ID within valid range [0, len(classes) - 1].
        - Coordinates and dimensions in normalized [0.0, 1.0].
        - Positive non-zero width and height.
        - Box boundaries within canvas (tolerance [-0.02, 1.02]).
        - No duplicate identical boxes (IoU > 0.95 with same class).
        - Optional image pairing check (existence, decodable, dimensions).
        """
        p = Path(txt_path)
        if not p.exists():
            return YOLOValidationResult(
                annotation_path=str(p),
                is_valid=False,
                errors=["FILE_NOT_FOUND"],
            )

        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return YOLOValidationResult(
                annotation_path=str(p),
                is_valid=False,
                errors=[f"READ_ERROR_{type(e).__name__}: {e}"],
            )

        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

        # Empty image handling: an empty .txt file or empty lines is a valid representation of 0 target objects
        if not lines:
            return YOLOValidationResult(
                annotation_path=str(p),
                is_valid=True,
                is_empty=True,
                num_boxes=0,
                boxes=[],
            )

        errors: list[str] = []
        warnings: list[str] = []
        valid_boxes: list[YOLOBoundingBox] = []

        for line_idx, line_str in enumerate(lines, start=1):
            parts = line_str.split()
            if len(parts) not in (5, 6):
                errors.append(f"Line {line_idx}: Expected 5 or 6 tokens, got {len(parts)} ('{line_str}')")
                continue

            try:
                cid = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if len(parts) == 6 else None
            except ValueError:
                errors.append(f"Line {line_idx}: Non-numeric tokens in line: '{line_str}'")
                continue

            # Class ID check
            if cid not in self.classes:
                errors.append(
                    f"Line {line_idx}: Invalid class_id {cid}. Allowed: {sorted(self.classes.keys())}"
                )
                continue

            # Normalized coordinate bounds check
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                errors.append(f"Line {line_idx}: Center coordinates ({xc}, {yc}) outside [0.0, 1.0]")
                continue

            # Dimensions check
            if w <= 0.0 or h <= 0.0:
                errors.append(f"Line {line_idx}: Non-positive dimension w={w}, h={h}")
                continue

            if w > 1.0 or h > 1.0:
                errors.append(f"Line {line_idx}: Dimensions ({w}, {h}) exceed 1.0")
                continue

            # Canvas overshoot check
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            if x1 < -self.canvas_margin or y1 < -self.canvas_margin or x2 > (1.0 + self.canvas_margin) or y2 > (1.0 + self.canvas_margin):
                errors.append(
                    f"Line {line_idx}: Box [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}] extends beyond canvas margin"
                )
                continue

            box = YOLOBoundingBox(
                class_id=cid,
                x_center=xc,
                y_center=yc,
                width=w,
                height=h,
                confidence=conf,
                class_name=self.classes[cid],
            )
            valid_boxes.append(box)

        # Duplicate box check
        for i in range(len(valid_boxes)):
            for j in range(i + 1, len(valid_boxes)):
                b1 = valid_boxes[i]
                b2 = valid_boxes[j]
                if b1.class_id == b2.class_id:
                    iou = b1.compute_iou(b2)
                    if iou > 0.95:
                        warnings.append(
                            f"Near-duplicate boxes detected for class '{b1.class_name}' with IoU={iou:.3f}"
                        )

        # Image pairing check
        if image_path is not None:
            img_p = Path(image_path)
            if not img_p.exists():
                errors.append(f"Paired image not found on disk: {img_p}")
            else:
                try:
                    with Image.open(img_p) as img:
                        img_w, img_h = img.size
                        if img_w <= 0 or img_h <= 0:
                            errors.append(f"Paired image has invalid dimensions: {img_w}x{img_h}")
                except Exception as e:
                    errors.append(f"Failed to open paired image {img_p}: {e}")

        is_valid = len(errors) == 0
        return YOLOValidationResult(
            annotation_path=str(p),
            is_valid=is_valid,
            is_empty=len(valid_boxes) == 0 and is_valid,
            num_boxes=len(valid_boxes),
            errors=errors,
            warnings=warnings,
            boxes=valid_boxes,
        )


class YOLOAnnotationManifestManager:
    """Manages the master annotation manifest and class/split audit statistics."""

    def __init__(
        self,
        manifest_csv_path: Path | str = "data/annotations/detection/annotation_manifest.csv",
        manifest_json_path: Path | str = "data/annotations/detection/annotation_manifest.json",
        detection_dir: Path | str = "data/annotations/detection",
    ) -> None:
        self.csv_path = Path(manifest_csv_path)
        self.json_path = Path(manifest_json_path)
        self.detection_dir = Path(detection_dir)
        self.validator = YOLOAnnotationValidator()

    def build_initial_manifest_from_corpus(
        self,
        corpus_parquet: Path | str = "data/processed/screenshots/visual_corpus.parquet",
        train_parquet: Path | str = "data/splits/visual/train.parquet",
        val_parquet: Path | str = "data/splits/visual/val.parquet",
        test_parquet: Path | str = "data/splits/visual/test.parquet",
        save: bool = True,
    ) -> pd.DataFrame:
        """Create the canonical annotation manifest for all 1,000 visual corpus screenshots.

        Strictly marks unannotated screenshots as 'UNANNOTATED' with 0 bounding boxes.
        Never fabricates labels, synthetic boxes, or fake annotations.
        """
        corpus_p = Path(corpus_parquet)
        if not corpus_p.exists():
            raise FileNotFoundError(f"Corpus parquet not found: {corpus_p}")

        df_corpus = pd.read_parquet(corpus_p)

        # Build split mapping
        split_map: dict[str, str] = {}
        for split_name, split_p in [("train", Path(train_parquet)), ("val", Path(val_parquet)), ("test", Path(test_parquet))]:
            if split_p.exists():
                df_split = pd.read_parquet(split_p)
                for sid in df_split["sample_id"]:
                    split_map[str(sid)] = split_name

        records = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for _, row in df_corpus.iterrows():
            sid = str(row["sample_id"])
            rel_path = str(row["screenshot_relative_path"])
            abs_path = str(row["screenshot_absolute_path"])
            domain = str(row["registrable_domain"]) if pd.notna(row["registrable_domain"]) else ""
            original_split = split_map.get(sid, "unassigned")
            raw_label = int(row["label"])

            # Determine if annotation text file exists on disk
            txt_file = self.detection_dir / f"{sid}.txt"
            status = "UNANNOTATED"
            num_boxes = 0
            boxes_per_class = {cname: 0 for cname in YOLO_DETECTION_CLASSES.values()}

            if txt_file.exists():
                val_res = self.validator.validate_yolo_file(txt_file)
                if val_res.is_valid:
                    if val_res.is_empty:
                        status = "ANNOTATED_EMPTY"
                    else:
                        status = "ANNOTATED_CANDIDATE"
                        num_boxes = val_res.num_boxes
                        for b in val_res.boxes:
                            boxes_per_class[b.class_name] = boxes_per_class.get(b.class_name, 0) + 1
                else:
                    status = "INVALID_ANNOTATION"

            record = {
                "sample_id": sid,
                "source_image_path": rel_path,
                "source_image_absolute_path": abs_path,
                "domain": domain,
                "dataset_label": raw_label,  # 0: legit, 1: phish
                "original_dataset_split": original_split,
                "annotation_status": status,
                "verification_status": "UNVERIFIED" if status.startswith("ANNOTATED") else "NONE",
                "annotator_or_verifier": "NONE",
                "num_brand_logo": boxes_per_class["brand_logo"],
                "num_login_form": boxes_per_class["login_form"],
                "num_username_field": boxes_per_class["username_field"],
                "num_password_field": boxes_per_class["password_field"],
                "num_submit_button": boxes_per_class["submit_button"],
                "num_payment_element": boxes_per_class["payment_element"],
                "total_boxes": num_boxes,
                "last_updated": now_iso,
            }
            records.append(record)

        manifest_df = pd.DataFrame(records)

        if save:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_df.to_csv(self.csv_path, index=False)
            manifest_dict = {
                "metadata": {
                    "project": "PhishVision AI",
                    "module": "M-04 YOLO Object Detection",
                    "total_images": len(manifest_df),
                    "created_at": now_iso,
                    "classes": YOLO_DETECTION_CLASSES,
                },
                "records": records,
            }
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(manifest_dict, f, indent=2)
            logger.info(f"Saved initial annotation manifest to {self.csv_path} and {self.json_path}")

        return manifest_df

    def compute_dataset_statistics(
        self,
        manifest_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive dataset statistics by class, split, and status."""
        if manifest_df is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")
            manifest_df = pd.read_csv(self.csv_path)

        total_images = len(manifest_df)
        status_counts = manifest_df["annotation_status"].value_counts().to_dict()
        verification_counts = manifest_df["verification_status"].value_counts().to_dict()

        split_summary = {}
        for split_name, group in manifest_df.groupby("original_dataset_split"):
            split_summary[str(split_name)] = {
                "total_images": len(group),
                "phishing_images": int((group["dataset_label"] == 1).sum()),
                "legitimate_images": int((group["dataset_label"] == 0).sum()),
                "annotated_count": int((group["annotation_status"].isin(["ANNOTATED_CANDIDATE", "ANNOTATED_EMPTY"])).sum()),
                "unannotated_count": int((group["annotation_status"] == "UNANNOTATED").sum()),
                "total_boxes": int(group["total_boxes"].sum()),
                "classes": {
                    "brand_logo": int(group["num_brand_logo"].sum()),
                    "login_form": int(group["num_login_form"].sum()),
                    "username_field": int(group["num_username_field"].sum()),
                    "password_field": int(group["num_password_field"].sum()),
                    "submit_button": int(group["num_submit_button"].sum()),
                    "payment_element": int(group["num_payment_element"].sum()),
                },
            }

        total_boxes_per_class = {
            "brand_logo": int(manifest_df["num_brand_logo"].sum()),
            "login_form": int(manifest_df["num_login_form"].sum()),
            "username_field": int(manifest_df["num_username_field"].sum()),
            "password_field": int(manifest_df["num_password_field"].sum()),
            "submit_button": int(manifest_df["num_submit_button"].sum()),
            "payment_element": int(manifest_df["num_payment_element"].sum()),
        }

        stats = {
            "total_images": total_images,
            "status_counts": status_counts,
            "verification_counts": verification_counts,
            "total_bounding_boxes": int(manifest_df["total_boxes"].sum()),
            "total_boxes_per_class": total_boxes_per_class,
            "split_summary": split_summary,
        }
        return stats
