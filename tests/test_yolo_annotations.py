"""Unit Tests for YOLO Object Detection Annotation & Manifest System (Phase 7.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.ml.vision.yolo_annotations import (
    CLASS_NAME_TO_ID,
    YOLO_DETECTION_CLASSES,
    YOLOAnnotationManifestManager,
    YOLOAnnotationValidator,
    YOLOBoundingBox,
    YOLOValidationResult,
)


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Creates a temporary dummy image for pairing validation tests."""
    img_path = tmp_path / "screen_test.png"
    arr = np.full((720, 1280, 3), 180, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


class TestYOLOBoundingBox:
    """Verifies bounding box geometry, normalization, conversions and IoU."""

    def test_bounding_box_init_and_line_generation(self):
        box = YOLOBoundingBox(
            class_id=0,
            x_center=0.5,
            y_center=0.2,
            width=0.3,
            height=0.1,
        )
        assert box.class_name == "brand_logo"
        line = box.to_yolo_line()
        assert line == "0 0.500000 0.200000 0.300000 0.100000"

    def test_bounding_box_xyxy_conversion(self):
        box = YOLOBoundingBox(
            class_id=2,  # username_field
            x_center=0.5,
            y_center=0.5,
            width=0.4,
            height=0.2,
        )
        # In a 1000x500 image:
        # width = 1000, height = 500
        # bx = 0.5 * 1000 - 0.2 * 1000 = 300
        # bw = 0.4 * 1000 = 400 => xmax = 700
        # by = 0.5 * 500 - 0.1 * 500 = 200
        # bh = 0.2 * 500 = 100 => ymax = 300
        xmin, ymin, xmax, ymax = box.to_xyxy(1000, 500)
        assert xmin == 300
        assert ymin == 200
        assert xmax == 700
        assert ymax == 300

    def test_iou_computation(self):
        b1 = YOLOBoundingBox(0, 0.5, 0.5, 0.4, 0.4)
        b2 = YOLOBoundingBox(0, 0.5, 0.5, 0.4, 0.4)
        # Identical boxes => IoU = 1.0
        assert pytest.approx(b1.compute_iou(b2), 1e-4) == 1.0

        # Disjoint boxes => IoU = 0.0
        b3 = YOLOBoundingBox(0, 0.1, 0.1, 0.1, 0.1)
        assert b1.compute_iou(b3) == 0.0

        # Partially overlapping boxes
        # b1: [0.3, 0.3, 0.7, 0.7], area = 0.16
        # b4: [0.5, 0.5, 0.9, 0.9], area = 0.16
        # intersection: [0.5, 0.5, 0.7, 0.7], area = 0.04
        # union = 0.16 + 0.16 - 0.04 = 0.28
        # iou = 0.04 / 0.28 = 1/7 ~= 0.142857
        b4 = YOLOBoundingBox(0, 0.7, 0.7, 0.4, 0.4)
        assert pytest.approx(b1.compute_iou(b4), 1e-4) == (1.0 / 7.0)


class TestYOLOAnnotationValidator:
    """Verifies strict validation of YOLO text files, boundary checks, and empty images."""

    def test_valid_annotation_file(self, tmp_path: Path):
        f = tmp_path / "valid.txt"
        f.write_text(
            "0 0.150000 0.100000 0.200000 0.080000\n"
            "1 0.500000 0.400000 0.400000 0.500000\n"
            "2 0.500000 0.300000 0.300000 0.060000\n"
            "3 0.500000 0.400000 0.300000 0.060000\n"
            "4 0.500000 0.500000 0.250000 0.080000\n"
            "5 0.500000 0.700000 0.350000 0.120000\n",
            encoding="utf-8",
        )
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is True
        assert res.is_empty is False
        assert res.num_boxes == 6
        assert [b.class_name for b in res.boxes] == [
            "brand_logo",
            "login_form",
            "username_field",
            "password_field",
            "submit_button",
            "payment_element",
        ]

    def test_empty_annotation_handling(self, tmp_path: Path):
        """Empty 0-byte file or whitespace-only is valid for empty images."""
        f_empty = tmp_path / "empty.txt"
        f_empty.write_text("", encoding="utf-8")

        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f_empty)
        assert res.is_valid is True
        assert res.is_empty is True
        assert res.num_boxes == 0

    def test_comments_and_blank_lines(self, tmp_path: Path):
        f = tmp_path / "comments.txt"
        f.write_text(
            "# Human verified annotation\n\n"
            "0 0.2 0.2 0.1 0.1\n\n"
            "# End of file\n",
            encoding="utf-8",
        )
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is True
        assert res.num_boxes == 1

    def test_invalid_class_id_rejected(self, tmp_path: Path):
        f = tmp_path / "bad_class.txt"
        f.write_text("6 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # Class 6 does not exist (only 0-5)
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is False
        assert any("Invalid class_id" in err for err in res.errors)

    def test_negative_coordinates_rejected(self, tmp_path: Path):
        f = tmp_path / "neg_coord.txt"
        f.write_text("0 -0.1 0.5 0.2 0.2\n", encoding="utf-8")
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is False
        assert any("outside [0.0, 1.0]" in err for err in res.errors)

    def test_non_positive_dimension_rejected(self, tmp_path: Path):
        f = tmp_path / "bad_dim.txt"
        f.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is False
        assert any("Non-positive dimension" in err for err in res.errors)

    def test_duplicate_box_warning(self, tmp_path: Path):
        f = tmp_path / "dup.txt"
        f.write_text(
            "0 0.500000 0.500000 0.200000 0.200000\n"
            "0 0.500000 0.500000 0.200000 0.200000\n",
            encoding="utf-8",
        )
        validator = YOLOAnnotationValidator()
        res = validator.validate_yolo_file(f)
        assert res.is_valid is True
        assert len(res.warnings) > 0
        assert any("Near-duplicate boxes detected" in w for w in res.warnings)

    def test_image_pairing_check(self, tmp_path: Path, dummy_image: Path):
        f = tmp_path / "pair.txt"
        f.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        validator = YOLOAnnotationValidator()
        # Valid pair
        res_ok = validator.validate_yolo_file(f, image_path=dummy_image)
        assert res_ok.is_valid is True

        # Missing image pair
        res_missing = validator.validate_yolo_file(f, image_path=tmp_path / "non_existent.png")
        assert res_missing.is_valid is False
        assert any("not found on disk" in err for err in res_missing.errors)


class TestAnnotationManifestManager:
    """Verifies manifest generation, audit integrity, split consistency, and stats."""

    def test_actual_manifest_on_disk(self):
        csv_path = Path("data/annotations/detection/annotation_manifest.csv")
        json_path = Path("data/annotations/detection/annotation_manifest.json")
        assert csv_path.exists(), "Manifest CSV must exist"
        assert json_path.exists(), "Manifest JSON must exist"

        df = pd.read_csv(csv_path)
        assert len(df) == 1000, f"Expected exactly 1,000 images, got {len(df)}"

        # Verify columns
        required_cols = [
            "sample_id",
            "source_image_path",
            "domain",
            "dataset_label",
            "original_dataset_split",
            "annotation_status",
            "verification_status",
            "total_boxes",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col}"

        # Strict integrity: All images must initially be UNANNOTATED
        assert (df["annotation_status"] == "UNANNOTATED").all()
        assert (df["total_boxes"] == 0).all()
        assert (df["verification_status"] == "NONE").all()

        # Split counts verification
        split_counts = df["original_dataset_split"].value_counts().to_dict()
        assert split_counts.get("train") == 624
        assert split_counts.get("val") == 262
        assert split_counts.get("test") == 114

        # Label counts verification
        label_counts = df["dataset_label"].value_counts().to_dict()
        assert label_counts.get(1) == 800  # Phishing
        assert label_counts.get(0) == 200  # Legitimate

    def test_compute_dataset_statistics(self):
        mgr = YOLOAnnotationManifestManager()
        stats = mgr.compute_dataset_statistics()
        assert stats["total_images"] == 1000
        assert stats["status_counts"]["UNANNOTATED"] == 1000
        assert stats["total_bounding_boxes"] == 0
        assert stats["split_summary"]["train"]["total_images"] == 624
        assert stats["split_summary"]["val"]["total_images"] == 262
        assert stats["split_summary"]["test"]["total_images"] == 114
