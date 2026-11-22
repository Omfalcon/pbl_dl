"""Unit Tests for Milestone 2C Annotation Validation & Visual Overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.datasets.generate_annotations import AnnotationManager
from scripts.datasets.validate_annotations import AnnotationValidator, BoundingBox
from scripts.datasets.visualize_annotations import AnnotationVisualizer


@pytest.fixture
def sample_canvas(tmp_path: Path):
    """Creates a synthetic image for bounding box rendering tests."""
    img_path = tmp_path / "canvas.png"
    arr = np.full((300, 400, 3), 200, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


class TestAnnotationValidator:
    """Verifies strict syntax and numerical boundary enforcement for YOLO & segmentation masks."""

    def test_valid_yolo_file(self, tmp_path: Path):
        txt_path = tmp_path / "valid.txt"
        txt_path.write_text(
            "0 0.150000 0.100000 0.200000 0.080000\n"
            "1 0.500000 0.400000 0.400000 0.060000\n"
            "2 0.500000 0.500000 0.400000 0.060000\n"
            "3 0.500000 0.600000 0.300000 0.080000\n",
            encoding="utf-8",
        )
        validator = AnnotationValidator()
        res = validator.validate_yolo_file(txt_path)
        assert res.is_valid is True
        assert res.num_boxes == 4
        assert res.boxes[0].class_name == "logo"
        assert res.boxes[1].class_name == "username_input"
        assert res.boxes[2].class_name == "password_input"
        assert res.boxes[3].class_name == "submit_button"

    def test_invalid_class_id_rejected(self, tmp_path: Path):
        txt_path = tmp_path / "bad_class.txt"
        txt_path.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        validator = AnnotationValidator()
        res = validator.validate_yolo_file(txt_path)
        assert res.is_valid is False
        assert any("Invalid class_id" in e for e in res.errors)

    def test_out_of_bounds_coordinate_rejected(self, tmp_path: Path):
        txt_path = tmp_path / "bad_coord.txt"
        txt_path.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
        validator = AnnotationValidator()
        res = validator.validate_yolo_file(txt_path)
        assert res.is_valid is False
        assert any("out of bounds" in e for e in res.errors)

    def test_non_positive_dimension_rejected(self, tmp_path: Path):
        txt_path = tmp_path / "bad_dim.txt"
        txt_path.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
        validator = AnnotationValidator()
        res = validator.validate_yolo_file(txt_path)
        assert res.is_valid is False
        assert any("Non-positive dimension" in e for e in res.errors)

    def test_segmentation_mask_valid(self, tmp_path: Path):
        mask_path = tmp_path / "mask.png"
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[:20, :] = 1  # Header
        mask_arr[20:80, :] = 4  # Main content
        mask_arr[80:, :] = 5  # Footer
        Image.fromarray(mask_arr).save(mask_path)

        validator = AnnotationValidator()
        is_val, errs = validator.validate_segmentation_mask(mask_path, expected_shape=(100, 100))
        assert is_val is True
        assert len(errs) == 0

    def test_segmentation_mask_invalid_classes(self, tmp_path: Path):
        mask_path = tmp_path / "mask_bad.png"
        mask_arr = np.zeros((50, 50), dtype=np.uint8)
        mask_arr[10:20, 10:20] = 42  # Class 42 does not exist
        Image.fromarray(mask_arr).save(mask_path)

        validator = AnnotationValidator()
        is_val, errs = validator.validate_segmentation_mask(mask_path)
        assert is_val is False
        assert any("invalid class pixel values" in e for e in errs)


class TestAnnotationVisualizer:
    """Verifies overlay rendering and image generation."""

    def test_render_detection_overlay(self, sample_canvas, tmp_path):
        boxes = [
            BoundingBox(class_id=0, x_center=0.2, y_center=0.2, width=0.1, height=0.1),
            BoundingBox(class_id=2, x_center=0.5, y_center=0.5, width=0.3, height=0.1),
        ]
        out_preview = tmp_path / "preview.png"
        ok = AnnotationVisualizer.render_detection_overlay(sample_canvas, boxes, out_preview)
        assert ok is True
        assert out_preview.exists()
        assert out_preview.stat().st_size > 0

    def test_annotation_manager_lifecycle(self, sample_canvas, tmp_path):
        mgr = AnnotationManager(annotations_dir=tmp_path / "annotations")
        boxes = [BoundingBox(class_id=1, x_center=0.5, y_center=0.4, width=0.3, height=0.08)]

        txt_p = mgr.save_yolo_annotation("canvas.png", boxes, is_verified=False)
        assert txt_p.exists()

        preview_p = mgr.generate_preview(sample_canvas, txt_p)
        assert preview_p is not None
        assert preview_p.exists()
