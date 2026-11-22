"""Unit & Regression Tests for YOLOv8-Seg Instance Segmentation (Phase 7.7).

Verifies 6-class schema, geometric polygon generation, YOLO segmentation string
formatting, mask rasterization, synthetic dataset structure, preview rendering,
and dual Box/Mask evaluation report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.train_yolo_seg import extract_evaluation_metrics
from app.ml.vision.yolo_seg import (
    CLASS_NAME_TO_ID,
    SPLIT_BRANDS,
    YOLO_SEG_CLASSES,
    SyntheticSegmentationGenerator,
    YOLOSegInferencer,
    YOLOSegmentationPolygon,
    diamond_polygon,
    ellipse_polygon,
    generate_segmentation_evaluation_report,
    rounded_rect_polygon,
    triangle_polygon,
)


class TestYOLOSegSchema:
    """Verifies that the segmentation class schema matches Phase 7 M-04 and M-05 exactly."""

    def test_six_class_schema(self) -> None:
        """Asserts exactly 6 classes matching M-04 YOLOv8n and M-05 Faster R-CNN."""
        assert len(YOLO_SEG_CLASSES) == 6
        expected_classes = [
            "brand_logo",
            "login_form",
            "username_field",
            "password_field",
            "submit_button",
            "payment_element",
        ]
        for cid, name in enumerate(expected_classes):
            assert YOLO_SEG_CLASSES[cid] == name
            assert CLASS_NAME_TO_ID[name] == cid


class TestGeometricPolygonCalculations:
    """Verifies programmatic generation of exact geometric polygon coordinates."""

    def test_rounded_rect_polygon(self) -> None:
        """Tests that rounded rectangle produces exact 16-point polygon within bounds."""
        x1, y1, x2, y2 = 50.0, 50.0, 250.0, 150.0
        pts = rounded_rect_polygon(x1, y1, x2, y2, radius=10.0, n_arc_points=4)
        assert len(pts) == 16
        for px, py in pts:
            assert x1 <= px <= x2
            assert y1 <= py <= y2

    def test_ellipse_polygon(self) -> None:
        """Tests that ellipse polygon produces requested vertex count."""
        cx, cy, rx, ry = 100.0, 100.0, 20.0, 20.0
        pts = ellipse_polygon(cx, cy, rx, ry, n_points=16)
        assert len(pts) == 16
        for px, py in pts:
            dist = np.hypot(px - cx, py - cy)
            assert dist == pytest.approx(20.0, rel=1e-3)

    def test_diamond_and_triangle_polygons(self) -> None:
        """Tests vertices for diamond and triangle geometric shapes."""
        diamond_pts = diamond_polygon(0.0, 0.0, 100.0, 100.0)
        assert len(diamond_pts) == 4
        assert diamond_pts[0] == (50.0, 0.0)
        assert diamond_pts[1] == (100.0, 50.0)

        triangle_pts = triangle_polygon(0.0, 0.0, 100.0, 100.0)
        assert len(triangle_pts) == 3
        assert triangle_pts[0] == (50.0, 0.0)


class TestYOLOSegmentationPolygon:
    """Verifies YOLO segmentation data structures, bounding box extraction, and mask rasterization."""

    def test_to_yolo_seg_line(self) -> None:
        """Asserts correct YOLO segmentation label line format: <class_id> x1 y1 x2 y2 ..."""
        pts = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
        poly = YOLOSegmentationPolygon(class_id=1, points=pts)
        line = poly.to_yolo_seg_line()
        parts = line.split()
        assert int(parts[0]) == 1
        assert len(parts) == 1 + 2 * len(pts)

    def test_to_bounding_box(self) -> None:
        """Asserts bounding box enclosing the polygon vertices."""
        pts = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
        poly = YOLOSegmentationPolygon(class_id=0, points=pts)
        bx1, by1, bx2, by2 = poly.to_bounding_box(img_width=1000, img_height=1000)
        assert bx1 == 100
        assert by1 == 200
        assert bx2 == 300
        assert by2 == 400

    def test_rasterize_mask(self) -> None:
        """Asserts binary mask rasterization produces valid uint8 mask with non-zero pixels."""
        pts = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
        poly = YOLOSegmentationPolygon(class_id=2, points=pts)
        mask = poly.rasterize_mask(img_width=100, img_height=100)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert mask.sum() > 0
        assert mask[40, 40] == 1
        assert mask[5, 5] == 0


class TestSyntheticSegmentationGenerator:
    """Verifies synthetic dataset generation and split brand isolation."""

    @pytest.fixture
    def generator(self) -> SyntheticSegmentationGenerator:
        return SyntheticSegmentationGenerator(seed=42)

    def test_generate_page_login(self, generator: SyntheticSegmentationGenerator) -> None:
        """Asserts generated standard login layout contains valid instance polygons."""
        layout = generator.generate_page(brand_name="TestBank", template_type="standard_login", width=512, height=512)
        assert not layout.is_empty
        assert len(layout.polygons) >= 4

        class_names = [p.class_name for p in layout.polygons]
        assert "brand_logo" in class_names
        assert "login_form" in class_names
        assert "username_field" in class_names
        assert "password_field" in class_names
        assert "submit_button" in class_names

        for p in layout.polygons:
            assert len(p.points) >= 3
            for x, y in p.points:
                assert 0.0 <= x <= 1.0
                assert 0.0 <= y <= 1.0

    def test_generate_empty_page(self, generator: SyntheticSegmentationGenerator) -> None:
        """Asserts 404 empty page produces exactly zero target polygons."""
        layout = generator.generate_page(brand_name="None", template_type="empty_page", width=512, height=512)
        assert layout.is_empty
        assert len(layout.polygons) == 0

    def test_zero_brand_leakage_between_splits(self) -> None:
        """Asserts disjoint brand dictionaries across train, val, and test splits."""
        train_brands = set(SPLIT_BRANDS["train"])
        val_brands = set(SPLIT_BRANDS["val"])
        test_brands = set(SPLIT_BRANDS["test"])

        assert len(train_brands.intersection(val_brands)) == 0
        assert len(train_brands.intersection(test_brands)) == 0
        assert len(val_brands.intersection(test_brands)) == 0


class TestYOLOSegInferencerAndPreview:
    """Verifies model loading, inference predictions, and semi-transparent mask preview rendering."""

    def test_inferencer_predict_and_preview(self, tmp_path: Path) -> None:
        """Tests that YOLOSegInferencer runs inference and saves annotated mask overlays."""
        weights_path = Path("yolov8n-seg.pt")
        if not weights_path.exists():
            pytest.skip("yolov8n-seg.pt weights not found locally.")

        inferencer = YOLOSegInferencer(model_weights=weights_path, device="cpu", conf_threshold=0.01)
        dummy_img = Image.new("RGB", (512, 512), color=(240, 240, 240))

        res = inferencer.predict(dummy_img)
        assert "image_size" in res
        assert res["image_size"] == [512, 512]
        assert "instances" in res
        assert isinstance(res["instances"], list)

        out_preview = tmp_path / "test_seg_preview.png"
        preview_path = inferencer.render_annotated_preview(dummy_img, output_path=out_preview)
        assert preview_path.exists()
        assert preview_path.stat().st_size > 0


class TestEvaluationReportGeneration:
    """Verifies that the dual Box/Mask evaluation report preserves metric separation."""

    def test_generate_segmentation_evaluation_report(self, tmp_path: Path) -> None:
        """Asserts that dual detection and segmentation metrics are properly written to JSON and Markdown."""
        mock_eval = {
            "status": "EVALUATION_COMPLETE",
            "model_id": "M-06",
            "architecture": "YOLOv8n-seg",
            "box_mAP50": 0.992,
            "box_mAP50_95": 0.955,
            "box_precision": 0.990,
            "box_recall": 0.995,
            "mask_mAP50": 0.985,
            "mask_mAP50_95": 0.940,
            "mask_precision": 0.982,
            "mask_recall": 0.990,
            "latency_ms": 11.5,
            "checkpoint_size_mb": 6.7,
            "class_metrics": {
                "0": {
                    "class_id": 0,
                    "class_name": "brand_logo",
                    "box_mAP50": 0.995,
                    "box_mAP50_95": 0.860,
                    "mask_mAP50": 0.988,
                    "mask_mAP50_95": 0.850,
                    "mask_precision": 0.990,
                    "mask_recall": 1.0,
                }
            },
        }

        generate_segmentation_evaluation_report(
            eval_metrics=mock_eval,
            output_dir=tmp_path,
            checkpoint_path="models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt",
        )

        metrics_json = tmp_path / "metrics.json"
        class_csv = tmp_path / "class_metrics.csv"
        report_md = tmp_path / "YOLO_SEG_EXPERIMENT_REPORT.md"

        assert metrics_json.exists()
        assert class_csv.exists()
        assert report_md.exists()

        with open(metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["box_mAP50"] == 0.992
            assert data["mask_mAP50"] == 0.985

    def test_extract_evaluation_metrics(self) -> None:
        """Asserts that extract_evaluation_metrics properly unpacks Box and Mask per-class metrics."""
        from ultralytics.utils.metrics import Metric

        class MockValResults:
            def __init__(self) -> None:
                self.box = Metric()
                self.box.nc = 6
                self.seg = Metric()
                self.seg.nc = 6

                p = np.array([0.90, 0.85, 0.92, 0.94, 0.88, 0.80])
                r = np.array([0.95, 0.88, 0.90, 0.91, 0.85, 0.82])
                f1 = 2 * p * r / (p + r)
                all_ap = np.ones((6, 10)) * 0.88
                all_ap[:, 0] = 0.95
                ap_class_index = np.array([0, 1, 2, 3, 4, 5])

                self.box.update((p, r, f1, all_ap, ap_class_index, None, None, None, None, None))
                self.seg.update((p * 0.95, r * 0.95, f1 * 0.95, all_ap * 0.92, ap_class_index, None, None, None, None, None))

        mock_val = MockValResults()
        eval_data = extract_evaluation_metrics(mock_val, "models/fake/best.pt", split_name="test")

        assert eval_data["status"] == "EVALUATION_COMPLETE"
        assert eval_data["model_id"] == "M-06"
        assert eval_data["split"] == "test"
        assert "class_metrics" in eval_data
        assert len(eval_data["class_metrics"]) == 6

        # Check brand_logo (cid 0)
        c0 = eval_data["class_metrics"]["0"]
        assert c0["class_name"] == "brand_logo"
        assert c0["box_precision"] == 0.90
        assert c0["box_recall"] == 0.95
        assert c0["box_mAP50"] == 0.95
        assert c0["mask_precision"] == round(0.90 * 0.95, 4)
        assert c0["mask_recall"] == round(0.95 * 0.95, 4)
