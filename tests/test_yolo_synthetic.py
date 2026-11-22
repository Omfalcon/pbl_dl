"""Unit Tests for Synthetic Dataset Generator & YOLOv8 Pipeline (Phase 7.2-7.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from app.ml.evaluate_yolov8 import load_config, run_evaluation
from app.ml.vision.synthetic_generator import (
    SPLIT_BRANDS,
    SyntheticPageLayout,
    SyntheticWebpageGenerator,
)
from app.ml.vision.yolo_annotations import (
    YOLO_DETECTION_CLASSES,
    YOLOAnnotationValidator,
)


class TestSyntheticWebpageGenerator:
    """Verifies synthetic generation, bounding box correctness, and class mapping."""

    def test_generator_all_templates(self):
        gen = SyntheticWebpageGenerator(seed=123)
        templates = [
            "standard_login",
            "banking_login",
            "social_login",
            "ecommerce_login",
            "payment_checkout",
            "empty_page",
        ]

        for tmpl in templates:
            layout = gen.generate_page(brand_name="TestBrand", template_type=tmpl, width=1280, height=800)
            assert isinstance(layout, SyntheticPageLayout)
            assert layout.image.size == (1280, 800)

            if tmpl == "empty_page":
                assert layout.is_empty is True
                assert len(layout.boxes) == 0
            else:
                assert layout.is_empty is False
                assert len(layout.boxes) >= 3

                # Validate every box geometry
                for b in layout.boxes:
                    assert b.class_id in YOLO_DETECTION_CLASSES
                    assert 0.0 <= b.x_center <= 1.0
                    assert 0.0 <= b.y_center <= 1.0
                    assert 0.0 < b.width <= 1.0
                    assert 0.0 < b.height <= 1.0

                    # Canvas containment
                    x1 = b.x_center - b.width / 2.0
                    y1 = b.y_center - b.height / 2.0
                    x2 = b.x_center + b.width / 2.0
                    y2 = b.y_center + b.height / 2.0
                    assert x1 >= -0.01 and x2 <= 1.01
                    assert y1 >= -0.01 and y2 <= 1.01

    def test_split_brand_isolation(self):
        """Strictly verifies that brand sets across train, val, and test have ZERO overlap."""
        train_brands = set(SPLIT_BRANDS["train"])
        val_brands = set(SPLIT_BRANDS["val"])
        test_brands = set(SPLIT_BRANDS["test"])

        assert len(train_brands & val_brands) == 0, "Train and Val brands must have zero overlap"
        assert len(train_brands & test_brands) == 0, "Train and Test brands must have zero overlap"
        assert len(val_brands & test_brands) == 0, "Val and Test brands must have zero overlap"


class TestSyntheticDatasetOnDisk:
    """Verifies that generated synthetic dataset on disk conforms to specifications."""

    def test_dataset_yaml_validity(self):
        yaml_p = Path("data/synthetic_yolo/dataset.yaml")
        assert yaml_p.exists(), "dataset.yaml must exist on disk"

        with open(yaml_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "names" in cfg
        assert len(cfg["names"]) == 6
        assert cfg["names"][0] == "brand_logo"
        assert cfg["names"][1] == "login_form"
        assert cfg["names"][2] == "username_field"
        assert cfg["names"][3] == "password_field"
        assert cfg["names"][4] == "submit_button"
        assert cfg["names"][5] == "payment_element"

    def test_file_counts_and_splits(self):
        base_p = Path("data/synthetic_yolo")
        assert base_p.exists()

        # Check image files count
        train_imgs = list((base_p / "images" / "train").glob("*.png"))
        val_imgs = list((base_p / "images" / "val").glob("*.png"))
        test_imgs = list((base_p / "images" / "test").glob("*.png"))

        assert len(train_imgs) == 560, f"Expected 560 train images, got {len(train_imgs)}"
        assert len(val_imgs) == 120, f"Expected 120 val images, got {len(val_imgs)}"
        assert len(test_imgs) == 120, f"Expected 120 test images, got {len(test_imgs)}"

        # Check label files count matches images exactly
        train_lbls = list((base_p / "labels" / "train").glob("*.txt"))
        val_lbls = list((base_p / "labels" / "val").glob("*.txt"))
        test_lbls = list((base_p / "labels" / "test").glob("*.txt"))

        assert len(train_lbls) == 560
        assert len(val_lbls) == 120
        assert len(test_lbls) == 120

    def test_manifest_and_statistics(self):
        manifest_p = Path("data/synthetic_yolo/manifest.csv")
        stats_p = Path("data/synthetic_yolo/dataset_statistics.json")

        assert manifest_p.exists()
        assert stats_p.exists()

        df = pd.read_csv(manifest_p)
        assert len(df) == 800

        with open(stats_p, "r", encoding="utf-8") as f:
            stats = json.load(f)

        assert stats["total_images"] == 800
        assert stats["split_counts"]["train"] == 560
        assert stats["split_counts"]["val"] == 120
        assert stats["split_counts"]["test"] == 120

    def test_annotation_validator_on_synthetic_samples(self):
        """Run YOLOAnnotationValidator on randomly sampled generated label files."""
        validator = YOLOAnnotationValidator()
        base_p = Path("data/synthetic_yolo")

        for split in ["train", "val", "test"]:
            lbl_dir = base_p / "labels" / split
            img_dir = base_p / "images" / split

            for sample_txt in list(lbl_dir.glob("*.txt"))[:15]:
                paired_img = img_dir / f"{sample_txt.stem}.png"
                res = validator.validate_yolo_file(sample_txt, image_path=paired_img)
                assert res.is_valid is True, f"Validation failed for {sample_txt.name}: {res.errors}"


class TestYOLOv8PipelineConfig:
    """Verifies training and evaluation configurations."""

    def test_training_config_loading(self):
        cfg = load_config("configs/yolov8n_synthetic.yaml")
        assert cfg["model"]["model_id"] == "M-04"
        assert cfg["model"]["architecture"] == "YOLOv8n"
        assert cfg["training"]["batch_size"] == 4
        assert cfg["data"]["imgsz"] == 512
        assert cfg["training"]["random_seed"] == 42

    def test_evaluation_preflight_report(self, tmp_path: Path):
        """Verifies that evaluation script safely outputs pre-flight status if checkpoint does not exist."""
        res = run_evaluation(
            checkpoint_path=tmp_path / "non_existent_best.pt",
            config_path="configs/yolov8n_synthetic.yaml",
            split="test",
            output_dir=tmp_path / "reports",
        )
        assert res["status"] == "PENDING_TRAINING_EXECUTION"
        assert (tmp_path / "reports" / "metrics.json").exists()
        assert (tmp_path / "reports" / "YOLO_EXPERIMENT_REPORT.md").exists()
