"""Unit tests for Multi-Region Screenshot Coverage Analysis (Phase 6.1.2).

Verifies vertical crop coordinate math, aspect ratio preservation, diagnostic views generation,
inference output shape, report files, and contact sheet generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from app.ml.vision.models import ResNet18ScreenshotClassifier
from app.ml.vision.multi_region import (
    compute_vertical_crop_coordinates,
    evaluate_image_tensor,
    extract_diagnostic_views,
)


class TestMultiRegionCoverageAnalysis:
    """Diagnostic test suite for multi-region coverage analysis."""

    def test_crop_coordinate_correctness_tall_image(self):
        """Verifies 5 sliding windows on a tall website screenshot (1280 x 3600)."""
        w, h = 1280, 3600
        coords = compute_vertical_crop_coordinates(w, h, viewport_ratio=1.0)

        assert "top" in coords
        assert "upper_middle" in coords
        assert "middle" in coords
        assert "lower_middle" in coords
        assert "bottom" in coords

        win_h = 1280
        max_y = h - win_h  # 2320

        # Check exact positions
        assert coords["top"] == (0, 0, 1280, 1280)
        assert coords["bottom"] == (0, max_y, 1280, max_y + win_h)

        # Check ordering and bounds
        last_y0 = -1
        for region, (x0, y0, x1, y1) in coords.items():
            assert x0 == 0
            assert x1 == 1280
            assert y0 >= 0
            assert y1 <= h
            assert y1 - y0 == win_h, f"Window height must be exactly {win_h} for {region}"
            assert y0 >= last_y0, "Crop positions must advance monotonically downwards"
            last_y0 = y0

    def test_crop_coordinate_correctness_short_image(self):
        """Verifies boundary handling when screenshot is shorter than viewport ratio."""
        w, h = 1280, 800
        coords = compute_vertical_crop_coordinates(w, h, viewport_ratio=1.0)

        # All regions must cleanly clamp to the full available image
        for region, box in coords.items():
            assert box == (0, 0, 1280, 800), f"Region {region} did not clamp correctly"

    def test_extract_diagnostic_views_aspect_ratio_preservation(self):
        """Verifies diagnostic views extraction produces all required representations without mutation."""
        synth_img = Image.new("RGB", (1000, 2500), color=(100, 150, 200))
        views = extract_diagnostic_views(synth_img, viewport_ratio=1.0, target_size=224)

        assert "original" in views
        assert "crops" in views
        assert "full_page" in views
        assert "crop_coordinates" in views

        # Check all 5 crops exist
        crops = views["crops"]
        assert len(crops) == 5
        for reg in ["top", "upper_middle", "middle", "lower_middle", "bottom"]:
            assert reg in crops
            assert isinstance(crops[reg], Image.Image)
            assert crops[reg].size == (1000, 1000)

        # Check full-page padded image is square 224x224
        fp_img = views["full_page"]
        assert fp_img.size == (224, 224)

        # Original image must remain unaltered
        assert synth_img.size == (1000, 2500)

    def test_evaluate_image_tensor_output_shape_and_probability(self):
        """Verifies inference forward pass outputs valid probabilities and labels."""
        model = ResNet18ScreenshotClassifier(pretrained=False)
        model.eval()

        dummy_img = Image.new("RGB", (224, 224), color=(50, 100, 150))
        result = evaluate_image_tensor(model, dummy_img, target_size=224)

        assert "pred_label" in result
        assert result["pred_label"] in (0, 1)
        assert "phishing_probability" in result
        assert "legitimate_probability" in result

        p_phish = result["phishing_probability"]
        p_legit = result["legitimate_probability"]
        assert 0.0 <= p_phish <= 1.0
        assert 0.0 <= p_legit <= 1.0
        assert np.isclose(p_phish + p_legit, 1.0, atol=1e-3)

    def test_multi_region_artifacts_existence_and_integrity(self):
        """Verifies all output artifacts are generated under data/reports/resnet18_error_analysis/multi_region/."""
        out_dir = Path("data/reports/resnet18_error_analysis/multi_region")
        assert out_dir.exists(), f"Output directory missing: {out_dir}"

        csv_file = out_dir / "multi_region_predictions.csv"
        json_file = out_dir / "multi_region_summary.json"
        sheet_file = out_dir / "multi_region_contact_sheet.png"
        md_file = out_dir / "MULTI_REGION_ANALYSIS.md"

        assert csv_file.exists() and csv_file.stat().st_size > 0
        assert json_file.exists() and json_file.stat().st_size > 0
        assert sheet_file.exists() and sheet_file.stat().st_size > 0
        assert md_file.exists() and md_file.stat().st_size > 0

        # Check CSV structure
        df = pd.read_csv(csv_file)
        assert len(df) == 43  # 39 FN + 4 FP
        assert "sample_id" in df.columns
        assert "top_crop_p_phish" in df.columns
        assert "middle_crop_p_phish" in df.columns
        assert "bottom_crop_p_phish" in df.columns
        assert "full_page_pad_p_phish" in df.columns
        assert "max_regional_p_phish" in df.columns

        # Check JSON schema
        with open(json_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_analyzed"] == 43
        assert summary["false_negatives"]["total_fn"] == 39
        assert summary["false_positives"]["total_fp"] == 4
        assert summary["false_negatives"]["tall_pages_aspect_ge_1_5"] == 29

        # Check contact sheet image
        with Image.open(sheet_file) as sheet_img:
            assert sheet_img.format == "PNG"
            assert sheet_img.size[0] > 1000 and sheet_img.size[1] > 1000
