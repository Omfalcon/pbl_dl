"""Unit tests for M-02 ResNet-18 Validation Error Analysis Pipeline.

Verifies checkpoint loading, validation split integrity, class mapping, probability calculations,
error identification (FN=39, FP=4), artifact exports, and that original screenshots remain unaltered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.vision.error_analysis import (
    analyze_image_characteristics,
    run_validation_error_analysis,
)
from app.ml.vision.inference import ResNet18Inference
from app.ml.vision.models import CLASS_TO_IDX, IDX_TO_CLASS


class TestResNet18ErrorAnalysis:
    """Diagnostic test suite for validation error analysis without retraining."""

    @pytest.fixture(scope="class")
    def checkpoint_path(self) -> Path:
        path = Path("models/checkpoints/resnet18/best_model.pt")
        assert path.exists(), f"Best checkpoint not found at: {path}"
        return path

    @pytest.fixture(scope="class")
    def val_parquet_path(self) -> Path:
        path = Path("data/splits/visual/val.parquet")
        assert path.exists(), f"Validation split not found at: {path}"
        return path

    @pytest.fixture(scope="class")
    def report_dir(self) -> Path:
        return Path("data/reports/resnet18_error_analysis")

    def test_checkpoint_loading(self, checkpoint_path: Path):
        """1. Verifies checkpoint loads onto evaluation mode without modifying weights."""
        infer = ResNet18Inference(checkpoint_path=checkpoint_path)
        assert infer.model is not None
        assert not infer.model.training  # In evaluation mode
        assert infer.checkpoint_metadata.get("epoch") == 9
        assert "val_macro_f1" in infer.checkpoint_metadata
        assert infer.crop_strategy == "top_viewport"
        assert infer.image_size == 224

    def test_validation_only_dataset_loading(self, val_parquet_path: Path):
        """2. Verifies validation partition exists, has 262 samples, and avoids test split."""
        df = pd.read_parquet(val_parquet_path)
        assert len(df) == 262
        assert "sample_id" in df.columns
        assert "label" in df.columns
        assert set(df["label"].unique()) == {0, 1}
        # Phishing count is 236, legitimate count is 26
        assert (df["label"] == 1).sum() == 236
        assert (df["label"] == 0).sum() == 26

    def test_correct_class_mapping(self, checkpoint_path: Path):
        """3. Verifies strict class mapping 0 = legitimate, 1 = phishing."""
        infer = ResNet18Inference(checkpoint_path=checkpoint_path)
        c2i = infer.checkpoint_metadata.get("class_to_idx", CLASS_TO_IDX)
        i2c = infer.checkpoint_metadata.get("idx_to_class", IDX_TO_CLASS)

        assert c2i["legitimate"] == 0
        assert c2i["phishing"] == 1
        assert i2c[0] == "legitimate"
        assert i2c[1] == "phishing"

    def test_probability_calculation_and_finiteness(self, checkpoint_path: Path):
        """4. Verifies calibrated output probabilities sum to 1.0 and are finite."""
        infer = ResNet18Inference(checkpoint_path=checkpoint_path)
        # Use synthetic dummy image
        from PIL import Image
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        pred = infer.predict(dummy_img)

        assert "probabilities" in pred
        p_legit = pred["probabilities"]["legitimate"]
        p_phish = pred["probabilities"]["phishing"]
        assert 0.0 <= p_legit <= 1.0
        assert 0.0 <= p_phish <= 1.0
        assert np.isclose(p_legit + p_phish, 1.0, atol=1e-3)
        assert pred["confidence"] == max(p_legit, p_phish)

    def test_false_positive_and_false_negative_identification(self, report_dir: Path):
        """5. Verifies error identification matches confusion matrix (FN=39, FP=4, TN=22, TP=197)."""
        summary_path = report_dir / "summary.json"
        assert summary_path.exists(), f"Summary JSON missing: {summary_path}"

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_validation_samples"] == 262
        assert summary["false_negatives"] == 39
        assert summary["false_positives"] == 4
        assert summary["correct_predictions"] == 219

        cm = summary["confusion_matrix"]
        assert cm["true_negative"] == 22
        assert cm["false_positive"] == 4
        assert cm["false_negative"] == 39
        assert cm["true_positive"] == 197

    def test_csv_export_structure(self, report_dir: Path):
        """6. Verifies CSV export columns, row count (43 errors), and sort order."""
        csv_path = report_dir / "error_analysis.csv"
        assert csv_path.exists(), f"Error analysis CSV missing: {csv_path}"

        df = pd.read_csv(csv_path)
        assert len(df) == 43  # 39 FN + 4 FP

        required_cols = [
            "sample_id",
            "image_path",
            "actual_label",
            "predicted_label",
            "legitimate_probability",
            "phishing_probability",
            "confidence",
            "error_type",
            "registrable_domain",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Verify False Negatives are sorted by phishing_probability ascending
        df_fn = df[df["error_type"] == "false_negative"]
        assert len(df_fn) == 39
        p_phish_values = df_fn["phishing_probability"].tolist()
        assert p_phish_values == sorted(p_phish_values), "False negatives must be sorted by phishing probability"

        df_fp = df[df["error_type"] == "false_positive"]
        assert len(df_fp) == 4

    def test_image_export_and_counts(self, report_dir: Path):
        """7. Verifies misclassified images are exported with meaningful names."""
        fn_dir = report_dir / "false_negatives"
        fp_dir = report_dir / "false_positives"

        assert fn_dir.exists() and fp_dir.exists()

        fn_files = list(fn_dir.glob("*.*"))
        fp_files = list(fp_dir.glob("*.*"))

        assert len(fn_files) == 39
        assert len(fp_files) == 4

        # Verify filename convention contains sample_id and probability
        for f in fn_files:
            assert f.name.startswith("FN_ext_vis_")
            assert "phish_" in f.name
        for f in fp_files:
            assert f.name.startswith("FP_ext_vis_")
            assert "phish_" in f.name

    def test_no_modification_of_original_screenshots(self, report_dir: Path):
        """8. Verifies that copying into error analysis folders did not modify original screenshots."""
        csv_path = report_dir / "error_analysis.csv"
        df = pd.read_csv(csv_path)

        # Check hash match for a sample of exported images
        for _, row in df.head(5).iterrows():
            orig_path = Path(row["image_path"])
            assert orig_path.exists(), f"Original file missing: {orig_path}"

            err_type = row["error_type"]
            folder = "false_negatives" if err_type == "false_negative" else "false_positives"
            prefix = "FN" if err_type == "false_negative" else "FP"
            exported_name = f"{prefix}_{row['sample_id']}_phish_{row['phishing_probability']:.4f}{orig_path.suffix}"
            exported_path = report_dir / folder / exported_name

            assert exported_path.exists(), f"Exported file missing: {exported_path}"

            # Calculate SHA256 of both
            with open(orig_path, "rb") as f1, open(exported_path, "rb") as f2:
                h_orig = hashlib.sha256(f1.read()).hexdigest()
                h_exp = hashlib.sha256(f2.read()).hexdigest()

            assert h_orig == h_exp, f"Original screenshot modified! Hash mismatch for {orig_path}"

    def test_contact_sheets_generation_and_validity(self, report_dir: Path):
        """9. Verifies visual contact sheets are created, non-empty, and valid images."""
        cs_dir = report_dir / "contact_sheets"
        assert cs_dir.exists(), f"Contact sheets directory missing: {cs_dir}"

        fn_sheet = cs_dir / "false_negatives_contact_sheet.png"
        fp_sheet = cs_dir / "false_positives_contact_sheet.png"

        assert fn_sheet.exists() and fn_sheet.stat().st_size > 0
        assert fp_sheet.exists() and fp_sheet.stat().st_size > 0

        # Load with PIL to verify integrity and dimensions
        from PIL import Image
        with Image.open(fn_sheet) as img_fn:
            assert img_fn.format == "PNG"
            assert img_fn.size == (1612, 3270)  # 5 cols x 8 rows
        with Image.open(fp_sheet) as img_fp:
            assert img_fp.format == "PNG"
            assert img_fp.size == (1296, 498)  # 4 cols x 1 row

    def test_contact_sheet_index_mapping_csv(self, report_dir: Path):
        """10. Verifies contact_sheet_index.csv maps all 43 misclassified screenshots."""
        index_csv = report_dir / "contact_sheets" / "contact_sheet_index.csv"
        assert index_csv.exists(), f"Contact sheet index CSV missing: {index_csv}"

        df_idx = pd.read_csv(index_csv)
        assert len(df_idx) == 43  # 39 FN + 4 FP

        required_cols = [
            "contact_sheet",
            "grid_index",
            "grid_row",
            "grid_col",
            "sample_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "phishing_probability",
            "legitimate_probability",
            "registrable_domain",
            "original_screenshot_path",
            "card_bounding_box",
            "thumbnail_bounding_box",
        ]
        for col in required_cols:
            assert col in df_idx.columns, f"Missing index column: {col}"

        fn_rows = df_idx[df_idx["contact_sheet"] == "false_negatives_contact_sheet.png"]
        fp_rows = df_idx[df_idx["contact_sheet"] == "false_positives_contact_sheet.png"]
        assert len(fn_rows) == 39
        assert len(fp_rows) == 4

        # Verify all original screenshot paths in index exist
        for orig_p in df_idx["original_screenshot_path"]:
            assert Path(orig_p).exists(), f"Referenced original file does not exist: {orig_p}"
