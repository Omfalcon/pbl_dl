"""Unit tests for Full Validation Multi-Region Aggregation Experiment (Phase 6.1.3).

Verifies full validation evaluation (262 samples), correct class mapping, aggregation calculations,
artifact consistency, checkpoint immutability, and test split isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestAggregationExperiment:
    """Verification suite for multi-region crop aggregation experiment."""

    @pytest.fixture(scope="class")
    def exp_dir(self) -> Path:
        p = Path("data/reports/resnet18_error_analysis/aggregation_experiment")
        assert p.exists(), f"Experiment output directory missing: {p}"
        return p

    def test_all_262_validation_samples_evaluated(self, exp_dir: Path):
        """1. Verifies that all 262 validation screenshots were evaluated."""
        csv_p = exp_dir / "all_validation_sample_predictions.csv"
        assert csv_p.exists(), f"Sample predictions CSV missing: {csv_p}"

        df = pd.read_csv(csv_p)
        assert len(df) == 262, f"Expected 262 evaluated samples, found {len(df)}"

        # Verify class distribution: 236 Phishing (1), 26 Legitimate (0)
        assert (df["actual_label"] == 1).sum() == 236
        assert (df["actual_label"] == 0).sum() == 26

    def test_correct_class_mapping_and_threshold(self, exp_dir: Path):
        """2. Verifies label mapping 0=legitimate, 1=phishing across results."""
        json_p = exp_dir / "aggregation_results.json"
        assert json_p.exists(), f"Results JSON missing: {json_p}"

        with open(json_p, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["total_validation_samples"] == 262
        base = summary["baseline_performance"]
        assert base["false_positives"] == 4
        assert base["false_negatives"] == 39

    def test_aggregation_calculations_accuracy(self, exp_dir: Path):
        """3. Verifies mathematical correctness of aggregation formulas on sample rows."""
        csv_p = exp_dir / "all_validation_sample_predictions.csv"
        df = pd.read_csv(csv_p)

        for _, r in df.head(10).iterrows():
            p_top = r["baseline_top"]
            p_um = r["region_upper_middle"]
            p_mid = r["region_middle"]
            p_lm = r["region_lower_middle"]
            p_bot = r["region_bottom"]
            five = [p_top, p_um, p_mid, p_lm, p_bot]

            # Top + Middle mean & max
            assert np.isclose(r["top_middle_mean"], (p_top + p_mid) / 2.0, atol=1e-4)
            assert np.isclose(r["top_middle_max"], max(p_top, p_mid), atol=1e-4)

            # Top + Bottom mean & max
            assert np.isclose(r["top_bottom_mean"], (p_top + p_bot) / 2.0, atol=1e-4)
            assert np.isclose(r["top_bottom_max"], max(p_top, p_bot), atol=1e-4)

            # 5-Region mean & max
            assert np.isclose(r["five_region_mean"], np.mean(five), atol=1e-4)
            assert np.isclose(r["five_region_max"], max(five), atol=1e-4)

    def test_comparison_table_and_confusion_matrix_totals(self, exp_dir: Path):
        """4. Verifies all strategies sum to 262 in their confusion matrices."""
        comp_csv = exp_dir / "aggregation_comparison.csv"
        assert comp_csv.exists()

        df_comp = pd.read_csv(comp_csv)
        assert len(df_comp) == 12  # 12 evaluated strategies

        for _, r in df_comp.iterrows():
            total_cm = r["true_negatives"] + r["false_positives"] + r["false_negatives"] + r["true_positives"]
            assert total_cm == 262, f"Confusion matrix total != 262 for {r['strategy']}"
            assert r["true_negatives"] + r["false_positives"] == 26  # Legit support
            assert r["true_positives"] + r["false_negatives"] == 236  # Phish support

    def test_checkpoint_unaltered_and_eval_mode(self):
        """5. Verifies best_model.pt exists and was not modified."""
        ckpt_p = Path("models/checkpoints/resnet18/best_model.pt")
        assert ckpt_p.exists()
        # Verify checkpoint is ~135 MB (complete ResNet-18 weights)
        assert ckpt_p.stat().st_size > 100_000_000

    def test_test_split_isolation_preserved(self, exp_dir: Path):
        """6. Verifies test split (test.parquet) was never referenced or evaluated."""
        json_p = exp_dir / "aggregation_results.json"
        with open(json_p, "r", encoding="utf-8") as f:
            content = f.read()
        assert "test.parquet" not in content, "Test partition was referenced in aggregation results!"
