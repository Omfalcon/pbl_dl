"""Lightweight Unit Tests for ResNet-18 Screenshot Classifier Pipeline (M-02).

Validates model output shapes, forward/backward passes on synthetic tensors,
preprocessing transforms, dataset partition integrity, checkpoint serialization,
and inference prediction schemas without running training epochs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.ml.vision.dataset import ScreenshotDataset
from app.ml.vision.evaluation import compute_vision_metrics
from app.ml.vision.inference import ResNet18Inference
from app.ml.vision.models import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ResNet18ScreenshotClassifier,
)
from app.ml.vision.preprocessing import (
    AspectRatioPadResize,
    TopViewportCrop,
    build_transforms,
    safe_load_image,
)


class TestResNet18Architecture:
    """Verifies ResNet-18 model architecture, tensor shapes, and backward pass on synthetic tensors."""

    def test_model_initialization_scratch(self):
        model = ResNet18ScreenshotClassifier(pretrained=False, dropout_p=0.3, num_classes=2)
        assert model.num_classes == 2
        assert not model.pretrained
        cfg = model.get_config()
        assert cfg["model_id"] == "M-02"
        assert cfg["architecture"] == "ResNet-18"

    def test_forward_output_shape_and_finiteness(self):
        model = ResNet18ScreenshotClassifier(pretrained=False)
        model.eval()

        # Batch of 3 synthetic RGB images (3, 224, 224)
        x = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (3, 2)
        assert torch.all(torch.isfinite(logits))

        # Check probability output
        with torch.no_grad():
            probs = model.predict_proba(x)

        assert probs.shape == (3, 2)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        # Probabilities sum to 1.0 per sample
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_single_synthetic_forward_backward_step(self):
        """Verifies loss computation, gradient accumulation, and parameter update on CPU."""
        model = ResNet18ScreenshotClassifier(pretrained=False)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 0.5]))

        # Record initial weights of classification head
        initial_weight = model.head[-1].weight.clone()

        # Synthetic batch: 2 samples
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        assert torch.isfinite(loss)
        loss.backward()

        # Verify gradients exist and are finite
        for param in model.head.parameters():
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))

        optimizer.step()

        # Weights must have changed after step
        assert not torch.allclose(initial_weight, model.head[-1].weight)

    def test_class_mapping_consistency(self):
        assert CLASS_TO_IDX["legitimate"] == 0
        assert CLASS_TO_IDX["phishing"] == 1
        assert IDX_TO_CLASS[0] == "legitimate"
        assert IDX_TO_CLASS[1] == "phishing"


class TestPreprocessingAndTransforms:
    """Verifies viewport cropping, aspect-ratio padding, and deterministic transforms."""

    def test_top_viewport_crop(self):
        # Tall image simulating a long webpage scroll (1280 x 3000)
        tall_img = Image.new("RGB", (1280, 3000), color=(100, 150, 200))
        crop_op = TopViewportCrop(viewport_ratio=1.0)
        cropped = crop_op(tall_img)

        # Output should preserve width and crop height to target ratio
        assert cropped.size == (1280, 1280)

    def test_aspect_ratio_pad_resize(self):
        # Wide image (1000 x 500)
        wide_img = Image.new("RGB", (1000, 500), color=(50, 50, 50))
        pad_op = AspectRatioPadResize(target_size=224)
        padded = pad_op(wide_img)

        assert padded.size == (224, 224)

    def test_deterministic_validation_transforms(self):
        tf_eval = build_transforms(crop_strategy="top_viewport", image_size=224, is_training=False)
        img = Image.new("RGB", (800, 1600), color=(120, 130, 140))

        t1 = tf_eval(img)
        t2 = tf_eval(img)

        assert t1.shape == (3, 224, 224)
        assert t2.shape == (3, 224, 224)
        # Deterministic: repeated application produces identical tensors
        assert torch.allclose(t1, t2)

    def test_safe_load_image_rgb_conversion(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create RGBA image
            rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
            rgba_img.save(tmp_path)

            loaded = safe_load_image(tmp_path)
            assert loaded.mode == "RGB"
            assert loaded.size == (100, 100)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_safe_load_image_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            safe_load_image("non_existent_image_path_12345.png")


class TestScreenshotDataset:
    """Verifies dataset loading, class weights calculation, and split integrity."""

    def test_synthetic_dataframe_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img1 = tmp_path / "img1.png"
            img2 = tmp_path / "img2.png"
            img3 = tmp_path / "img3.png"

            Image.new("RGB", (200, 200), (255, 0, 0)).save(img1)
            Image.new("RGB", (200, 200), (0, 255, 0)).save(img2)
            Image.new("RGB", (200, 200), (0, 0, 255)).save(img3)

            df = pd.DataFrame({
                "sample_id": ["s1", "s2", "s3"],
                "screenshot_relative_path": [str(img1), str(img2), str(img3)],
                "label": [1, 1, 0],
            })

            ds = ScreenshotDataset(data_source=df, repo_root=tmp_path)
            assert len(ds) == 3

            tensor_img, label_tensor, sample_id = ds[0]
            assert tensor_img.shape == (3, 224, 224)
            assert label_tensor.item() == 1
            assert sample_id == "s1"

            # Dynamic class weights: 2 positive, 1 negative -> w0 = 3/2=1.5, w1 = 3/4=0.75
            weights = ds.compute_class_weights()
            assert weights.shape == (2,)
            assert weights[0] > weights[1]

    def test_actual_visual_splits_integrity_on_disk(self):
        """Verifies physical split files on disk preserve zero leakage."""
        train_file = Path("data/splits/visual/train.parquet")
        val_file = Path("data/splits/visual/val.parquet")
        test_file = Path("data/splits/visual/test.parquet")

        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        df_tr = pd.read_parquet(train_file)
        df_va = pd.read_parquet(val_file)
        df_te = pd.read_parquet(test_file)

        assert len(df_tr) == 624
        assert len(df_va) == 262
        assert len(df_te) == 114
        assert len(df_tr) + len(df_va) + len(df_te) == 1000

        # Domain overlap assertion
        tr_dom = set(df_tr["registrable_domain"].dropna())
        va_dom = set(df_va["registrable_domain"].dropna())
        te_dom = set(df_te["registrable_domain"].dropna())

        assert len(tr_dom & va_dom) == 0
        assert len(tr_dom & te_dom) == 0
        assert len(va_dom & te_dom) == 0


class TestEvaluationAndMetrics:
    """Verifies metrics computation including accuracy, F1, ROC-AUC, and edge cases."""

    def test_compute_metrics_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]

        metrics = compute_vision_metrics(y_true, y_pred, y_prob=y_prob, loss=0.05)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_avg"]["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["confusion_matrix"]["true_positive"] == 2
        assert metrics["confusion_matrix"]["true_negative"] == 2

    def test_compute_metrics_single_class_edge_case(self):
        # Only class 1 present (e.g. subset edge case)
        y_true = [1, 1, 1]
        y_pred = [1, 1, 1]
        y_prob = [0.9, 0.8, 0.7]

        metrics = compute_vision_metrics(y_true, y_pred, y_prob=y_prob)
        assert metrics["accuracy"] == 1.0
        # ROC-AUC cannot be calculated for 1 class; must safely return None without crashing
        assert metrics["roc_auc"] is None


class TestCheckpointAndInference:
    """Verifies checkpoint serialization, model restoration, and inference prediction format."""

    def test_checkpoint_roundtrip_and_inference(self):
        model = ResNet18ScreenshotClassifier(pretrained=False)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            checkpoint_data = {
                "model_id": "M-02",
                "architecture": "ResNet-18",
                "epoch": 5,
                "val_macro_f1": 0.885,
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "preprocessing_config": {"crop_strategy": "top_viewport", "image_size": 224},
            }
            torch.save(checkpoint_data, tmp_path)

            # Load into inference engine
            engine = ResNet18Inference(checkpoint_path=tmp_path, device="cpu")
            assert engine.crop_strategy == "top_viewport"
            assert engine.image_size == 224
            assert engine.checkpoint_metadata["epoch"] == 5

            # Predict on synthetic PIL image
            dummy_img = Image.new("RGB", (600, 800), (200, 100, 50))
            pred = engine.predict(dummy_img, threshold=0.5)

            assert pred["model_id"] == "M-02"
            assert pred["predicted_label"] in ("legitimate", "phishing")
            assert pred["predicted_class_id"] in (0, 1)
            assert 0.0 <= pred["probabilities"]["legitimate"] <= 1.0
            assert 0.0 <= pred["probabilities"]["phishing"] <= 1.0
            assert np.isclose(
                pred["probabilities"]["legitimate"] + pred["probabilities"]["phishing"],
                1.0,
                atol=1e-4,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_config_file_validity(self):
        import yaml
        config_path = Path("configs/resnet18.yaml")
        assert config_path.exists()

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "model" in cfg
        assert "data" in cfg
        assert "preprocessing" in cfg
        assert "training" in cfg
        assert cfg["model"]["architecture"] == "resnet18"
        assert cfg["preprocessing"]["image_size"] == 224
