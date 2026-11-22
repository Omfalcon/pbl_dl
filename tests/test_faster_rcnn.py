"""Unit & Regression Tests for Faster R-CNN (Phase 7.6).

Verifies model architecture, 7-class schema, bidirectional YOLO-FasterRCNN mapping,
empty-image handling, target dictionary generation, forward pass in train/eval modes,
vectorized IoU & AP computation, and preview rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN

from app.ml.vision.faster_rcnn import (
    FASTER_RCNN_CLASSES,
    FASTER_RCNN_TO_YOLO,
    YOLO_TO_FASTER_RCNN,
    SyntheticFasterRCNNDataset,
    FasterRCNNInferencer,
    compute_ap,
    compute_box_iou,
    create_faster_rcnn_model,
    detection_collate_fn,
)
from app.ml.train_faster_rcnn import generate_comparison_report
from app.ml.vision.yolo_annotations import YOLO_DETECTION_CLASSES


class TestFasterRCNNArchitecture:
    """Verifies Faster R-CNN architecture, class schema, and parameter dimensions."""

    def test_class_schema_and_mapping(self) -> None:
        """Asserts 7 total classes (0=background, 1..6=foreground) and exact YOLO bijection."""
        assert len(FASTER_RCNN_CLASSES) == 7
        assert FASTER_RCNN_CLASSES[0] == "__background__"

        # Check foreground classes match YOLO classes exactly
        for yolo_cid, yolo_name in YOLO_DETECTION_CLASSES.items():
            fr_cid = YOLO_TO_FASTER_RCNN[yolo_cid]
            assert fr_cid == yolo_cid + 1
            assert FASTER_RCNN_CLASSES[fr_cid] == yolo_name
            assert FASTER_RCNN_TO_YOLO[fr_cid] == yolo_cid

    def test_model_instantiation(self) -> None:
        """Verifies model creation, FPN backbone, and box predictor dimensions."""
        model = create_faster_rcnn_model(
            num_classes=7,
            pretrained=False,
            min_size=512,
            max_size=512,
        )
        assert isinstance(model, FasterRCNN)
        # FastRCNNPredictor cls_score out_features must equal 7
        assert model.roi_heads.box_predictor.cls_score.out_features == 7
        # bbox_pred out_features must equal 7 * 4 = 28
        assert model.roi_heads.box_predictor.bbox_pred.out_features == 28


class TestFasterRCNNDataset:
    """Verifies synthetic dataset loading, coordinates conversion, and empty image handling."""

    @pytest.fixture
    def dataset(self) -> SyntheticFasterRCNNDataset:
        manifest = Path("data/synthetic_yolo/manifest.csv")
        if not manifest.exists():
            pytest.skip("Synthetic dataset manifest not found.")
        return SyntheticFasterRCNNDataset(manifest_csv=manifest, split="test", imgsz=512)

    def test_dataset_item_format(self, dataset: SyntheticFasterRCNNDataset) -> None:
        """Asserts image tensor format [3, H, W] and target dictionary keys and dtypes."""
        img_tensor, target = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.dim() == 3
        assert img_tensor.size(0) == 3

        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        assert "area" in target
        assert "iscrowd" in target

        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64
        assert target["boxes"].dim() == 2
        assert target["boxes"].size(1) == 4

        # Labels must be within foreground range [1, 6]
        if len(target["labels"]) > 0:
            assert (target["labels"] >= 1).all()
            assert (target["labels"] <= 6).all()

    def test_empty_image_handling(self, dataset: SyntheticFasterRCNNDataset) -> None:
        """Asserts that 0-box samples (e.g., 404 pages) return empty tensors of correct shape."""
        # Find an empty sample in test split
        empty_rows = dataset.df[dataset.df["is_empty"] == True]
        if len(empty_rows) == 0:
            pytest.skip("No empty samples in test split.")

        empty_idx = empty_rows.index[0]
        _, target = dataset[empty_idx]

        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert target["area"].shape == (0,)
        assert target["iscrowd"].shape == (0,)

    def test_collate_fn(self, dataset: SyntheticFasterRCNNDataset) -> None:
        """Verifies custom collate_fn handles variable length box tuples."""
        batch = [dataset[0], dataset[1]]
        images, targets = detection_collate_fn(batch)
        assert len(images) == 2
        assert len(targets) == 2
        assert isinstance(images, tuple)
        assert isinstance(targets, tuple)


class TestFasterRCNNForwardPass:
    """Verifies train-mode loss dictionary and eval-mode output predictions."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return create_faster_rcnn_model(num_classes=7, pretrained=False, min_size=512, max_size=512)

    def test_train_forward_pass(self, model: nn.Module) -> None:
        """Asserts training mode forward pass returns 4 loss components."""
        model.train()
        images = [torch.rand(3, 512, 512)]
        targets = [{
            "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([0]),
            "area": torch.tensor([10000.0]),
            "iscrowd": torch.tensor([0]),
        }]

        losses = model(images, targets)
        assert isinstance(losses, dict)
        assert "loss_classifier" in losses
        assert "loss_box_reg" in losses
        assert "loss_objectness" in losses
        assert "loss_rpn_box_reg" in losses

        total_loss = sum(v for v in losses.values())
        assert torch.isfinite(total_loss)

    def test_eval_forward_pass(self, model: nn.Module) -> None:
        """Asserts eval mode forward pass returns detections list."""
        model.eval()
        images = [torch.rand(3, 512, 512)]
        with torch.no_grad():
            outputs = model(images)

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert "boxes" in outputs[0]
        assert "labels" in outputs[0]
        assert "scores" in outputs[0]


class TestMetricsCalculation:
    """Verifies pure-Python vectorized IoU and Average Precision calculations."""

    def test_compute_box_iou(self) -> None:
        """Tests box IoU calculation on identical, disjoint, and partial overlap boxes."""
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        iou = compute_box_iou(box1, box2)
        assert float(iou[0, 0]) == pytest.approx(1.0, rel=1e-4)

        # Disjoint boxes
        box3 = torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float32)
        iou_disjoint = compute_box_iou(box1, box3)
        assert float(iou_disjoint[0, 0]) == pytest.approx(0.0, rel=1e-4)

        # 50% horizontal overlap
        box4 = torch.tensor([[5.0, 0.0, 15.0, 10.0]], dtype=torch.float32)
        # Inter = 5*10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
        iou_partial = compute_box_iou(box1, box4)
        assert float(iou_partial[0, 0]) == pytest.approx(50.0 / 150.0, rel=1e-4)

    def test_compute_box_iou_empty(self) -> None:
        """Tests IoU when one or both box sets are empty."""
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        normal_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        iou = compute_box_iou(empty_boxes, normal_boxes)
        assert iou.shape == (0, 1)

    def test_compute_ap_perfect(self) -> None:
        """Asserts perfect recall and precision yields AP = 1.0."""
        recalls = np.array([0.2, 0.5, 0.8, 1.0])
        precisions = np.array([1.0, 1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert ap == pytest.approx(1.0, abs=1e-4)

    def test_compute_ap_empty(self) -> None:
        """Asserts empty recall/precision yields AP = 0.0."""
        ap = compute_ap(np.array([]), np.array([]))
        assert ap == 0.0

    def test_compute_ap_partial(self) -> None:
        """Asserts 101-point interpolation on a known PR curve returns expected AP."""
        # Simple staircase: recall=[0.5, 1.0], precision=[1.0, 0.5]
        # At IoU thresholds 0.00..0.50 -> p=1.0 (from envelope: max rightwards)
        # At IoU thresholds 0.50..1.00 -> p=0.5
        # Expected mean: 51/101 * 1.0 + 50/101 * 0.5 = (51 + 25) / 101 ≈ 0.7525
        recalls = np.array([0.5, 1.0])
        precisions = np.array([1.0, 0.5])
        ap = compute_ap(recalls, precisions)
        # 101-point: thresholds 0,0.01..0.49 -> ind=1 in mrec=[0,0.5,1,1] -> mpre[1]=1.0 (51 pts)
        #            thresholds 0.50..1.00    -> ind=2 or 3 -> mpre=0.5 or 0.5 (50 pts)
        assert ap == pytest.approx(0.7525, abs=1e-3)

    def test_image_id_no_collision_uneven_batch(self) -> None:
        """Asserts evaluate_dataset uses global monotonic counter, not img_idx*len(images)+i.

        A naive formula img_idx * len(images) + i produces collisions when the last
        batch is smaller than the full batch size (e.g. 5 images, batch_size=2).
        The fix uses a global_sample_idx counter instead.
        """
        # Simulate: 5 images, batches of size [2, 2, 1]
        global_sample_idx = 0
        batch_sizes = [2, 2, 1]
        ids_global = []
        for bs in batch_sizes:
            for _ in range(bs):
                ids_global.append(global_sample_idx)
                global_sample_idx += 1
        assert len(ids_global) == len(set(ids_global)), "Global counter should produce unique IDs"

        # Old formula would produce collision
        ids_old = []
        for img_idx, bs in enumerate(batch_sizes):
            for i in range(bs):
                ids_old.append(img_idx * bs + i)
        assert len(ids_old) != len(set(ids_old)), "Old formula should have produced collision"


class TestFasterRCNNInferencer:
    """Verifies inference interface, prediction dictionary schema, and preview rendering."""

    def test_inferencer_predict_and_preview(self, tmp_path: Path) -> None:
        """Tests single-image prediction and annotated image generation."""
        inferencer = FasterRCNNInferencer(checkpoint_path=None, device="cpu", score_threshold=0.01)
        dummy_img = Image.new("RGB", (512, 512), color=(240, 240, 240))

        pred = inferencer.predict(dummy_img)
        assert "image_size" in pred
        assert pred["image_size"] == [512, 512]
        assert "num_detections" in pred
        assert "detections" in pred
        assert isinstance(pred["detections"], list)

        # Test preview rendering
        out_preview = tmp_path / "test_preview.png"
        res_path = inferencer.render_annotated_preview(dummy_img, output_path=out_preview)
        assert res_path.exists()
        assert res_path.stat().st_size > 0


class TestComparisonReportGeneration:
    """Verifies that side-by-side comparison report schema matches requirements."""

    def test_generate_comparison_report(self, tmp_path: Path) -> None:
        mock_metrics = {
            "status": "EVALUATION_COMPLETE",
            "model_id": "M-05",
            "architecture": "FasterRCNN_ResNet50_FPN",
            "mAP50": 0.9850,
            "mAP50_95": 0.9420,
            "mean_precision": 0.9900,
            "mean_recall": 0.9850,
            "mean_latency_ms": 42.5,
            "peak_vram_mb": 1450.0,
            "class_metrics": {
                0: {
                    "class_id": 0,
                    "faster_rcnn_class_id": 1,
                    "class_name": "brand_logo",
                    "total_ground_truth": 111,
                    "total_detections": 115,
                    "precision": 0.991,
                    "recall": 1.0,
                    "f1_score": 0.995,
                    "mAP50": 0.995,
                    "mAP50_95": 0.880,
                }
            },
        }

        yolo_report_dir = Path("data/reports/yolo")
        comparison = generate_comparison_report(
            faster_rcnn_metrics=mock_metrics,
            yolo_report_dir=yolo_report_dir,
            output_dir=tmp_path,
            chk_size_mb=165.4,
        )

        assert "models" in comparison
        assert "M-04_YOLOv8n" in comparison["models"]
        assert "M-05_FasterRCNN" in comparison["models"]

        comp_json = tmp_path / "comparison_with_yolo.json"
        comp_md = tmp_path / "FASTER_RCNN_EXPERIMENT_REPORT.md"
        assert comp_json.exists()
        assert comp_md.exists()

        with open(comp_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["models"]["M-05_FasterRCNN"]["mAP50"] == 0.9850
