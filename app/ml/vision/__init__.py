"""Vision Intelligence and Deep Neural Network Package (M-02 ResNet-18).

Provides ResNet-18 screenshot classifier, configurable top-viewport and aspect-ratio
preprocessing, dataset partitioning loaders, evaluation metrics, and inference engine.
"""

from __future__ import annotations

from app.ml.vision.dataset import ScreenshotDataset
from app.ml.vision.evaluation import compute_vision_metrics, save_evaluation_report
from app.ml.vision.inference import ResNet18Inference
from app.ml.vision.models import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ResNet18ScreenshotClassifier,
)
from app.ml.vision.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AspectRatioPadResize,
    TopViewportCrop,
    build_transforms,
    safe_load_image,
)
from app.ml.vision.trainer import ResNet18Trainer

__all__ = [
    "ResNet18ScreenshotClassifier",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "ScreenshotDataset",
    "build_transforms",
    "safe_load_image",
    "TopViewportCrop",
    "AspectRatioPadResize",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "compute_vision_metrics",
    "save_evaluation_report",
    "ResNet18Trainer",
    "ResNet18Inference",
]
