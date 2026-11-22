"""Vision Intelligence and Deep Neural Network Package.

Provides:
- M-02 ResNet-18 screenshot classifier, viewport preprocessing, dataset loaders,
  evaluation metrics, error diagnostics, contact sheets, and inference engine.
- M-03 Siamese CNN Brand Impersonation Matcher (ResNet-18 shared-weight Siamese,
  256-d L2 normalized embeddings, Contrastive and Triplet losses, synthetic pair loaders,
  brand reference acquisition and auditing, similarity metrics, and brand verification).
"""

from __future__ import annotations

from app.ml.vision.aggregation_experiment import (
    STRATEGY_DESCRIPTIONS,
    run_full_validation_aggregation_experiment,
)
from app.ml.vision.brand_acquisition import (
    BrandReferenceAcquisitionPipeline,
    audit_brand_reference_gallery,
)
from app.ml.vision.contact_sheet import (
    generate_all_contact_sheets,
    render_contact_sheet,
)
from app.ml.vision.dataset import ScreenshotDataset
from app.ml.vision.error_analysis import (
    analyze_image_characteristics,
    run_validation_error_analysis,
)
from app.ml.vision.evaluation import compute_vision_metrics, save_evaluation_report
from app.ml.vision.inference import ResNet18Inference
from app.ml.vision.models import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ResNet18ScreenshotClassifier,
)
from app.ml.vision.multi_region import (
    compute_vertical_crop_coordinates,
    evaluate_image_tensor,
    extract_diagnostic_views,
    run_multi_region_analysis,
)
from app.ml.vision.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AspectRatioPadResize,
    TopViewportCrop,
    build_transforms,
    safe_load_image,
)
from app.ml.vision.siamese import (
    ContrastiveLoss,
    CosineContrastiveLoss,
    SiameseResNet18,
    TripletLoss,
)
from app.ml.vision.siamese_dataset import (
    SiamesePairDataset,
    SiameseTripletDataset,
    generate_synthetic_brand_pairs,
)
from app.ml.vision.siamese_inference import SiameseBrandMatcher
from app.ml.vision.multimodal_pipeline import MultimodalPhishingPipeline
from app.ml.vision.siamese_trainer import (
    SiameseTrainer,
    compute_similarity_metrics,
)
from app.ml.vision.trainer import ResNet18Trainer
from app.ml.vision.yolo_annotations import (
    CLASS_NAME_TO_ID,
    YOLO_DETECTION_CLASSES,
    YOLOAnnotationManifestManager,
    YOLOAnnotationValidator,
    YOLOBoundingBox,
    YOLOValidationResult,
)
from app.ml.vision.yolo_inference import YOLOUIElementDetector


__all__ = [
    # M-02 ResNet-18 Classifier
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
    "run_validation_error_analysis",
    "analyze_image_characteristics",
    "render_contact_sheet",
    "generate_all_contact_sheets",
    "compute_vertical_crop_coordinates",
    "extract_diagnostic_views",
    "evaluate_image_tensor",
    "run_multi_region_analysis",
    "run_full_validation_aggregation_experiment",
    "STRATEGY_DESCRIPTIONS",
    # M-03 Siamese CNN Brand Matcher
    "SiameseResNet18",
    "ContrastiveLoss",
    "CosineContrastiveLoss",
    "TripletLoss",
    "SiamesePairDataset",
    "SiameseTripletDataset",
    "generate_synthetic_brand_pairs",
    "SiameseTrainer",
    "compute_similarity_metrics",
    "SiameseBrandMatcher",
    "audit_brand_reference_gallery",
    "BrandReferenceAcquisitionPipeline",
    # M-04 YOLO Object Detection
    "YOLO_DETECTION_CLASSES",
    "CLASS_NAME_TO_ID",
    "YOLOBoundingBox",
    "YOLOAnnotationValidator",
    "YOLOValidationResult",
    "YOLOAnnotationManifestManager",
    "YOLOUIElementDetector",
    # Multimodal Pipeline
    "MultimodalPhishingPipeline",
]
