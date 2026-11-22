"""Production Inference Engine for ResNet-18 Binary Screenshot Classifier (M-02).

Loads trained checkpoint, restores preprocessing configuration, evaluates single images
or batches, and returns calibrated classification probabilities and risk scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image

from app.ml.vision.models import CLASS_TO_IDX, IDX_TO_CLASS, ResNet18ScreenshotClassifier
from app.ml.vision.preprocessing import build_transforms, safe_load_image

logger = logging.getLogger("phishvision.ml.vision.inference")


class ResNet18Inference:
    """Production inference wrapper for M-02 ResNet-18 website screenshot classification."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        crop_strategy: str = "top_viewport",
        image_size: int = 224,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.crop_strategy = crop_strategy
        self.image_size = image_size
        self.model: ResNet18ScreenshotClassifier | None = None
        self.checkpoint_metadata: dict[str, Any] = {}

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            # Instantiate an uninitialized model architecture if no checkpoint provided
            self.model = ResNet18ScreenshotClassifier(pretrained=False)
            self.model.to(self.device)
            self.model.eval()

        # Build deterministic evaluation transforms
        self.transform = build_transforms(
            crop_strategy=self.crop_strategy,
            image_size=self.image_size,
            is_training=False,
            horizontal_flip=False,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load weights and restore exact preprocessing configuration from a checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ResNet-18 checkpoint file not found at: {path}")

        logger.info(f"Loading ResNet-18 checkpoint from {path} onto {self.device}...")
        checkpoint = torch.load(path, map_location=self.device)

        # Restore model configuration
        model_cfg = checkpoint.get("model_config", {})
        pretrained = model_cfg.get("pretrained", False)
        dropout_p = model_cfg.get("dropout_p", 0.3)
        num_classes = model_cfg.get("num_classes", 2)

        self.model = ResNet18ScreenshotClassifier(
            pretrained=pretrained,
            dropout_p=dropout_p,
            num_classes=num_classes,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Restore preprocessing configuration
        prep_cfg = checkpoint.get("preprocessing_config", {})
        self.crop_strategy = prep_cfg.get("crop_strategy", self.crop_strategy)
        self.image_size = prep_cfg.get("image_size", self.image_size)

        self.transform = build_transforms(
            crop_strategy=self.crop_strategy,
            image_size=self.image_size,
            is_training=False,
            horizontal_flip=False,
        )

        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "val_macro_f1": checkpoint.get("val_macro_f1"),
            "class_to_idx": checkpoint.get("class_to_idx", CLASS_TO_IDX),
            "idx_to_class": checkpoint.get("idx_to_class", IDX_TO_CLASS),
        }
        logger.info(f"Checkpoint successfully loaded (trained epoch: {self.checkpoint_metadata['epoch']})")

    @torch.no_grad()
    def predict(
        self,
        image_input: str | Path | Image.Image,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Perform screenshot classification inference on a single image.
        
        Args:
            image_input: File path string, Path object, or PIL Image.
            threshold: Decision threshold for phishing classification (default: 0.5).
            
        Returns:
            Structured prediction dictionary containing probabilities and decision.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Please call load_checkpoint().")

        try:
            pil_img = safe_load_image(image_input)
        except Exception as e:
            logger.error(f"Failed to process input image for inference: {e}")
            return {
                "error": str(e),
                "is_phishing": False,
                "confidence": 0.0,
                "probabilities": {"legitimate": 0.5, "phishing": 0.5},
                "predicted_label": "error",
            }

        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_legit = float(probs[0])
        p_phish = float(probs[1])

        is_phishing = bool(p_phish >= threshold)
        predicted_label = "phishing" if is_phishing else "legitimate"
        confidence = p_phish if is_phishing else p_legit

        return {
            "model_id": "M-02",
            "architecture": "ResNet-18",
            "predicted_label": predicted_label,
            "predicted_class_id": 1 if is_phishing else 0,
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "probabilities": {
                "legitimate": round(p_legit, 4),
                "phishing": round(p_phish, 4),
            },
            "threshold_applied": threshold,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[str | Path | Image.Image],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Perform batch classification inference."""
        return [self.predict(img, threshold=threshold) for img in images]
