"""ResNet-18 Binary Screenshot Classifier (M-02) for PhishVision AI.

Implements PyTorch ResNet-18 architecture with configurable ImageNet initialization,
feature extraction head, and calibrated class mapping for website screenshot classification.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger("phishvision.ml.vision.models")

CLASS_TO_IDX: dict[str, int] = {
    "legitimate": 0,
    "phishing": 1,
}

IDX_TO_CLASS: dict[int, str] = {
    0: "legitimate",
    1: "phishing",
}


class ResNet18ScreenshotClassifier(nn.Module):
    """ResNet-18 binary website screenshot classifier.
    
    Architecture:
      Backbone: ResNet-18 (convolutional stages 1 through 4)
      Classification Head:
        AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.dropout_p = dropout_p
        self.num_classes = num_classes

        # Load backbone with configurable ImageNet pretraining
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            logger.info("Initializing ResNet-18 backbone with ImageNet pretrained weights.")
        else:
            weights = None
            logger.info("Initializing ResNet-18 backbone from scratch (Kaiming normal initialization).")

        base_model = models.resnet18(weights=weights)

        # Isolate feature extractor backbone up to layer4
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Custom classification head as required:
        # AdaptiveAvgPool2d(1) -> Flatten -> Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

        # Initialize head layers using Kaiming normal
        self._initialize_head()

    def _initialize_head(self) -> None:
        """Initialize classification head weights with He (Kaiming) normal initialization."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from the ResNet-18 convolutional backbone."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized raw class logits.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
            
        Returns:
            Logits tensor of shape (batch_size, 2). Softmax is NOT applied inside forward.
        """
        features = self.forward_features(x)
        logits = self.head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities via Softmax along the class dimension.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
            
        Returns:
            Probability tensor of shape (batch_size, 2) where probabilities sum to 1.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_config(self) -> dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_id": "M-02",
            "architecture": "ResNet-18",
            "pretrained": self.pretrained,
            "dropout_p": self.dropout_p,
            "num_classes": self.num_classes,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
        }
