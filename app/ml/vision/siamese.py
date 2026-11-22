"""Siamese ResNet-18 CNN Brand Impersonation Matcher (M-03).

Provides shared-weight Siamese CNN architecture, L2-normalized 256-dimensional embeddings,
cosine similarity computation, and configurable contrastive / triplet margin loss functions.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger("phishvision.ml.vision.siamese")


class SiameseResNet18(nn.Module):
    """Shared-weight Siamese CNN with ResNet-18 backbone and 256-dim L2-normalized embedding head."""

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout_p: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # 1. Convolutional Backbone (ResNet-18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Truncate fc layer, keep layers up to avgpool
        self.conv_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        if freeze_backbone:
            for param in self.conv_stem.parameters():
                param.requires_grad = False

        # 2. Embedding Projection Head: AdaptiveAvgPool -> Flatten -> Dropout -> Linear -> L2 Normalization
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.projection = nn.Linear(512, embedding_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial convolutional feature map."""
        return self.conv_stem(x)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Computes 256-dimensional L2-normalized embedding for a single image branch."""
        features = self.conv_stem(x)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        dropped = self.dropout(flat)
        projected = self.projection(dropped)
        # L2-normalization: ||z||_2 = 1.0
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of input images.
        
        Args:
            x1: First image batch (B, 3, H, W).
            x2: Second image batch (B, 3, H, W).
            
        Returns:
            Tuple of (z1, z2, cosine_similarity):
              - z1: L2-normalized embeddings for x1 (B, 256).
              - z2: L2-normalized embeddings for x2 (B, 256).
              - cosine_similarity: Pairwise cosine similarities in range [-1.0, 1.0] (B,).
        """
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        # Since z1 and z2 are unit-normalized, dot product equals cosine similarity
        cosine_sim = torch.sum(z1 * z2, dim=-1)
        return z1, z2, cosine_sim

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for triplet learning."""
        za = self.forward_one(anchor)
        zp = self.forward_one(positive)
        zn = self.forward_one(negative)
        return za, zp, zn


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for Siamese pair metric learning.
    
    Supports distance-based contrastive loss:
      L(z1, z2, y) = y * 0.5 * D^2 + (1 - y) * 0.5 * max(0, margin - D)^2
    where D = ||z1 - z2||_2, y = 1 for positive pairs, y = 0 for negative pairs.
    """

    def __init__(self, margin: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1: Normalized embeddings (B, D).
            z2: Normalized embeddings (B, D).
            label: 1 for same-brand (positive), 0 for different-brand (negative).
        """
        # Euclidean distance between normalized embeddings
        d = torch.sqrt(torch.sum((z1 - z2) ** 2, dim=-1) + self.eps)
        loss_pos = label * 0.5 * (d ** 2)
        loss_neg = (1.0 - label) * 0.5 * (torch.clamp(self.margin - d, min=0.0) ** 2)
        return torch.mean(loss_pos + loss_neg)


class CosineContrastiveLoss(nn.Module):
    """Cosine-similarity contrastive loss:
    
    L(z1, z2, y) = y * (1 - S) + (1 - y) * max(0, S - margin)
    where S = z1 . z2.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        cosine_sim = torch.sum(z1 * z2, dim=-1)
        loss_pos = label * (1.0 - cosine_sim)
        loss_neg = (1.0 - label) * torch.clamp(cosine_sim - self.margin, min=0.0)
        return torch.mean(loss_pos + loss_neg)


class TripletLoss(nn.Module):
    """Triplet Margin Loss with L2 distance on normalized embeddings:
    
    L(a, p, n) = max(0, ||za - zp||_2^2 - ||za - zn||_2^2 + alpha)
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        za: torch.Tensor,
        zp: torch.Tensor,
        zn: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = torch.sum((za - zp) ** 2, dim=-1)
        d_neg = torch.sum((za - zn) ** 2, dim=-1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return torch.mean(loss)
