"""Unit and Contract Tests for M-03 Siamese ResNet-18 Brand Matcher (Phase 6.2).

Verifies:
1. SiameseResNet18 architecture, weight sharing, forward pass, and embedding shapes.
2. Embedding L2 normalization (unit hypersphere ||z||_2 = 1.0).
3. Freeze backbone option and parameter gradient flags.
4. Cosine similarity output boundedness [-1.0, 1.0].
5. ContrastiveLoss, CosineContrastiveLoss, and TripletLoss computation and gradient backpropagation.
6. Synthetic pair dataset generation, item schema, and DataLoader collation with dummy image tensors.
7. SiameseBrandMatcher inference interface:
   - Cosine similarity matching between image crops.
   - Graceful INSUFFICIENT_REFERENCE_DATA handling when gallery is empty on disk.
   - Domain authorization verification logic.
8. Physical brand reference gallery audit function and empirical asset check.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.ml.vision.brand_acquisition import audit_brand_reference_gallery
from app.ml.vision.siamese import (
    ContrastiveLoss,
    CosineContrastiveLoss,
    SiameseResNet18,
    TripletLoss,
)
from app.ml.vision.siamese_dataset import (
    SiamesePairDataset,
    generate_synthetic_brand_pairs,
)
from app.ml.vision.siamese_inference import SiameseBrandMatcher
from app.ml.vision.siamese_trainer import compute_similarity_metrics


@pytest.fixture
def dummy_siamese_model() -> SiameseResNet18:
    """Fixture returning a SiameseResNet18 with pretrained=False for fast deterministic testing."""
    torch.manual_seed(42)
    return SiameseResNet18(pretrained=False, embedding_dim=256, freeze_backbone=False)


class TestSiameseArchitecture:
    """Test suite for SiameseResNet18 architecture and embedding representation."""

    def test_model_initialization(self, dummy_siamese_model: SiameseResNet18) -> None:
        """Verify model initializes with expected layers and finite parameters."""
        assert dummy_siamese_model.embedding_dim == 256
        assert dummy_siamese_model.pretrained is False
        for param in dummy_siamese_model.parameters():
            assert torch.isfinite(param).all()

    def test_freeze_backbone(self) -> None:
        """Verify freeze_backbone=True sets requires_grad=False on conv_stem layers."""
        frozen_model = SiameseResNet18(pretrained=False, embedding_dim=256, freeze_backbone=True)
        for param in frozen_model.conv_stem.parameters():
            assert param.requires_grad is False
        # Projection head parameters must remain trainable
        for param in frozen_model.projection.parameters():
            assert param.requires_grad is True

    def test_forward_one_shape_and_l2_norm(
        self, dummy_siamese_model: SiameseResNet18
    ) -> None:
        """Verify single image forward pass produces (B, 256) L2-normalized embeddings."""
        dummy_siamese_model.eval()
        batch_size = 4
        x = torch.randn(batch_size, 3, 128, 128)

        with torch.no_grad():
            emb = dummy_siamese_model.forward_one(x)

        assert emb.shape == (batch_size, 256)

        # L2 norm must equal 1.0 within numerical precision
        norms = torch.linalg.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_pair_weight_sharing(
        self, dummy_siamese_model: SiameseResNet18
    ) -> None:
        """Verify forward(x1, x2) uses shared weights and produces identical embeddings for identical inputs."""
        dummy_siamese_model.eval()
        x = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            z1, z2, cos_sim = dummy_siamese_model(x, x)

        assert z1.shape == (2, 256)
        assert z2.shape == (2, 256)
        # Identical inputs must yield identical embeddings across both branches
        assert torch.allclose(z1, z2, atol=1e-6)

        # Compute cosine similarity for identical inputs: must equal 1.0
        assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-5)

    def test_forward_triplet(self, dummy_siamese_model: SiameseResNet18) -> None:
        """Verify triplet forward pass produces (anchor, positive, negative) embeddings."""
        dummy_siamese_model.eval()
        anchor = torch.randn(2, 3, 128, 128)
        pos = torch.randn(2, 3, 128, 128)
        neg = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            za, zp, zn = dummy_siamese_model.forward_triplet(anchor, pos, neg)

        assert za.shape == (2, 256)
        assert zp.shape == (2, 256)
        assert zn.shape == (2, 256)
        assert torch.allclose(torch.linalg.norm(za, dim=-1), torch.ones(2), atol=1e-5)


class TestSiameseLossFunctions:
    """Test suite for contrastive, cosine contrastive, and triplet loss functions."""

    def test_cosine_contrastive_loss(
        self, dummy_siamese_model: SiameseResNet18
    ) -> None:
        """Verify CosineContrastiveLoss calculates positive/negative loss and backpropagates."""
        criterion = CosineContrastiveLoss(margin=0.5)

        # Unit vectors
        z1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        z2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
        # label: 1 = same brand (positive), 0 = different brand (negative)
        labels = torch.tensor([1.0, 0.0])

        loss = criterion(z1, z2, labels)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None

    def test_contrastive_loss(self) -> None:
        """Verify Euclidean ContrastiveLoss operates properly."""
        criterion = ContrastiveLoss(margin=1.0)
        z1 = torch.tensor([[0.5, 0.5]], requires_grad=True)
        z2 = torch.tensor([[0.5, 0.5]], requires_grad=True)
        labels = torch.tensor([1.0])

        loss = criterion(z1, z2, labels)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_triplet_loss(self) -> None:
        """Verify TripletLoss with margin."""
        criterion = TripletLoss(margin=0.3)
        # anchor identical to positive, far from negative -> loss should be 0
        za = torch.tensor([[1.0, 0.0]], requires_grad=True)
        zp = torch.tensor([[1.0, 0.0]], requires_grad=True)
        zn = torch.tensor([[0.0, 1.0]], requires_grad=True)

        loss = criterion(za, zp, zn)
        assert torch.isfinite(loss)
        loss.backward()
        assert za.grad is not None


class TestSiameseDatasetAndMetrics:
    """Test suite for pair dataset loading, batching, and similarity metrics."""

    def test_synthetic_pair_generation(self) -> None:
        """Verify synthetic pair generation creates valid DataFrame with labels."""
        df = generate_synthetic_brand_pairs(
            brands=["google", "microsoft", "paypal"],
            num_samples_per_brand=3,
            negative_ratio=1.0,
            seed=42,
        )
        assert len(df) > 0
        assert "label" in df.columns
        assert "brand_1" in df.columns
        assert "brand_2" in df.columns
        assert "image_path_1" in df.columns
        assert "image_path_2" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})

    def test_pair_dataset_loading_and_batching(self) -> None:
        """Verify SiamesePairDataset loads images from disk and DataLoaders batch correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create 2 tiny dummy images
            img1_path = tmp_path / "img1.png"
            img2_path = tmp_path / "img2.png"
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img1_path)
            Image.new("RGB", (32, 32), color=(0, 255, 0)).save(img2_path)

            pairs_records = [
                {
                    "pair_id": "pair_0",
                    "brand_1": "brandA",
                    "brand_2": "brandA",
                    "image_path_1": str(img1_path),
                    "image_path_2": str(img1_path),
                    "label": 1.0,
                    "pair_type": "authentic_same_brand",
                },
                {
                    "pair_id": "pair_1",
                    "brand_1": "brandA",
                    "brand_2": "brandB",
                    "image_path_1": str(img1_path),
                    "image_path_2": str(img2_path),
                    "label": 0.0,
                    "pair_type": "cross_brand_negative",
                },
            ]

            dataset = SiamesePairDataset(
                pairs_data=pairs_records,
                repo_root=tmp_path,
                image_size=64,
            )
            assert len(dataset) == 2

            item = dataset[0]
            assert len(item) == 4
            t1, t2, label, meta = item
            assert t1.shape == (3, 64, 64)
            assert t2.shape == (3, 64, 64)
            assert label.item() == 1.0
            assert meta["brand_1"] == "brandA"

            # Test DataLoader batching
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            b_t1, b_t2, b_labels, b_meta = batch
            assert b_t1.shape == (2, 3, 64, 64)
            assert b_t2.shape == (2, 3, 64, 64)
            assert b_labels.shape == (2,)

    def test_compute_similarity_metrics(self) -> None:
        """Verify similarity metric calculations at default threshold tau=0.82."""
        y_true = np.array([1, 1, 0, 0])
        y_sim = np.array([0.95, 0.88, 0.40, 0.10])

        metrics = compute_similarity_metrics(y_true, y_sim, threshold=0.82)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["threshold_applied"] == 0.82
        assert metrics["confusion_matrix"]["true_positive"] == 2
        assert metrics["confusion_matrix"]["true_negative"] == 2


class TestSiameseInferenceInterface:
    """Test suite for SiameseBrandMatcher inference engine."""

    def test_insufficient_reference_data_on_empty_gallery(self) -> None:
        """Verify matcher returns INSUFFICIENT_REFERENCE_DATA when gallery has 0 images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_gallery = Path(tmp_dir)
            matcher = SiameseBrandMatcher(
                gallery_dir=empty_gallery,
                similarity_threshold=0.82,
                device="cpu",
            )
            dummy_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
            result = matcher.match_brand(dummy_img, candidate_brand="paypal")

            assert result["status"] == "INSUFFICIENT_REFERENCE_DATA"
            assert result["predicted_brand"] == "UNIDENTIFIED_OR_NEUTRAL"
            assert result["max_similarity"] == 0.0
            assert result["is_brand_impersonation"] is False
            assert "0 authentic images found" in result["message"]

    def test_verify_brand_authorization_logic(self) -> None:
        """Verify domain authorization verification: authorized domain is legitimate; unauthorized is impersonation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_gallery = Path(tmp_dir)
            matcher = SiameseBrandMatcher(
                gallery_dir=empty_gallery,
                similarity_threshold=0.82,
                device="cpu",
            )

            # Inject mock brand metadata for testing domain authorization logic
            matcher.brand_metadata["paypal"] = {
                "brand_id": "paypal",
                "display_name": "PayPal",
                "official_registrable_domains": ["paypal.com", "paypal.me"],
            }
            # Inject simulated reference embeddings for paypal that match query embedding
            dummy_query = Image.new("RGB", (64, 64), color=(100, 100, 100))
            q_emb = matcher.compute_embedding(dummy_query)
            matcher.reference_embeddings["paypal"] = [q_emb]
            matcher.has_reference_data = True

            # Query with authorized domain
            legit_res = matcher.verify_brand_authorization(
                query_image=dummy_query,
                target_registrable_domain="paypal.com",
                candidate_brand="paypal",
            )
            assert legit_res["is_authorized_domain"] is True
            assert legit_res["is_brand_impersonation"] is False
            assert legit_res["verdict"] == "LEGITIMATE_OFFICIAL_BRAND"

            # Query with phishing/unauthorized domain
            phish_res = matcher.verify_brand_authorization(
                query_image=dummy_query,
                target_registrable_domain="paypal-security-update.xyz",
                candidate_brand="paypal",
            )
            assert phish_res["is_authorized_domain"] is False
            assert phish_res["is_brand_impersonation"] is True
            assert phish_res["verdict"] == "CRITICAL_BRAND_IMPERSONATION"

    def test_physical_gallery_audit_integrity(self) -> None:
        """Verify audit_brand_reference_gallery runs against physical data directory and reports empirical inventory."""
        gallery_dir = Path("data/brands/reference_gallery")
        audit = audit_brand_reference_gallery(gallery_dir)

        assert audit["target_brands_count"] == 18
        assert audit["brands_with_workspaces"] == 18
        assert audit["brands_with_reference_images"] == 18
        assert audit["total_reference_images_on_disk"] >= 18
        assert audit["is_gallery_populated"] is True
        assert "NONE" in audit["blocker_status"]
        assert len(audit["brands_audit"]) == 18

        brand_ids = [b["brand_id"] for b in audit["brands_audit"]]
        assert "sbi" in brand_ids
        assert "paypal" in brand_ids
        assert "microsoft" in brand_ids
        assert "google" in brand_ids
