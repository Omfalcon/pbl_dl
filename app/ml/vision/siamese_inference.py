"""Siamese ResNet-18 Brand Impersonation Inference Interface (M-03 Phase 6.2).

Provides pairwise visual verification and gallery matching against authentic brand assets.
Computes cosine similarity, predicts brand match with similarity thresholds, checks
domain authorization, and explicitly reports INSUFFICIENT_REFERENCE_DATA when assets are absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from app.ml.vision.brand_acquisition import TARGET_18_BRANDS, audit_brand_reference_gallery
from app.ml.vision.preprocessing import IMAGENET_MEAN, IMAGENET_STD, safe_load_image
from app.ml.vision.siamese import SiameseResNet18

logger = logging.getLogger("phishvision.ml.vision.siamese_inference")


class SiameseBrandMatcher:
    """Production and diagnostic inference interface for M-03 Siamese Brand Impersonation Matcher."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        gallery_dir: str | Path = "data/brands/reference_gallery",
        similarity_threshold: float = 0.82,
        image_size: int = 128,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.similarity_threshold = similarity_threshold
        self.image_size = image_size
        self.gallery_dir = Path(gallery_dir)

        # 1. Initialize or load Siamese architecture
        self.model = SiameseResNet18(pretrained=False, embedding_dim=256)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.similarity_threshold = ckpt.get("similarity_threshold", self.similarity_threshold)
            logger.info(f"Loaded Siamese checkpoint from {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        # 2. Evaluation Transform
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # 3. Load Brand Gallery Metadata and Audit Assets
        self.brand_metadata: dict[str, dict[str, Any]] = {}
        self.reference_embeddings: dict[str, list[torch.Tensor]] = {}
        self._load_gallery()

    def _load_gallery(self) -> None:
        """Loads available reference metadata and precomputes gallery embeddings if assets exist."""
        manifest_p = self.gallery_dir / "manifest_brands.json"
        if manifest_p.exists():
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for b in data.get("brands", []):
                        self.brand_metadata[b["brand_id"]] = b
            except Exception as e:
                logger.warning(f"Failed to read brand manifest: {e}")

        # Audit disk for actual images
        audit = audit_brand_reference_gallery(gallery_dir=self.gallery_dir, manifest_path=manifest_p)
        self.has_reference_data = bool(audit["total_reference_images_on_disk"] > 0)

        if not self.has_reference_data:
            logger.info("Brand reference gallery is currently EMPTY on disk (0 images found). Status: INSUFFICIENT_REFERENCE_DATA")
            return

        # Precompute embeddings for all available images
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        for brand_id in TARGET_18_BRANDS:
            b_dir = self.gallery_dir / brand_id
            if not b_dir.exists():
                continue

            brand_imgs: list[Path] = []
            for sub in ["logos", "canonical_screenshots"]:
                sd = b_dir / sub
                if sd.exists():
                    brand_imgs.extend([p for p in sd.iterdir() if p.suffix.lower() in valid_exts])

            if brand_imgs:
                self.reference_embeddings[brand_id] = []
                for img_p in brand_imgs:
                    try:
                        emb = self.compute_embedding(img_p)
                        self.reference_embeddings[brand_id].append(emb)
                    except Exception as e:
                        logger.warning(f"Skipping un-decodable gallery image {img_p}: {e}")

    @torch.no_grad()
    def compute_embedding(self, image_input: str | Path | Image.Image) -> torch.Tensor:
        """Extracts 256-dimensional L2-normalized unit embedding vector."""
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = safe_load_image(image_input)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        embedding = self.model.forward_one(tensor).squeeze(0)
        return embedding

    @torch.no_grad()
    def compute_similarity(
        self,
        image_1: str | Path | Image.Image,
        image_2: str | Path | Image.Image,
    ) -> float:
        """Computes cosine similarity between two image crops in range [-1.0, 1.0]."""
        z1 = self.compute_embedding(image_1)
        z2 = self.compute_embedding(image_2)
        sim = float(torch.sum(z1 * z2).cpu().item())
        return round(sim, 4)

    def match_brand(
        self,
        query_image: str | Path | Image.Image,
        candidate_brand: str | None = None,
    ) -> dict[str, Any]:
        """Matches query logo or website crop against reference gallery assets.
        
        Args:
            query_image: Candidate logo or screenshot crop.
            candidate_brand: Optional prior brand hint from lexical URL analysis.
            
        Returns:
            Structured match result with similarity score and predicted brand.
        """
        if not self.has_reference_data:
            return {
                "status": "INSUFFICIENT_REFERENCE_DATA",
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "confidence": 0.0,
                "is_brand_impersonation": False,
                "message": (
                    "Brand reference gallery is empty on disk (0 authentic images found). "
                    "Cannot match visual query against reference gallery. "
                    "Acquisition required before visual brand matching is operational."
                ),
                "candidate_matches": [],
            }

        q_emb = self.compute_embedding(query_image)
        candidate_scores: list[dict[str, Any]] = []

        brands_to_check = [candidate_brand] if candidate_brand and candidate_brand in self.reference_embeddings else list(self.reference_embeddings.keys())

        best_brand = "UNIDENTIFIED_OR_NEUTRAL"
        best_sim = -1.0

        for b_id in brands_to_check:
            ref_embs = self.reference_embeddings.get(b_id, [])
            if not ref_embs:
                continue

            # Maximum cosine similarity against any reference asset for this brand
            sims = [float(torch.sum(q_emb * r_emb).cpu().item()) for r_emb in ref_embs]
            max_b_sim = max(sims)
            candidate_scores.append({
                "brand_id": b_id,
                "display_name": self.brand_metadata.get(b_id, {}).get("display_name", b_id),
                "similarity": round(max_b_sim, 4),
            })

            if max_b_sim > best_sim:
                best_sim = max_b_sim
                if max_b_sim >= self.similarity_threshold:
                    best_brand = b_id

        candidate_scores.sort(key=lambda x: x["similarity"], reverse=True)

        is_matched = bool(best_brand != "UNIDENTIFIED_OR_NEUTRAL")

        return {
            "status": "MATCHED" if is_matched else "NO_MATCH",
            "predicted_brand": best_brand,
            "max_similarity": round(best_sim, 4),
            "confidence": round(max(best_sim, 0.0), 4),
            "threshold_applied": self.similarity_threshold,
            "candidate_matches": candidate_scores[:5],
        }

    def verify_brand_authorization(
        self,
        query_image: str | Path | Image.Image,
        target_registrable_domain: str,
        candidate_brand: str | None = None,
    ) -> dict[str, Any]:
        """Cross-references visual brand match against domain authorization list.
        
        Crucial Security Rule: Visual brand similarity alone does NOT imply phishing.
        Phishing occurs if and only if visual brand is identified AND target domain is UNAUTHORIZED.
        """
        match_res = self.match_brand(query_image=query_image, candidate_brand=candidate_brand)

        if match_res["status"] == "INSUFFICIENT_REFERENCE_DATA":
            return {
                **match_res,
                "is_brand_impersonation": False,
                "domain_authorization": "UNKNOWN_INSUFFICIENT_DATA",
            }

        matched_brand = match_res["predicted_brand"]
        if matched_brand == "UNIDENTIFIED_OR_NEUTRAL":
            return {
                **match_res,
                "is_brand_impersonation": False,
                "domain_authorization": "NEUTRAL_OR_UNIDENTIFIED",
            }

        # Check official domains in metadata
        b_meta = self.brand_metadata.get(matched_brand, {})
        auth_domains = b_meta.get("official_registrable_domains", [])
        clean_target = target_registrable_domain.strip().lower()

        is_authorized = clean_target in [d.lower() for d in auth_domains]
        is_impersonation = not is_authorized

        return {
            **match_res,
            "target_registrable_domain": clean_target,
            "authorized_domains": auth_domains,
            "is_authorized_domain": is_authorized,
            "is_brand_impersonation": is_impersonation,
            "verdict": "CRITICAL_BRAND_IMPERSONATION" if is_impersonation else "LEGITIMATE_OFFICIAL_BRAND",
        }
