"""Curated Brand Reference Gallery Acquisition & Audit Pipeline (M-03 Phase 6.2).

Audits the 18 target brand workspaces, inspects reference gallery assets,
and provides a reproducible pipeline for acquiring, validating, hashing (SHA-256),
and registering authentic brand logos and screenshots from official domain endpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("phishvision.ml.vision.brand_acquisition")

TARGET_18_BRANDS = [
    "sbi",
    "hdfc",
    "paypal",
    "microsoft",
    "google",
    "apple",
    "amazon",
    "chase",
    "wellsfargo",
    "bofa",
    "facebook",
    "instagram",
    "whatsapp",
    "netflix",
    "linkedin",
    "adobe",
    "dhl",
    "fedex",
]


def audit_brand_reference_gallery(
    gallery_dir: str | Path = "data/brands/reference_gallery",
    manifest_path: str | Path = "data/brands/reference_gallery/manifest_brands.json",
) -> dict[str, Any]:
    """Audits the physical brand reference gallery on disk for the 18 target brands.
    
    Verifies existence of manifests, folders, logos, and canonical screenshots,
    returning an empirical inventory without fabricating any data.
    """
    g_dir = Path(gallery_dir)
    m_path = Path(manifest_path)

    manifest_exists = m_path.exists()
    manifest_brands: list[str] = []
    if manifest_exists:
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            manifest_brands = [b["brand_id"] for b in data.get("brands", [])]

    brand_audit_records: list[dict[str, Any]] = []
    total_images_found = 0
    brands_with_images = 0

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

    for brand_id in TARGET_18_BRANDS:
        b_dir = g_dir / brand_id
        dir_exists = b_dir.exists()
        meta_exists = (b_dir / "metadata.json").exists() if dir_exists else False

        logos_dir = b_dir / "logos" if dir_exists else None
        shots_dir = b_dir / "canonical_screenshots" if dir_exists else None

        logo_files: list[str] = []
        if logos_dir and logos_dir.exists():
            logo_files = [f.name for f in logos_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

        shot_files: list[str] = []
        if shots_dir and shots_dir.exists():
            shot_files = [f.name for f in shots_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

        img_count = len(logo_files) + len(shot_files)
        total_images_found += img_count
        if img_count > 0:
            brands_with_images += 1

        brand_audit_records.append({
            "brand_id": brand_id,
            "workspace_directory_exists": dir_exists,
            "metadata_json_exists": meta_exists,
            "logo_images_count": len(logo_files),
            "logo_files": logo_files,
            "canonical_screenshots_count": len(shot_files),
            "canonical_screenshot_files": shot_files,
            "total_assets_on_disk": img_count,
            "status": "ready" if img_count > 0 else "awaiting_assets",
        })

    is_gallery_populated = total_images_found > 0

    return {
        "manifest_path": str(m_path),
        "manifest_exists": manifest_exists,
        "target_brands_count": len(TARGET_18_BRANDS),
        "manifest_brands_count": len(manifest_brands),
        "brands_with_workspaces": sum(1 for r in brand_audit_records if r["workspace_directory_exists"]),
        "brands_with_reference_images": brands_with_images,
        "total_reference_images_on_disk": total_images_found,
        "total_images": total_images_found,
        "brands": [r["brand_id"] for r in brand_audit_records],
        "is_gallery_populated": is_gallery_populated,
        "blocker_status": (
            "NONE: Gallery is populated."
            if is_gallery_populated
            else "BLOCKED: Reference images (logos and canonical screenshots) are missing on disk. "
                  "0 images found across all 18 brand directories. Acquisition required before training."
        ),
        "brands_audit": brand_audit_records,
    }


def compute_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of an authentic brand reference file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BrandReferenceAcquisitionPipeline:
    """Reproducible pipeline for acquiring, validating, hashing, and storing authentic brand assets."""

    def __init__(
        self,
        gallery_dir: str | Path = "data/brands/reference_gallery",
        registry_path: str | Path = "data/brands/reference_gallery/brand_assets_registry.json",
    ) -> None:
        self.gallery_dir = Path(gallery_dir)
        self.registry_path = Path(registry_path)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, list[dict[str, Any]]] = self._load_registry()

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {b: [] for b in TARGET_18_BRANDS}

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def register_brand_asset(
        self,
        brand_id: str,
        asset_type: str,  # "logo_primary", "logo_icon", "canonical_login", "canonical_home"
        image_input: str | Path | Image.Image,
        source_url: str,
        notes: str = "",
        save_as_name: str | None = None,
    ) -> dict[str, Any]:
        """Validates and registers an authentic brand asset, recording SHA-256 and metadata.
        
        Args:
            brand_id: Identifier from TARGET_18_BRANDS.
            asset_type: Semantic classification of asset.
            image_input: File path or PIL Image.
            source_url: Official brand origin URL where the asset was captured.
            notes: Additional context (e.g. 'Desktop Chrome 1280x720 capture').
            save_as_name: Optional explicit filename.
            
        Returns:
            Registered asset record.
        """
        if brand_id not in TARGET_18_BRANDS:
            raise ValueError(f"Unknown brand ID '{brand_id}'. Must be one of: {TARGET_18_BRANDS}")

        target_subfolder = "logos" if "logo" in asset_type else "canonical_screenshots"
        brand_folder = self.gallery_dir / brand_id / target_subfolder
        brand_folder.mkdir(parents=True, exist_ok=True)

        # Load and validate PIL Image
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            fname = save_as_name or f"{asset_type}.png"
            dest_path = brand_folder / fname
            pil_img.save(dest_path, format="PNG")
        else:
            src_path = Path(image_input)
            if not src_path.exists():
                raise FileNotFoundError(f"Source asset not found: {src_path}")
            pil_img = Image.open(src_path).convert("RGB")
            fname = save_as_name or f"{asset_type}{src_path.suffix}"
            dest_path = brand_folder / fname
            # Copy/save
            pil_img.save(dest_path, format="PNG")

        w, h = pil_img.size
        sha256 = compute_sha256(dest_path)
        capture_date = datetime.now(timezone.utc).isoformat()

        record = {
            "asset_id": f"{brand_id}_{asset_type}_{sha256[:8]}",
            "brand_id": brand_id,
            "asset_type": asset_type,
            "relative_path": str(dest_path.as_posix()),
            "source_url": source_url,
            "capture_date": capture_date,
            "sha256": sha256,
            "width": w,
            "height": h,
            "channels": 3,
            "notes": notes,
        }

        if brand_id not in self.registry:
            self.registry[brand_id] = []
        self.registry[brand_id].append(record)
        self._save_registry()

        logger.info(f"Registered brand asset '{asset_type}' for {brand_id} (SHA256: {sha256[:12]}...)")
        return record
