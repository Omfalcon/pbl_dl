"""Image Integrity Validation, Blank Detection & Standardization Module."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.validate_images")


@dataclass
class ImageValidationResult:
    """Detailed validation status of an individual image file."""
    file_path: str
    is_valid: bool
    error_reason: str | None = None
    width: int = 0
    height: int = 0
    channels: int = 0
    is_blank: bool = False
    std_dev: float = 0.0
    aspect_ratio: float = 0.0
    file_size_bytes: int = 0


@dataclass
class BatchImageValidationSummary:
    """Aggregate statistics from validating a batch of image screenshots."""
    total_scanned: int = 0
    valid_count: int = 0
    corrupt_count: int = 0
    empty_count: int = 0
    blank_count: int = 0
    too_small_count: int = 0
    results: list[ImageValidationResult] = field(default_factory=list)


class ImageValidator:
    """Validates screenshot image files, identifies corrupt or blank captures, and standardizes formats."""

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        blank_std_threshold: float = 3.0,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.blank_std_threshold = blank_std_threshold

    def validate_file(self, file_path: Path | str) -> ImageValidationResult:
        """Inspect a single image file for corruption, emptiness, or blank rendering."""
        p = Path(file_path)
        if not p.exists():
            return ImageValidationResult(
                file_path=str(p),
                is_valid=False,
                error_reason="FILE_NOT_FOUND",
            )

        file_size = p.stat().st_size
        if file_size == 0:
            return ImageValidationResult(
                file_path=str(p),
                is_valid=False,
                error_reason="EMPTY_FILE_ZERO_BYTES",
                file_size_bytes=0,
            )

        try:
            with Image.open(p) as img:
                img.verify()  # Fast check for header corruption
        except Exception as e:
            return ImageValidationResult(
                file_path=str(p),
                is_valid=False,
                error_reason=f"CORRUPT_HEADER_{type(e).__name__}",
                file_size_bytes=file_size,
            )

        # Full decode test and pixel statistics
        try:
            with Image.open(p) as img:
                # Convert to RGB to ensure uniform channel representation
                rgb_img = img.convert("RGB")
                width, height = rgb_img.size
                channels = len(rgb_img.getbands())

                if width < self.min_width or height < self.min_height:
                    return ImageValidationResult(
                        file_path=str(p),
                        is_valid=False,
                        error_reason=f"TOO_SMALL_{width}x{height}",
                        width=width,
                        height=height,
                        channels=channels,
                        file_size_bytes=file_size,
                        aspect_ratio=round(width / max(height, 1), 2),
                    )

                # Convert to grayscale array to test variance/blankness
                gray_arr = np.array(ImageOps.grayscale(rgb_img), dtype=np.float32)
                std_dev = float(np.std(gray_arr))

                # Standard deviation below threshold indicates a single solid color (all white, black, or grey)
                is_blank = std_dev < self.blank_std_threshold
                if is_blank:
                    return ImageValidationResult(
                        file_path=str(p),
                        is_valid=False,
                        error_reason="BLANK_UNIFORM_SCREENSHOT",
                        width=width,
                        height=height,
                        channels=channels,
                        is_blank=True,
                        std_dev=round(std_dev, 2),
                        file_size_bytes=file_size,
                        aspect_ratio=round(width / max(height, 1), 2),
                    )

                return ImageValidationResult(
                    file_path=str(p),
                    is_valid=True,
                    width=width,
                    height=height,
                    channels=channels,
                    is_blank=False,
                    std_dev=round(std_dev, 2),
                    file_size_bytes=file_size,
                    aspect_ratio=round(width / max(height, 1), 2),
                )
        except Exception as e:
            return ImageValidationResult(
                file_path=str(p),
                is_valid=False,
                error_reason=f"DECODE_FAILURE_{type(e).__name__}",
                file_size_bytes=file_size,
            )

    def validate_directory(
        self,
        directory_path: Path | str,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> BatchImageValidationSummary:
        """Recursively scan and validate all image screenshots inside a directory."""
        dir_p = Path(directory_path)
        summary = BatchImageValidationSummary()

        if not dir_p.exists():
            logger.warning(f"Directory {directory_path} does not exist.")
            return summary

        image_files = [f for f in dir_p.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
        summary.total_scanned = len(image_files)

        for img_p in image_files:
            res = self.validate_file(img_p)
            summary.results.append(res)
            if res.is_valid:
                summary.valid_count += 1
            else:
                reason = res.error_reason or ""
                if "ZERO_BYTES" in reason or "FILE_NOT_FOUND" in reason:
                    summary.empty_count += 1
                elif "BLANK" in reason:
                    summary.blank_count += 1
                elif "TOO_SMALL" in reason:
                    summary.too_small_count += 1
                else:
                    summary.corrupt_count += 1

        logger.info(
            f"Validation complete for {dir_p.name}: {summary.valid_count}/{summary.total_scanned} valid "
            f"({summary.blank_count} blank, {summary.corrupt_count} corrupt, {summary.too_small_count} too small)"
        )
        return summary

    @staticmethod
    def standardize_image(
        input_path: Path | str,
        output_path: Path | str,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = False,
    ) -> tuple[bool, str | None]:
        """Resize image using high-quality Lanczos resampling and convert to standardized RGB."""
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(in_p) as img:
                rgb = img.convert("RGB")
                resized = rgb.resize(target_size, resample=Image.Resampling.LANCZOS)
                resized.save(out_p, format="PNG", optimize=True)
                return True, None
        except Exception as e:
            return False, f"Standardization failed: {e}"
