"""Image Preprocessing and Augmentation Pipeline for Website Screenshots (M-02).

Implements configurable viewport cropping (top viewport crop vs. aspect-ratio pad),
deterministic evaluation transforms, safe PIL loading, and ImageNet standardization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger("phishvision.ml.vision.preprocessing")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TopViewportCrop:
    """Crops the top viewport region of a website screenshot where branding and credential forms concentrate."""

    def __init__(self, viewport_ratio: float = 1.0) -> None:
        """
        Args:
            viewport_ratio: Height-to-width ratio of the initial fold to preserve.
                            1.0 preserves a square top viewport fold (0, 0, W, W).
                            0.75 preserves a 4:3 viewport fold.
                            0.5625 preserves a 16:9 viewport fold.
        """
        self.viewport_ratio = viewport_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_h = int(w * self.viewport_ratio)
        if h > target_h:
            # Crop top fold (left=0, top=0, right=w, bottom=target_h)
            return img.crop((0, 0, w, target_h))
        return img


class AspectRatioPadResize:
    """Resizes an image preserving aspect ratio and pads to a square target dimension."""

    def __init__(self, target_size: int = 224, fill_color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.target_size, self.target_size), self.fill_color)

        scale = min(self.target_size / w, self.target_size / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))
        return padded


def build_transforms(
    crop_strategy: str = "top_viewport",
    image_size: int = 224,
    is_training: bool = False,
    horizontal_flip: bool = False,
    color_jitter: bool = True,
    viewport_ratio: float = 1.0,
) -> T.Compose:
    """Build PyTorch torchvision transform pipeline.
    
    Args:
        crop_strategy: "top_viewport" or "aspect_ratio_pad".
        image_size: Output spatial resolution (default: 224).
        is_training: If True, adds controlled non-destructive augmentations.
        horizontal_flip: If True, enables random horizontal flipping (default: False).
        color_jitter: If True and is_training, adds mild brightness/contrast jitter (+-0.1).
        viewport_ratio: Aspect ratio for top viewport fold crop.
        
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transforms_list: list[Any] = []

    # 1. Aspect ratio / Viewport cropping strategy
    if crop_strategy == "top_viewport":
        transforms_list.append(TopViewportCrop(viewport_ratio=viewport_ratio))
        transforms_list.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))
    elif crop_strategy == "aspect_ratio_pad":
        transforms_list.append(AspectRatioPadResize(target_size=image_size))
    else:
        # Fallback standard direct resize
        transforms_list.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))

    # 2. Training augmentations (controlled and non-destructive)
    if is_training:
        if horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        if color_jitter:
            transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05))

    # 3. Conversion to Tensor and ImageNet Standardization
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms_list)


def safe_load_image(image_input: str | Path | Image.Image) -> Image.Image:
    """Safely load and convert an image input into a verified RGB PIL Image.
    
    Handles file paths, existing PIL images, corrupted bytes, and missing files.
    """
    if isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":
            return image_input.convert("RGB")
        return image_input

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Screenshot image file does not exist: {path}")

    try:
        with Image.open(path) as raw_img:
            # Force load into memory to catch truncation/corruption early
            raw_img.load()
            if raw_img.mode != "RGB":
                return raw_img.convert("RGB")
            return raw_img.copy()
    except Exception as e:
        raise ValueError(f"Failed to read or decode image file at {path}: {e}") from e
