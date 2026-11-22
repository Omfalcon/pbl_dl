"""Visualization Tools for Bounding Box Annotations and Semantic Segmentation Overlays."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .validate_annotations import DETECTION_CLASSES, SEGMENTATION_CLASSES, AnnotationValidator, BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.visualize_annotations")

# Bounding box color palette (BGR for OpenCV)
CLASS_COLORS = {
    0: (255, 50, 50),     # Logo: Blue
    1: (50, 200, 50),     # Username: Green
    2: (0, 0, 255),       # Password: Red
    3: (0, 215, 255),     # Submit: Yellow/Gold
    4: (255, 0, 255),     # QR Code: Magenta
}

# Segmentation color palette (RGB)
SEGMENTATION_PALETTE = {
    0: (0, 0, 0),         # Background: Black
    1: (66, 133, 244),    # Header: Google Blue
    2: (52, 168, 83),     # Navigation: Green
    3: (234, 67, 53),     # Credential Form: Red
    4: (251, 188, 4),     # Main Content: Yellow
    5: (154, 160, 166),   # Footer: Grey
}


class AnnotationVisualizer:
    """Renders visual overlays of bounding boxes and semantic segmentation masks for human review."""

    @staticmethod
    def render_detection_overlay(
        image_path: Path | str,
        boxes: list[BoundingBox],
        output_path: Path | str,
        thickness: int = 2,
    ) -> bool:
        """Draw YOLO bounding boxes and class labels onto an image and save preview."""
        img_p = Path(image_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_p))
        if img is None:
            logger.error(f"Failed to load image for visualization: {image_path}")
            return False

        h, w = img.shape[:2]

        for box in boxes:
            # Denormalize YOLO coordinates
            bx = int((box.x_center - box.width / 2.0) * w)
            by = int((box.y_center - box.height / 2.0) * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w - 1, bx + bw)
            y2 = min(h - 1, by + bh)

            color = CLASS_COLORS.get(box.class_id, (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = f"{box.class_name} [{box.class_id}]"
            # Background pill for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(out_p), img)
        return True

    @staticmethod
    def render_segmentation_overlay(
        image_path: Path | str,
        mask_path: Path | str,
        output_path: Path | str,
        alpha: float = 0.5,
    ) -> bool:
        """Blend semantic segmentation mask colors over original image."""
        img_p = Path(image_path)
        mask_p = Path(mask_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_p))
        if img is None:
            return False

        mask_img = Image.open(mask_p)
        mask_arr = np.array(mask_img)

        # Resize mask if dimensions don't match image
        if mask_arr.shape[:2] != img.shape[:2]:
            mask_arr = cv2.resize(mask_arr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        colored_mask = np.zeros_like(img, dtype=np.uint8)
        for class_id, rgb_col in SEGMENTATION_PALETTE.items():
            if class_id == 0:
                continue  # Keep background transparent
            bgr_col = (rgb_col[2], rgb_col[1], rgb_col[0])
            colored_mask[mask_arr == class_id] = bgr_col

        # Blend mask over image where mask is non-zero
        overlay = img.copy()
        mask_fg = mask_arr > 0
        overlay[mask_fg] = cv2.addWeighted(img, 1.0 - alpha, colored_mask, alpha, 0)[mask_fg]

        cv2.imwrite(str(out_p), overlay)
        return True
