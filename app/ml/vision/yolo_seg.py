"""YOLOv8 Instance Segmentation Pipeline for Synthetic UI Elements (Phase 7.7).

Implements M-06 YOLOv8-Seg for detecting UI components and generating pixel-level
instance segmentation masks. Provides programmatic geometric polygon label generation,
semi-transparent mask overlay rendering, and dual Box/Mask mAP evaluation reporting.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("phishvision.yolo_seg")

# Standardized Phase 7 6-Class UI Schema
YOLO_SEG_CLASSES: dict[int, str] = {
    0: "brand_logo",
    1: "login_form",
    2: "username_field",
    3: "password_field",
    4: "submit_button",
    5: "payment_element",
}

CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in YOLO_SEG_CLASSES.items()}

# UI Palette for Visual Overlays (matching Phase 7 standards)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "brand_logo": (37, 99, 235),       # Vibrant Blue
    "login_form": (16, 185, 129),      # Emerald Green
    "username_field": (245, 158, 11),   # Amber/Orange
    "password_field": (239, 68, 68),    # Crimson Red
    "submit_button": (139, 92, 246),    # Indigo/Purple
    "payment_element": (236, 72, 153),  # Magenta/Pink
}

# Disjoint Fictional Brand Names per Split to guarantee ZERO brand leakage
SPLIT_BRANDS: dict[str, list[str]] = {
    "train": [
        "ApexBank", "CloudNexus", "SwiftCart", "NovaStream", "OrbitSocial",
        "PaySphere", "VertexPay", "QuantumSecure", "StarlightID", "ZenithPortal",
        "EchoPay", "AegisTrust", "HyperLink", "PulseNet", "CyberKey", "StratumCore",
    ],
    "val": [
        "KryptonPay", "SolariaBank", "VortexChat", "AeroCart",
    ],
    "test": [
        "TitanVault", "BeaconAuth", "OmniPay", "PrismSocial",
    ],
}

BRAND_COLORS: list[tuple[int, int, int]] = [
    (24, 119, 242),   # Modern blue
    (15, 118, 110),   # Corporate teal
    (13, 148, 136),   # Cyan
    (79, 70, 229),    # Indigo
    (124, 58, 237),   # Violet
    (217, 70, 239),   # Fuchsia
    (225, 29, 72),    # Rose
    (234, 88, 12),    # Orange
    (16, 185, 129),   # Emerald green
    (30, 41, 59),     # Slate dark
]


# =========================================================================
# Geometric Polygon Helpers
# =========================================================================

def rounded_rect_polygon(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float = 8.0,
    n_arc_points: int = 4,
) -> list[tuple[float, float]]:
    """Compute exact boundary polygon vertices for a rectangle with rounded corners."""
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    r = min(radius, w / 2.0, h / 2.0)
    pts: list[tuple[float, float]] = []

    # Top-right arc (270° to 360°)
    cx, cy = x2 - r, y1 + r
    for a in np.linspace(math.pi * 1.5, math.pi * 2.0, n_arc_points, endpoint=True):
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    # Bottom-right arc (0° to 90°)
    cx, cy = x2 - r, y2 - r
    for a in np.linspace(0.0, math.pi * 0.5, n_arc_points, endpoint=True):
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    # Bottom-left arc (90° to 180°)
    cx, cy = x1 + r, y2 - r
    for a in np.linspace(math.pi * 0.5, math.pi, n_arc_points, endpoint=True):
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    # Top-left arc (180° to 270°)
    cx, cy = x1 + r, y1 + r
    for a in np.linspace(math.pi, math.pi * 1.5, n_arc_points, endpoint=True):
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    return pts


def ellipse_polygon(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    n_points: int = 16,
) -> list[tuple[float, float]]:
    """Compute polygon vertices approximating an ellipse or circle."""
    pts: list[tuple[float, float]] = []
    for a in np.linspace(0.0, math.pi * 2.0, n_points, endpoint=False):
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts


def diamond_polygon(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    """Compute 4 vertices of a diamond polygon inscribed in [x1, y1, x2, y2]."""
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    return [
        (mid_x, y1),
        (x2, mid_y),
        (mid_x, y2),
        (x1, mid_y),
    ]


def triangle_polygon(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    """Compute 3 vertices of an isosceles triangle inscribed in [x1, y1, x2, y2]."""
    mid_x = (x1 + x2) / 2.0
    return [
        (mid_x, y1),
        (x2, y2),
        (x1, y2),
    ]


# =========================================================================
# Segmentation Annotation Dataclass
# =========================================================================

@dataclass
class YOLOSegmentationPolygon:
    """Represents an instance segmentation polygon in normalized [0.0, 1.0] coordinates."""
    class_id: int
    points: list[tuple[float, float]]
    class_name: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.class_name and self.class_id in YOLO_SEG_CLASSES:
            self.class_name = YOLO_SEG_CLASSES[self.class_id]

    def to_yolo_seg_line(self) -> str:
        """Format as standard YOLO segmentation text line: <class_id> x1 y1 x2 y2 ..."""
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords_str}"

    def to_pixel_polygon(self, img_width: int, img_height: int) -> list[tuple[int, int]]:
        """Convert normalized coordinates to absolute integer pixel vertices."""
        pixel_pts = []
        for x, y in self.points:
            px = max(0, min(img_width - 1, int(round(x * img_width))))
            py = max(0, min(img_height - 1, int(round(y * img_height))))
            pixel_pts.append((px, py))
        return pixel_pts

    def to_bounding_box(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Compute pixel bounding box (xmin, ymin, xmax, ymax) enclosing the polygon."""
        pts = self.to_pixel_polygon(img_width, img_height)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def rasterize_mask(self, img_width: int, img_height: int) -> np.ndarray:
        """Rasterize polygon into a binary uint8 mask of shape (img_height, img_width)."""
        mask_img = Image.new("L", (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask_img)
        pts = self.to_pixel_polygon(img_width, img_height)
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
        return np.array(mask_img, dtype=np.uint8)


@dataclass
class SyntheticSegmentationPageLayout:
    """Stores generated layout elements and their exact segmentation polygons."""
    image: Image.Image
    polygons: list[YOLOSegmentationPolygon]
    template_name: str
    brand_name: str
    is_empty: bool = False


# =========================================================================
# Synthetic Segmentation Dataset Generator
# =========================================================================

class SyntheticSegmentationGenerator:
    """Generates synthetic website screenshots with exact programmatic polygon segmentation masks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        try:
            self.font_large = ImageFont.truetype("arial.ttf", 22)
            self.font_medium = ImageFont.truetype("arial.ttf", 16)
            self.font_small = ImageFont.truetype("arial.ttf", 13)
            self.font_title = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_title = ImageFont.load_default()

    def _normalize_polygon(
        self,
        pts: list[tuple[float, float]],
        canvas_w: int,
        canvas_h: int,
        class_id: int,
        class_name: str,
    ) -> YOLOSegmentationPolygon:
        """Convert pixel points to clamped normalized [0.0, 1.0] coordinates."""
        norm_pts = []
        for x, y in pts:
            nx = max(0.0, min(1.0, x / float(canvas_w)))
            ny = max(0.0, min(1.0, y / float(canvas_h)))
            norm_pts.append((round(nx, 6), round(ny, 6)))

        return YOLOSegmentationPolygon(
            class_id=class_id,
            points=norm_pts,
            class_name=class_name,
        )

    def _render_logo_segmentation(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        brand_name: str,
        color: tuple[int, int, int],
        canvas_w: int,
        canvas_h: int,
    ) -> YOLOSegmentationPolygon:
        """Draw brand logo with exact geometric polygon (circle/diamond/triangle/square)."""
        sym_size = 28
        shape_type = self.rng.choice(["circle", "square", "diamond", "triangle"])

        if shape_type == "circle":
            draw.ellipse([x, y, x + sym_size, y + sym_size], fill=color)
            pts = ellipse_polygon(x + sym_size / 2.0, y + sym_size / 2.0, sym_size / 2.0, sym_size / 2.0, n_points=16)
        elif shape_type == "square":
            draw.rounded_rectangle([x, y, x + sym_size, y + sym_size], radius=4, fill=color)
            pts = rounded_rect_polygon(x, y, x + sym_size, y + sym_size, radius=4)
        elif shape_type == "diamond":
            mid_x = x + sym_size // 2
            mid_y = y + sym_size // 2
            draw.polygon([(mid_x, y), (x + sym_size, mid_y), (mid_x, y + sym_size), (x, mid_y)], fill=color)
            pts = diamond_polygon(x, y, x + sym_size, y + sym_size)
        else:  # triangle
            mid_x = x + sym_size // 2
            draw.polygon([(mid_x, y), (x + sym_size, y + sym_size), (x, y + sym_size)], fill=color)
            pts = triangle_polygon(x, y, x + sym_size, y + sym_size)

        # Draw inner icon accent
        inner_m = 7
        draw.ellipse([x + inner_m, y + inner_m, x + sym_size - inner_m, y + sym_size - inner_m], fill=(255, 255, 255))

        # Draw brand text lockup
        text_x = x + sym_size + 8
        text_y = y + 3
        draw.text((text_x, text_y), brand_name, fill=(20, 25, 35), font=self.font_large)

        # Build full logo lockup polygon encompassing icon + text
        text_w = len(brand_name) * 12 + 6
        full_pts = rounded_rect_polygon(x - 2, y - 2, text_x + text_w, y + sym_size + 2, radius=4)
        return self._normalize_polygon(full_pts, canvas_w, canvas_h, 0, "brand_logo")

    def _render_standard_login(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> SyntheticSegmentationPageLayout:
        """Render standard single-card login layout with exact polygon segmentation masks."""
        polys: list[YOLOSegmentationPolygon] = []

        # Header bar
        header_h = 56
        draw.rectangle([0, 0, width, header_h], fill=(255, 255, 255), outline=(226, 232, 240))
        logo_poly = self._render_logo_segmentation(draw, 32, 14, brand_name, brand_color, width, height)
        polys.append(logo_poly)

        # Card container
        card_w, card_h = 420, 480
        cx1 = (width - card_w) // 2
        cy1 = (height - card_h) // 2 + 10
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=10, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        form_poly = self._normalize_polygon(rounded_rect_polygon(cx1, cy1, cx2, cy2, radius=10), width, height, 1, "login_form")
        polys.append(form_poly)

        # Card Title
        draw.text((cx1 + 32, cy1 + 32), f"Sign in to {brand_name}", fill=(15, 23, 42), font=self.font_title)
        draw.text((cx1 + 32, cy1 + 72), "Enter your credentials to continue", fill=(100, 116, 139), font=self.font_small)

        # Username field
        fld_y = cy1 + 115
        draw.text((cx1 + 32, fld_y - 20), "Username or Email", fill=(51, 65, 85), font=self.font_small)
        draw.rounded_rectangle([cx1 + 32, fld_y, cx2 - 32, fld_y + 42], radius=6, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        draw.text((cx1 + 44, fld_y + 12), "user@example.com", fill=(148, 163, 184), font=self.font_medium)
        user_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, fld_y, cx2 - 32, fld_y + 42, radius=6), width, height, 2, "username_field")
        polys.append(user_poly)

        # Password field
        pwd_y = fld_y + 75
        draw.text((cx1 + 32, pwd_y - 20), "Password", fill=(51, 65, 85), font=self.font_small)
        draw.rounded_rectangle([cx1 + 32, pwd_y, cx2 - 32, pwd_y + 42], radius=6, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        draw.text((cx1 + 44, pwd_y + 14), "••••••••••••", fill=(100, 116, 139), font=self.font_medium)
        pwd_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, pwd_y, cx2 - 32, pwd_y + 42, radius=6), width, height, 3, "password_field")
        polys.append(pwd_poly)

        # Submit button
        btn_y = pwd_y + 80
        draw.rounded_rectangle([cx1 + 32, btn_y, cx2 - 32, btn_y + 46], radius=6, fill=brand_color)
        draw.text((cx1 + 155, btn_y + 13), "Sign In", fill=(255, 255, 255), font=self.font_large)
        btn_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, btn_y, cx2 - 32, btn_y + 46, radius=6), width, height, 4, "submit_button")
        polys.append(btn_poly)

        return SyntheticSegmentationPageLayout(img, polys, "standard_login", brand_name)

    def _render_payment_checkout(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> SyntheticSegmentationPageLayout:
        """Render checkout payment layout with payment_element polygons."""
        polys: list[YOLOSegmentationPolygon] = []

        # Header
        draw.rectangle([0, 0, width, 56], fill=(255, 255, 255), outline=(226, 232, 240))
        logo_poly = self._render_logo_segmentation(draw, 32, 14, brand_name, brand_color, width, height)
        polys.append(logo_poly)

        # Form card
        card_w, card_h = 520, 520
        cx1 = (width - card_w) // 2
        cy1 = (height - card_h) // 2 + 10
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=10, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        form_poly = self._normalize_polygon(rounded_rect_polygon(cx1, cy1, cx2, cy2, radius=10), width, height, 1, "login_form")
        polys.append(form_poly)

        draw.text((cx1 + 32, cy1 + 24), "Payment Information", fill=(15, 23, 42), font=self.font_title)

        # Credit card field (payment_element)
        fld_y = cy1 + 80
        draw.text((cx1 + 32, fld_y - 18), "Card Number", fill=(51, 65, 85), font=self.font_small)
        draw.rounded_rectangle([cx1 + 32, fld_y, cx2 - 32, fld_y + 42], radius=6, fill=(248, 250, 252), outline=(203, 213, 225))
        draw.text((cx1 + 44, fld_y + 12), "4532 •••• •••• 8912", fill=(100, 116, 139), font=self.font_medium)
        card_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, fld_y, cx2 - 32, fld_y + 42, radius=6), width, height, 5, "payment_element")
        polys.append(card_poly)

        # Expiration date (payment_element)
        sub_y = fld_y + 65
        mid_x = (cx1 + cx2) // 2
        draw.text((cx1 + 32, sub_y - 18), "Expiration", fill=(51, 65, 85), font=self.font_small)
        draw.rounded_rectangle([cx1 + 32, sub_y, mid_x - 10, sub_y + 42], radius=6, fill=(248, 250, 252), outline=(203, 213, 225))
        draw.text((cx1 + 44, sub_y + 12), "12/28", fill=(100, 116, 139), font=self.font_medium)
        exp_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, sub_y, mid_x - 10, sub_y + 42, radius=6), width, height, 5, "payment_element")
        polys.append(exp_poly)

        # CVV (payment_element)
        draw.text((mid_x + 10, sub_y - 18), "Security Code (CVV)", fill=(51, 65, 85), font=self.font_small)
        draw.rounded_rectangle([mid_x + 10, sub_y, cx2 - 32, sub_y + 42], radius=6, fill=(248, 250, 252), outline=(203, 213, 225))
        draw.text((mid_x + 22, sub_y + 12), "•••", fill=(100, 116, 139), font=self.font_medium)
        cvv_poly = self._normalize_polygon(rounded_rect_polygon(mid_x + 10, sub_y, cx2 - 32, sub_y + 42, radius=6), width, height, 5, "payment_element")
        polys.append(cvv_poly)

        # Submit button
        btn_y = sub_y + 80
        draw.rounded_rectangle([cx1 + 32, btn_y, cx2 - 32, btn_y + 46], radius=6, fill=brand_color)
        draw.text((cx1 + 175, btn_y + 13), "Pay Now", fill=(255, 255, 255), font=self.font_large)
        btn_poly = self._normalize_polygon(rounded_rect_polygon(cx1 + 32, btn_y, cx2 - 32, btn_y + 46, radius=6), width, height, 4, "submit_button")
        polys.append(btn_poly)

        return SyntheticSegmentationPageLayout(img, polys, "payment_checkout", brand_name)

    def _render_empty_page(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
    ) -> SyntheticSegmentationPageLayout:
        """Render 404/Empty negative control page with zero target polygons."""
        cx = width // 2
        cy = height // 2 - 40
        draw.text((cx - 80, cy), "404 Not Found", fill=(148, 163, 184), font=self.font_title)
        draw.text((cx - 130, cy + 45), "The requested URL was not found on this server.", fill=(148, 163, 184), font=self.font_medium)
        return SyntheticSegmentationPageLayout(img, [], "empty_page", "None", is_empty=True)

    def generate_page(
        self,
        brand_name: str,
        template_type: str,
        width: int = 512,
        height: int = 512,
    ) -> SyntheticSegmentationPageLayout:
        """Generate a single synthetic webpage layout with exact instance polygon annotations."""
        img = Image.new("RGB", (width, height), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        brand_color = self.rng.choice(BRAND_COLORS)

        if template_type == "empty_page":
            return self._render_empty_page(img, draw, width, height)
        elif template_type == "payment_checkout":
            return self._render_payment_checkout(img, draw, brand_name, brand_color, width, height)
        else:
            return self._render_standard_login(img, draw, brand_name, brand_color, width, height)

    def generate_dataset(
        self,
        output_dir: str | Path = "data/synthetic_yolo_seg",
        counts: dict[str, int] | None = None,
        width: int = 512,
        height: int = 512,
    ) -> dict[str, Any]:
        """Generate complete synthetic segmentation dataset with exact polygon labels and masks."""
        if counts is None:
            counts = {"train": 560, "val": 120, "test": 120}

        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        manifest_records = []
        total_generated = 0
        split_stats: dict[str, Any] = {}

        for split_name, num_samples in counts.items():
            img_dir = out_p / "images" / split_name
            lbl_dir = out_p / "labels" / split_name
            msk_dir = out_p / "masks" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)

            brands = SPLIT_BRANDS[split_name]
            templates = ["standard_login", "payment_checkout", "empty_page"]
            weights = [0.70, 0.20, 0.10]

            split_poly_counts = {c: 0 for c in YOLO_SEG_CLASSES.values()}
            split_empty_count = 0

            for idx in range(num_samples):
                sample_id = f"synth_seg_{split_name}_{idx:04d}"
                brand = self.rng.choice(brands)
                template = self.rng.choices(templates, weights=weights, k=1)[0]

                layout = self.generate_page(brand, template, width=width, height=height)

                # Save image
                img_path = img_dir / f"{sample_id}.png"
                layout.image.save(img_path, format="PNG")

                # Save YOLO segmentation label file
                lbl_path = lbl_dir / f"{sample_id}.txt"
                with open(lbl_path, "w", encoding="utf-8") as f:
                    for poly in layout.polygons:
                        f.write(poly.to_yolo_seg_line() + "\n")

                # Save 8-bit multi-class semantic mask
                msk_path = msk_dir / f"{sample_id}_mask.png"
                mask_canvas = Image.new("L", (width, height), 0)
                mask_draw = ImageDraw.Draw(mask_canvas)
                for poly in layout.polygons:
                    pts = poly.to_pixel_polygon(width, height)
                    if len(pts) >= 3:
                        mask_draw.polygon(pts, fill=poly.class_id + 1)
                mask_canvas.save(msk_path, format="PNG")

                if layout.is_empty:
                    split_empty_count += 1
                else:
                    for poly in layout.polygons:
                        split_poly_counts[poly.class_name] += 1

                manifest_records.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "template": template,
                    "brand": brand,
                    "is_empty": layout.is_empty,
                    "num_polygons": len(layout.polygons),
                    "image_path": str(img_path.relative_to(out_p.parent.parent)),
                    "label_path": str(lbl_path.relative_to(out_p.parent.parent)),
                    "mask_path": str(msk_path.relative_to(out_p.parent.parent)),
                })
                total_generated += 1

            split_stats[split_name] = {
                "total_images": num_samples,
                "empty_images": split_empty_count,
                "total_polygons": sum(split_poly_counts.values()),
                "classes": split_poly_counts,
                "brands_used": sorted(brands),
            }

        # Write dataset.yaml
        dataset_yaml_content = f"""# Ultralytics YOLOv8 Segmentation Dataset Configuration — PhishVision AI (M-06)
path: {out_p.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: brand_logo
  1: login_form
  2: username_field
  3: password_field
  4: submit_button
  5: payment_element
"""
        yaml_path = out_p / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(dataset_yaml_content)

        # Write manifest CSV
        df_manifest = pd.DataFrame(manifest_records)
        df_manifest.to_csv(out_p / "manifest.csv", index=False)

        # Write dataset statistics
        meta = {
            "project": "PhishVision AI",
            "module": "M-06 YOLOv8n-Seg Synthetic UI Elements",
            "seed": self.seed,
            "total_images": total_generated,
            "classes": YOLO_SEG_CLASSES,
            "split_counts": counts,
            "split_stats": split_stats,
        }
        with open(out_p / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Generated {total_generated} synthetic segmentation images under {out_p}")
        return meta


# =========================================================================
# YOLO-Seg Inference Interface & Annotated Preview
# =========================================================================

class YOLOSegInferencer:
    """Inference engine for M-06 YOLOv8-Seg instance segmentation."""

    def __init__(
        self,
        model_weights: str | Path = "yolov8n-seg.pt",
        device: str | int | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(str(model_weights))
        logger.info(f"Loaded YOLO-Seg model from: {model_weights}")

    def predict(
        self,
        image: Image.Image | Path | str,
        conf_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform instance segmentation on an input image."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        else:
            img_pil = image.convert("RGB")

        orig_w, orig_h = img_pil.size
        results = self.model.predict(
            source=img_pil,
            conf=conf,
            device=self.device,
            verbose=False,
        )

        instances = []
        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            masks = res.masks

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cids = boxes.cls.cpu().numpy().astype(int)

                # Parse masks if available
                poly_list = []
                if masks is not None and masks.xy is not None:
                    poly_list = masks.xy

                for i in range(len(boxes)):
                    cid = int(cids[i])
                    cname = YOLO_SEG_CLASSES.get(cid, f"class_{cid}")
                    score = float(confs[i])
                    box = [int(round(coord)) for coord in xyxy[i]]

                    poly_pts = []
                    if i < len(poly_list) and len(poly_list[i]) > 0:
                        poly_pts = [[float(p[0]), float(p[1])] for p in poly_list[i]]

                    instances.append({
                        "class_id": cid,
                        "class_name": cname,
                        "confidence": round(score, 4),
                        "box_xyxy": box,
                        "polygon_points": poly_pts,
                    })

        return {
            "image_size": [orig_w, orig_h],
            "num_instances": len(instances),
            "instances": instances,
        }

    def render_annotated_preview(
        self,
        image: Image.Image | Path | str,
        output_path: Path | str,
        conf_threshold: float | None = None,
        mask_alpha: float = 0.40,
    ) -> Path:
        """Render semi-transparent filled segmentation masks and bounding box labels."""
        if isinstance(image, (str, Path)):
            base_img = Image.open(image).convert("RGBA")
        else:
            base_img = image.convert("RGBA")

        w, h = base_img.size
        pred = self.predict(base_img, conf_threshold=conf_threshold)

        # Create overlay layer for translucent masks
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_base = ImageDraw.Draw(base_img)

        for inst in pred["instances"]:
            cname = inst["class_name"]
            conf = inst["confidence"]
            box = inst["box_xyxy"]
            pts = inst["polygon_points"]
            color_rgb = CLASS_COLORS.get(cname, (239, 68, 68))
            alpha_val = int(round(mask_alpha * 255))
            color_rgba = (color_rgb[0], color_rgb[1], color_rgb[2], alpha_val)

            # Draw filled polygon mask
            if len(pts) >= 3:
                flat_pts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
                draw_overlay.polygon(flat_pts, fill=color_rgba)
                draw_base.line(flat_pts + [flat_pts[0]], fill=color_rgb, width=2)
            else:
                # Fallback to box outline if mask is degenerate
                draw_base.rectangle(box, outline=color_rgb, width=2)

            # Draw label banner
            x1, y1, x2, y2 = box
            label_text = f"{cname} {conf:.2f}"
            banner_w = len(label_text) * 7 + 4
            draw_base.rectangle([x1, max(0, y1 - 18), x1 + banner_w, y1], fill=color_rgb)
            draw_base.text((x1 + 2, max(0, y1 - 16)), label_text, fill=(255, 255, 255))

        # Alpha composite overlay onto base image
        composed = Image.alpha_composite(base_img, overlay).convert("RGB")
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        composed.save(out_p, format="PNG")
        return out_p


# =========================================================================
# Evaluation & Metric Reporting Helper
# =========================================================================

def generate_segmentation_evaluation_report(
    eval_metrics: dict[str, Any],
    output_dir: Path | str = "data/reports/yolo_seg",
    checkpoint_path: Path | str = "models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt",
) -> dict[str, Any]:
    """Generate dual Box and Mask evaluation report with markdown summary."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    # Class metrics CSV
    class_rows = []
    for cid_str, info in eval_metrics.get("class_metrics", {}).items():
        class_rows.append({
            "class_id": int(cid_str),
            "class_name": info["class_name"],
            "box_precision": info.get("box_precision", 0.0),
            "box_recall": info.get("box_recall", 0.0),
            "box_mAP50": info.get("box_mAP50", 0.0),
            "box_mAP50_95": info.get("box_mAP50_95", 0.0),
            "mask_precision": info.get("mask_precision", 0.0),
            "mask_recall": info.get("mask_recall", 0.0),
            "mask_mAP50": info.get("mask_mAP50", 0.0),
            "mask_mAP50_95": info.get("mask_mAP50_95", 0.0),
        })

    if class_rows:
        df = pd.DataFrame(class_rows)
        df.to_csv(out_dir / "class_metrics.csv", index=False)

    md_content = f"""# M-06 YOLOv8n-Seg Synthetic UI Elements — Instance Segmentation Report

**Phase:** Phase 7.7 YOLOv8 Instance Segmentation  
**Architecture:** Ultralytics YOLOv8n-seg (Nano Segmentation)  
**Evaluated Checkpoint:** `{checkpoint_path}`  
**Test Split:** `test` (`data/synthetic_yolo_seg/images/test`, 120 synthetic images)  
**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Dual Evaluation: Bounding Box Detection vs. Pixel Instance Segmentation

| Metric Tier | mAP@50 | mAP@50-95 | Mean Precision | Mean Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Bounding Box Detection** | **{eval_metrics.get('box_mAP50', 'N/A')}** | **{eval_metrics.get('box_mAP50_95', 'N/A')}** | {eval_metrics.get('box_precision', 'N/A')} | {eval_metrics.get('box_recall', 'N/A')} |
| **Instance Mask Segmentation** | **{eval_metrics.get('mask_mAP50', 'N/A')}** | **{eval_metrics.get('mask_mAP50_95', 'N/A')}** | {eval_metrics.get('mask_precision', 'N/A')} | {eval_metrics.get('mask_recall', 'N/A')} |

---

## 2. Per-Class Mask Segmentation Accuracy

| Class ID | Class Name | Mask Precision | Mask Recall | Mask mAP@50 | Mask mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    for r in class_rows:
        md_content += f"| {r['class_id']} | `{r['class_name']}` | {r['mask_precision']} | {r['mask_recall']} | {r['mask_mAP50']} | {r['mask_mAP50_95']} |\n"

    md_content += f"""
---

## 3. Operational & Latency Telemetry

- **Mean Inference Latency:** {eval_metrics.get('latency_ms', 'N/A')} ms / image
- **Checkpoint Model Size:** {eval_metrics.get('checkpoint_size_mb', 'N/A')} MB
- **Hardware Profile:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA)

---

## 4. Controlled Synthetic Experiment & Generalization Disclaimer

> [!WARNING]
> **Synthetic Dataset Notice:**  
> This benchmark evaluates bounding box and pixel segmentation mask accuracy on programmatically rendered synthetic webpage templates. While it establishes that YOLOv8n-seg learns exact geometric contours (rounded card containers, input fields, pill buttons, and brand logo symbols), **these synthetic benchmark numbers do NOT represent real-world phishing detection accuracy**.
> 
> Real-world phishing pages exhibit dynamic DOM styling, obfuscated CSS, nested iframes, responsive canvas scaling, and zero-day design variations.
"""

    report_p = out_dir / "YOLO_SEG_EXPERIMENT_REPORT.md"
    with open(report_p, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved YOLO-Seg experiment report to: {report_p}")
    return eval_metrics
