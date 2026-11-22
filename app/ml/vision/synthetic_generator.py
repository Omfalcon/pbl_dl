"""Synthetic Webpage Generator & Exact YOLO Annotation Engine (Phase 7.2).

Generates visually rich, diverse synthetic website screenshots across multiple
archetypes (standard login, banking, social, ecommerce, payment checkout, and empty pages)
with pixel-exact, automatically normalized YOLO bounding-box annotations.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.ml.vision.yolo_annotations import (
    CLASS_NAME_TO_ID,
    YOLO_DETECTION_CLASSES,
    YOLOAnnotationValidator,
    YOLOBoundingBox,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.synthetic_generator")

# Disjoint Fictional Brand Names per Split to guarantee ZERO brand/template leakage
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

# Color palettes per brand archetype
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


@dataclass
class SyntheticPageLayout:
    """Stores generated layout elements and their exact bounding boxes."""
    image: Image.Image
    boxes: list[YOLOBoundingBox]
    template_name: str
    brand_name: str
    is_empty: bool = False


class SyntheticWebpageGenerator:
    """Generates synthetic website screenshots and programmatic exact YOLO annotations."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Load fonts
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

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
        radius: int = 6,
    ) -> None:
        """Draw a rectangle with rounded corners."""
        x1, y1, x2, y2 = box
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

    def _draw_logo(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        brand_name: str,
        color: tuple[int, int, int],
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[tuple[int, int, int, int], YOLOBoundingBox]:
        """Draw brand logo symbol and text lockup, returning pixel box and YOLO annotation."""
        sym_size = 28
        shape_type = self.rng.choice(["circle", "square", "triangle", "diamond"])

        # Draw symbol
        if shape_type == "circle":
            draw.ellipse([x, y, x + sym_size, y + sym_size], fill=color)
        elif shape_type == "square":
            self._draw_rounded_rect(draw, (x, y, x + sym_size, y + sym_size), fill=color, radius=4)
        elif shape_type == "diamond":
            mid_x = x + sym_size // 2
            mid_y = y + sym_size // 2
            draw.polygon([(mid_x, y), (x + sym_size, mid_y), (mid_x, y + sym_size), (x, mid_y)], fill=color)
        else:  # triangle
            draw.polygon([(x + sym_size // 2, y), (x + sym_size, y + sym_size), (x, y + sym_size)], fill=color)

        # Draw inner icon accent
        inner_m = 7
        draw.ellipse([x + inner_m, y + inner_m, x + sym_size - inner_m, y + sym_size - inner_m], fill=(255, 255, 255))

        # Draw brand text lockup
        text_x = x + sym_size + 8
        text_y = y + 3
        draw.text((text_x, text_y), brand_name, fill=(20, 25, 35), font=self.font_large)

        # Estimate text width
        text_w = len(brand_name) * 12 + 6
        x2 = text_x + text_w
        y2 = y + sym_size

        pixel_box = (x, y, x2, y2)
        yolo_box = self._to_yolo_box(0, pixel_box, canvas_w, canvas_h, "brand_logo")
        return pixel_box, yolo_box

    def _to_yolo_box(
        self,
        class_id: int,
        pixel_box: tuple[int, int, int, int],
        canvas_w: int,
        canvas_h: int,
        class_name: str,
    ) -> YOLOBoundingBox:
        """Convert pixel coordinates (x1, y1, x2, y2) to normalized YOLOBoundingBox."""
        x1, y1, x2, y2 = pixel_box
        x_center = (x1 + x2) / (2.0 * canvas_w)
        y_center = (y1 + y2) / (2.0 * canvas_h)
        width = (x2 - x1) / float(canvas_w)
        height = (y2 - y1) / float(canvas_h)

        return YOLOBoundingBox(
            class_id=class_id,
            x_center=round(x_center, 6),
            y_center=round(y_center, 6),
            width=round(width, 6),
            height=round(height, 6),
            class_name=class_name,
        )

    def generate_page(
        self,
        brand_name: str,
        template_type: str,
        width: int = 1280,
        height: int = 800,
    ) -> SyntheticPageLayout:
        """Generate a single synthetic webpage layout with exact annotations."""
        img = Image.new("RGB", (width, height), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        boxes: list[YOLOBoundingBox] = []

        brand_color = self.rng.choice(BRAND_COLORS)

        if template_type == "empty_page":
            return self._render_empty_page(img, draw, width, height)

        if template_type == "standard_login":
            return self._render_standard_login(img, draw, brand_name, brand_color, width, height)
        elif template_type == "banking_login":
            return self._render_banking_login(img, draw, brand_name, brand_color, width, height)
        elif template_type == "social_login":
            return self._render_social_login(img, draw, brand_name, brand_color, width, height)
        elif template_type == "ecommerce_login":
            return self._render_ecommerce_login(img, draw, brand_name, brand_color, width, height)
        elif template_type == "payment_checkout":
            return self._render_payment_checkout(img, draw, brand_name, brand_color, width, height)
        else:
            return self._render_standard_login(img, draw, brand_name, brand_color, width, height)

    def _render_standard_login(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        boxes: list[YOLOBoundingBox] = []

        # Background styling: optional soft gradient or subtle tint
        bg_color = self.rng.choice([(245, 247, 250), (241, 245, 249), (238, 242, 246), (255, 255, 255)])
        draw.rectangle([0, 0, w, h], fill=bg_color)

        # Card dimensions and placement
        card_w = self.rng.randint(380, 440)
        card_h = self.rng.randint(460, 520)
        card_x = (w - card_w) // 2 + self.rng.randint(-30, 30)
        card_y = (h - card_h) // 2 + self.rng.randint(-20, 20)

        # Card shadow / card border
        self._draw_rounded_rect(
            draw,
            (card_x - 1, card_y - 1, card_x + card_w + 1, card_y + card_h + 1),
            fill=None,
            outline=(220, 226, 235),
            radius=10,
        )
        self._draw_rounded_rect(
            draw,
            (card_x, card_y, card_x + card_w, card_y + card_h),
            fill=(255, 255, 255),
            radius=10,
        )
        # Class 1: login_form
        boxes.append(self._to_yolo_box(1, (card_x, card_y, card_x + card_w, card_y + card_h), w, h, "login_form"))

        # Inside card: Logo
        logo_x = card_x + (card_w - (len(brand_name) * 12 + 50)) // 2
        logo_y = card_y + 35
        _, logo_box = self._draw_logo(draw, logo_x, logo_y, brand_name, brand_color, w, h)
        boxes.append(logo_box)

        # Header title text
        title_y = logo_y + 45
        draw.text((card_x + 35, title_y), "Sign in to your account", fill=(30, 41, 59), font=self.font_medium)

        # Username input field
        in_w = card_w - 70
        in_h = 42
        u_label_y = title_y + 32
        draw.text((card_x + 35, u_label_y), "Email or username", fill=(71, 85, 105), font=self.font_small)

        u_box_y = u_label_y + 20
        u_box_pixel = (card_x + 35, u_box_y, card_x + 35 + in_w, u_box_y + in_h)
        self._draw_rounded_rect(draw, u_box_pixel, fill=(248, 250, 252), outline=(203, 213, 225), radius=6)
        draw.text((card_x + 45, u_box_y + 12), "user@domain.com", fill=(148, 163, 184), font=self.font_small)
        # Class 2: username_field
        boxes.append(self._to_yolo_box(2, u_box_pixel, w, h, "username_field"))

        # Password input field
        p_label_y = u_box_y + in_h + 16
        draw.text((card_x + 35, p_label_y), "Password", fill=(71, 85, 105), font=self.font_small)

        p_box_y = p_label_y + 20
        p_box_pixel = (card_x + 35, p_box_y, card_x + 35 + in_w, p_box_y + in_h)
        self._draw_rounded_rect(draw, p_box_pixel, fill=(248, 250, 252), outline=(203, 213, 225), radius=6)
        draw.text((card_x + 45, p_box_y + 12), "••••••••••••", fill=(100, 116, 139), font=self.font_small)
        # Eye icon
        eye_x = card_x + 35 + in_w - 28
        draw.ellipse([eye_x, p_box_y + 14, eye_x + 14, p_box_y + 26], outline=(148, 163, 184), width=1)
        # Class 3: password_field
        boxes.append(self._to_yolo_box(3, p_box_pixel, w, h, "password_field"))

        # Submit button
        btn_y = p_box_y + in_h + 24
        btn_h = 44
        btn_pixel = (card_x + 35, btn_y, card_x + 35 + in_w, btn_y + btn_h)
        self._draw_rounded_rect(draw, btn_pixel, fill=brand_color, radius=6)
        btn_text = self.rng.choice(["Sign In", "Log In", "Continue"])
        draw.text((card_x + card_w // 2 - 25, btn_y + 12), btn_text, fill=(255, 255, 255), font=self.font_medium)
        # Class 4: submit_button
        boxes.append(self._to_yolo_box(4, btn_pixel, w, h, "submit_button"))

        # Footer link
        draw.text((card_x + 35, btn_y + btn_h + 15), "Forgot password?", fill=brand_color, font=self.font_small)

        return SyntheticPageLayout(img, boxes, "standard_login", brand_name)

    def _render_banking_login(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        boxes: list[YOLOBoundingBox] = []

        # Corporate header bar
        header_h = 70
        draw.rectangle([0, 0, w, header_h], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        # Logo in header
        _, logo_box = self._draw_logo(draw, 50, 20, brand_name, brand_color, w, h)
        boxes.append(logo_box)

        # Nav items
        draw.text((w - 320, 26), "Personal | Business | Commercial", fill=(100, 116, 139), font=self.font_small)

        # Background below header
        draw.rectangle([0, header_h, w, h], fill=(241, 245, 249))

        # Split layout: Hero advisory on left, Banking login card on right
        advisory_x = 80
        advisory_y = 120
        draw.text((advisory_x, advisory_y), f"Welcome to {brand_name} Online Banking", fill=(15, 23, 42), font=self.font_title)
        draw.text((advisory_x, advisory_y + 45), "Secure 256-bit encrypted authentication portal.", fill=(71, 85, 105), font=self.font_medium)
        # Security badge
        self._draw_rounded_rect(draw, (advisory_x, advisory_y + 90, advisory_x + 360, advisory_y + 170), fill=(255, 255, 255), outline=(203, 213, 225), radius=8)
        draw.text((advisory_x + 20, advisory_y + 110), "Security Guarantee", fill=brand_color, font=self.font_medium)
        draw.text((advisory_x + 20, advisory_y + 135), "Never share your OTP or PIN with anyone.", fill=(100, 116, 139), font=self.font_small)

        # Login card on right
        card_w = 400
        card_h = 440
        card_x = w - card_w - 90
        card_y = 120

        self._draw_rounded_rect(draw, (card_x, card_y, card_x + card_w, card_y + card_h), fill=(255, 255, 255), outline=(203, 213, 225), radius=8)
        # Class 1: login_form
        boxes.append(self._to_yolo_box(1, (card_x, card_y, card_x + card_w, card_y + card_h), w, h, "login_form"))

        draw.text((card_x + 30, card_y + 25), "Online Banking Login", fill=(15, 23, 42), font=self.font_medium)

        # Username / Customer ID field
        in_w = card_w - 60
        in_h = 42
        u_label_y = card_y + 70
        draw.text((card_x + 30, u_label_y), "User ID / Customer Number", fill=(51, 65, 85), font=self.font_small)
        u_box_pixel = (card_x + 30, u_label_y + 20, card_x + 30 + in_w, u_label_y + 20 + in_h)
        self._draw_rounded_rect(draw, u_box_pixel, fill=(248, 250, 252), outline=(203, 213, 225), radius=4)
        draw.text((card_x + 40, u_label_y + 32), "Enter Customer ID", fill=(148, 163, 184), font=self.font_small)
        # Class 2: username_field
        boxes.append(self._to_yolo_box(2, u_box_pixel, w, h, "username_field"))

        # Password / PIN field
        p_label_y = u_label_y + 20 + in_h + 16
        draw.text((card_x + 30, p_label_y), "Password / IPIN", fill=(51, 65, 85), font=self.font_small)
        p_box_pixel = (card_x + 30, p_label_y + 20, card_x + 30 + in_w, p_label_y + 20 + in_h)
        self._draw_rounded_rect(draw, p_box_pixel, fill=(248, 250, 252), outline=(203, 213, 225), radius=4)
        draw.text((card_x + 40, p_label_y + 32), "••••••••", fill=(100, 116, 139), font=self.font_small)
        # Class 3: password_field
        boxes.append(self._to_yolo_box(3, p_box_pixel, w, h, "password_field"))

        # Submit button
        btn_y = p_label_y + 20 + in_h + 26
        btn_h = 44
        btn_pixel = (card_x + 30, btn_y, card_x + 30 + in_w, btn_y + btn_h)
        self._draw_rounded_rect(draw, btn_pixel, fill=brand_color, radius=4)
        draw.text((card_x + card_w // 2 - 40, btn_y + 12), "LOGIN SECURELY", fill=(255, 255, 255), font=self.font_medium)
        # Class 4: submit_button
        boxes.append(self._to_yolo_box(4, btn_pixel, w, h, "submit_button"))

        return SyntheticPageLayout(img, boxes, "banking_login", brand_name)

    def _render_social_login(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        boxes: list[YOLOBoundingBox] = []

        # Dark or vibrant modern background
        draw.rectangle([0, 0, w, h], fill=(15, 23, 42))

        # Centered modern card
        card_w = 420
        card_h = 480
        card_x = (w - card_w) // 2
        card_y = (h - card_h) // 2

        self._draw_rounded_rect(draw, (card_x, card_y, card_x + card_w, card_y + card_h), fill=(30, 41, 59), outline=(51, 65, 85), radius=12)
        # Class 1: login_form
        boxes.append(self._to_yolo_box(1, (card_x, card_y, card_x + card_w, card_y + card_h), w, h, "login_form"))

        # Logo inside card
        logo_x = card_x + (card_w - (len(brand_name) * 12 + 50)) // 2
        logo_y = card_y + 35
        _, logo_box = self._draw_logo(draw, logo_x, logo_y, brand_name, brand_color, w, h)
        boxes.append(logo_box)

        # Username field
        in_w = card_w - 60
        in_h = 44
        u_box_pixel = (card_x + 30, card_y + 110, card_x + 30 + in_w, card_y + 110 + in_h)
        self._draw_rounded_rect(draw, u_box_pixel, fill=(15, 23, 42), outline=(71, 85, 105), radius=6)
        draw.text((card_x + 42, card_y + 122), "Phone number, username, or email", fill=(148, 163, 184), font=self.font_small)
        # Class 2: username_field
        boxes.append(self._to_yolo_box(2, u_box_pixel, w, h, "username_field"))

        # Password field
        p_box_pixel = (card_x + 30, card_y + 172, card_x + 30 + in_w, card_y + 172 + in_h)
        self._draw_rounded_rect(draw, p_box_pixel, fill=(15, 23, 42), outline=(71, 85, 105), radius=6)
        draw.text((card_x + 42, card_y + 184), "Password", fill=(148, 163, 184), font=self.font_small)
        # Class 3: password_field
        boxes.append(self._to_yolo_box(3, p_box_pixel, w, h, "password_field"))

        # Submit button
        btn_pixel = (card_x + 30, card_y + 236, card_x + 30 + in_w, card_y + 236 + in_h)
        self._draw_rounded_rect(draw, btn_pixel, fill=brand_color, radius=6)
        draw.text((card_x + card_w // 2 - 25, card_y + 248), "Log In", fill=(255, 255, 255), font=self.font_medium)
        # Class 4: submit_button
        boxes.append(self._to_yolo_box(4, btn_pixel, w, h, "submit_button"))

        # Divider line
        draw.line([(card_x + 30, card_y + 310), (card_x + card_w - 30, card_y + 310)], fill=(51, 65, 85), width=1)
        draw.text((card_x + card_w // 2 - 10, card_y + 302), "OR", fill=(148, 163, 184), font=self.font_small)

        # Secondary action button
        sec_btn = (card_x + 30, card_y + 335, card_x + 30 + in_w, card_y + 335 + in_h)
        self._draw_rounded_rect(draw, sec_btn, fill=None, outline=(71, 85, 105), radius=6)
        draw.text((card_x + card_w // 2 - 55, card_y + 347), "Create New Account", fill=(226, 232, 240), font=self.font_small)

        return SyntheticPageLayout(img, boxes, "social_login", brand_name)

    def _render_ecommerce_login(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        boxes: list[YOLOBoundingBox] = []

        # Header bar
        header_h = 75
        draw.rectangle([0, 0, w, header_h], fill=(15, 23, 42))

        # Logo in header
        _, logo_box = self._draw_logo(draw, 40, 22, brand_name, brand_color, w, h)
        boxes.append(logo_box)

        # Search bar in header (IMPORTANT: NOT a username field!)
        search_box = (320, 18, w - 240, 56)
        self._draw_rounded_rect(draw, search_box, fill=(255, 255, 255), radius=4)
        draw.text((335, 27), "Search products, categories, or brands...", fill=(148, 163, 184), font=self.font_small)

        # Cart icon text
        draw.text((w - 180, 28), "Orders | Cart (0)", fill=(255, 255, 255), font=self.font_medium)

        # Background
        draw.rectangle([0, header_h, w, h], fill=(243, 244, 246))

        # Centered auth card
        card_w = 400
        card_h = 440
        card_x = (w - card_w) // 2
        card_y = header_h + 40

        self._draw_rounded_rect(draw, (card_x, card_y, card_x + card_w, card_y + card_h), fill=(255, 255, 255), outline=(229, 231, 235), radius=8)
        # Class 1: login_form
        boxes.append(self._to_yolo_box(1, (card_x, card_y, card_x + card_w, card_y + card_h), w, h, "login_form"))

        draw.text((card_x + 30, card_y + 30), "Sign in to complete order", fill=(17, 24, 39), font=self.font_large)

        # Username input
        in_w = card_w - 60
        in_h = 40
        u_box_pixel = (card_x + 30, card_y + 90, card_x + 30 + in_w, card_y + 90 + in_h)
        self._draw_rounded_rect(draw, u_box_pixel, fill=(255, 255, 255), outline=(209, 213, 219), radius=4)
        draw.text((card_x + 40, card_y + 102), "Email or mobile phone number", fill=(156, 163, 175), font=self.font_small)
        # Class 2: username_field
        boxes.append(self._to_yolo_box(2, u_box_pixel, w, h, "username_field"))

        # Password input
        p_box_pixel = (card_x + 30, card_y + 150, card_x + 30 + in_w, card_y + 150 + in_h)
        self._draw_rounded_rect(draw, p_box_pixel, fill=(255, 255, 255), outline=(209, 213, 219), radius=4)
        draw.text((card_x + 40, card_y + 162), "Password", fill=(156, 163, 175), font=self.font_small)
        # Class 3: password_field
        boxes.append(self._to_yolo_box(3, p_box_pixel, w, h, "password_field"))

        # Submit button
        btn_pixel = (card_x + 30, card_y + 215, card_x + 30 + in_w, card_y + 215 + in_h)
        self._draw_rounded_rect(draw, btn_pixel, fill=(245, 158, 11), radius=4)
        draw.text((card_x + card_w // 2 - 50, card_y + 226), "Continue to Checkout", fill=(17, 24, 39), font=self.font_medium)
        # Class 4: submit_button
        boxes.append(self._to_yolo_box(4, btn_pixel, w, h, "submit_button"))

        return SyntheticPageLayout(img, boxes, "ecommerce_login", brand_name)

    def _render_payment_checkout(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        brand_name: str,
        brand_color: tuple[int, int, int],
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        boxes: list[YOLOBoundingBox] = []

        # Clean checkout background
        draw.rectangle([0, 0, w, h], fill=(248, 250, 252))

        # Header with brand logo
        header_h = 65
        draw.rectangle([0, 0, w, header_h], fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        _, logo_box = self._draw_logo(draw, 60, 18, brand_name, brand_color, w, h)
        boxes.append(logo_box)
        draw.text((w - 220, 24), "🔒 SSL 256-Bit Encrypted", fill=(16, 185, 129), font=self.font_small)

        # Checkout form card
        card_w = 460
        card_h = 500
        card_x = (w - card_w) // 2
        card_y = 100

        self._draw_rounded_rect(draw, (card_x, card_y, card_x + card_w, card_y + card_h), fill=(255, 255, 255), outline=(203, 213, 225), radius=10)
        # Class 1: login_form / payment_form container
        boxes.append(self._to_yolo_box(1, (card_x, card_y, card_x + card_w, card_y + card_h), w, h, "login_form"))

        draw.text((card_x + 35, card_y + 25), "Payment Details", fill=(15, 23, 42), font=self.font_large)
        draw.text((card_x + 35, card_y + 55), "Complete your transaction securely.", fill=(100, 116, 139), font=self.font_small)

        # Payment element 1: Credit Card Number
        in_w = card_w - 70
        in_h = 42
        c_label_y = card_y + 85
        draw.text((card_x + 35, c_label_y), "Card Number", fill=(51, 65, 85), font=self.font_small)
        card_input_box = (card_x + 35, c_label_y + 20, card_x + 35 + in_w, c_label_y + 20 + in_h)
        self._draw_rounded_rect(draw, card_input_box, fill=(248, 250, 252), outline=(203, 213, 225), radius=6)
        draw.text((card_x + 45, c_label_y + 32), "4532 •••• •••• 8892", fill=(71, 85, 105), font=self.font_small)
        # Class 5: payment_element
        boxes.append(self._to_yolo_box(5, card_input_box, w, h, "payment_element"))

        # Payment element 2: Expiry & CVV row
        sub_w = (in_w - 20) // 2
        exp_y = c_label_y + 20 + in_h + 16
        draw.text((card_x + 35, exp_y), "Expiration Date", fill=(51, 65, 85), font=self.font_small)
        exp_box = (card_x + 35, exp_y + 20, card_x + 35 + sub_w, exp_y + 20 + in_h)
        self._draw_rounded_rect(draw, exp_box, fill=(248, 250, 252), outline=(203, 213, 225), radius=6)
        draw.text((card_x + 45, exp_y + 32), "MM / YY", fill=(148, 163, 184), font=self.font_small)
        # Class 5: payment_element
        boxes.append(self._to_yolo_box(5, exp_box, w, h, "payment_element"))

        # CVV
        draw.text((card_x + 35 + sub_w + 20, exp_y), "Security Code (CVV)", fill=(51, 65, 85), font=self.font_small)
        cvv_box = (card_x + 35 + sub_w + 20, exp_y + 20, card_x + 35 + in_w, exp_y + 20 + in_h)
        self._draw_rounded_rect(draw, cvv_box, fill=(248, 250, 252), outline=(203, 213, 225), radius=6)
        draw.text((card_x + 45 + sub_w + 20, exp_y + 32), "CVC", fill=(148, 163, 184), font=self.font_small)
        # Class 5: payment_element
        boxes.append(self._to_yolo_box(5, cvv_box, w, h, "payment_element"))

        # Submit button
        btn_y = exp_y + 20 + in_h + 30
        btn_h = 46
        btn_pixel = (card_x + 35, btn_y, card_x + 35 + in_w, btn_y + btn_h)
        self._draw_rounded_rect(draw, btn_pixel, fill=(16, 185, 129), radius=6)
        draw.text((card_x + card_w // 2 - 45, btn_y + 14), "PAY $49.99 NOW", fill=(255, 255, 255), font=self.font_medium)
        # Class 4: submit_button
        boxes.append(self._to_yolo_box(4, btn_pixel, w, h, "submit_button"))

        return SyntheticPageLayout(img, boxes, "payment_checkout", brand_name)

    def _render_empty_page(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        w: int,
        h: int,
    ) -> SyntheticPageLayout:
        """Render a negative sample (404 error page or informational blog) with ZERO target objects."""
        page_type = self.rng.choice(["404_error", "article_page", "terms_page"])

        if page_type == "404_error":
            draw.rectangle([0, 0, w, h], fill=(241, 245, 249))
            draw.text((w // 2 - 80, h // 2 - 100), "404", fill=(100, 116, 139), font=self.font_title)
            draw.text((w // 2 - 120, h // 2 - 30), "Page Not Found", fill=(51, 65, 85), font=self.font_large)
            draw.text((w // 2 - 170, h // 2 + 10), "The requested URL was not found on this server.", fill=(148, 163, 184), font=self.font_small)
        else:
            draw.rectangle([0, 0, w, h], fill=(255, 255, 255))
            draw.rectangle([0, 0, w, 50], fill=(248, 250, 252), outline=(226, 232, 240))
            draw.text((40, 16), "Documentation & Knowledge Base", fill=(100, 116, 139), font=self.font_medium)
            # Paragraph text blocks
            for line_i in range(8):
                draw.line([(80, 100 + line_i * 28), (w - 120, 100 + line_i * 28)], fill=(226, 232, 240), width=10)

        # Boxes is strictly EMPTY
        return SyntheticPageLayout(img, [], "empty_page", "None", is_empty=True)

    def generate_dataset(
        self,
        output_dir: Path | str = "data/synthetic_yolo",
        counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Generate full synthetic YOLO dataset with train, val, and test splits.

        Default counts:
        - train: 560
        - val: 120
        - test: 120
        Total: 800 images.
        """
        out_p = Path(output_dir)
        counts = counts or {"train": 560, "val": 120, "test": 120}

        validator = YOLOAnnotationValidator()
        manifest_records: list[dict[str, Any]] = []

        total_generated = 0
        split_stats: dict[str, Any] = {}

        for split_name, num_samples in counts.items():
            img_dir = out_p / "images" / split_name
            lbl_dir = out_p / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            brands = SPLIT_BRANDS[split_name]
            templates = [
                "standard_login",
                "banking_login",
                "social_login",
                "ecommerce_login",
                "payment_checkout",
                "empty_page",
            ]
            # Weights: ~20% each for the 4 login types, 12% payment, 8% empty
            weights = [0.20, 0.20, 0.20, 0.20, 0.12, 0.08]

            split_box_counts = {c: 0 for c in YOLO_DETECTION_CLASSES.values()}
            split_empty_count = 0

            for idx in range(num_samples):
                sample_id = f"synth_{split_name}_{idx:04d}"
                brand = self.rng.choice(brands)
                template = self.rng.choices(templates, weights=weights, k=1)[0]

                # Generate layout
                layout = self.generate_page(brand, template)

                # Save image
                img_path = img_dir / f"{sample_id}.png"
                layout.image.save(img_path, format="PNG")

                # Save YOLO label file
                lbl_path = lbl_dir / f"{sample_id}.txt"
                with open(lbl_path, "w", encoding="utf-8") as f:
                    for b in layout.boxes:
                        f.write(b.to_yolo_line() + "\n")

                # Validate
                val_res = validator.validate_yolo_file(lbl_path, image_path=img_path)
                if not val_res.is_valid:
                    raise RuntimeError(f"Generated invalid annotation for {sample_id}: {val_res.errors}")

                if layout.is_empty:
                    split_empty_count += 1
                else:
                    for b in layout.boxes:
                        split_box_counts[b.class_name] += 1

                manifest_records.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "template": template,
                    "brand": brand,
                    "is_empty": layout.is_empty,
                    "num_boxes": len(layout.boxes),
                    "image_path": str(img_path.relative_to(out_p.parent.parent)),
                    "label_path": str(lbl_path.relative_to(out_p.parent.parent)),
                })
                total_generated += 1

            split_stats[split_name] = {
                "total_images": num_samples,
                "empty_images": split_empty_count,
                "total_boxes": sum(split_box_counts.values()),
                "classes": split_box_counts,
                "brands_used": sorted(brands),
            }

        # Write dataset.yaml
        dataset_yaml_content = f"""# Ultralytics YOLOv8 Dataset Configuration — PhishVision AI Synthetic UI Elements
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
        import pandas as pd
        df_manifest = pd.DataFrame(manifest_records)
        manifest_csv_path = out_p / "manifest.csv"
        df_manifest.to_csv(manifest_csv_path, index=False)

        # Write generation config & stats
        gen_meta = {
            "project": "PhishVision AI",
            "module": "M-04 YOLOv8n Synthetic UI Elements",
            "seed": self.seed,
            "total_images": total_generated,
            "classes": YOLO_DETECTION_CLASSES,
            "split_counts": counts,
            "split_stats": split_stats,
        }
        stats_path = out_p / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(gen_meta, f, indent=2)

        logger.info(f"Generated {total_generated} synthetic images under {out_p}")
        return gen_meta
