"""Visual Contact Sheet Generator for Error Analysis (M-02 ResNet-18).

Generates high-resolution multi-panel contact sheets for manual visual inspection
of misclassified screenshots (False Negatives and False Positives), preserving
aspect ratio, overlaying calibrated prediction metadata, and emitting a cross-reference index.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.ml.vision.preprocessing import safe_load_image

logger = logging.getLogger("phishvision.ml.vision.contact_sheet")


def _get_font(size: int = 12) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Attempts to load Arial or falls back to Pillow default font."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def calculate_file_sha256(path: str | Path) -> str:
    """Computes SHA-256 digest of a file to verify immutability."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def render_contact_sheet(
    records: list[dict[str, Any]],
    sheet_title: str,
    output_image_path: str | Path,
    cols: int = 5,
    cell_w: int = 300,
    cell_h: int = 380,
    thumb_max_w: int = 270,
    thumb_max_h: int = 270,
    bg_color: tuple[int, int, int] = (245, 247, 250),
    card_color: tuple[int, int, int] = (255, 255, 255),
    card_border: tuple[int, int, int] = (210, 215, 225),
) -> list[dict[str, Any]]:
    """Renders a grid contact sheet for a list of screenshot error records.
    
    Args:
        records: List of dictionaries from error analysis containing image_path, sample_id, etc.
        sheet_title: Banner title rendered at top of sheet.
        output_image_path: Target path for contact sheet PNG.
        cols: Number of grid columns.
        cell_w: Width of each thumbnail card cell.
        cell_h: Height of each thumbnail card cell.
        thumb_max_w: Maximum width bounding box for thumbnail.
        thumb_max_h: Maximum height bounding box for thumbnail.
        bg_color: Canvas background color.
        card_color: Cell card background color.
        card_border: Cell card border stroke color.
        
    Returns:
        List of index records with bounding box coordinates on the contact sheet.
    """
    out_path = Path(output_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    if n == 0:
        logger.warning(f"No records provided for contact sheet: {sheet_title}")
        return []

    rows = (n + cols - 1) // cols
    margin = 24
    header_h = 70
    sheet_w = margin * 2 + cols * cell_w + (cols - 1) * 16
    sheet_h = margin * 2 + header_h + rows * cell_h + (rows - 1) * 16

    # Create canvas
    canvas = Image.new("RGB", (sheet_w, sheet_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(22)
    meta_font = _get_font(12)
    meta_bold = _get_font(13)

    # Draw header banner
    draw.rectangle([(margin, margin), (sheet_w - margin, margin + header_h - 10)], fill=(30, 41, 59))
    draw.text((margin + 20, margin + 12), sheet_title, fill=(255, 255, 255), font=title_font)
    subtitle = f"Total Items: {n} | Grid: {rows}x{cols} | Model: ResNet-18 M-02 (Val Split)"
    draw.text((margin + 20, margin + 42), subtitle, fill=(148, 163, 184), font=meta_font)

    index_entries: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        r = i // cols
        c = i % cols

        x0 = margin + c * (cell_w + 16)
        y0 = margin + header_h + r * (cell_h + 16)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        # Draw card background and border
        draw.rectangle([(x0, y0), (x1, y1)], fill=card_color, outline=card_border, width=1)

        # Header bar in card
        err_type = rec.get("error_type", "")
        is_fn = err_type == "false_negative"
        bar_color = (220, 38, 38) if is_fn else (217, 119, 6)  # Red for FN (phishing missed), Amber for FP
        draw.rectangle([(x0, y0), (x1, y0 + 6)], fill=bar_color)

        # Metadata text
        sid = rec.get("sample_id", f"item_{i}")
        act_label = rec.get("actual_label", 0)
        pred_label = rec.get("predicted_label", 0)
        p_phish = float(rec.get("phishing_probability", 0.0))
        p_legit = float(rec.get("legitimate_probability", 0.0))
        dom = str(rec.get("registrable_domain", "N/A"))
        if len(dom) > 28:
            dom = dom[:25] + "..."

        act_str = "Phish" if act_label == 1 else "Legit"
        pred_str = "Phish" if pred_label == 1 else "Legit"

        draw.text((x0 + 10, y0 + 12), f"#{i+1} ID: {sid}", fill=(15, 23, 42), font=meta_bold)
        draw.text((x0 + 10, y0 + 30), f"Actual: {act_str} | Pred: {pred_str}", fill=(71, 85, 105), font=meta_font)
        draw.text((x0 + 10, y0 + 46), f"P(Phish): {p_phish:.4f} | P(Legit): {p_legit:.4f}", fill=(71, 85, 105), font=meta_font)
        draw.text((x0 + 10, y0 + 62), f"Domain: {dom}", fill=(100, 116, 139), font=meta_font)

        # Thumbnail area: y from y0 + 82 to y1 - 10
        thumb_box_x = x0 + (cell_w - thumb_max_w) // 2
        thumb_box_y = y0 + 82
        thumb_box_w = thumb_max_w
        thumb_box_h = thumb_max_h

        # Light gray inner frame for image
        draw.rectangle(
            [(thumb_box_x, thumb_box_y), (thumb_box_x + thumb_box_w, thumb_box_y + thumb_box_h)],
            fill=(241, 245, 249),
            outline=(226, 232, 240),
            width=1,
        )

        img_path = Path(rec.get("image_path", ""))
        thumb_x_on_sheet = thumb_box_x
        thumb_y_on_sheet = thumb_box_y
        thumb_actual_w = 0
        thumb_actual_h = 0

        if img_path.exists():
            try:
                # Load safely preserving original
                orig_img = safe_load_image(img_path)
                orig_w, orig_h = orig_img.size

                # Create copy and scale preserving aspect ratio
                scaled_img = orig_img.copy()
                scaled_img.thumbnail((thumb_box_w - 4, thumb_box_h - 4), Image.Resampling.LANCZOS)
                thumb_actual_w, thumb_actual_h = scaled_img.size

                # Center inside the box
                offset_x = thumb_box_x + (thumb_box_w - thumb_actual_w) // 2
                offset_y = thumb_box_y + (thumb_box_h - thumb_actual_h) // 2
                canvas.paste(scaled_img, (offset_x, offset_y))

                thumb_x_on_sheet = offset_x
                thumb_y_on_sheet = offset_y

                # Dimension tag at bottom of thumbnail box
                dim_str = f"{orig_w}x{orig_h}"
                draw.text((thumb_box_x + 4, thumb_box_y + thumb_box_h - 14), dim_str, fill=(148, 163, 184), font=meta_font)
            except Exception as e:
                logger.error(f"Failed to render thumbnail for {img_path}: {e}")
                draw.text((thumb_box_x + 10, thumb_box_y + 40), f"Error loading image:\n{e}", fill=(220, 38, 38), font=meta_font)
        else:
            draw.text((thumb_box_x + 10, thumb_box_y + 40), f"File not found:\n{img_path.name}", fill=(220, 38, 38), font=meta_font)

        index_entries.append({
            "contact_sheet": out_path.name,
            "grid_row": r,
            "grid_col": c,
            "grid_index": i + 1,
            "sample_id": sid,
            "actual_label": act_label,
            "actual_label_name": act_str,
            "predicted_label": pred_label,
            "predicted_label_name": pred_str,
            "phishing_probability": p_phish,
            "legitimate_probability": p_legit,
            "registrable_domain": dom,
            "original_screenshot_path": str(img_path),
            "card_bounding_box": f"({x0},{y0},{x1},{y1})",
            "thumbnail_bounding_box": f"({thumb_x_on_sheet},{thumb_y_on_sheet},{thumb_x_on_sheet + thumb_actual_w},{thumb_y_on_sheet + thumb_actual_h})",
        })

    # Save contact sheet
    canvas.save(out_path, format="PNG", optimize=True)
    logger.info(f"Saved contact sheet to: {out_path} ({sheet_w}x{sheet_h} px)")
    return index_entries


def generate_all_contact_sheets(
    error_csv_path: str | Path = "data/reports/resnet18_error_analysis/error_analysis.csv",
    output_dir: str | Path = "data/reports/resnet18_error_analysis/contact_sheets",
) -> dict[str, Any]:
    """Generates false-negative and false-positive contact sheets and CSV index mapping.
    
    Args:
        error_csv_path: Path to existing error_analysis.csv.
        output_dir: Destination directory for contact sheets.
        
    Returns:
        Dictionary with output paths, verification hashes, and index details.
    """
    csv_p = Path(error_csv_path)
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    if not csv_p.exists():
        raise FileNotFoundError(f"Error analysis CSV not found at: {csv_p}")

    import pandas as pd
    df = pd.read_csv(csv_p)

    df_fn = df[df["error_type"] == "false_negative"].copy()
    df_fp = df[df["error_type"] == "false_positive"].copy()

    # Pre-hash original images to guarantee zero modification
    pre_hashes = {}
    for _, r in df.iterrows():
        p = Path(r["image_path"])
        if p.exists():
            pre_hashes[str(p)] = calculate_file_sha256(p)

    fn_records = df_fn.to_dict(orient="records")
    fp_records = df_fp.to_dict(orient="records")

    # 1. False Negatives Contact Sheet (39 images, 5 columns x 8 rows)
    fn_sheet_path = out_d / "false_negatives_contact_sheet.png"
    fn_index = render_contact_sheet(
        records=fn_records,
        sheet_title="M-02 ResNet-18: False Negatives Contact Sheet (39 Phishing Websites Missed)",
        output_image_path=fn_sheet_path,
        cols=5,
        cell_w=300,
        cell_h=380,
    )

    # 2. False Positives Contact Sheet (4 images, 4 columns x 1 row or 2x2)
    fp_sheet_path = out_d / "false_positives_contact_sheet.png"
    fp_index = render_contact_sheet(
        records=fp_records,
        sheet_title="M-02 ResNet-18: False Positives Contact Sheet (4 Legitimate Websites Misclassified)",
        output_image_path=fp_sheet_path,
        cols=4,
        cell_w=300,
        cell_h=380,
    )

    # 3. Combined Contact Sheet Index CSV
    all_index = fn_index + fp_index
    index_csv_path = out_d / "contact_sheet_index.csv"

    index_fieldnames = [
        "contact_sheet",
        "grid_index",
        "grid_row",
        "grid_col",
        "sample_id",
        "actual_label",
        "actual_label_name",
        "predicted_label",
        "predicted_label_name",
        "phishing_probability",
        "legitimate_probability",
        "registrable_domain",
        "original_screenshot_path",
        "card_bounding_box",
        "thumbnail_bounding_box",
    ]

    with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_fieldnames)
        writer.writeheader()
        writer.writerows(all_index)

    logger.info(f"Saved contact sheet index CSV to: {index_csv_path}")

    # Post-hash verification
    post_hashes = {}
    for p_str in pre_hashes:
        post_hashes[p_str] = calculate_file_sha256(p_str)
        assert pre_hashes[p_str] == post_hashes[p_str], f"Immutability violated for {p_str}!"

    return {
        "fn_contact_sheet": str(fn_sheet_path),
        "fp_contact_sheet": str(fp_sheet_path),
        "index_csv": str(index_csv_path),
        "fn_count": len(fn_records),
        "fp_count": len(fp_records),
        "total_indexed": len(all_index),
        "immutability_verified": True,
    }
