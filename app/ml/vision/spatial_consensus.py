"""Spatial Consensus and Deduplication Engine for UI Element Detectors (Phase 8).

Unifies detections from M-04 (YOLOv8n), M-05 (Faster R-CNN), and M-06 (YOLOv8-Seg)
into a single non-duplicative spatial evidence layer using Intersection-over-Union (IoU)
clustering, multi-detector agreement tracking, and polygon mask preservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

logger = logging.getLogger("phishvision.spatial_consensus")

# Standard 6-Class UI Schema
UI_CLASSES: dict[int, str] = {
    0: "brand_logo",
    1: "login_form",
    2: "username_field",
    3: "password_field",
    4: "submit_button",
    5: "payment_element",
}

CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in UI_CLASSES.items()}


def compute_box_iou(box_a: list[int | float], box_b: list[int | float]) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class FusedUIElement:
    """Deduplicated UI component detected across one or more visual models."""
    class_id: int
    class_name: str
    bbox_pixel: list[int]
    confidence: float
    detectors: list[str] = field(default_factory=list)
    consensus_count: int = 1
    segmentation_polygon: list[list[float]] | None = None
    mask_area: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_pixel": self.bbox_pixel,
            "confidence": round(self.confidence, 4),
            "detectors": self.detectors,
            "consensus_count": self.consensus_count,
            "has_mask": self.segmentation_polygon is not None,
            "segmentation_polygon": self.segmentation_polygon,
            "mask_area": self.mask_area,
        }


class SpatialConsensusEngine:
    """Merges bounding boxes and instance segmentation masks from M-04, M-05, and M-06.

    Prevents duplicate evidence counting when multiple detectors observe the same physical
    UI component on the page.
    """

    def __init__(self, iou_threshold: float = 0.45) -> None:
        self.iou_threshold = iou_threshold

    def fuse_detections(
        self,
        yolo_detections: list[dict[str, Any]] | None = None,
        faster_rcnn_detections: list[dict[str, Any]] | None = None,
        yolo_seg_instances: list[dict[str, Any]] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Cluster and deduplicate detections from all available detectors.

        Args:
            yolo_detections: Detections from M-04 YOLOv8n (with 'bbox_pixel', 'class_id', 'confidence').
            faster_rcnn_detections: Detections from M-05 Faster R-CNN (with 'bbox_pixel', 'class_id', 'confidence').
            yolo_seg_instances: Instances from M-06 YOLOv8-Seg (with 'box_xyxy', 'class_id', 'confidence', 'polygon_pts').
            image_size: (width, height) of the source image.

        Returns:
            Dict containing deduplicated elements, per-class counts, UI presence flags, and detector agreement stats.
        """
        raw_candidates: list[dict[str, Any]] = []

        # Standardize M-04 YOLO candidates
        if yolo_detections:
            for det in yolo_detections:
                raw_candidates.append({
                    "class_id": int(det["class_id"]),
                    "class_name": det.get("class_name", UI_CLASSES.get(int(det["class_id"]), "unknown")),
                    "bbox_pixel": [int(round(c)) for c in det.get("bbox_pixel", [0, 0, 0, 0])],
                    "confidence": float(det.get("confidence", 0.0)),
                    "source": "M-04_yolov8n",
                    "polygon": None,
                    "mask_area": None,
                })

        # Standardize M-05 Faster R-CNN candidates
        if faster_rcnn_detections:
            for det in faster_rcnn_detections:
                cid = int(det["class_id"])
                # Note: Faster R-CNN inferencer maps internally to YOLO 0..5 class IDs
                raw_candidates.append({
                    "class_id": cid,
                    "class_name": det.get("class_name", UI_CLASSES.get(cid, "unknown")),
                    "bbox_pixel": [int(round(c)) for c in det.get("bbox_pixel", [0, 0, 0, 0])],
                    "confidence": float(det.get("confidence", 0.0)),
                    "source": "M-05_faster_rcnn",
                    "polygon": None,
                    "mask_area": None,
                })

        # Standardize M-06 YOLO-Seg candidates
        if yolo_seg_instances:
            for inst in yolo_seg_instances:
                cid = int(inst["class_id"])
                raw_candidates.append({
                    "class_id": cid,
                    "class_name": inst.get("class_name", UI_CLASSES.get(cid, "unknown")),
                    "bbox_pixel": [int(round(c)) for c in inst.get("box_xyxy", [0, 0, 0, 0])],
                    "confidence": float(inst.get("confidence", 0.0)),
                    "source": "M-06_yolov8_seg",
                    "polygon": inst.get("polygon_pts"),
                    "mask_area": inst.get("mask_area_pixels"),
                })

        # Group by class ID to enforce class-specific spatial clustering, filtering oversized artifacts
        by_class: dict[int, list[dict[str, Any]]] = {}
        for c in raw_candidates:
            # Filter oversized background hallucinations (boxes covering > 45% of image area)
            box = c["bbox_pixel"]
            bw = max(0, box[2] - box[0])
            bh = max(0, box[3] - box[1])
            box_area = bw * bh
            if image_size and image_size[0] > 0 and image_size[1] > 0:
                if box_area > 0.45 * (image_size[0] * image_size[1]):
                    continue
            elif bw > 1000 and bh > 400:
                continue

            by_class.setdefault(c["class_id"], []).append(c)

        fused_elements: list[FusedUIElement] = []

        for cid, candidates in by_class.items():
            # Sort candidates by descending confidence
            candidates.sort(key=lambda x: x["confidence"], reverse=True)
            clusters: list[list[dict[str, Any]]] = []

            for cand in candidates:
                matched_cluster = None
                for cluster in clusters:
                    # Check IoU against cluster representative (highest confidence candidate)
                    rep = cluster[0]
                    iou = compute_box_iou(cand["bbox_pixel"], rep["bbox_pixel"])
                    if iou >= self.iou_threshold:
                        matched_cluster = cluster
                        break

                if matched_cluster is not None:
                    # Do not duplicate detector source within the same cluster
                    matched_cluster.append(cand)
                else:
                    clusters.append([cand])

            # Resolve each cluster into a single fused UI component
            for cluster in clusters:
                # Representative is the highest confidence detection
                rep = cluster[0]
                sources = list({item["source"] for item in cluster})
                max_conf = max(item["confidence"] for item in cluster)

                # Use representative bounding box
                fused_box = rep["bbox_pixel"]

                # If any member of cluster has segmentation mask (e.g. from M-06), attach it
                polygon = None
                mask_area = None
                for item in cluster:
                    if item.get("polygon") is not None:
                        polygon = item["polygon"]
                        mask_area = item.get("mask_area")
                        break

                fused_elements.append(
                    FusedUIElement(
                        class_id=cid,
                        class_name=UI_CLASSES.get(cid, f"class_{cid}"),
                        bbox_pixel=fused_box,
                        confidence=max_conf,
                        detectors=sorted(sources),
                        consensus_count=len(sources),
                        segmentation_polygon=polygon,
                        mask_area=mask_area,
                    )
                )

        # Aggregate counts and semantic presence flags
        class_counts: dict[str, int] = {cname: 0 for cname in UI_CLASSES.values()}
        for elem in fused_elements:
            class_counts[elem.class_name] = class_counts.get(elem.class_name, 0) + 1

        has_login_form = class_counts.get("login_form", 0) > 0
        has_user_field = class_counts.get("username_field", 0) > 0
        has_pass_field = class_counts.get("password_field", 0) > 0
        has_submit_btn = class_counts.get("submit_button", 0) > 0
        has_payment_el = class_counts.get("payment_element", 0) > 0

        # Credential inputs strictly require a password field (or DOM password field)
        has_credential_inputs = has_pass_field

        return {
            "total_fused_elements": len(fused_elements),
            "elements": [e.to_dict() for e in fused_elements],
            "class_counts": class_counts,
            "has_login_form": has_login_form,
            "has_credential_inputs": has_credential_inputs,
            "has_password_field": has_pass_field,
            "has_username_field": has_user_field,
            "has_submit_button": has_submit_btn,
            "has_payment_element": has_payment_el,
            "raw_detection_counts": {
                "M-04_yolov8n": len(yolo_detections or []),
                "M-05_faster_rcnn": len(faster_rcnn_detections or []),
                "M-06_yolov8_seg": len(yolo_seg_instances or []),
            },
        }

    def extract_fused_logo_crops(
        self,
        image: Image.Image,
        fused_elements: list[dict[str, Any]],
        padding_pct: float = 0.10,
    ) -> list[dict[str, Any]]:
        """Extract cropped logo images with ~10% contextual padding from fused logo elements.

        Args:
            image: Rendered webpage screenshot.
            fused_elements: Output of `fuse_detections()['elements']`.
            padding_pct: Margin percentage applied to width and height.

        Returns:
            List of dicts with 'crop_image', 'bbox_pixel', 'confidence', 'detectors', etc.
        """
        w, h = image.size
        crops = []

        for elem in fused_elements:
            if elem.get("class_id") == 0 or elem.get("class_name") == "brand_logo":
                box = elem["bbox_pixel"]
                bx1, by1, bx2, by2 = box
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)

                pad_w = int(round(bw * padding_pct))
                pad_h = int(round(bh * padding_pct))

                cx1 = max(0, bx1 - pad_w)
                cy1 = max(0, by1 - pad_h)
                cx2 = min(w, bx2 + pad_w)
                cy2 = min(h, by2 + pad_h)

                if cx2 > cx1 and cy2 > cy1:
                    crop = image.crop((cx1, cy1, cx2, cy2))
                    crops.append({
                        "crop_image": crop,
                        "original_bbox": box,
                        "padded_bbox": [cx1, cy1, cx2, cy2],
                        "confidence": elem.get("confidence", 0.0),
                        "detectors": elem.get("detectors", []),
                        "has_mask": elem.get("has_mask", False),
                        "segmentation_polygon": elem.get("segmentation_polygon"),
                    })

        return crops
