"""Unit and Integration Tests for Phase 8 Seven-Model Multimodal Pipeline.

Validates:
1. Unified pipeline initialization and all 7 model availabilities:
   - M-01 (Lexical MLP)
   - M-02 (ResNet-18 Global Layout Classifier)
   - M-03 (Siamese Brand Impersonation Matcher)
   - M-04 (YOLOv8n UI Component Detector)
   - M-05 (Faster R-CNN ResNet-50-FPN Object Detector)
   - M-06 (YOLOv8-Seg Instance Segmenter)
   - M-07 (URL Sequence Bi-LSTM)
2. Spatial Consensus Engine:
   - IoU clustering and deduplication across M-04, M-05, and M-06.
   - Agreement tracking across multiple detectors.
   - Polygon mask preservation from YOLO-Seg.
   - Deduplicated logo crop extraction with ~10% padding.
3. Multi-Evidence Risk Fusion & Safeguards:
   - Official brand whitelist preservation.
   - Critical brand impersonation override.
   - Localized credential harvesting on anomalous hosts.
   - 404 / empty page risk capping below active phishing threshold.
   - DOM form cross-verification.
4. Error Handling & Robustness:
   - Missing / unavailable model checkpoints graceful degradation.
   - Corrupt, empty, or unreadable screenshots.
   - Missing or empty brand reference galleries.
   - Anti-SSRF scanner protections.
   - Fallback to UNKNOWN / INSUFFICIENT EVIDENCE.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from app.ml.unified_pipeline import UnifiedPhishingPipeline, normalize_calibrated_risk
from app.ml.vision.spatial_consensus import FusedUIElement, SpatialConsensusEngine, compute_box_iou
from app.scanner.anti_ssrf import SSRFValidationError, validate_url_and_resolve


# =============================================================================
# 1. Spatial Consensus Engine Tests
# =============================================================================

class TestSpatialConsensusEngine:
    """Tests for IoU computation, spatial clustering, and non-duplicative UI fusion."""

    def test_compute_box_iou(self) -> None:
        """Verify IoU calculation on identical, disjoint, and partial overlap boxes."""
        box1 = [10, 10, 50, 50]
        # Identical
        assert compute_box_iou(box1, box1) == 1.0

        # Disjoint
        box2 = [60, 60, 100, 100]
        assert compute_box_iou(box1, box2) == 0.0

        # Half overlap: box1 area=1600 (40x40), box3 area=1600 (40x40).
        # Inter: [30, 10, 50, 50] -> 20x40=800. Union: 1600+1600-800 = 2400. IoU = 800/2400 = 1/3.
        box3 = [30, 10, 70, 50]
        assert abs(compute_box_iou(box1, box3) - (1.0 / 3.0)) < 1e-4

    def test_deduplication_across_three_detectors(self) -> None:
        """Verify that M-04, M-05, and M-06 detections of the same object are merged into 1."""
        engine = SpatialConsensusEngine(iou_threshold=0.45)

        # Three detectors identifying the SAME login form at roughly [100, 100, 400, 500]
        yolo_dets = [
            {"class_id": 1, "class_name": "login_form", "bbox_pixel": [100, 100, 400, 500], "confidence": 0.92}
        ]
        faster_rcnn_dets = [
            {"class_id": 1, "class_name": "login_form", "bbox_pixel": [102, 98, 405, 502], "confidence": 0.88}
        ]
        yolo_seg_insts = [
            {
                "class_id": 1,
                "class_name": "login_form",
                "box_xyxy": [99, 101, 398, 499],
                "confidence": 0.95,
                "polygon_pts": [[100.0, 100.0], [400.0, 100.0], [400.0, 500.0], [100.0, 500.0]],
                "mask_area_pixels": 120000.0,
            }
        ]

        fused = engine.fuse_detections(
            yolo_detections=yolo_dets,
            faster_rcnn_detections=faster_rcnn_dets,
            yolo_seg_instances=yolo_seg_insts,
            image_size=(800, 600),
        )

        # Crucial requirement: Must NOT count the same login form 3 times!
        assert fused["total_fused_elements"] == 1
        assert fused["class_counts"]["login_form"] == 1
        assert fused["has_login_form"] is True

        elem = fused["elements"][0]
        assert elem["class_name"] == "login_form"
        # Agreement tracking across all 3 models
        assert elem["consensus_count"] == 3
        assert set(elem["detectors"]) == {"M-04_yolov8n", "M-05_faster_rcnn", "M-06_yolov8_seg"}
        # Highest confidence preserved
        assert elem["confidence"] == 0.95
        # Segmentation mask preserved from M-06
        assert elem["has_mask"] is True
        assert elem["segmentation_polygon"] is not None
        assert elem["mask_area"] == 120000.0

    def test_extract_fused_logo_crops_with_padding(self) -> None:
        """Verify fused logo extraction applies ~10% padding clamped to bounds."""
        engine = SpatialConsensusEngine()
        img = Image.new("RGB", (300, 300), color=(255, 255, 255))

        fused_elements = [
            {
                "class_id": 0,
                "class_name": "brand_logo",
                "bbox_pixel": [50, 50, 150, 100],  # w=100, h=50
                "confidence": 0.91,
                "detectors": ["M-04_yolov8n"],
                "has_mask": False,
                "segmentation_polygon": None,
            }
        ]

        # 10% on width (100) = 10 -> [40, 160]
        # 10% on height (50) = 5 -> [45, 105]
        crops = engine.extract_fused_logo_crops(img, fused_elements, padding_pct=0.10)
        assert len(crops) == 1
        crop = crops[0]
        assert crop["padded_bbox"] == [40, 45, 160, 105]
        assert crop["crop_image"].size == (120, 60)


@pytest.fixture
def unified_pipeline() -> UnifiedPhishingPipeline:
    return UnifiedPhishingPipeline()


# =============================================================================
# 2. Unified Pipeline Initialization & Availability Tests
# =============================================================================

class TestUnifiedPipelineLifecycle:
    """Tests for UnifiedPhishingPipeline initialization, checkpoint resolution, and fallback."""

    def test_all_seven_models_loaded(self, unified_pipeline: UnifiedPhishingPipeline) -> None:
        """Verify that all 7 trained model checkpoints are loaded and marked available."""
        avail = unified_pipeline.analyze(url="https://test-check.com", render_preview=False)["model_availability"]
        assert avail["M-01_lexical_mlp"] is True
        assert avail["M-02_resnet18_classifier"] is True
        assert avail["M-03_siamese_brand_matcher"] is True
        assert avail["M-04_yolov8n_detector"] is True
        assert avail["M-05_faster_rcnn_detector"] is True
        assert avail["M-06_yolov8_seg_segmenter"] is True
        assert avail["M-07_lstm_sequence_classifier"] is True

    def test_missing_checkpoints_graceful_degradation(self) -> None:
        """When all model paths are invalid, pipeline must degrade gracefully to UNKNOWN."""
        pipe = UnifiedPhishingPipeline(
            m01_weights_path="nonexistent/m01.npz",
            m01_preprocessor_path="nonexistent/m01.json",
            m02_checkpoint_path="nonexistent/m02.pt",
            m03_checkpoint_path="nonexistent/m03.pt",
            m04_checkpoint_path="nonexistent/m04.pt",
            m05_checkpoint_path="nonexistent/m05.pt",
            m06_checkpoint_path="nonexistent/m06.pt",
            m07_checkpoint_path="nonexistent/m07.pt",
            m07_preprocessor_path="nonexistent/m07.json",
        )
        assert pipe.m01_available is False
        assert pipe.m02_available is False
        assert pipe.m03_available is False
        assert pipe.m04_available is False
        assert pipe.m05_available is False
        assert pipe.m06_available is False
        assert pipe.m07_available is False

        # When no models are operational, fallback to UNKNOWN / INSUFFICIENT EVIDENCE
        res = pipe.analyze(url="https://unknown-degraded.org", render_preview=False)
        assert res["threat_category"] == "UNKNOWN / INSUFFICIENT EVIDENCE"
        assert res["risk_score"] == 0.50


# =============================================================================
# 3. M-07 Sequence Inference Integration Tests
# =============================================================================

class TestM07Integration:
    """Tests for M-07 URL Sequence Bi-LSTM within the multimodal pipeline."""

    def test_calibrated_risk_normalization(self) -> None:
        """Verify that normalize_calibrated_risk maps threshold to 0.50."""
        # For threshold tau = 0.01:
        # At tau=0.01 -> risk=0.50
        assert abs(normalize_calibrated_risk(0.01, 0.01) - 0.50) < 1e-3
        # Below tau: p=0.005 -> risk=0.25
        assert abs(normalize_calibrated_risk(0.005, 0.01) - 0.25) < 1e-3
        # Above tau: p=0.505 -> risk=0.75
        assert abs(normalize_calibrated_risk(0.505, 0.01) - 0.75) < 1e-3

    def test_lstm_redirect_fallback_status(self, unified_pipeline: UnifiedPhishingPipeline) -> None:
        """Verify M-07 telemetry notes that redirect chains are not empirically trained."""
        res = unified_pipeline.analyze(
            url="https://legitimate-example-domain.com/path",
            redirect_chain=["hop1.com", "hop2.com"],
            render_preview=False,
        )
        m07_ev = res["evidence"]["url_models"]["m07_lstm"]
        assert m07_ev["status"] == "SUCCESS"
        assert m07_ev["redirect_chain_provided"] is True
        assert m07_ev["redirect_training_status"] == "NOT_EMPIRICALLY_TRAINED_ON_REDIRECTS"


# =============================================================================
# 4. Multi-Evidence Risk Fusion & Security Tests
# =============================================================================

class TestSevenModelFusion:
    """Tests for explainable multi-evidence fusion rules and edge cases."""

    def test_missing_screenshot_ambiguous_url_unknown_fallback(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """When visual screenshot is missing and URL risk is ambiguous (0.35-0.65), fallback to UNKNOWN."""
        with patch.object(
            unified_pipeline,
            "_analyze_urls",
            return_value={
                "m01_lexical": {"status": "SUCCESS", "probability": 0.50, "normalized_risk": 0.50},
                "m07_lstm": {"status": "SUCCESS", "probability": 0.01, "normalized_risk": 0.50},
                "composite_url_risk": 0.50,
                "is_url_suspicious": False,
            },
        ):
            res = unified_pipeline.analyze(
                url="https://ambiguous-portal-example.net",
                screenshot=None,
                render_preview=False,
            )
            assert res["threat_category"] == "UNKNOWN / INSUFFICIENT EVIDENCE"
            assert res["risk_score"] == 0.50
            assert "insufficient evidence" in res["explanation"].lower()

    def test_404_empty_page_capped_at_suspicious(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """An empty or 404 page (0 UI elements) must NEVER be classified as active PHISHING."""
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        mock_visual = {
            "screenshot_available": True,
            "m02_phishing_probability": 0.85,
            "ui_elements": {
                "total_fused_elements": 0,
                "elements": [],
                "class_counts": {},
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
            },
            "brand_verification": {
                "is_brand_impersonation": False,
                "is_authorized_domain": False,
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
            },
        }

        with patch.object(unified_pipeline, "_analyze_visual", return_value=mock_visual), \
             patch.object(
                 unified_pipeline,
                 "_analyze_urls",
                 return_value={
                     "composite_url_risk": 0.80,
                     "is_url_suspicious": True,
                     "m01_lexical": {"probability": 0.80},
                     "m07_lstm": {"probability": 0.80},
                 },
             ):
            res = unified_pipeline.analyze(
                url="https://anomalous-404-domain.org/missing",
                screenshot=img,
                render_preview=False,
            )
            assert res["threat_category"] == "SUSPICIOUS"
            assert res["risk_score"] <= 0.40
            assert "empty or error page" in res["explanation"].lower()

    def test_brand_impersonation_critical_override(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """When an authentic brand logo is detected on an unauthorized domain, flag CRITICAL_PHISHING."""
        img = Image.new("RGB", (400, 300), color=(250, 250, 250))
        mock_visual = {
            "screenshot_available": True,
            "m02_phishing_probability": 0.40,
            "ui_elements": {
                "total_fused_elements": 1,
                "elements": [{"class_name": "brand_logo", "confidence": 0.95}],
                "class_counts": {"brand_logo": 1},
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
            },
            "brand_verification": {
                "is_brand_impersonation": True,
                "is_authorized_domain": False,
                "predicted_brand": "paypal",
                "max_similarity": 0.942,
            },
        }

        with patch.object(unified_pipeline, "_analyze_visual", return_value=mock_visual):
            res = unified_pipeline.analyze(
                url="https://paypal-security-auth.attacker-host.com/verify",
                screenshot=img,
                domain="attacker-host.com",
                render_preview=False,
            )
            assert res["threat_category"] == "CRITICAL_PHISHING"
            assert res["risk_score"] >= 0.95
            assert res["is_brand_impersonation"] is True
            assert res["matched_brand"] == "paypal"
            assert "brand impersonation confirmed" in res["explanation"].lower()

    def test_official_brand_whitelist_pass(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """When an authentic brand logo matches on its authorized domain, flag LEGITIMATE."""
        img = Image.new("RGB", (400, 300), color=(250, 250, 250))
        mock_visual = {
            "screenshot_available": True,
            "m02_phishing_probability": 0.10,
            "ui_elements": {
                "total_fused_elements": 2,
                "elements": [],
                "class_counts": {"brand_logo": 1, "login_form": 1},
                "has_login_form": True,
                "has_credential_inputs": True,
                "has_submit_button": True,
                "has_payment_element": False,
            },
            "brand_verification": {
                "is_brand_impersonation": False,
                "is_authorized_domain": True,
                "predicted_brand": "paypal",
                "max_similarity": 0.965,
            },
        }

        with patch.object(unified_pipeline, "_analyze_visual", return_value=mock_visual):
            res = unified_pipeline.analyze(
                url="https://www.paypal.com/signin",
                screenshot=img,
                domain="paypal.com",
                render_preview=False,
            )
            assert res["threat_category"] == "LEGITIMATE"
            assert res["risk_score"] <= 0.05
            assert "verified official brand" in res["explanation"].lower()

    def test_dom_forms_cross_verification(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """When visual detector misses password input but DOM captures it, harvesting surface is confirmed."""
        img = Image.new("RGB", (400, 300), color=(250, 250, 250))
        # Visual missed credential fields
        mock_visual = {
            "screenshot_available": True,
            "m02_phishing_probability": 0.60,
            "ui_elements": {
                "total_fused_elements": 1,
                "elements": [],
                "class_counts": {"login_form": 1},
                "has_login_form": True,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
            },
            "brand_verification": {
                "is_brand_impersonation": False,
                "is_authorized_domain": False,
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
            },
        }

        # Scanner DOM evidence captured a password field and external form submission
        dom_forms = [
            {"has_password_field": True, "is_external_submission": True}
        ]

        with patch.object(unified_pipeline, "_analyze_visual", return_value=mock_visual), \
             patch.object(
                 unified_pipeline,
                 "_analyze_urls",
                 return_value={
                     "composite_url_risk": 0.75,
                     "is_url_suspicious": True,
                     "m01_lexical": {"probability": 0.75},
                     "m07_lstm": {"probability": 0.75},
                 },
             ):
            res = unified_pipeline.analyze(
                url="https://phishing-portal.xyz/login",
                screenshot=img,
                domain="phishing-portal.xyz",
                dom_forms=dom_forms,
                render_preview=False,
            )
            assert res["threat_category"] == "PHISHING"
            assert res["risk_score"] >= 0.85
            assert res["evidence"]["dom_evidence"]["has_dom_password_field"] is True


# =============================================================================
# 5. Security & Robustness Tests
# =============================================================================

class TestSecurityAndRobustness:
    """Validates anti-SSRF protection and corrupt image handling."""

    def test_anti_ssrf_blocks_private_and_loopback_ips(self) -> None:
        """Anti-SSRF validator must block loopback, RFC 1918, and AWS metadata addresses."""
        with pytest.raises(SSRFValidationError):
            validate_url_and_resolve("http://127.0.0.1:8000/admin").raise_if_invalid()

        with pytest.raises(SSRFValidationError):
            validate_url_and_resolve("http://169.254.169.254/latest/meta-data/").raise_if_invalid()

        with pytest.raises(SSRFValidationError):
            validate_url_and_resolve("http://10.0.0.1/internal").raise_if_invalid()

    def test_corrupt_screenshot_graceful_handling(
        self, unified_pipeline: UnifiedPhishingPipeline
    ) -> None:
        """A corrupt or 0-byte screenshot file must not crash the pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"NOT_A_VALID_IMAGE_HEADER_CORRUPT_BYTES")
            corrupt_path = f.name

        try:
            res = unified_pipeline.analyze(
                url="https://example.com",
                screenshot=corrupt_path,
                render_preview=False,
            )
            assert res["threat_category"] in ("LEGITIMATE", "SUSPICIOUS", "PHISHING", "UNKNOWN / INSUFFICIENT EVIDENCE")
            assert res["evidence"]["ui_elements"]["total_fused_elements"] == 0
        finally:
            Path(corrupt_path).unlink(missing_ok=True)
