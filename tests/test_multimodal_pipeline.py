"""Unit and regression tests for Phase 7.5.1 Multimodal Phishing Pipeline.

Validates:
1. YOLO UI element detection and 10% context padding calculation.
2. Siamese brand matcher forwarding from detected logo crops.
3. Fallback logic for UNKNOWN / INSUFFICIENT EVIDENCE.
4. Multimodal risk fusion across M-01 (Lexical), M-02 (ResNet-18), M-03 (Siamese), and M-04 (YOLO).
5. Regression test suite:
   - Legitimate neutral URLs.
   - Suspicious-looking but legitimate URLs.
   - Synthetic phishing URLs with credential harvesting.
   - 404 / empty error pages (preventing false positive phishing classification).
   - Missing screenshots.
   - Missing reference gallery graceful handling.
   - Unavailable model checkpoints graceful degradation.
   - Brand similarity without domain mismatch (official authorized brand whitelist pass).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from app.ml.vision.multimodal_pipeline import MultimodalPhishingPipeline
from app.ml.vision.yolo_inference import YOLOUIElementDetector


class TestYOLOInference:
    """Tests for YOLOUIElementDetector."""

    def test_detector_schema_and_classes(self) -> None:
        """Verify the 6-class schema and color mappings."""
        detector = YOLOUIElementDetector()
        assert len(detector.classes) == 6
        assert detector.classes[0] == "brand_logo"
        assert detector.classes[1] == "login_form"
        assert detector.classes[2] == "username_field"
        assert detector.classes[3] == "password_field"
        assert detector.classes[4] == "submit_button"
        assert detector.classes[5] == "payment_element"

    def test_extract_logo_crops_with_padding(self) -> None:
        """Verify that extract_logo_crops applies ~10% padding and clamps to image bounds."""
        detector = YOLOUIElementDetector()

        # Create a 200x200 test image
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))

        mock_detect_result = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 2,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "brand_logo",
                    "confidence": 0.95,
                    "bbox_pixel": [50, 50, 150, 100],  # width 100, height 50
                    "bbox_normalized": [0.5, 0.375, 0.5, 0.25],
                },
                {
                    "class_id": 1,
                    "class_name": "login_form",
                    "confidence": 0.90,
                    "bbox_pixel": [10, 10, 190, 190],
                    "bbox_normalized": [0.5, 0.5, 0.9, 0.9],
                },
            ],
        }

        # 10% padding on width (100) is 10px -> x1=40, x2=160
        # 10% padding on height (50) is 5px -> y1=45, y2=105
        with patch.object(detector, "detect", return_value=mock_detect_result):
            crops = detector.extract_logo_crops(img, padding_pct=0.10)
            assert len(crops) == 1
            crop_data = crops[0]

            assert crop_data["confidence"] == 0.95
            assert crop_data["original_bbox"] == [50, 50, 150, 100]
            assert crop_data["padded_bbox"] == [40, 45, 160, 105]
            assert crop_data["crop_image"].size == (120, 60)

    def test_extract_logo_crops_boundary_clamping(self) -> None:
        """Verify that padding does not exceed image dimensions (0 to width/height)."""
        detector = YOLOUIElementDetector()
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))

        mock_detect_result = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 1,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "brand_logo",
                    "confidence": 0.92,
                    "bbox_pixel": [2, 2, 30, 30],
                    "bbox_normalized": [0.16, 0.16, 0.28, 0.28],
                }
            ],
        }

        with patch.object(detector, "detect", return_value=mock_detect_result):
            crops = detector.extract_logo_crops(img, padding_pct=0.20)
            assert len(crops) == 1
            padded_box = crops[0]["padded_bbox"]
            assert padded_box[0] >= 0
            assert padded_box[1] >= 0
            assert padded_box[2] <= 99
            assert padded_box[3] <= 99

    def test_render_annotated_preview(self) -> None:
        """Verify preview rendering returns a valid PIL Image with expected dimensions."""
        detector = YOLOUIElementDetector()
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        detections = [
            {
                "class_id": 0,
                "class_name": "brand_logo",
                "confidence": 0.91,
                "bbox_pixel": [20, 20, 80, 50],
            },
            {
                "class_id": 4,
                "class_name": "submit_button",
                "confidence": 0.88,
                "bbox_pixel": [100, 150, 200, 180],
            },
        ]
        preview = detector.render_annotated_preview(img, detections=detections)
        assert isinstance(preview, Image.Image)
        assert preview.size == (300, 200)


class TestMultimodalPipeline:
    """Tests for MultimodalPhishingPipeline logic, fusion calibration, and regression scenarios."""

    @pytest.fixture
    def pipeline(self) -> MultimodalPhishingPipeline:
        """Construct pipeline instance with existing checkpoints."""
        return MultimodalPhishingPipeline()

    def test_pipeline_model_availability(self, pipeline: MultimodalPhishingPipeline) -> None:
        """Ensure all 4 models are correctly identified on the filesystem."""
        assert pipeline.m01_available is True
        assert pipeline.m02_available is True
        assert pipeline.m03_available is True
        assert pipeline.m04_available is True

    def test_fallback_insufficient_evidence_when_inconclusive(
        self, pipeline: MultimodalPhishingPipeline
    ) -> None:
        """When visual screenshot is missing and lexical probability is ambiguous (0.35-0.65),
        pipeline must output UNKNOWN / INSUFFICIENT EVIDENCE with neutral score 0.50.
        """
        with patch.object(
            pipeline,
            "_analyze_lexical",
            return_value={"status": "SUCCESS", "probability": 0.55, "prediction": "PHISHING"},
        ):
            result = pipeline.analyze(
                url="https://ambiguous-corporate-example.net/portal",
                screenshot=None,
                domain="ambiguous-corporate-example.net",
            )
            assert result["threat_category"] == "UNKNOWN / INSUFFICIENT EVIDENCE"
            assert "insufficient evidence" in result["explanation"].lower()
            assert result["multimodal_risk_score"] == 0.50

    def test_fallback_decisive_lexical_when_screenshot_missing(
        self, pipeline: MultimodalPhishingPipeline
    ) -> None:
        """When visual screenshot is missing but lexical probability is decisively high (>0.65),
        threat decision is PHISHING.
        """
        with patch.object(
            pipeline,
            "_analyze_lexical",
            return_value={"status": "SUCCESS", "probability": 0.95, "prediction": "PHISHING"},
        ):
            result = pipeline.analyze(
                url="https://paypal-update-account-security-alert.xyz/verify",
                screenshot=None,
                domain="paypal-update-account-security-alert.xyz",
            )
            assert result["threat_category"] == "PHISHING"
            assert result["multimodal_risk_score"] == 0.95

    def test_404_error_page_not_phishing(self, pipeline: MultimodalPhishingPipeline) -> None:
        """A 404 or empty error page (0 UI elements) must NEVER be classified as active PHISHING,
        even if lexical score is elevated. Risk must be capped at 0.40 (SUSPICIOUS / INACTIVE).
        """
        img = Image.new("RGB", (300, 200), color=(255, 255, 255))
        mock_ui = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 0,
            "class_counts": {},
            "has_login_form": False,
            "has_credential_inputs": False,
            "has_submit_button": False,
            "has_payment_element": False,
            "detections": [],
        }

        with patch.object(pipeline, "_analyze_lexical", return_value={"status": "SUCCESS", "probability": 1.0}), \
             patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui), \
             patch.object(pipeline.resnet_infer, "predict", return_value={"probabilities": {"phishing": 1.0, "legitimate": 0.0}}):

            result = pipeline.analyze(
                url="https://help-docs.cloudsupport-knowledge.org/articles/error-404",
                screenshot=img,
                domain="cloudsupport-knowledge.org",
                render_preview=False,
            )

            assert result["threat_category"] == "SUSPICIOUS"
            assert result["multimodal_risk_score"] <= 0.40
            assert "empty or 404 error page" in result["explanation"].lower()

    def test_legitimate_neutral_url(self, pipeline: MultimodalPhishingPipeline) -> None:
        """Legitimate neutral website without credential inputs must be classified as LEGITIMATE."""
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        mock_ui = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 1,
            "class_counts": {"brand_logo": 1},
            "has_login_form": False,
            "has_credential_inputs": False,
            "has_submit_button": False,
            "has_payment_element": False,
            "detections": [{"class_id": 0, "class_name": "brand_logo", "confidence": 0.85, "bbox_pixel": [10, 10, 50, 50]}],
        }

        with patch.object(pipeline, "_analyze_lexical", return_value={"status": "SUCCESS", "probability": 0.20}), \
             patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui), \
             patch.object(pipeline.yolo_detector, "extract_logo_crops", return_value=[]), \
             patch.object(pipeline.resnet_infer, "predict", return_value={"probabilities": {"phishing": 0.25, "legitimate": 0.75}}):

            result = pipeline.analyze(
                url="https://www.example-company.org/about-us",
                screenshot=img,
                domain="example-company.org",
                render_preview=False,
            )

            assert result["threat_category"] == "LEGITIMATE"
            assert result["multimodal_risk_score"] <= 0.30

    def test_suspicious_looking_legitimate_url(self, pipeline: MultimodalPhishingPipeline) -> None:
        """When lexical model is somewhat elevated on an unusual URL but visual telemetry is completely benign,
        page is classified as LEGITIMATE with risk capped <= 0.30.
        """
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        mock_ui = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 1,
            "class_counts": {"brand_logo": 1},
            "has_login_form": False,
            "has_credential_inputs": False,
            "has_submit_button": False,
            "has_payment_element": False,
            "detections": [{"class_id": 0, "class_name": "brand_logo", "confidence": 0.88, "bbox_pixel": [10, 10, 50, 50]}],
        }

        # Elevated lexical (0.80) but low ResNet (0.25) and clean visual layout (no inputs)
        with patch.object(pipeline, "_analyze_lexical", return_value={"status": "SUCCESS", "probability": 0.80}), \
             patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui), \
             patch.object(pipeline.yolo_detector, "extract_logo_crops", return_value=[]), \
             patch.object(pipeline.resnet_infer, "predict", return_value={"probabilities": {"phishing": 0.25, "legitimate": 0.75}}):

            result = pipeline.analyze(
                url="https://t.me",
                screenshot=img,
                domain="t.me",
                render_preview=False,
            )

            assert result["threat_category"] == "LEGITIMATE"
            assert result["multimodal_risk_score"] <= 0.30

    def test_synthetic_phishing_credential_harvest(
        self, pipeline: MultimodalPhishingPipeline
    ) -> None:
        """When active login credential inputs exist on a suspicious host, decision must be PHISHING."""
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        mock_ui = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 4,
            "class_counts": {"login_form": 1, "username_field": 1, "password_field": 1, "submit_button": 1},
            "has_login_form": True,
            "has_credential_inputs": True,
            "has_submit_button": True,
            "has_payment_element": False,
            "detections": [
                {"class_id": 1, "class_name": "login_form", "confidence": 0.95, "bbox_pixel": [20, 20, 200, 200]},
                {"class_id": 2, "class_name": "username_field", "confidence": 0.92, "bbox_pixel": [30, 40, 180, 70]},
                {"class_id": 3, "class_name": "password_field", "confidence": 0.92, "bbox_pixel": [30, 80, 180, 110]},
                {"class_id": 4, "class_name": "submit_button", "confidence": 0.90, "bbox_pixel": [30, 120, 180, 150]},
            ],
        }

        with patch.object(pipeline, "_analyze_lexical", return_value={"status": "SUCCESS", "probability": 0.95}), \
             patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui), \
             patch.object(pipeline.yolo_detector, "extract_logo_crops", return_value=[]), \
             patch.object(pipeline.resnet_infer, "predict", return_value={"probabilities": {"phishing": 0.90, "legitimate": 0.10}}):

            result = pipeline.analyze(
                url="https://secure-login.titanvault-portal.net/auth/signin",
                screenshot=img,
                domain="titanvault-portal.net",
                render_preview=False,
            )

            assert result["threat_category"] == "PHISHING"
            assert result["multimodal_risk_score"] >= 0.85
            assert "credential fields" in result["explanation"].lower()

    def test_brand_similarity_without_domain_mismatch(
        self, pipeline: MultimodalPhishingPipeline
    ) -> None:
        """When visual logo matches a brand and domain IS in the authorized list,
        decision must be LEGITIMATE with minimal risk (<= 0.05).
        """
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        mock_crop = {
            "crop_image": Image.new("RGB", (50, 50)),
            "confidence": 0.98,
            "original_bbox": [50, 20, 150, 70],
            "padded_bbox": [40, 15, 160, 75],
            "crop_dimensions": [50, 50],
        }
        mock_brand_result = {
            "is_brand_impersonation": False,
            "is_authorized_domain": True,
            "predicted_brand": "paypal",
            "matched_brand": "paypal",
            "target_registrable_domain": "paypal.com",
            "authorized_domains": ["paypal.com", "paypal-community.com"],
            "max_similarity": 0.96,
            "similarity_score": 0.96,
            "verdict": "LEGITIMATE_OFFICIAL_BRAND",
        }
        mock_ui = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 1,
            "class_counts": {"brand_logo": 1},
            "has_login_form": False,
            "has_credential_inputs": False,
            "has_submit_button": False,
            "has_payment_element": False,
            "detections": [{"class_id": 0, "class_name": "brand_logo", "confidence": 0.98, "bbox_pixel": [50, 20, 150, 70]}],
        }

        with patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui), \
             patch.object(pipeline.yolo_detector, "extract_logo_crops", return_value=[mock_crop]), \
             patch.object(pipeline.siamese_matcher, "verify_brand_authorization", return_value=mock_brand_result):

            result = pipeline.analyze(
                url="https://www.paypal.com/signin",
                screenshot=img,
                domain="paypal.com",
                render_preview=False,
            )

            assert result["threat_category"] == "LEGITIMATE"
            assert result["multimodal_risk_score"] <= 0.05
            assert "verified official brand" in result["explanation"].lower()

    def test_multimodal_brand_impersonation_detection(
        self, pipeline: MultimodalPhishingPipeline
    ) -> None:
        """Test brand impersonation escalation when Siamese detects unauthorized brand match."""
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))

        mock_crop = {
            "crop_image": Image.new("RGB", (50, 50)),
            "confidence": 0.98,
            "original_bbox": [50, 20, 150, 70],
            "padded_bbox": [40, 15, 160, 75],
            "crop_dimensions": [50, 50],
        }

        mock_brand_result = {
            "is_brand_impersonation": True,
            "is_authorized_domain": False,
            "predicted_brand": "paypal",
            "matched_brand": "paypal",
            "matched_domain": "paypal.com",
            "max_similarity": 0.92,
            "similarity_score": 0.92,
            "decision": "UNAUTHORIZED_IMPERSONATION",
            "authorized_domains": ["paypal.com", "paypal-communication.com"],
        }

        mock_ui_summary = {
            "status": "SUCCESS",
            "is_available": True,
            "total_detections": 2,
            "class_counts": {
                "brand_logo": 1,
                "login_form": 1,
                "username_field": 0,
                "password_field": 0,
                "submit_button": 0,
                "payment_element": 0,
            },
            "has_login_form": True,
            "has_credential_inputs": False,
            "has_submit_button": False,
            "has_payment_element": False,
            "detections": [
                {"class_id": 0, "class_name": "brand_logo", "confidence": 0.98, "bbox_pixel": [50, 20, 150, 70]},
                {"class_id": 1, "class_name": "login_form", "confidence": 0.95, "bbox_pixel": [50, 100, 350, 250]},
            ],
        }

        mock_resnet_result = {
            "predicted_label": "phishing",
            "probabilities": {"phishing": 0.85, "legitimate": 0.15},
            "status": "SUCCESS",
        }

        with patch.object(pipeline.yolo_detector, "detect", return_value=mock_ui_summary), \
             patch.object(pipeline.yolo_detector, "extract_logo_crops", return_value=[mock_crop]), \
             patch.object(pipeline.siamese_matcher, "verify_brand_authorization", return_value=mock_brand_result), \
             patch.object(pipeline.resnet_infer, "predict", return_value=mock_resnet_result):
            
            result = pipeline.analyze(
                url="https://pay-pal-security-verification.com/login",
                screenshot=img,
                domain="pay-pal-security-verification.com",
                render_preview=False,
            )

            assert result["threat_category"] in ("PHISHING", "CRITICAL_PHISHING")
            assert result["brand_verification"]["is_brand_impersonation"] is True
            assert result["brand_verification"]["matched_brand"] == "paypal"
            assert result["multimodal_risk_score"] >= 0.90
            assert "brand impersonation" in result["explanation"].lower()

    def test_missing_reference_gallery_graceful_handling(self) -> None:
        """Pipeline must handle a missing brand gallery directory gracefully without crashing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            pipe = MultimodalPhishingPipeline(brand_gallery_dir=empty_dir)
            # Siamese matcher should initialize with empty gallery embeddings
            assert pipe.m03_available is True
            img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            res = pipe.analyze(url="https://example.com", screenshot=img, domain="example.com")
            assert "decision" in res
            assert res["brand_verification"]["predicted_brand"] == "UNIDENTIFIED_OR_NEUTRAL"

    def test_unavailable_checkpoints_graceful_degradation(self) -> None:
        """When checkpoints do not exist, pipeline flags model availability as False and falls back safely."""
        pipe = MultimodalPhishingPipeline(
            m01_weights_path="nonexistent/m01.npz",
            m01_preprocessor_path="nonexistent/prep.json",
            m02_checkpoint_path="nonexistent/m02.pt",
            m03_checkpoint_path="nonexistent/m03.pt",
            m04_checkpoint_path="nonexistent/m04.pt",
        )
        assert pipe.m01_available is False
        assert pipe.m02_available is False
        assert pipe.m03_available is False
        assert pipe.m04_available is False

        # When all models are unavailable, fallback to UNKNOWN / INSUFFICIENT EVIDENCE
        res = pipe.analyze(url="https://unknown-host.net")
        assert res["threat_category"] == "UNKNOWN / INSUFFICIENT EVIDENCE"
        assert res["risk_score"] == 0.50
