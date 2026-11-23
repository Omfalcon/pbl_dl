"""Unified Seven-Model Multimodal Phishing Analysis Pipeline (Phase 8).

Integrates all 7 PhishVision AI model pipelines:
- M-01: URL Lexical Multi-Layer Perceptron (fast-path, threshold=0.05)
- M-02: ResNet-18 Global Webpage Screenshot Classifier
- M-03: Siamese CNN Brand Impersonation Matcher (threshold=0.82)
- M-04: YOLOv8n UI Component & Logo Detector
- M-05: Faster R-CNN Object Detector (ResNet-50-FPN)
- M-06: YOLOv8-Seg UI Component Instance Segmenter
- M-07: URL Sequence Bi-LSTM (threshold=0.01)

Features:
1. Spatial Consensus Engine: Prevents duplicate element counting across M-04, M-05, and M-06.
2. Calibrated Decision Normalization: Calibrated thresholds for cost-sensitive models (M-01, M-07).
3. DOM & Visual Cross-Verification: Fuses scanner DOM form evidence with visual detectors.
4. Explainable Multi-Evidence Threat Scoring: Transparent rationale without arbitrary black-box averages.
5. Strict Fallbacks: Preserves UNKNOWN / INSUFFICIENT EVIDENCE and SSRF security controls.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.intelligence.lexical import LexicalFeatureExtractor
from app.intelligence.lexical_model import LexicalMLP
from app.intelligence.preprocessing import LexicalPreprocessor
from app.intelligence.url_domain.psl_extractor import PSLExtractor
from app.ml.sequence.lstm_inference import LSTMInferencer
from app.ml.vision.faster_rcnn import FasterRCNNInferencer
from app.ml.vision.inference import ResNet18Inference
from app.ml.vision.preprocessing import safe_load_image
from app.ml.vision.siamese_inference import SiameseBrandMatcher
from app.ml.vision.spatial_consensus import SpatialConsensusEngine, UI_CLASSES
from app.ml.vision.yolo_inference import YOLOUIElementDetector
from app.ml.vision.yolo_seg import YOLOSegInferencer

logger = logging.getLogger("phishvision.unified_pipeline")


def normalize_calibrated_risk(probability: float, threshold: float) -> float:
    """Map a raw model probability to a normalized risk scale in [0.0, 1.0].

    Maps raw decision threshold to 0.50 so that:
    - Below threshold: scales from 0.0 to 0.499.
    - At or above threshold: scales from 0.50 to 1.00.
    """
    if probability is None or np.isnan(probability):
        return 0.50

    p = float(np.clip(probability, 0.0, 1.0))
    t = float(np.clip(threshold, 0.001, 0.999))

    if p < t:
        return round(0.50 * (p / t), 4)
    else:
        return round(0.50 + 0.50 * ((p - t) / (1.0 - t)), 4)


class UnifiedPhishingPipeline:
    """Unified production and forensic inference engine orchestrating all seven models."""

    def __init__(
        self,
        m01_weights_path: str | Path = "models/lexical/model_weights.npz",
        m01_preprocessor_path: str | Path = "models/lexical/preprocessor.json",
        m02_checkpoint_path: str | Path = "models/checkpoints/resnet18/best_model.pt",
        m03_checkpoint_path: str | Path = "models/checkpoints/siamese/best_siamese.pt",
        m04_checkpoint_path: str | Path = "models/checkpoints/yolov8n_synthetic/run1/weights/best.pt",
        m05_checkpoint_path: str | Path = "models/checkpoints/faster_rcnn_synthetic/best_faster_rcnn.pt",
        m06_checkpoint_path: str | Path = "models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt",
        m07_checkpoint_path: str | Path = "models/checkpoints/lstm/best_lstm.pt",
        m07_preprocessor_path: str | Path = "models/checkpoints/lstm/preprocessor.json",
        brand_gallery_dir: str | Path = "data/brands/reference_gallery",
        device: str | torch.device | None = None,
        enable_faster_rcnn: bool = True,
        enable_yolo_seg: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Environment variable overrides
        m01_weights_path = os.getenv("PHISHVISION_M01_WEIGHTS", str(m01_weights_path))
        m01_preprocessor_path = os.getenv("PHISHVISION_M01_PREPROCESSOR", str(m01_preprocessor_path))
        m02_checkpoint_path = os.getenv("PHISHVISION_M02_CHECKPOINT", str(m02_checkpoint_path))
        m03_checkpoint_path = os.getenv("PHISHVISION_M03_CHECKPOINT", str(m03_checkpoint_path))
        m04_checkpoint_path = os.getenv("PHISHVISION_M04_CHECKPOINT", str(m04_checkpoint_path))
        m05_checkpoint_path = os.getenv("PHISHVISION_M05_CHECKPOINT", str(m05_checkpoint_path))
        m06_checkpoint_path = os.getenv("PHISHVISION_M06_CHECKPOINT", str(m06_checkpoint_path))
        m07_checkpoint_path = os.getenv("PHISHVISION_M07_CHECKPOINT", str(m07_checkpoint_path))
        m07_preprocessor_path = os.getenv("PHISHVISION_M07_PREPROCESSOR", str(m07_preprocessor_path))
        brand_gallery_dir = os.getenv("PHISHVISION_BRAND_GALLERY", str(brand_gallery_dir))

        self.checkpoint_paths = {
            "m01_weights": m01_weights_path,
            "m01_preprocessor": m01_preprocessor_path,
            "m02_checkpoint": m02_checkpoint_path,
            "m03_checkpoint": m03_checkpoint_path,
            "m04_checkpoint": m04_checkpoint_path,
            "m05_checkpoint": m05_checkpoint_path,
            "m06_checkpoint": m06_checkpoint_path,
            "m07_checkpoint": m07_checkpoint_path,
            "m07_preprocessor": m07_preprocessor_path,
            "brand_gallery": brand_gallery_dir,
        }

        # ---------------------------------------------------------------------
        # 1. Initialize M-01 URL Lexical MLP
        # ---------------------------------------------------------------------
        self.m01_available = False
        self.lexical_extractor = LexicalFeatureExtractor()
        self.lexical_preprocessor = None
        self.lexical_mlp = None
        self.m01_threshold = 0.050

        prep_p = Path(m01_preprocessor_path)
        weights_p = Path(m01_weights_path)
        if prep_p.exists() and weights_p.exists():
            try:
                self.lexical_preprocessor = LexicalPreprocessor.load(prep_p)
                self.lexical_mlp = LexicalMLP(in_features=40, hidden_dim1=64, hidden_dim2=32)
                self.lexical_mlp.load_weights(weights_p)
                self.m01_available = True
                logger.info("Initialized M-01 Lexical MLP pipeline.")
            except Exception as e:
                logger.warning(f"Failed to load M-01 Lexical model: {e}")

        # ---------------------------------------------------------------------
        # 2. Initialize M-02 ResNet-18 Global Screenshot Classifier
        # ---------------------------------------------------------------------
        self.m02_available = False
        self.resnet_infer = None
        res_p = Path(m02_checkpoint_path)
        if res_p.exists():
            try:
                self.resnet_infer = ResNet18Inference(checkpoint_path=res_p, device=self.device)
                self.m02_available = True
                logger.info("Initialized M-02 ResNet-18 visual classifier.")
            except Exception as e:
                logger.warning(f"Failed to load M-02 ResNet-18 model: {e}")

        # ---------------------------------------------------------------------
        # 3. Initialize M-03 Siamese Brand Impersonation Matcher
        # ---------------------------------------------------------------------
        self.m03_available = False
        self.siamese_matcher = None
        self.m03_threshold = 0.820
        siamese_p = Path(m03_checkpoint_path)
        if siamese_p.exists():
            try:
                self.siamese_matcher = SiameseBrandMatcher(
                    checkpoint_path=siamese_p,
                    gallery_dir=brand_gallery_dir,
                    device=self.device,
                )
                self.m03_available = True
                logger.info("Initialized M-03 Siamese Brand Matcher.")
            except Exception as e:
                logger.warning(f"Failed to load M-03 Siamese model: {e}")

        # ---------------------------------------------------------------------
        # 4. Initialize M-04 YOLOv8n UI Component Detector
        # ---------------------------------------------------------------------
        self.m04_available = False
        self.yolo_detector = None
        m04_p = Path(m04_checkpoint_path)
        if m04_p.exists():
            try:
                self.yolo_detector = YOLOUIElementDetector(checkpoint_path=m04_p, device=self.device)
                self.m04_available = self.yolo_detector.is_available
                if self.m04_available:
                    logger.info("Initialized M-04 YOLOv8n UI Component Detector.")
            except Exception as e:
                logger.warning(f"Failed to load M-04 YOLOv8n detector: {e}")

        # ---------------------------------------------------------------------
        # 5. Initialize M-05 Faster R-CNN UI Component Detector
        # ---------------------------------------------------------------------
        self.m05_available = False
        self.faster_rcnn_infer = None
        m05_p = Path(m05_checkpoint_path)
        if enable_faster_rcnn and m05_p.exists():
            try:
                self.faster_rcnn_infer = FasterRCNNInferencer(checkpoint_path=m05_p, device=self.device)
                self.m05_available = True
                logger.info("Initialized M-05 Faster R-CNN Object Detector.")
            except Exception as e:
                logger.warning(f"Failed to load M-05 Faster R-CNN detector: {e}")

        # ---------------------------------------------------------------------
        # 6. Initialize M-06 YOLOv8-Seg Instance Segmenter
        # ---------------------------------------------------------------------
        self.m06_available = False
        self.yolo_seg_infer = None
        m06_p = Path(m06_checkpoint_path)
        if enable_yolo_seg and m06_p.exists():
            try:
                self.yolo_seg_infer = YOLOSegInferencer(model_weights=m06_p, device=self.device)
                self.m06_available = True
                logger.info("Initialized M-06 YOLOv8-Seg Instance Segmenter.")
            except Exception as e:
                logger.warning(f"Failed to load M-06 YOLOv8-Seg: {e}")

        # ---------------------------------------------------------------------
        # 7. Initialize M-07 URL Sequence Bi-LSTM
        # ---------------------------------------------------------------------
        self.m07_available = False
        self.lstm_infer = None
        self.m07_threshold = 0.010
        m07_p = Path(m07_checkpoint_path)
        m07_prep_p = Path(m07_preprocessor_path)
        if m07_p.exists() and m07_prep_p.exists():
            try:
                self.lstm_infer = LSTMInferencer(
                    checkpoint_path=m07_p,
                    preprocessor_path=m07_prep_p,
                    device=self.device,
                    threshold=self.m07_threshold,
                )
                self.m07_threshold = self.lstm_infer.threshold
                self.m07_available = True
                logger.info(f"Initialized M-07 URL Sequence Bi-LSTM (threshold={self.m07_threshold}).")
            except Exception as e:
                logger.warning(f"Failed to load M-07 LSTM model: {e}")

        # Spatial Consensus Engine for Non-Duplicative Visual Detection
        self.spatial_engine = SpatialConsensusEngine(iou_threshold=0.45)

    def get_checkpoint_status(self) -> dict[str, dict[str, Any]]:
        """Return physical existence and operational status for all seven models.

        Ensures full visibility into checkpoint deployment and prevents claiming
        models are operational when weights are missing.
        """
        models_info = {
            "M-01_lexical_mlp": {
                "name": "M-01 URL Lexical MLP",
                "weights_path": str(self.checkpoint_paths["m01_weights"]),
                "weights_exist": Path(self.checkpoint_paths["m01_weights"]).exists(),
                "auxiliary_exist": Path(self.checkpoint_paths["m01_preprocessor"]).exists(),
                "status": "LOADED" if self.m01_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m01_weights"]).exists() else "LOAD_FAILED"),
            },
            "M-02_resnet18": {
                "name": "M-02 ResNet-18 Global Classifier",
                "weights_path": str(self.checkpoint_paths["m02_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m02_checkpoint"]).exists(),
                "status": "LOADED" if self.m02_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m02_checkpoint"]).exists() else "LOAD_FAILED"),
            },
            "M-03_siamese": {
                "name": "M-03 Siamese CNN Brand Matcher",
                "weights_path": str(self.checkpoint_paths["m03_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m03_checkpoint"]).exists(),
                "status": "LOADED" if self.m03_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m03_checkpoint"]).exists() else "LOAD_FAILED"),
            },
            "M-04_yolov8n": {
                "name": "M-04 YOLOv8n UI Detector",
                "weights_path": str(self.checkpoint_paths["m04_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m04_checkpoint"]).exists(),
                "status": "LOADED" if self.m04_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m04_checkpoint"]).exists() else "LOAD_FAILED"),
            },
            "M-05_faster_rcnn": {
                "name": "M-05 Faster R-CNN (ResNet-50-FPN)",
                "weights_path": str(self.checkpoint_paths["m05_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m05_checkpoint"]).exists(),
                "status": "LOADED" if self.m05_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m05_checkpoint"]).exists() else "LOAD_FAILED"),
            },
            "M-06_yolov8_seg": {
                "name": "M-06 YOLOv8-Seg Instance Segmenter",
                "weights_path": str(self.checkpoint_paths["m06_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m06_checkpoint"]).exists(),
                "status": "LOADED" if self.m06_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m06_checkpoint"]).exists() else "LOAD_FAILED"),
            },
            "M-07_lstm": {
                "name": "M-07 URL Sequence Bi-LSTM",
                "weights_path": str(self.checkpoint_paths["m07_checkpoint"]),
                "weights_exist": Path(self.checkpoint_paths["m07_checkpoint"]).exists(),
                "auxiliary_exist": Path(self.checkpoint_paths["m07_preprocessor"]).exists(),
                "status": "LOADED" if self.m07_available else ("MISSING_WEIGHTS" if not Path(self.checkpoint_paths["m07_checkpoint"]).exists() else "LOAD_FAILED"),
            },
        }
        return models_info

    def analyze(
        self,
        url: str,
        screenshot: str | Path | Image.Image | None = None,
        domain: str | None = None,
        sample_id: str | None = None,
        redirect_chain: list[str] | None = None,
        dom_forms: list[dict[str, Any]] | None = None,
        render_preview: bool = True,
        preview_output_dir: str | Path = "data/reports/pipeline_previews",
    ) -> dict[str, Any]:
        """Perform unified multi-modal phishing forensic analysis across all 7 models.

        Args:
            url: Target URL string.
            screenshot: Webpage screenshot (file path or PIL Image).
            domain: Optional override for target registrable domain.
            sample_id: Identifier used for preview image filenames.
            redirect_chain: Optional list of intermediate redirect hostnames.
            dom_forms: Optional list of DOM forms captured by scanner.
            render_preview: Whether to render an annotated visual preview.
            preview_output_dir: Directory where preview images are saved.

        Returns:
            Structured explainable forensic report.
        """
        # Resolve target registrable domain
        if not domain:
            try:
                psl_extractor = getattr(self.lexical_extractor, "psl_extractor", None) or PSLExtractor()
                psl_res = psl_extractor.extract_from_url(url)
                target_domain = psl_res.registrable_domain or psl_res.normalized_hostname or "unknown_domain"
            except Exception:
                target_domain = "unknown_domain"
        else:
            target_domain = domain.strip().lower()

        sid = sample_id or (
            Path(str(screenshot)).stem if screenshot and not isinstance(screenshot, Image.Image) else "sample"
        )

        # ---------------------------------------------------------------------
        # Tier 1: URL Evidence (M-01 Lexical MLP + M-07 Sequence Bi-LSTM)
        # ---------------------------------------------------------------------
        url_evidence = self._analyze_urls(url=url, redirect_chain=redirect_chain)

        # ---------------------------------------------------------------------
        # Tier 2: Visual Evidence (M-02, M-04, M-05, M-06, M-03)
        # ---------------------------------------------------------------------
        screenshot_loaded = None
        screenshot_valid = False

        if screenshot is not None:
            try:
                if isinstance(screenshot, Image.Image):
                    screenshot_loaded = screenshot.convert("RGB")
                    screenshot_valid = True
                else:
                    sc_p = Path(screenshot)
                    if sc_p.exists() and sc_p.stat().st_size > 0:
                        screenshot_loaded = safe_load_image(sc_p)
                        screenshot_valid = True
            except Exception as e:
                logger.warning(f"Could not load screenshot: {e}")
                screenshot_loaded = None
                screenshot_valid = False

        if not screenshot_valid or screenshot_loaded is None:
            visual_evidence = self._build_empty_visual_evidence()
        else:
            visual_evidence = self._analyze_visual(
                screenshot=screenshot_loaded,
                target_domain=target_domain,
                sample_id=sid,
                render_preview=render_preview,
                preview_output_dir=preview_output_dir,
            )

        # ---------------------------------------------------------------------
        # Tier 3: DOM Evidence Cross-Check (from Playwright scanner if available)
        # ---------------------------------------------------------------------
        dom_evidence = self._analyze_dom_forms(dom_forms, target_domain)

        # ---------------------------------------------------------------------
        # Tier 4: Explainable Multi-Evidence Threat Scoring & Risk Fusion
        # ---------------------------------------------------------------------
        fusion_result = self._fuse_seven_models(
            url_evidence=url_evidence,
            visual_evidence=visual_evidence,
            dom_evidence=dom_evidence,
            target_domain=target_domain,
        )

        brand_verif = visual_evidence.get("brand_verification", {})
        matched_brand = brand_verif.get("predicted_brand") or brand_verif.get("matched_brand", "UNIDENTIFIED_OR_NEUTRAL")
        brand_sim = brand_verif.get("max_similarity", 0.0)

        return {
            "url": url,
            "target_registrable_domain": target_domain,
            "decision": fusion_result["threat_category"],
            "threat_category": fusion_result["threat_category"],
            "risk_score": fusion_result["risk_score"],
            "multimodal_risk_score": fusion_result["risk_score"],
            "is_brand_impersonation": brand_verif.get("is_brand_impersonation", False),
            "matched_brand": matched_brand,
            "brand_similarity": brand_sim,
            "brand_verification": brand_verif,
            "model_availability": {
                "M-01_lexical_mlp": self.m01_available,
                "M-02_resnet18_classifier": self.m02_available,
                "M-03_siamese_brand_matcher": self.m03_available,
                "M-04_yolov8n_detector": self.m04_available,
                "M-05_faster_rcnn_detector": self.m05_available,
                "M-06_yolov8_seg_segmenter": self.m06_available,
                "M-07_lstm_sequence_classifier": self.m07_available,
            },
            "evidence": {
                "url_models": url_evidence,
                "lexical": url_evidence.get("m01_lexical", {}),
                "visual_global": {
                    "m02_probability": visual_evidence.get("m02_phishing_probability"),
                    "m02_prediction": visual_evidence.get("m02_prediction"),
                },
                "ui_elements": visual_evidence.get("ui_elements", {}),
                "brand_verification": brand_verif,
                "dom_evidence": dom_evidence,
            },
            "explanation": fusion_result["explanation"],
            "disclaimer": (
                "Notice: M-04, M-05, and M-06 were trained on synthetic UI elements. "
                "Synthetic mAP does not establish real-world detection accuracy. "
                "M-07 redirect branch operates in fallback mode as labelled multi-hop data was not in historical corpus."
            ),
        }

    # -------------------------------------------------------------------------
    # Helper: URL Analysis (M-01 + M-07)
    # -------------------------------------------------------------------------
    def _analyze_urls(self, url: str, redirect_chain: list[str] | None = None) -> dict[str, Any]:
        """Runs M-01 (Lexical MLP) and M-07 (URL Sequence Bi-LSTM)."""
        m01_res: dict[str, Any] = {"status": "UNAVAILABLE", "probability": None}
        m07_res: dict[str, Any] = {"status": "UNAVAILABLE", "probability": None}

        # 1. M-01 Lexical
        if self.m01_available and self.lexical_mlp is not None and self.lexical_preprocessor is not None:
            try:
                eval_url = url.strip()
                if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", eval_url):
                    eval_url = "http://" + eval_url
                parsed = urllib.parse.urlsplit(eval_url)
                if not parsed.path:
                    eval_url = urllib.parse.urlunsplit(
                        (parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment)
                    )

                feats = self.lexical_extractor.extract_features(eval_url)
                vector = np.array([feats[name] for name in self.lexical_extractor.feature_names], dtype=np.float64)
                norm_x = self.lexical_preprocessor.transform(vector.reshape(1, -1))
                proba = float(self.lexical_mlp.predict_proba(norm_x)[0, 0])
                pred = int(proba >= self.m01_threshold)

                norm_risk = normalize_calibrated_risk(proba, self.m01_threshold)

                m01_res = {
                    "status": "SUCCESS",
                    "probability": round(proba, 4),
                    "normalized_risk": norm_risk,
                    "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
                    "calibrated_threshold": self.m01_threshold,
                    "is_above_calibrated_threshold": bool(proba >= self.m01_threshold),
                    "url_entropy": round(feats.get("url_entropy", 0.0), 3),
                    "has_ip_hostname": bool(feats.get("has_ip_hostname", False)),
                    "is_punycode": bool(feats.get("is_punycode", False)),
                    "suspicious_keywords_count": int(feats.get("count_suspicious_keywords", 0)),
                    "brand_in_subdomain_or_path": bool(feats.get("brand_in_subdomain_or_path", False)),
                }
            except Exception as e:
                logger.warning(f"Error in M-01 lexical evaluation: {e}")
                m01_res = {"status": "ERROR", "error": str(e), "probability": None}

        # 2. M-07 Sequence LSTM
        if self.m07_available and self.lstm_infer is not None:
            try:
                pred_lstm = self.lstm_infer.predict(
                    url=url,
                    redirect_chain=redirect_chain,
                    threshold=self.m07_threshold,
                )
                lstm_prob = float(pred_lstm.get("phishing_probability", 0.0))
                norm_risk = normalize_calibrated_risk(lstm_prob, self.m07_threshold)

                m07_res = {
                    "status": "SUCCESS",
                    "probability": round(lstm_prob, 4),
                    "normalized_risk": norm_risk,
                    "prediction": pred_lstm.get("predicted_label", "UNKNOWN"),
                    "calibrated_threshold": self.m07_threshold,
                    "is_above_calibrated_threshold": bool(lstm_prob >= self.m07_threshold),
                    "sequence_length": pred_lstm.get("url_sequence_length", len(url)),
                    "redirect_chain_provided": bool(redirect_chain and len(redirect_chain) > 0),
                    "redirect_training_status": "NOT_EMPIRICALLY_TRAINED_ON_REDIRECTS",
                }
            except Exception as e:
                logger.warning(f"Error in M-07 LSTM evaluation: {e}")
                m07_res = {"status": "ERROR", "error": str(e), "probability": None}

        # Combined URL risk score
        valid_risks = [
            r["normalized_risk"]
            for r in (m01_res, m07_res)
            if r.get("status") == "SUCCESS" and "normalized_risk" in r
        ]
        composite_url_risk = round(float(np.mean(valid_risks)), 4) if valid_risks else None

        return {
            "m01_lexical": m01_res,
            "m07_lstm": m07_res,
            "composite_url_risk": composite_url_risk,
            "is_url_suspicious": any(
                r.get("is_above_calibrated_threshold", False) for r in (m01_res, m07_res)
            ),
        }

    # -------------------------------------------------------------------------
    # Helper: Visual Evidence (M-02, M-04, M-05, M-06, M-03)
    # -------------------------------------------------------------------------
    def _analyze_visual(
        self,
        screenshot: Image.Image,
        target_domain: str,
        sample_id: str = "sample",
        render_preview: bool = True,
        preview_output_dir: str | Path = "data/reports/pipeline_previews",
    ) -> dict[str, Any]:
        """Runs M-02 ResNet, M-04 YOLO, M-05 Faster R-CNN, M-06 YOLO-Seg, and M-03 Siamese."""
        res: dict[str, Any] = {"screenshot_available": True}

        # 1. M-02 ResNet-18
        if self.m02_available and self.resnet_infer is not None:
            try:
                m02_pred = self.resnet_infer.predict(screenshot)
                phish_prob = m02_pred.get("probabilities", {}).get("phishing", 0.0)
                res["m02_phishing_probability"] = round(phish_prob, 4)
                res["m02_prediction"] = m02_pred.get(
                    "predicted_label", "phishing" if phish_prob >= 0.50 else "legitimate"
                )
                res["m02_status"] = "SUCCESS"
            except Exception as e:
                logger.warning(f"M-02 prediction failed: {e}")
                res["m02_phishing_probability"] = None
                res["m02_status"] = f"ERROR: {e}"
        else:
            res["m02_phishing_probability"] = None
            res["m02_status"] = "UNAVAILABLE"

        # 2. Spatial Detectors: M-04 (YOLO), M-05 (Faster R-CNN), M-06 (YOLO-Seg)
        yolo_dets = []
        if self.m04_available and self.yolo_detector is not None:
            try:
                det_res = self.yolo_detector.detect(screenshot)
                yolo_dets = det_res.get("detections", [])
            except Exception as e:
                logger.warning(f"M-04 YOLO detection failed: {e}")

        faster_rcnn_dets = []
        if self.m05_available and self.faster_rcnn_infer is not None:
            try:
                fr_res = self.faster_rcnn_infer.predict(screenshot)
                faster_rcnn_dets = fr_res.get("detections", [])
            except Exception as e:
                logger.warning(f"M-05 Faster R-CNN detection failed: {e}")

        yolo_seg_insts = []
        if self.m06_available and self.yolo_seg_infer is not None:
            try:
                seg_res = self.yolo_seg_infer.predict(screenshot)
                yolo_seg_insts = seg_res.get("instances", [])
            except Exception as e:
                logger.warning(f"M-06 YOLO-Seg prediction failed: {e}")

        # 3. Spatial Consensus & Deduplication
        fused_spatial = self.spatial_engine.fuse_detections(
            yolo_detections=yolo_dets,
            faster_rcnn_detections=faster_rcnn_dets,
            yolo_seg_instances=yolo_seg_insts,
            image_size=screenshot.size,
        )
        res["ui_elements"] = fused_spatial

        # Preview rendering
        if render_preview and self.m04_available and self.yolo_detector is not None:
            try:
                preview_path = Path(preview_output_dir) / f"{sample_id}_annotated.png"
                self.yolo_detector.render_annotated_preview(
                    screenshot,
                    detections=fused_spatial["elements"],
                    output_path=preview_path,
                )
                res["ui_elements"]["preview_image_path"] = str(preview_path)
            except Exception as e:
                logger.warning(f"Failed to render preview: {e}")

        # 4. M-03 Siamese Brand Impersonation Verification from Fused Logo Crops
        logo_crops = self.spatial_engine.extract_fused_logo_crops(
            image=screenshot,
            fused_elements=fused_spatial["elements"],
            padding_pct=0.10,
        )

        if not logo_crops or not self.m03_available or self.siamese_matcher is None:
            res["brand_verification"] = {
                "status": "NO_LOGO_CROPS" if not logo_crops else "SIAMESE_UNAVAILABLE",
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "is_brand_impersonation": False,
                "is_authorized_domain": False,
                "num_logos_found": len(logo_crops),
            }
        else:
            best_match: dict[str, Any] | None = None
            best_sim = -1.0

            for crop_info in logo_crops:
                auth_res = self.siamese_matcher.verify_brand_authorization(
                    query_image=crop_info["crop_image"],
                    target_registrable_domain=target_domain,
                )
                sim = auth_res.get("max_similarity", 0.0)
                if sim > best_sim:
                    best_sim = sim
                    best_match = auth_res

            res["brand_verification"] = best_match or {
                "status": "NO_MATCH",
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "is_brand_impersonation": False,
                "is_authorized_domain": False,
            }

        return res

    # -------------------------------------------------------------------------
    # Helper: Empty Visual Evidence
    # -------------------------------------------------------------------------
    def _build_empty_visual_evidence(self) -> dict[str, Any]:
        """Construct empty visual evidence when screenshot is missing or corrupted."""
        return {
            "screenshot_available": False,
            "m02_status": "SKIPPED_NO_SCREENSHOT",
            "m02_phishing_probability": None,
            "ui_elements": {
                "total_fused_elements": 0,
                "elements": [],
                "class_counts": {cname: 0 for cname in UI_CLASSES.values()},
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_password_field": False,
                "has_username_field": False,
                "has_submit_button": False,
                "has_payment_element": False,
                "raw_detection_counts": {"M-04_yolov8n": 0, "M-05_faster_rcnn": 0, "M-06_yolov8_seg": 0},
            },
            "brand_verification": {
                "status": "NO_SCREENSHOT",
                "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                "max_similarity": 0.0,
                "is_brand_impersonation": False,
                "is_authorized_domain": False,
                "num_logos_found": 0,
            },
        }

    # -------------------------------------------------------------------------
    # Helper: DOM Evidence Analysis
    # -------------------------------------------------------------------------
    def _analyze_dom_forms(
        self,
        dom_forms: list[dict[str, Any]] | None,
        target_domain: str,
    ) -> dict[str, Any]:
        """Extract password fields and external form action telemetry from scanner DOM."""
        if not dom_forms:
            return {
                "dom_telemetry_available": False,
                "has_dom_password_field": False,
                "has_external_submission": False,
                "form_count": 0,
            }

        has_pwd = False
        has_ext = False
        for f in dom_forms:
            if f.get("has_password_field", False):
                has_pwd = True
            if f.get("is_external_submission", False):
                has_ext = True

        return {
            "dom_telemetry_available": True,
            "has_dom_password_field": has_pwd,
            "has_external_submission": has_ext,
            "form_count": len(dom_forms),
        }

    # -------------------------------------------------------------------------
    # Helper: Explainable Seven-Model Risk Fusion
    # -------------------------------------------------------------------------
    def _fuse_seven_models(
        self,
        url_evidence: dict[str, Any],
        visual_evidence: dict[str, Any],
        dom_evidence: dict[str, Any],
        target_domain: str,
        tls_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesizes risk score, category, and preserves UNKNOWN / INSUFFICIENT EVIDENCE.

        Preserves:
        1. Official verified brand whitelist pass.
        2. Critical brand impersonation override.
        3. Localized credential/payment harvesting surface on anomalous hosts.
        4. Capping 404 / empty error pages below phishing threshold.
        5. Clean visual and neutral domain protection.
        6. Missing visual telemetry fallback to UNKNOWN or decisive URL score.

        TLS evidence is treated as a contextual signal:
        - Valid TLS: mitigating factor (reduces risk slightly for borderline cases).
        - Expired cert: MEDIUM risk contribution (+0.05 to risk in weighted fallback).
        - Invalid cert (handshake failure): HIGH risk contribution (+0.10 to risk).
        - TLS unavailable / HTTP: no contribution to fusion (reported as UNAVAILABLE).
        Note: TLS alone NEVER proves legitimacy; it cannot override impersonation rules.
        """
        m01_res = url_evidence.get("m01_lexical", {})
        m07_res = url_evidence.get("m07_lstm", {})
        p_lex = m01_res.get("probability")
        p_lstm = m07_res.get("probability")
        p_res = visual_evidence.get("m02_phishing_probability")
        composite_url_risk = url_evidence.get("composite_url_risk")

        brand_verif = visual_evidence.get("brand_verification", {})
        is_impersonation = brand_verif.get("is_brand_impersonation", False)
        matched_brand = brand_verif.get("predicted_brand", "UNIDENTIFIED_OR_NEUTRAL")
        max_sim = brand_verif.get("max_similarity", 0.0)
        is_auth_domain = brand_verif.get("is_authorized_domain", False)

        screenshot_available = visual_evidence.get("screenshot_available", False)
        ui_elements = visual_evidence.get("ui_elements", {})
        total_ui_elements = ui_elements.get("total_fused_elements", 0)

        has_login = ui_elements.get("has_login_form", False)
        has_inputs = ui_elements.get("has_credential_inputs", False) or dom_evidence.get("has_dom_password_field", False)
        has_submit = ui_elements.get("has_submit_button", False)
        has_payment = ui_elements.get("has_payment_element", False)

        # ---------------------------------------------------------------------
        # Rule 1: Missing Visual Telemetry (No Screenshot / Load Error)
        # ---------------------------------------------------------------------
        if not screenshot_available:
            if composite_url_risk is None:
                # Fallback to single model if composite is None
                url_prob = p_lex if p_lex is not None else p_lstm
                if url_prob is None:
                    return {
                        "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                        "risk_score": 0.50,
                        "explanation": "No visual telemetry available and URL models were unserviceable.",
                    }
                if 0.35 <= url_prob <= 0.65:
                    return {
                        "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                        "risk_score": 0.50,
                        "explanation": (
                            f"Visual telemetry is unavailable, and URL risk score ({url_prob:.2f}) "
                            "resides in the ambiguous confidence band. Defaulting safely to INSUFFICIENT EVIDENCE."
                        ),
                    }
                if url_prob > 0.65:
                    return {
                        "threat_category": "PHISHING",
                        "risk_score": round(url_prob, 4),
                        "explanation": f"High URL phishing confidence ({url_prob:.2f}) despite missing visual capture.",
                    }
                return {
                    "threat_category": "LEGITIMATE",
                    "risk_score": round(url_prob, 4),
                    "explanation": f"Low URL phishing risk ({url_prob:.2f}); visual capture was unavailable.",
                }

            # We have composite URL risk
            if 0.35 <= composite_url_risk <= 0.65:
                return {
                    "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                    "risk_score": 0.50,
                    "explanation": (
                        f"Visual capture unavailable, and combined URL risk ({composite_url_risk:.2f}) "
                        "is ambiguous. Preserving UNKNOWN / INSUFFICIENT EVIDENCE state."
                    ),
                }
            if composite_url_risk > 0.65:
                return {
                    "threat_category": "PHISHING",
                    "risk_score": composite_url_risk,
                    "explanation": f"Decisive URL risk ({composite_url_risk:.2f}) across M-01/M-07 without visual capture.",
                }
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": composite_url_risk,
                "explanation": f"Clean URL indicators ({composite_url_risk:.2f}); visual capture was unavailable.",
            }

        # ---------------------------------------------------------------------
        # Rule 2: Official Verified Brand Whitelist Pass
        # ---------------------------------------------------------------------
        if is_auth_domain:
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": 0.05,
                "explanation": (
                    f"Verified Official Brand: Visual logo matches '{matched_brand}' and domain '{target_domain}' "
                    "is authorized in the official brand registry."
                ),
            }

        # ---------------------------------------------------------------------
        # Rule 3: Critical Brand Impersonation Override
        # ---------------------------------------------------------------------
        if is_impersonation and matched_brand != "UNIDENTIFIED_OR_NEUTRAL":
            base_risk = composite_url_risk if composite_url_risk is not None else (p_lex or 0.0)
            return {
                "threat_category": "CRITICAL_PHISHING",
                "risk_score": max(0.95, round(base_risk, 4)),
                "explanation": (
                    f"Brand Impersonation Confirmed: Detected logo matching '{matched_brand}' "
                    f"(Cosine Similarity={max_sim:.4f}), but host domain '{target_domain}' is NOT "
                    f"in authorized domains list."
                ),
            }

        # Check whether an active credential or payment harvesting surface is present
        has_active_harvesting = (
            has_inputs
            or has_payment
            or is_impersonation
        )

        # ---------------------------------------------------------------------
        # Rule 4: Non-Harvesting / Empty / Error / Clean Pages
        # When page contains NO active credential fields, NO password field,
        # NO payment elements, and NO brand impersonation:
        # ---------------------------------------------------------------------
        if not has_active_harvesting:
            # If ResNet-18 confirms legitimate visual layout (e.g. t.me preview, blogs)
            if p_res is not None and p_res <= 0.40:
                url_term = composite_url_risk if composite_url_risk is not None else (p_lex if p_lex is not None else 0.50)
                risk = round(min(0.30, 0.30 * url_term + 0.70 * p_res), 4)
                return {
                    "threat_category": "LEGITIMATE",
                    "risk_score": risk,
                    "explanation": (
                        f"Clean visual page on '{target_domain}' with no credential harvesting elements, "
                        "no brand impersonation, and legitimate visual layout profile."
                    ),
                }

            # If page is empty / error / minimal, or has non-harvesting elements:
            # Even if URL or layout triggered heuristic suspicion, lack of credential theft or brand spoofing
            # caps the threat at SUSPICIOUS (or LEGITIMATE if URL is clean).
            is_suspicious_url = (
                (composite_url_risk is not None and composite_url_risk >= 0.50)
                or (p_lex is not None and p_lex >= self.m01_threshold)
                or (p_lstm is not None and p_lstm >= self.m07_threshold)
                or (p_res is not None and p_res >= 0.50)
            )

            if is_suspicious_url:
                return {
                    "threat_category": "SUSPICIOUS",
                    "risk_score": 0.40,
                    "explanation": (
                        f"Suspicious URL or visual layout indicators detected on '{target_domain}', but page contains "
                        "no active credential fields, login form, payment elements, or brand logos (empty or error page). "
                        "Absence of harvesting surface caps risk at SUSPICIOUS."
                    ),
                }
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": min(0.20, composite_url_risk if composite_url_risk is not None else 0.10),
                "explanation": (
                    f"Non-harvesting page on '{target_domain}' with no credential fields or brand logos, "
                    "and clean URL indicators."
                ),
            }

        # ---------------------------------------------------------------------
        # Rule 5: Active Credential / Payment Harvesting Surface Present
        # ---------------------------------------------------------------------
        if has_inputs or has_payment:
            is_suspicious_host = (
                url_evidence.get("is_url_suspicious", False)
                or (p_lex is not None and p_lex >= self.m01_threshold)
                or (p_lstm is not None and p_lstm >= self.m07_threshold)
                or (p_res is not None and p_res >= 0.50)
                or dom_evidence.get("has_external_submission", False)
            )

            if is_suspicious_host:
                surface_type = "payment inputs" if has_payment else "login credential fields"
                url_term = composite_url_risk if composite_url_risk is not None else 0.50
                res_term = p_res if p_res is not None else 0.50
                risk = max(0.85, round(0.40 * url_term + 0.35 * res_term + 0.25 * 0.90, 4))
                return {
                    "threat_category": "PHISHING",
                    "risk_score": risk,
                    "explanation": (
                        f"Active {surface_type} detected on suspicious domain '{target_domain}'. "
                        "Threat confirmed from localized UI elements, spatial consensus, and host telemetry."
                    ),
                }
            else:
                # Login/form interface on neutral/clean domain
                res_term = p_res if p_res is not None else 0.20
                risk = round(0.35 + 0.30 * res_term, 4)
                return {
                    "threat_category": "SUSPICIOUS" if risk >= 0.50 else "LEGITIMATE",
                    "risk_score": risk,
                    "explanation": (
                        f"Standard login/form interface detected on neutral domain '{target_domain}'. "
                        "No brand impersonation or malicious URL pattern detected."
                    ),
                }

        # ---------------------------------------------------------------------
        # Rule 6: Multi-Evidence Fallback
        # ---------------------------------------------------------------------
        weights = []
        values = []

        if composite_url_risk is not None:
            weights.append(0.50)
            values.append(composite_url_risk)
        elif p_lex is not None:
            weights.append(0.50)
            values.append(p_lex)

        if p_res is not None:
            weights.append(0.50)
            values.append(p_res)

        if not weights:
            return {
                "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                "risk_score": 0.50,
                "explanation": "No operational models available to score request.",
            }

        combined_risk = round(sum(w * v for w, v in zip(weights, values)) / sum(weights), 4)

        # Apply TLS context adjustment in the weighted fallback path only.
        # TLS adjustment is bounded and cannot trigger an impersonation override.
        tls_adjustment = 0.0
        tls_note = ""
        if tls_evidence:
            if tls_evidence.get("is_expired"):
                tls_adjustment = +0.05
                tls_note = " Expired TLS certificate raises risk marginally (+0.05)."
            elif tls_evidence.get("tls_available") and not tls_evidence.get("certificate_valid"):
                tls_adjustment = +0.10
                tls_note = " Invalid TLS certificate (handshake error) raises risk (+0.10)."
            elif tls_evidence.get("certificate_valid"):
                tls_adjustment = -0.03
                tls_note = " Valid TLS certificate from trusted CA marginally mitigates risk (-0.03)."
        combined_risk = round(max(0.0, min(1.0, combined_risk + tls_adjustment)), 4)

        if combined_risk >= 0.65:
            cat = "PHISHING"
            exp = f"Elevated phishing risk ({combined_risk:.2f}) from combined URL and visual evidence.{tls_note}"
        elif combined_risk >= 0.35:
            cat = "SUSPICIOUS"
            exp = f"Moderate suspicion ({combined_risk:.2f}). Anomalous domain or layout indicators.{tls_note}"
        else:
            cat = "LEGITIMATE"
            exp = f"Low risk profile ({combined_risk:.2f}). No credential harvesting or brand misuse detected.{tls_note}"

        return {
            "threat_category": cat,
            "risk_score": combined_risk,
            "explanation": exp,
        }
