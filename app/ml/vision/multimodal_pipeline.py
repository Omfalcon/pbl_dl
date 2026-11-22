"""Multimodal Phishing Analysis Pipeline (Phase 8 Seven-Model Integration).

Integrates all 7 PhishVision AI model pipelines:
- M-01: URL Lexical Multi-Layer Perceptron (fast-path, threshold=0.05)
- M-02: ResNet-18 Global Webpage Screenshot Classifier
- M-03: Siamese CNN Brand Impersonation Matcher (threshold=0.82)
- M-04: YOLOv8n UI Component & Logo Detector
- M-05: Faster R-CNN Object Detector (ResNet-50-FPN)
- M-06: YOLOv8-Seg UI Component Instance Segmenter
- M-07: URL Sequence Bi-LSTM (threshold=0.01)

Features:
- Non-duplicative spatial consensus engine combining M-04, M-05, and M-06.
- Deduplicated logo crop extraction with ~10% padding for Siamese brand matching.
- URL sequence analysis via M-07 with explicit disclaimer that redirect sequences
  were not empirically trained on historical redirect-chain data.
- Explainable multi-evidence threat scoring preserving UNKNOWN / INSUFFICIENT EVIDENCE.
- Fully backward-compatible with Phase 7.5 API contracts.
"""

from __future__ import annotations

import logging
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
from app.ml.vision.spatial_consensus import SpatialConsensusEngine
from app.ml.vision.yolo_inference import YOLOUIElementDetector
from app.ml.vision.yolo_seg import YOLOSegInferencer

logger = logging.getLogger("phishvision.ml.vision.multimodal_pipeline")


def normalize_calibrated_risk(probability: float, threshold: float) -> float:
    """Map a raw model probability to a normalized risk scale in [0.0, 1.0]."""
    if probability is None or np.isnan(probability):
        return 0.50

    p = float(np.clip(probability, 0.0, 1.0))
    t = float(np.clip(threshold, 0.001, 0.999))

    if p < t:
        return round(0.50 * (p / t), 4)
    else:
        return round(0.50 + 0.50 * ((p - t) / (1.0 - t)), 4)


class MultimodalPhishingPipeline:
    """Production and diagnostic orchestrator combining URL lexical, global screenshot,
    spatial UI detection consensus (YOLOv8, Faster R-CNN, YOLO-Seg), Siamese brand verification,
    and sequential LSTM analysis.
    """

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
        enable_faster_rcnn: bool = False,
        enable_yolo_seg: bool = False,
        enable_lstm: bool = False,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.enable_faster_rcnn = enable_faster_rcnn
        self.enable_yolo_seg = enable_yolo_seg
        self.enable_lstm = enable_lstm

        # 1. Initialize M-01 Lexical Engine
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

        # 2. Initialize M-02 ResNet-18 Classifier
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

        # 3. Initialize M-03 Siamese Brand Matcher
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

        # 4. Initialize M-04 YOLOv8n Detector
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

        # 5. Initialize M-05 Faster R-CNN Detector
        self.m05_available = False
        self.faster_rcnn_infer = None
        m05_p = Path(m05_checkpoint_path)
        if self.enable_faster_rcnn and m05_p.exists():
            try:
                self.faster_rcnn_infer = FasterRCNNInferencer(checkpoint_path=m05_p, device=self.device)
                self.m05_available = True
                logger.info("Initialized M-05 Faster R-CNN Object Detector.")
            except Exception as e:
                logger.warning(f"Failed to load M-05 Faster R-CNN detector: {e}")

        # 6. Initialize M-06 YOLOv8-Seg Instance Segmenter
        self.m06_available = False
        self.yolo_seg_infer = None
        m06_p = Path(m06_checkpoint_path)
        if self.enable_yolo_seg and m06_p.exists():
            try:
                self.yolo_seg_infer = YOLOSegInferencer(model_weights=m06_p, device=self.device)
                self.m06_available = True
                logger.info("Initialized M-06 YOLOv8-Seg Instance Segmenter.")
            except Exception as e:
                logger.warning(f"Failed to load M-06 YOLOv8-Seg: {e}")

        # 7. Initialize M-07 URL Sequence Bi-LSTM
        self.m07_available = False
        self.lstm_infer = None
        self.m07_threshold = 0.010
        m07_p = Path(m07_checkpoint_path)
        m07_prep_p = Path(m07_preprocessor_path)
        if self.enable_lstm and m07_p.exists() and m07_prep_p.exists():
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
        """Runs the full multimodal analysis pipeline across all available models."""
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

        # Step 1: Lexical URL Analysis (M-01) and Sequence Analysis (M-07)
        lexical_result = self._analyze_lexical(url)
        lstm_result = self._analyze_lstm(url, redirect_chain=redirect_chain) if self.enable_lstm else None

        # Step 2 & 3: Visual & Spatial Analysis
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
            visual_result = {
                "screenshot_available": False,
                "m02_status": "SKIPPED_NO_SCREENSHOT",
                "m04_status": "SKIPPED_NO_SCREENSHOT",
                "m05_status": "SKIPPED_NO_SCREENSHOT",
                "m06_status": "SKIPPED_NO_SCREENSHOT",
                "m03_status": "SKIPPED_NO_SCREENSHOT",
                "m02_phishing_probability": None,
                "yolo_detections": [],
                "yolo_class_counts": {},
                "has_login_form": False,
                "has_credential_inputs": False,
                "has_submit_button": False,
                "has_payment_element": False,
                "brand_verification": {
                    "is_brand_impersonation": False,
                    "is_authorized_domain": False,
                    "predicted_brand": "UNIDENTIFIED_OR_NEUTRAL",
                    "max_similarity": 0.0,
                    "status": "NO_SCREENSHOT",
                },
            }
        else:
            visual_result = self._analyze_visual(
                screenshot=screenshot_loaded,
                target_domain=target_domain,
                render_preview=render_preview,
                preview_output_dir=preview_output_dir,
                sample_id=sid,
            )

        # Step 4: Decision Synthesis & Risk Fusion
        fusion_result = self._fuse_evidence(
            lexical_result=lexical_result,
            visual_result=visual_result,
            target_domain=target_domain,
            lstm_result=lstm_result,
            dom_forms=dom_forms,
        )

        brand_verif = visual_result.get("brand_verification", {})
        matched_brand = brand_verif.get("predicted_brand") or brand_verif.get("matched_brand", "UNIDENTIFIED_OR_NEUTRAL")
        brand_sim = brand_verif.get("max_similarity") if brand_verif.get("max_similarity") is not None else brand_verif.get("similarity_score", 0.0)

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
                "lexical": lexical_result,
                "lstm_sequence": lstm_result,
                "visual_global": {
                    "m02_probability": visual_result.get("m02_phishing_probability"),
                    "m02_prediction": visual_result.get("m02_prediction"),
                },
                "ui_elements": {
                    "total_detections": len(visual_result.get("yolo_detections", [])),
                    "class_counts": visual_result.get("yolo_class_counts", {}),
                    "has_login_form": visual_result.get("has_login_form", False),
                    "has_credential_inputs": visual_result.get("has_credential_inputs", False),
                    "has_submit_button": visual_result.get("has_submit_button", False),
                    "has_payment_element": visual_result.get("has_payment_element", False),
                    "preview_image_path": visual_result.get("preview_image_path"),
                    "spatial_consensus": visual_result.get("spatial_consensus"),
                },
                "brand_verification": visual_result.get("brand_verification"),
            },
            "explanation": fusion_result["explanation"],
            "disclaimer": (
                "Notice: M-04, M-05, and M-06 were trained on synthetic UI elements (proof-of-concept). "
                "Synthetic mAP does not establish real-world detection accuracy. "
                "M-07 redirect branch operates in fallback mode as labelled multi-hop data was not in historical corpus."
            ),
        }

    def _analyze_lexical(self, url: str) -> dict[str, Any]:
        """Extract lexical features and compute M-01 phishing probability."""
        if not self.m01_available or self.lexical_mlp is None or self.lexical_preprocessor is None:
            return {
                "status": "UNAVAILABLE",
                "probability": None,
                "prediction": None,
            }

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

            return {
                "status": "SUCCESS",
                "probability": round(proba, 4),
                "normalized_risk": norm_risk,
                "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
                "canonical_url_evaluated": eval_url,
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
            return {
                "status": "ERROR",
                "probability": None,
                "error": str(e),
            }

    def _analyze_lstm(self, url: str, redirect_chain: list[str] | None = None) -> dict[str, Any]:
        """Perform sequential phishing prediction with M-07 URL Sequence Bi-LSTM."""
        if not self.m07_available or self.lstm_infer is None:
            return {"status": "UNAVAILABLE", "probability": None}

        try:
            pred_lstm = self.lstm_infer.predict(
                url=url,
                redirect_chain=redirect_chain,
                threshold=self.m07_threshold,
            )
            prob = float(pred_lstm.get("phishing_probability", 0.0))
            norm_risk = normalize_calibrated_risk(prob, self.m07_threshold)

            return {
                "status": "SUCCESS",
                "probability": round(prob, 4),
                "normalized_risk": norm_risk,
                "prediction": pred_lstm.get("predicted_label", "UNKNOWN"),
                "calibrated_threshold": self.m07_threshold,
                "is_above_calibrated_threshold": bool(prob >= self.m07_threshold),
                "redirect_chain_provided": bool(redirect_chain and len(redirect_chain) > 0),
                "redirect_training_status": "NOT_EMPIRICALLY_TRAINED_ON_REDIRECTS",
            }
        except Exception as e:
            logger.warning(f"Error in M-07 LSTM evaluation: {e}")
            return {"status": "ERROR", "probability": None, "error": str(e)}

    def _analyze_visual(
        self,
        screenshot: Image.Image,
        target_domain: str,
        render_preview: bool = True,
        preview_output_dir: str | Path = "data/reports/pipeline_previews",
        sample_id: str = "sample",
    ) -> dict[str, Any]:
        """Executes ResNet-18, YOLOv8n, Faster R-CNN, YOLO-Seg, and Siamese brand matching on screenshot."""
        res: dict[str, Any] = {
            "screenshot_available": True,
        }

        # 1. M-02 ResNet-18
        if self.m02_available and self.resnet_infer is not None:
            try:
                m02_pred = self.resnet_infer.predict(screenshot)
                probs = m02_pred.get("probabilities", {})
                phish_prob = probs.get("phishing", 0.0)
                res["m02_phishing_probability"] = round(phish_prob, 4)
                res["m02_prediction"] = m02_pred.get("predicted_label", "phishing" if phish_prob >= 0.50 else "legitimate")
                res["m02_status"] = "SUCCESS"
            except Exception as e:
                logger.warning(f"M-02 prediction failed: {e}")
                res["m02_phishing_probability"] = None
                res["m02_status"] = f"ERROR: {e}"
        else:
            res["m02_phishing_probability"] = None
            res["m02_status"] = "UNAVAILABLE"

        # 2. Spatial Detectors (M-04 YOLOv8n, M-05 Faster R-CNN, M-06 YOLO-Seg)
        yolo_dets = []
        det_res = {}
        if self.m04_available and self.yolo_detector is not None:
            try:
                det_res = self.yolo_detector.detect(screenshot)
                yolo_dets = det_res.get("detections", [])
                res["m04_status"] = "SUCCESS"
            except Exception as e:
                logger.warning(f"M-04 YOLO detection failed: {e}")
                res["m04_status"] = f"ERROR: {e}"
        else:
            res["m04_status"] = "UNAVAILABLE"

        faster_rcnn_dets = []
        if self.enable_faster_rcnn and self.m05_available and self.faster_rcnn_infer is not None:
            try:
                fr_res = self.faster_rcnn_infer.predict(screenshot)
                faster_rcnn_dets = fr_res.get("detections", [])
                res["m05_status"] = "SUCCESS"
            except Exception as e:
                logger.warning(f"M-05 Faster R-CNN detection failed: {e}")
                res["m05_status"] = f"ERROR: {e}"
        else:
            res["m05_status"] = "UNAVAILABLE"

        yolo_seg_insts = []
        if self.enable_yolo_seg and self.m06_available and self.yolo_seg_infer is not None:
            try:
                seg_res = self.yolo_seg_infer.predict(screenshot)
                yolo_seg_insts = seg_res.get("instances", [])
                res["m06_status"] = "SUCCESS"
            except Exception as e:
                logger.warning(f"M-06 YOLO-Seg prediction failed: {e}")
                res["m06_status"] = f"ERROR: {e}"
        else:
            res["m06_status"] = "UNAVAILABLE"

        # 3. Spatial Resolution
        if self.enable_faster_rcnn or self.enable_yolo_seg:
            fused_spatial = self.spatial_engine.fuse_detections(
                yolo_detections=yolo_dets,
                faster_rcnn_detections=faster_rcnn_dets,
                yolo_seg_instances=yolo_seg_insts,
                image_size=screenshot.size,
            )
            res["spatial_consensus"] = fused_spatial
            res["yolo_detections"] = fused_spatial["elements"]
            res["yolo_class_counts"] = fused_spatial["class_counts"]
            res["has_login_form"] = fused_spatial["has_login_form"]
            res["has_credential_inputs"] = fused_spatial["has_credential_inputs"]
            res["has_submit_button"] = fused_spatial["has_submit_button"]
            res["has_payment_element"] = fused_spatial["has_payment_element"]
            display_detections = fused_spatial["elements"]

            logo_crops = self.spatial_engine.extract_fused_logo_crops(
                image=screenshot,
                fused_elements=fused_spatial["elements"],
                padding_pct=0.10,
            )
        else:
            # Phase 7.5 backwards compatible direct YOLO mapping
            res["yolo_detections"] = yolo_dets
            res["yolo_class_counts"] = det_res.get("class_counts", {})
            res["has_login_form"] = det_res.get("has_login_form", False)
            res["has_credential_inputs"] = det_res.get("has_credential_inputs", False)
            res["has_submit_button"] = det_res.get("has_submit_button", False)
            res["has_payment_element"] = det_res.get("has_payment_element", False)
            display_detections = yolo_dets

            logo_crops = (
                self.yolo_detector.extract_logo_crops(screenshot, padding_pct=0.10)
                if self.m04_available and self.yolo_detector
                else []
            )

        # Preview rendering
        if render_preview and self.m04_available and self.yolo_detector is not None:
            try:
                preview_path = Path(preview_output_dir) / f"{sample_id}_annotated.png"
                self.yolo_detector.render_annotated_preview(
                    screenshot,
                    detections=display_detections,
                    output_path=preview_path,
                )
                res["preview_image_path"] = str(preview_path)
            except Exception as e:
                logger.warning(f"Failed to render preview: {e}")

        # 4. M-03 Siamese Brand Impersonation Matching

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

    def _fuse_evidence(
        self,
        lexical_result: dict[str, Any],
        visual_result: dict[str, Any],
        target_domain: str,
        lstm_result: dict[str, Any] | None = None,
        dom_forms: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Synthesizes risk score, category, and preserves UNKNOWN / INSUFFICIENT EVIDENCE."""
        p_lex = lexical_result.get("probability")
        p_res = visual_result.get("m02_phishing_probability")
        brand_verif = visual_result.get("brand_verification", {})
        is_impersonation = brand_verif.get("is_brand_impersonation", False)
        matched_brand = brand_verif.get("predicted_brand", "UNIDENTIFIED_OR_NEUTRAL")
        max_sim = brand_verif.get("max_similarity", 0.0)
        is_auth_domain = brand_verif.get("is_authorized_domain", False)

        screenshot_available = visual_result.get("screenshot_available", False)
        yolo_dets = visual_result.get("yolo_detections", [])
        total_ui_elements = len(yolo_dets)
        has_login = visual_result.get("has_login_form", False)
        has_inputs = visual_result.get("has_credential_inputs", False)
        has_submit = visual_result.get("has_submit_button", False)
        has_payment = visual_result.get("has_payment_element", False)

        # Check DOM form evidence
        has_dom_pwd = False
        if dom_forms:
            for f in dom_forms:
                if f.get("has_password_field", False):
                    has_dom_pwd = True
        has_inputs = has_inputs or has_dom_pwd

        # Check LSTM risk if available
        p_lstm = lstm_result.get("probability") if lstm_result else None
        norm_lex = lexical_result.get("normalized_risk", p_lex)
        norm_lstm = lstm_result.get("normalized_risk", p_lstm) if lstm_result else None

        valid_url_risks = [r for r in (norm_lex, norm_lstm) if r is not None]
        composite_url_risk = float(np.mean(valid_url_risks)) if valid_url_risks else None

        # 1. Fallback State Preservation: Missing Visual Telemetry
        if not screenshot_available:
            score_to_check = composite_url_risk if composite_url_risk is not None else p_lex
            if score_to_check is None:
                return {
                    "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                    "risk_score": 0.50,
                    "explanation": "No visual telemetry available and URL models were unserviceable.",
                }
            if 0.35 <= score_to_check <= 0.65:
                return {
                    "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                    "risk_score": 0.50,
                    "explanation": (
                        f"Visual telemetry is unavailable, and URL risk ({score_to_check:.2f}) "
                        "resides in the ambiguous confidence band. Defaulting safely to INSUFFICIENT EVIDENCE."
                    ),
                }
            if score_to_check > 0.65:
                return {
                    "threat_category": "PHISHING",
                    "risk_score": round(score_to_check, 4),
                    "explanation": f"High URL phishing confidence ({score_to_check:.2f}) despite missing visual capture.",
                }
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": round(score_to_check, 4),
                "explanation": f"Low URL phishing score ({score_to_check:.2f}); visual capture was unavailable.",
            }

        # 2. Official Verified Brand Whitelist Pass
        if is_auth_domain:
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": 0.05,
                "explanation": (
                    f"Verified Official Brand: Visual logo matches '{matched_brand}' and domain '{target_domain}' "
                    "is authorized in the official brand registry."
                ),
            }

        # 3. Critical Brand Impersonation Override
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

        # 4. Empty / Error / 404 Pages Handling (0 UI Elements & no DOM password fields)
        if total_ui_elements == 0 and not has_dom_pwd:
            is_susp_url = (composite_url_risk is not None and composite_url_risk >= 0.50) or (
                p_lex is not None and p_lex >= 0.50
            ) or (p_lstm is not None and p_lstm >= self.m07_threshold)

            if is_susp_url:
                return {
                    "threat_category": "SUSPICIOUS",
                    "risk_score": 0.40,
                    "explanation": (
                        "Anomalous URL pattern detected, but visual screenshot contains no credential fields, "
                        "forms, or brand logos (empty or 404 error page). Inactive visual state caps risk below active phishing threshold."
                    ),
                }
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": min(0.20, composite_url_risk if composite_url_risk is not None else (p_lex if p_lex is not None else 0.10)),
                "explanation": (
                    "Empty or minimal error page with no credential harvesting elements or brand logos, "
                    "and low URL risk."
                ),
            }

        # 5. Credential / Payment Harvesting Surface Present
        if has_inputs or has_payment or (has_login and has_submit):
            is_suspicious_host = (
                (p_lex is not None and p_lex >= self.m01_threshold)
                or (p_lstm is not None and p_lstm >= self.m07_threshold)
                or (composite_url_risk is not None and composite_url_risk >= 0.50)
                or (p_res is not None and p_res >= 0.50)
            )
            if is_suspicious_host:
                surface_type = "payment inputs" if has_payment else "login credential fields"
                url_term = composite_url_risk if composite_url_risk is not None else (p_lex if p_lex is not None else 0.50)
                res_term = p_res if p_res is not None else 0.50
                risk = max(0.85, round(0.40 * url_term + 0.35 * res_term + 0.25 * 0.90, 4))
                return {
                    "threat_category": "PHISHING",
                    "risk_score": risk,
                    "explanation": (
                        f"Active {surface_type} detected on suspicious domain '{target_domain}'. "
                        "High threat confirmed from localized UI elements, spatial consensus, and host telemetry."
                    ),
                }
            else:
                # Login/form interface on neutral domain
                res_term = p_res if p_res is not None else 0.20
                risk = round(0.35 + 0.30 * res_term, 4)
                return {
                    "threat_category": "SUSPICIOUS" if risk >= 0.50 else "LEGITIMATE",
                    "risk_score": risk,
                    "explanation": (
                        f"Standard login/form interface detected on neutral domain '{target_domain}'. "
                        "No brand impersonation or malicious lexical pattern detected."
                    ),
                }

        # 6. Neutral Visual Pages without Harvesting Elements (e.g. t.me, informative articles)
        # When page has elements (e.g. logo, text) but NO credential inputs and NO brand impersonation:
        if p_res is not None and p_res <= 0.40:
            lex_term = composite_url_risk if composite_url_risk is not None else (p_lex if p_lex is not None else 0.50)
            risk = round(min(0.30, 0.30 * lex_term + 0.70 * p_res), 4)
            return {
                "threat_category": "LEGITIMATE",
                "risk_score": risk,
                "explanation": (
                    f"Clean visual page on '{target_domain}' with no credential harvesting elements, "
                    "no brand impersonation, and legitimate visual layout profile."
                ),
            }

        # 7. Fallback Weighted Multi-Evidence Fusion
        weights = []
        values = []
        if composite_url_risk is not None:
            weights.append(0.45)
            values.append(composite_url_risk)
        elif p_lex is not None:
            weights.append(0.45)
            values.append(p_lex)

        if p_res is not None:
            weights.append(0.55)
            values.append(p_res)

        if not weights:
            return {
                "threat_category": "UNKNOWN / INSUFFICIENT EVIDENCE",
                "risk_score": 0.50,
                "explanation": "No operational models available to score request.",
            }

        combined_risk = round(sum(w * v for w, v in zip(weights, values)) / sum(weights), 4)
        if combined_risk >= 0.65:
            cat = "PHISHING"
            exp = f"Elevated phishing risk ({combined_risk:.2f}) from combined URL and visual evidence."
        elif combined_risk >= 0.35:
            cat = "SUSPICIOUS"
            exp = f"Moderate suspicion ({combined_risk:.2f}). Anomalous domain or layout indicators."
        else:
            cat = "LEGITIMATE"
            exp = f"Low risk profile ({combined_risk:.2f}). No credential harvesting or brand misuse detected."

        return {
            "threat_category": cat,
            "risk_score": combined_risk,
            "explanation": exp,
        }
