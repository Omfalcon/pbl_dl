"""End-to-End Evaluation Script for Phase 8 Seven-Model Phishing Pipeline.

Executes unified inference across:
- M-01: URL Lexical Multi-Layer Perceptron (fast-path)
- M-02: ResNet-18 Global Webpage Screenshot Classifier
- M-03: Siamese CNN Brand Impersonation Matcher
- M-04: YOLOv8n UI Component & Logo Detector
- M-05: Faster R-CNN Object Detector (ResNet-50-FPN)
- M-06: YOLOv8-Seg UI Component Instance Segmenter
- M-07: URL Sequence Bi-LSTM

Evaluates spatial consensus deduplication, multi-detector agreement, calibrated risk
normalization, and fallback state preservation on controlled benchmark test cases.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from app.ml.unified_pipeline import UnifiedPhishingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.run_seven_models")


def main() -> None:
    logger.info("Initializing Unified Seven-Model Phishing Pipeline...")
    pipeline = UnifiedPhishingPipeline()

    avail = pipeline.analyze(url="https://probe-init.com", render_preview=False)["model_availability"]
    logger.info(f"Seven-Model Availability Status: {avail}")

    preview_dir = Path("data/reports/seven_model_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    test_cases = []

    # 1. Synthetic Login with Brand Logo & Credentials (TitanVault)
    synth_test_manifest = Path("data/synthetic_yolo/manifest.csv")
    if synth_test_manifest.exists():
        df_synth = pd.read_csv(synth_test_manifest)
        login_rows = df_synth[(df_synth["split"] == "test") & (df_synth["template"] == "standard_login")]
        if len(login_rows) > 0:
            login_row = login_rows.iloc[0]
            test_cases.append({
                "name": "Synthetic Standard Login (TitanVault)",
                "sample_id": f"{login_row['sample_id']}",
                "url": "https://secure-login.titanvault-portal.net/auth/signin?token=9281a",
                "screenshot": f"data/synthetic_yolo/images/test/{login_row['sample_id']}.png",
                "domain": "titanvault-portal.net",
                "expected_type": "Synthetic UI Form",
            })

        # 2. Synthetic Payment Checkout (OmniPay)
        pay_rows = df_synth[(df_synth["split"] == "test") & (df_synth["template"] == "payment_checkout")]
        if len(pay_rows) > 0:
            pay_row = pay_rows.iloc[0]
            test_cases.append({
                "name": "Synthetic Payment Checkout (OmniPay)",
                "sample_id": f"{pay_row['sample_id']}",
                "url": "https://billing-checkout.omnipays-gateway.com/pay/invoice_4910",
                "screenshot": f"data/synthetic_yolo/images/test/{pay_row['sample_id']}.png",
                "domain": "omnipays-gateway.com",
                "expected_type": "Synthetic Payment Form",
            })

        # 3. Synthetic Empty Page (Negative Control)
        empty_rows = df_synth[(df_synth["split"] == "test") & (df_synth["is_empty"] == True)]
        if len(empty_rows) > 0:
            empty_row = empty_rows.iloc[0]
            test_cases.append({
                "name": "Synthetic Empty / 404 Page (Negative Control)",
                "sample_id": f"{empty_row['sample_id']}",
                "url": "https://help-docs.cloudsupport-knowledge.org/articles/error-404",
                "screenshot": f"data/synthetic_yolo/images/test/{empty_row['sample_id']}.png",
                "domain": "cloudsupport-knowledge.org",
                "expected_type": "Empty Negative Sample",
            })

    # 4. Real-world Phishing Screenshot from Validation Split
    val_parquet = Path("data/splits/visual/val.parquet")
    if val_parquet.exists():
        df_val = pd.read_parquet(val_parquet)
        phish_samples = df_val[df_val["label"] == 1]
        legit_samples = df_val[df_val["label"] == 0]

        if len(phish_samples) > 0:
            p_row = phish_samples.iloc[0]
            test_cases.append({
                "name": f"Real-World Phishing Screenshot ({p_row['sample_id']})",
                "sample_id": f"{p_row['sample_id']}",
                "url": str(p_row["original_url"]),
                "screenshot": str(p_row["screenshot_relative_path"]),
                "domain": str(p_row["registrable_domain"]),
                "expected_type": "Real-World Phishing",
            })

        if len(legit_samples) > 0:
            l_row = legit_samples.iloc[0]
            test_cases.append({
                "name": f"Real-World Legitimate Screenshot ({l_row['sample_id']})",
                "sample_id": f"{l_row['sample_id']}",
                "url": str(l_row["original_url"]),
                "screenshot": str(l_row["screenshot_relative_path"]),
                "domain": str(l_row["registrable_domain"]),
                "expected_type": "Real-World Legitimate",
            })

    # 5. Missing Screenshot / Ambiguous Inconclusive Fallback
    test_cases.append({
        "name": "Missing Screenshot / Ambiguous Lexical (Insufficient Evidence Fallback)",
        "sample_id": "missing_screenshot_fallback",
        "url": "https://medium.com/@user/story",
        "screenshot": None,
        "domain": "medium.com",
        "expected_type": "Fallback Test",
    })

    # Run evaluations
    evaluation_results = []
    logger.info(f"Executing seven-model pipeline across {len(test_cases)} benchmark test cases...")

    for idx, tc in enumerate(test_cases, start=1):
        logger.info(f"--- Running Test Case {idx}: {tc['name']} ---")
        res = pipeline.analyze(
            url=tc["url"],
            screenshot=tc["screenshot"],
            domain=tc["domain"],
            sample_id=tc["sample_id"],
            render_preview=True,
            preview_output_dir=preview_dir,
        )

        ui_elem_ev = res["evidence"]["ui_elements"]
        url_models = res["evidence"]["url_models"]

        tc_summary = {
            "test_case_name": tc["name"],
            "expected_type": tc["expected_type"],
            "input_url": tc["url"],
            "target_domain": res["target_registrable_domain"],
            "threat_decision": res["decision"],
            "risk_score": res["risk_score"],
            "is_brand_impersonation": res["is_brand_impersonation"],
            "matched_brand": res["matched_brand"],
            "brand_similarity": res["brand_similarity"],
            "url_models": {
                "m01_lexical_prob": url_models.get("m01_lexical", {}).get("probability"),
                "m01_normalized_risk": url_models.get("m01_lexical", {}).get("normalized_risk"),
                "m07_lstm_prob": url_models.get("m07_lstm", {}).get("probability"),
                "m07_normalized_risk": url_models.get("m07_lstm", {}).get("normalized_risk"),
                "composite_url_risk": url_models.get("composite_url_risk"),
            },
            "visual_models": {
                "m02_resnet_prob": res["evidence"]["visual_global"].get("m02_probability"),
                "total_fused_elements": ui_elem_ev.get("total_fused_elements"),
                "fused_class_counts": ui_elem_ev.get("class_counts"),
                "raw_detector_counts": ui_elem_ev.get("raw_detection_counts"),
            },
            "explanation": res["explanation"],
            "preview_saved": ui_elem_ev.get("preview_image_path"),
        }
        evaluation_results.append(tc_summary)

        logger.info(f"Result: Decision={tc_summary['threat_decision']}, Risk={tc_summary['risk_score']:.2f}, "
                    f"Brand={tc_summary['matched_brand']} (S={tc_summary['brand_similarity']:.2f}), "
                    f"Fused UI Elements={tc_summary['visual_models']['total_fused_elements']}")

    out_json = Path("data/reports/seven_model_pipeline_evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Saved seven-model pipeline evaluation to: {out_json}")


if __name__ == "__main__":
    main()
