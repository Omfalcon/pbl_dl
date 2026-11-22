"""Test and Evaluation Script for Full Multimodal Phishing Pipeline (Phase 7.5).

Runs end-to-end integration across M-01 (Lexical), M-02 (ResNet-18), M-03 (Siamese),
and M-04 (YOLOv8n) on controlled test cases and exports evaluation reports.
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

from app.ml.vision.multimodal_pipeline import MultimodalPhishingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.run_pipeline")


def main() -> None:
    logger.info("Initializing Multimodal Phishing Pipeline...")
    pipeline = MultimodalPhishingPipeline()

    logger.info(f"Pipeline model availability: {pipeline.m01_available=}, {pipeline.m02_available=}, "
                f"{pipeline.m03_available=}, {pipeline.m04_available=}")

    preview_dir = Path("data/reports/pipeline_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather controlled test cases
    test_cases = []

    # Case 1: Synthetic Login with Brand Logo & Credentials
    synth_test_manifest = Path("data/synthetic_yolo/manifest.csv")
    if synth_test_manifest.exists():
        df_synth = pd.read_csv(synth_test_manifest)
        # Find a login page from test split
        login_row = df_synth[(df_synth["split"] == "test") & (df_synth["template"] == "standard_login")].iloc[0]
        test_cases.append({
            "name": "Synthetic Standard Login (TitanVault)",
            "sample_id": f"{login_row['sample_id']}",
            "url": "https://secure-login.titanvault-portal.net/auth/signin?token=9281a",
            "screenshot": f"data/synthetic_yolo/images/test/{login_row['sample_id']}.png",
            "domain": "titanvault-portal.net",
            "expected_type": "Synthetic UI Form",
        })

        # Case 2: Synthetic Payment Checkout
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

        # Case 3: Synthetic Empty Page (Negative Control)
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

    # Case 4: Real-world Phishing Screenshot from Validation Split
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

    # Case 5: Missing Screenshot / Inconclusive Fallback
    test_cases.append({
        "name": "Missing Screenshot / Ambiguous Lexical (Insufficient Evidence Fallback)",
        "sample_id": "missing_screenshot_fallback",
        "url": "https://medium.com/@user/story",
        "screenshot": None,  # No screenshot available
        "domain": "medium.com",
        "expected_type": "Fallback Test",
    })

    # Run evaluations
    evaluation_results = []
    logger.info(f"Executing multimodal pipeline across {len(test_cases)} controlled test cases...")

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
            "lexical_score": res["evidence"]["lexical"].get("probability"),
            "resnet_score": res["evidence"]["visual_global"].get("m02_probability"),
            "ui_elements_detected": res["evidence"]["ui_elements"].get("total_detections"),
            "ui_class_counts": res["evidence"]["ui_elements"].get("class_counts"),
            "explanation": res["explanation"],
            "preview_saved": res["evidence"]["ui_elements"].get("preview_image_path"),
        }
        evaluation_results.append(tc_summary)

        logger.info(f"Result: Decision={tc_summary['threat_decision']}, Risk={tc_summary['risk_score']:.2f}, "
                    f"Brand={tc_summary['matched_brand']} (S={tc_summary['brand_similarity']:.2f}), "
                    f"UI Boxes={tc_summary['ui_elements_detected']}")

    # Save summary report
    out_json = Path("data/reports/pipeline_evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Saved full multimodal pipeline evaluation to: {out_json}")


if __name__ == "__main__":
    main()
