"""External Checkpoint Configuration and Download Manager for PhishVision AI.

Manages external model weights, checkpoints, and vocabulary preprocessors
for all seven models in the PhishVision AI architecture:
- M-01: URL Lexical Multi-Layer Perceptron (model_weights.npz, preprocessor.json)
- M-02: ResNet-18 Global Webpage Screenshot Classifier (best_model.pt)
- M-03: Siamese CNN Brand Impersonation Matcher (best_siamese.pt)
- M-04: YOLOv8n UI Component & Logo Detector (best.pt)
- M-05: Faster R-CNN Object Detector (best_faster_rcnn.pt)
- M-06: YOLOv8-Seg UI Component Instance Segmenter (best.pt)
- M-07: URL Sequence Bi-LSTM (best_lstm.pt, preprocessor.json)

Ensures that large model binaries are not tracked in git and provides structured
verification, status reporting, and automated retrieval from remote storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.download_checkpoints")

CHECKPOINT_REGISTRY: dict[str, dict[str, Any]] = {
    "M-01_weights": {
        "model_id": "M-01",
        "description": "Lexical MLP NumPy Weights (40 -> 64 -> 32 -> 1)",
        "env_var": "PHISHVISION_M01_WEIGHTS",
        "default_path": "models/lexical/model_weights.npz",
        "remote_filename": "m01_lexical_model_weights.npz",
        "expected_sha256": None,
        "required": True,
    },
    "M-01_preprocessor": {
        "model_id": "M-01",
        "description": "Lexical Feature Standard Scaler and Imputer Metadata",
        "env_var": "PHISHVISION_M01_PREPROCESSOR",
        "default_path": "models/lexical/preprocessor.json",
        "remote_filename": "m01_lexical_preprocessor.json",
        "expected_sha256": None,
        "required": True,
    },
    "M-02_checkpoint": {
        "model_id": "M-02",
        "description": "ResNet-18 Global Webpage Screenshot Classifier",
        "env_var": "PHISHVISION_M02_CHECKPOINT",
        "default_path": "models/checkpoints/resnet18/best_model.pt",
        "remote_filename": "m02_resnet18_best_model.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-03_checkpoint": {
        "model_id": "M-03",
        "description": "Siamese CNN Metric Learning Brand Impersonation Matcher",
        "env_var": "PHISHVISION_M03_CHECKPOINT",
        "default_path": "models/checkpoints/siamese/best_siamese.pt",
        "remote_filename": "m03_siamese_best_model.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-04_checkpoint": {
        "model_id": "M-04",
        "description": "YOLOv8n UI Component & Logo Object Detector",
        "env_var": "PHISHVISION_M04_CHECKPOINT",
        "default_path": "models/checkpoints/yolov8n_synthetic/run1/weights/best.pt",
        "remote_filename": "m04_yolov8n_synthetic_best.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-05_checkpoint": {
        "model_id": "M-05",
        "description": "Faster R-CNN with ResNet-50-FPN UI Component Detector",
        "env_var": "PHISHVISION_M05_CHECKPOINT",
        "default_path": "models/checkpoints/faster_rcnn_synthetic/best_faster_rcnn.pt",
        "remote_filename": "m05_faster_rcnn_synthetic_best.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-06_checkpoint": {
        "model_id": "M-06",
        "description": "YOLOv8-Seg UI Component Instance Segmenter",
        "env_var": "PHISHVISION_M06_CHECKPOINT",
        "default_path": "models/checkpoints/yolov8n_seg_synthetic/run1/weights/best.pt",
        "remote_filename": "m06_yolov8n_seg_synthetic_best.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-07_checkpoint": {
        "model_id": "M-07",
        "description": "URL Sequence Bidirectional LSTM with Character & Token Embeddings",
        "env_var": "PHISHVISION_M07_CHECKPOINT",
        "default_path": "models/checkpoints/lstm/best_lstm.pt",
        "remote_filename": "m07_lstm_best_model.pt",
        "expected_sha256": None,
        "required": True,
    },
    "M-07_preprocessor": {
        "model_id": "M-07",
        "description": "URL & Subdomain Character Vocabulary Preprocessor",
        "env_var": "PHISHVISION_M07_PREPROCESSOR",
        "default_path": "models/checkpoints/lstm/preprocessor.json",
        "remote_filename": "m07_lstm_preprocessor.json",
        "expected_sha256": None,
        "required": True,
    },
}

DEFAULT_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB default per artifact


def get_max_download_bytes() -> int:
    """Return maximum permitted artifact download size in bytes."""
    return int(os.getenv("PHISHVISION_MAX_DOWNLOAD_BYTES", str(DEFAULT_MAX_DOWNLOAD_BYTES)))


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoints() -> dict[str, Any]:
    """Inspect all configured checkpoints and return deployment status."""
    report: dict[str, Any] = {
        "total_artifacts": len(CHECKPOINT_REGISTRY),
        "present_count": 0,
        "missing_count": 0,
        "artifacts": {},
    }

    for key, spec in CHECKPOINT_REGISTRY.items():
        configured_path_str = os.getenv(spec["env_var"], spec["default_path"])
        p = Path(configured_path_str)
        exists = p.exists() and p.is_file()

        info: dict[str, Any] = {
            "model_id": spec["model_id"],
            "description": spec["description"],
            "configured_path": str(p),
            "env_variable": spec["env_var"],
            "exists": exists,
        }

        if exists:
            report["present_count"] += 1
            info["size_bytes"] = p.stat().st_size
            info["size_mb"] = round(p.stat().st_size / (1024 * 1024), 2)
            info["status"] = "PRESENT"
        else:
            report["missing_count"] += 1
            info["status"] = "MISSING"

        report["artifacts"][key] = info

    return report


def download_artifact(url: str, dest_path: Path, expected_sha256: str | None = None) -> None:
    """Download an external artifact with progress reporting, size bounds, and SHA-256 integrity verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f".tmp_{os.getpid()}")
    max_bytes = get_max_download_bytes()

    hasher = hashlib.sha256()
    downloaded_bytes = 0
    chunk_size = 64 * 1024

    logger.info(f"Downloading {url} -> {dest_path} (max size: {max_bytes // (1024 * 1024)} MB)...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhishVision-Artifact-Manager/1.0"})
        with urllib.request.urlopen(req) as response, temp_path.open("wb") as out_file:
            while chunk := response.read(chunk_size):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_bytes:
                    raise ValueError(
                        f"Download exceeded maximum permitted size of {max_bytes} bytes "
                        f"({max_bytes // (1024 * 1024)} MB). Aborting."
                    )
                hasher.update(chunk)
                out_file.write(chunk)

        computed_sha256 = hasher.hexdigest().lower()

        if expected_sha256:
            if computed_sha256 != expected_sha256.lower():
                raise ValueError(
                    f"SHA-256 verification failed for {dest_path.name}: "
                    f"expected {expected_sha256}, got {computed_sha256}"
                )
            logger.info(f"SHA-256 integrity verified for {dest_path.name}: {computed_sha256}")
        else:
            logger.warning(
                f"Verification unavailable: no expected SHA-256 hash configured for {dest_path.name}. "
                f"(Computed hash: {computed_sha256})"
            )

        # Atomically replace destination only after complete, verified download
        temp_path.replace(dest_path)
        logger.info(f"Successfully saved {dest_path} ({dest_path.stat().st_size} bytes)")

    except Exception as e:
        logger.error(f"Download failed for {dest_path.name}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PhishVision AI Checkpoint Deployment & Verification Manager")
    parser.add_argument("--verify", action="store_true", help="Verify presence and sizes of all registered checkpoints")
    parser.add_argument("--download", action="store_true", help="Download missing checkpoints from remote repository")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHISHVISION_ARTIFACTS_BASE_URL", ""),
        help="Base remote URL where external checkpoints are hosted (e.g. https://storage.googleapis.com/.../weights)",
    )
    parser.add_argument("--json", action="store_true", help="Output status report in structured JSON format")
    args = parser.parse_args()

    status = verify_checkpoints()

    if args.json:
        print(json.dumps(status, indent=2))
        return

    logger.info("=== PhishVision AI Checkpoint Deployment Status ===")
    logger.info(f"Artifacts Present: {status['present_count']}/{status['total_artifacts']}")

    for key, art in status["artifacts"].items():
        if art["exists"]:
            logger.info(f"  [OK] {art['model_id']} ({key}): {art['configured_path']} ({art['size_mb']} MB)")
        else:
            logger.warning(f"  [MISSING] {art['model_id']} ({key}): {art['configured_path']} (Env: {art['env_variable']})")

    if args.download:
        if not args.base_url:
            logger.error(
                "Remote base URL not specified. Provide --base-url or set PHISHVISION_ARTIFACTS_BASE_URL."
            )
            sys.exit(1)

        base_url = args.base_url.rstrip("/")
        for key, spec in CHECKPOINT_REGISTRY.items():
            target_p = Path(os.getenv(spec["env_var"], spec["default_path"]))
            if not target_p.exists():
                remote_url = f"{base_url}/{spec['remote_filename']}"
                try:
                    download_artifact(
                        url=remote_url,
                        dest_path=target_p,
                        expected_sha256=spec.get("expected_sha256"),
                    )
                except Exception as e:
                    logger.error(f"Failed to download {key} from {remote_url}: {e}")

        logger.info("Download step complete. Re-verifying checkpoints...")
        new_status = verify_checkpoints()
        logger.info(f"Artifacts Present after download: {new_status['present_count']}/{new_status['total_artifacts']}")


if __name__ == "__main__":
    main()
