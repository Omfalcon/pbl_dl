"""Unit and regression tests for scripts/download_checkpoints.py.

Verifies checkpoint verification, download size enforcement, SHA-256 integrity,
and atomic file replacement behaviors.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
import pytest

from scripts.download_checkpoints import (
    CHECKPOINT_REGISTRY,
    download_artifact,
    verify_checkpoints,
)


class MockHTTPResponse(io.BytesIO):
    """Mock urllib response object."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_verify_checkpoints_structure() -> None:
    """Verify verify_checkpoints returns standard status dictionary with all models."""
    status = verify_checkpoints()
    assert "total_artifacts" in status
    assert "present_count" in status
    assert "missing_count" in status
    assert "artifacts" in status
    assert status["total_artifacts"] == len(CHECKPOINT_REGISTRY)

    for spec in CHECKPOINT_REGISTRY.values():
        assert "expected_sha256" in spec
        # Critical security & integrity rule: Never invent synthetic hashes
        assert spec["expected_sha256"] is None or isinstance(spec["expected_sha256"], str)


def test_download_artifact_respects_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify download exceeding PHISHVISION_MAX_DOWNLOAD_BYTES raises ValueError and removes temp file."""
    monkeypatch.setenv("PHISHVISION_MAX_DOWNLOAD_BYTES", "500")  # 500 bytes max

    payload = b"A" * 1024
    dest = tmp_path / "test_model.pt"

    def mock_urlopen(req):
        return MockHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(ValueError, match="Download exceeded maximum permitted size"):
        download_artifact("http://mock-storage.test/weights.pt", dest)

    assert not dest.exists()
    # Ensure no lingering .tmp files
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0


def test_download_artifact_sha256_match_and_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify download with matching SHA-256 succeeds and atomically replaces destination."""
    payload = b"checkpoint binary model weights payload"
    expected_hash = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "best_model.pt"

    def mock_urlopen(req):
        return MockHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    download_artifact("http://mock-storage.test/weights.pt", dest, expected_sha256=expected_hash)

    assert dest.exists()
    assert dest.read_bytes() == payload
    # Check no temp file left
    assert len(list(tmp_path.glob("*.tmp*"))) == 0


def test_download_artifact_sha256_mismatch_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify SHA-256 mismatch raises ValueError and does NOT overwrite or delete existing valid file."""
    existing_content = b"original valid checkpoint"
    dest = tmp_path / "existing_model.pt"
    dest.write_bytes(existing_content)

    corrupted_payload = b"tampered binary payload"
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    def mock_urlopen(req):
        return MockHTTPResponse(corrupted_payload)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        download_artifact("http://mock-storage.test/weights.pt", dest, expected_sha256=wrong_hash)

    # Destination MUST remain untouched
    assert dest.exists()
    assert dest.read_bytes() == existing_content
    # Temp files must be cleaned up
    assert len(list(tmp_path.glob("*.tmp*"))) == 0


def test_download_artifact_without_sha256_logs_warning_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify download without configured SHA-256 succeeds without crashing (verification unavailable)."""
    payload = b"unverified weights"
    dest = tmp_path / "unverified.pt"

    def mock_urlopen(req):
        return MockHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    download_artifact("http://mock-storage.test/weights.pt", dest, expected_sha256=None)
    assert dest.exists()
    assert dest.read_bytes() == payload
