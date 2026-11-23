"""Six-Stage Scan Pipeline Progress Tracker for PhishVision AI.

Provides deterministic, real-time stage tracking for the scan lifecycle:

    URL_CHECK -> REDIRECT_INVESTIGATION -> TLS_INSPECTION ->
    PLAYWRIGHT_CAPTURE -> VISUAL_ANALYSIS -> EVIDENCE_FUSION -> COMPLETED

Each stage has an honest status (PENDING, IN_PROGRESS, COMPLETED, FAILED,
SKIPPED, UNAVAILABLE), timestamps, duration, and a human-readable message.
Progress percentage is derived from actual completed stages, not timers.

Design principles:
- Never use fake timer-based progress.
- Stage failures are isolated; optional stages do not kill the scan.
- All stage data is serializable to JSON for the status API.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"


STAGE_NAMES = [
    "URL_CHECK",
    "REDIRECT_INVESTIGATION",
    "TLS_INSPECTION",
    "PLAYWRIGHT_CAPTURE",
    "VISUAL_ANALYSIS",
    "EVIDENCE_FUSION",
]

# Stages that contribute to progress percentage when complete
_PROGRESS_STAGES = STAGE_NAMES


class ScanPipelineTracker:
    """Tracks real-time progress of a scan across six pipeline stages."""

    def __init__(self, scan_id: str, fast_mode: bool = False) -> None:
        self.scan_id = scan_id
        self.fast_mode = fast_mode
        self._stages: dict[str, dict[str, Any]] = {}

        for name in STAGE_NAMES:
            self._stages[name] = {
                "name": name,
                "status": StageStatus.PENDING,
                "started_at": None,
                "completed_at": None,
                "duration_ms": None,
                "message": f"Stage {name} is pending.",
                "error": None,
                "data": {},
            }

        # In fast_mode, browser-dependent stages are skipped immediately
        if fast_mode:
            for stage in ("REDIRECT_INVESTIGATION", "TLS_INSPECTION", "PLAYWRIGHT_CAPTURE", "VISUAL_ANALYSIS"):
                self._stages[stage]["status"] = StageStatus.SKIPPED
                self._stages[stage]["message"] = "Skipped: fast_mode=True (lexical-only analysis)."

    def begin(self, stage: str, message: str | None = None) -> None:
        """Mark a stage as IN_PROGRESS with current timestamp."""
        if stage not in self._stages:
            return
        self._stages[stage]["status"] = StageStatus.IN_PROGRESS
        self._stages[stage]["started_at"] = datetime.now(timezone.utc).isoformat()
        self._stages[stage]["message"] = message or f"{stage} is in progress."

    def complete(self, stage: str, message: str | None = None, data: dict[str, Any] | None = None) -> None:
        """Mark a stage as COMPLETED with timestamp and optional result data."""
        if stage not in self._stages:
            return
        s = self._stages[stage]
        s["status"] = StageStatus.COMPLETED
        s["completed_at"] = datetime.now(timezone.utc).isoformat()
        s["message"] = message or f"{stage} completed successfully."
        if data:
            s["data"].update(data)
        # Calculate duration if started_at is set
        if s["started_at"]:
            try:
                start = datetime.fromisoformat(s["started_at"])
                end = datetime.fromisoformat(s["completed_at"])
                s["duration_ms"] = round((end - start).total_seconds() * 1000, 1)
            except Exception:
                pass

    def fail(self, stage: str, error: str, message: str | None = None) -> None:
        """Mark a stage as FAILED with error details."""
        if stage not in self._stages:
            return
        s = self._stages[stage]
        s["status"] = StageStatus.FAILED
        s["completed_at"] = datetime.now(timezone.utc).isoformat()
        s["error"] = error
        s["message"] = message or f"{stage} failed: {error}"
        if s["started_at"]:
            try:
                start = datetime.fromisoformat(s["started_at"])
                end = datetime.fromisoformat(s["completed_at"])
                s["duration_ms"] = round((end - start).total_seconds() * 1000, 1)
            except Exception:
                pass

    def skip(self, stage: str, message: str | None = None) -> None:
        """Mark a stage as SKIPPED (e.g. TLS skip for HTTP URLs)."""
        if stage not in self._stages:
            return
        self._stages[stage]["status"] = StageStatus.SKIPPED
        self._stages[stage]["message"] = message or f"{stage} was skipped."

    def unavailable(self, stage: str, message: str | None = None) -> None:
        """Mark a stage as UNAVAILABLE (e.g. model checkpoint missing)."""
        if stage not in self._stages:
            return
        self._stages[stage]["status"] = StageStatus.UNAVAILABLE
        self._stages[stage]["message"] = message or f"{stage} is unavailable."

    def set_data(self, stage: str, key: str, value: Any) -> None:
        """Store structured result data in a stage."""
        if stage in self._stages:
            self._stages[stage]["data"][key] = value

    def get_current_stage(self) -> str:
        """Return the name of the currently IN_PROGRESS stage, or last COMPLETED."""
        for name in STAGE_NAMES:
            if self._stages[name]["status"] == StageStatus.IN_PROGRESS:
                return name
        # Return the last completed stage
        for name in reversed(STAGE_NAMES):
            if self._stages[name]["status"] == StageStatus.COMPLETED:
                return name
        return STAGE_NAMES[0]

    def get_progress_percent(self) -> float:
        """Calculate progress percentage from actual stage completions (0.0 – 100.0)."""
        completed = sum(
            1 for name in _PROGRESS_STAGES
            if self._stages[name]["status"] in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        )
        total = len(_PROGRESS_STAGES)
        return round(100.0 * completed / total, 1) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize all stage information to a JSON-compatible dictionary."""
        return {
            "stages": {name: dict(self._stages[name]) for name in STAGE_NAMES},
            "current_stage": self.get_current_stage(),
            "progress_percent": self.get_progress_percent(),
        }

    def get_stage_data(self, stage: str) -> dict[str, Any]:
        """Return structured data accumulated in a stage."""
        return self._stages.get(stage, {}).get("data", {})
