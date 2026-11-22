"""Feature Preprocessing and Standardization Pipeline for Lexical Intelligence.

Ensures zero information leakage between training, validation, and holdout splits.
All statistics (mean, scale, clipping bounds) are learned strictly on training partitions
and saved to disk for reproducible deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


class LexicalPreprocessor:
    """Standardizes numerical feature matrices with zero-leakage guarantees.
    
    Fits sample mean and standard deviation on training data, applying:
        z = np.clip((x - mean_) / scale_, clip_min, clip_max)
    """

    def __init__(
        self,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        eps: float = 1e-8,
        feature_names: list[str] | None = None,
    ) -> None:
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.eps = float(eps)
        self.feature_names = feature_names or []
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int = 0

    def fit(self, x: np.ndarray) -> LexicalPreprocessor:
        """Compute empirical mean and standard deviation on training data."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D array of shape (n_samples, n_features); got {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit preprocessor on empty dataset")

        self.n_features_in_ = x.shape[1]
        self.n_samples_seen_ = x.shape[0]

        # Handle any possible non-finite entries in raw input safely
        valid_mask = np.isfinite(x)
        clean_x = np.where(valid_mask, x, 0.0)

        # Compute mean and standard deviation
        mean = np.mean(clean_x, axis=0)
        # Avoid zero division on constant features: scale = max(std, eps)
        var = np.var(clean_x, axis=0)
        scale = np.sqrt(var)
        scale = np.where(scale < self.eps, 1.0, scale)

        self.mean_ = mean.astype(np.float64)
        self.scale_ = scale.astype(np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply standardization and outlier clipping using fitted statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D array; got {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {x.shape[1]}"
            )

        # Replace non-finite entries with fitted mean
        clean_x = np.where(np.isfinite(x), x, self.mean_)
        z = (clean_x - self.mean_) / self.scale_
        z_clipped = np.clip(z, self.clip_min, self.clip_max)
        return z_clipped.astype(np.float64)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit to training data, then transform it."""
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted parameters to a serializable dictionary."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot serialize unfitted preprocessor")
        return {
            "n_features_in": self.n_features_in_,
            "n_samples_seen": self.n_samples_seen_,
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LexicalPreprocessor:
        """Construct and initialize fitted preprocessor from serialized dictionary."""
        prep = cls(
            clip_min=data.get("clip_min", -10.0),
            clip_max=data.get("clip_max", 10.0),
            eps=data.get("eps", 1e-8),
            feature_names=data.get("feature_names", []),
        )
        prep.n_features_in_ = int(data["n_features_in"])
        prep.n_samples_seen_ = int(data["n_samples_seen"])
        prep.mean_ = np.array(data["mean"], dtype=np.float64)
        prep.scale_ = np.array(data["scale"], dtype=np.float64)
        return prep

    def save(self, filepath: str | Path) -> None:
        """Save fitted preprocessor to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> LexicalPreprocessor:
        """Load fitted preprocessor from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
