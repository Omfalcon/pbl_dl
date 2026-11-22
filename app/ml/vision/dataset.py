"""PyTorch Dataset Implementation for Website Screenshot Classification (M-02).

Safely loads screenshots from parquet split manifests, converts to RGB, applies configurable
transforms, and provides dynamic class-weight calculation for cost-sensitive loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.ml.vision.preprocessing import build_transforms, safe_load_image

logger = logging.getLogger("phishvision.ml.vision.dataset")


class ScreenshotDataset(Dataset):
    """PyTorch Dataset loading website screenshots from parquet partitions or DataFrame."""

    def __init__(
        self,
        data_source: str | Path | pd.DataFrame,
        repo_root: str | Path = ".",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        crop_strategy: str = "top_viewport",
        image_size: int = 224,
        viewport_ratio: float = 1.0,
        is_training: bool = False,
        horizontal_flip: bool = False,
        color_jitter: bool = False,
        handle_invalid: str = "fallback",  # "fallback" or "raise"
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.handle_invalid = handle_invalid
        self.crop_strategy = crop_strategy
        self.image_size = image_size
        self.viewport_ratio = viewport_ratio
        self.is_training = is_training

        # Load DataFrame
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        else:
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset split file not found: {path}")
            if path.suffix == ".parquet":
                self.df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                self.df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported data source format: {path}")
            self.df = self.df.reset_index(drop=True)

        # Configure transforms if not explicitly passed
        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_transforms(
                crop_strategy=crop_strategy,
                image_size=image_size,
                viewport_ratio=viewport_ratio,
                is_training=is_training,
                horizontal_flip=horizontal_flip,
                color_jitter=color_jitter,
            )

        # Identify image path and label columns
        self.path_col = "screenshot_relative_path"
        if self.path_col not in self.df.columns:
            if "file_path" in self.df.columns:
                self.path_col = "file_path"
            elif "screenshot_absolute_path" in self.df.columns:
                self.path_col = "screenshot_absolute_path"
            else:
                raise KeyError(f"Missing image path column in dataset columns: {list(self.df.columns)}")

        if "label" not in self.df.columns:
            raise KeyError(f"Missing 'label' column in dataset columns: {list(self.df.columns)}")

        # Convert labels to int64
        self.labels = self.df["label"].astype(int).values
        self.sample_ids = (
            self.df["sample_id"].astype(str).tolist()
            if "sample_id" in self.df.columns
            else [f"sample_{i:05d}" for i in range(len(self.df))]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        raw_path = str(row[self.path_col])
        label = int(row["label"])
        sample_id = self.sample_ids[idx]

        # Resolve relative to repo root if path is not absolute
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = self.repo_root / file_path

        try:
            pil_img = safe_load_image(file_path)
        except Exception as e:
            if self.handle_invalid == "raise":
                raise RuntimeError(f"Error loading image at index {idx} ({file_path}): {e}") from e
            logger.warning(f"Failed to load image {file_path}; using neutral blank image fallback: {e}")
            pil_img = Image.new("RGB", (224, 224), (255, 255, 255))

        tensor_img = self.transform(pil_img)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return tensor_img, label_tensor, sample_id

    def compute_class_weights(self) -> torch.Tensor:
        """Compute balanced class weights dynamically from dataset labels.
        
        Formula:
            w_c = N / (2 * N_c)
            
        Returns:
            torch.Tensor of shape (2,) with [w_legitimate, w_phishing].
        """
        n_total = len(self.labels)
        if n_total == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())

        w0 = float(n_total / (2.0 * max(n_neg, 1)))
        w1 = float(n_total / (2.0 * max(n_pos, 1)))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def get_summary(self) -> dict[str, Any]:
        """Return dataset partition statistics summary."""
        n_total = len(self.labels)
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        domains = self.df["registrable_domain"].nunique() if "registrable_domain" in self.df.columns else None

        return {
            "total_samples": n_total,
            "phishing_count": n_pos,
            "legitimate_count": n_neg,
            "phishing_ratio": round(n_pos / max(n_total, 1), 4),
            "unique_domains": domains,
            "class_weights": [round(w, 4) for w in self.compute_class_weights().tolist()],
        }
