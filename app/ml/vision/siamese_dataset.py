"""Siamese Pair and Triplet Dataset & Generation Pipeline (M-03 Phase 6.2).

Provides PyTorch Dataset implementations for Siamese pairs (positive/negative)
and triplets (anchor/positive/negative), plus automated pair-generation with
domain-isolated partitioning to prevent data leakage.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from app.ml.vision.preprocessing import IMAGENET_MEAN, IMAGENET_STD, safe_load_image

logger = logging.getLogger("phishvision.ml.vision.siamese_dataset")


def build_siamese_transforms(
    image_size: int = 128,
    is_training: bool = False,
) -> Callable[[Image.Image], torch.Tensor]:
    """Builds standard evaluation or augmentation transforms for logo and screenshot crops."""
    transforms_list: list[Any] = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
    ]
    if is_training:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1))

    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(transforms_list)


class SiamesePairDataset(Dataset):
    """PyTorch Dataset yielding image pairs (x1, x2) with binary similarity label (1=positive, 0=negative)."""

    def __init__(
        self,
        pairs_data: pd.DataFrame | list[dict[str, Any]] | str | Path | None = None,
        manifest_csv: str | Path | None = None,
        split: str | None = None,
        repo_root: str | Path = ".",
        image_size: int = 128,
        is_training: bool = False,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.image_size = image_size
        self.is_training = is_training
        self.split = split

        target_data = pairs_data if pairs_data is not None else manifest_csv
        if target_data is None:
            raise ValueError("Either pairs_data or manifest_csv must be provided.")

        if isinstance(target_data, (str, Path)):
            p = Path(target_data)
            if not p.exists():
                raise FileNotFoundError(f"Pairs file not found: {p}")
            if p.suffix == ".parquet":
                self.df = pd.read_parquet(p)
            else:
                self.df = pd.read_csv(p)
        elif isinstance(target_data, pd.DataFrame):
            self.df = target_data.copy().reset_index(drop=True)
        elif isinstance(target_data, list):
            self.df = pd.DataFrame(target_data).reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported pairs data type: {type(target_data)}")

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].copy().reset_index(drop=True)

        self.transform = transform or build_siamese_transforms(image_size=image_size, is_training=is_training)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.df.iloc[idx]
        p1 = Path(row["image_path_1"])
        p2 = Path(row["image_path_2"])
        label = float(row["label"])

        # Resolve paths against repo_root if relative
        full_p1 = p1 if p1.is_absolute() else self.repo_root / p1
        full_p2 = p2 if p2.is_absolute() else self.repo_root / p2

        img1 = safe_load_image(full_p1)
        img2 = safe_load_image(full_p2)

        t1 = self.transform(img1)
        t2 = self.transform(img2)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        meta = {
            "brand_1": str(row.get("brand_1", "")),
            "brand_2": str(row.get("brand_2", "")),
            "pair_type": str(row.get("pair_type", "")),
            "pair_id": str(row.get("pair_id", f"pair_{idx}")),
        }
        return t1, t2, label_tensor, meta


class SiameseTripletDataset(Dataset):
    """PyTorch Dataset yielding triplets (anchor, positive, negative)."""

    def __init__(
        self,
        triplets_data: pd.DataFrame | list[dict[str, Any]],
        repo_root: str | Path = ".",
        image_size: int = 128,
        is_training: bool = False,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.df = pd.DataFrame(triplets_data).reset_index(drop=True)
        self.transform = transform or build_siamese_transforms(image_size=image_size, is_training=is_training)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        pa = Path(row["anchor_path"])
        pp = Path(row["positive_path"])
        pn = Path(row["negative_path"])
        brand = str(row.get("brand_id", ""))

        fa = pa if pa.is_absolute() else self.repo_root / pa
        fp = pp if pp.is_absolute() else self.repo_root / pp
        fn = pn if pn.is_absolute() else self.repo_root / pn

        ta = self.transform(safe_load_image(fa))
        tp = self.transform(safe_load_image(fp))
        tn = self.transform(safe_load_image(fn))
        return ta, tp, tn, brand


def generate_synthetic_brand_pairs(
    brands: list[str] | None = None,
    num_samples_per_brand: int = 4,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic pair metadata DataFrame for testing pipeline mechanics without requiring real images on disk."""
    rng = random.Random(seed)
    brand_list = brands or ["google", "microsoft", "paypal", "apple", "amazon"]

    records: list[dict[str, Any]] = []

    # Positive pairs: same brand
    for b in brand_list:
        for i in range(num_samples_per_brand):
            for j in range(i + 1, num_samples_per_brand):
                records.append({
                    "pair_id": f"pos_{b}_{i}_{j}",
                    "brand_1": b,
                    "brand_2": b,
                    "image_path_1": f"data/brands/reference_gallery/{b}/logos/logo_var_{i}.png",
                    "image_path_2": f"data/brands/reference_gallery/{b}/logos/logo_var_{j}.png",
                    "label": 1,
                    "pair_type": "authentic_same_brand",
                })

    num_pos = len(records)
    num_neg = int(round(num_pos * negative_ratio))

    # Negative pairs: different brands
    neg_count = 0
    while neg_count < num_neg:
        b1, b2 = rng.sample(brand_list, 2)
        i = rng.randint(0, num_samples_per_brand - 1)
        j = rng.randint(0, num_samples_per_brand - 1)
        records.append({
            "pair_id": f"neg_{b1}_{b2}_{i}_{j}",
            "brand_1": b1,
            "brand_2": b2,
            "image_path_1": f"data/brands/reference_gallery/{b1}/logos/logo_var_{i}.png",
            "image_path_2": f"data/brands/reference_gallery/{b2}/logos/logo_var_{j}.png",
            "label": 0,
            "pair_type": "cross_brand_negative",
        })
        neg_count += 1

    rng.shuffle(records)
    return pd.DataFrame(records)
