"""Siamese ResNet-18 Metric Learning Trainer (M-03 Phase 6.2).

Provides full training and validation loop for Siamese pair representation learning,
featuring AdamW optimization, Cosine Annealing learning rate schedule, early stopping,
checkpoint serialization, and similarity metrics evaluation (ROC-AUC, PR-AUC, F1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from app.ml.vision.siamese import ContrastiveLoss, CosineContrastiveLoss, SiameseResNet18

logger = logging.getLogger("phishvision.ml.vision.siamese_trainer")


def compute_similarity_metrics(
    y_true: np.ndarray | list[int | float],
    similarities: np.ndarray | list[float],
    threshold: float = 0.82,
) -> dict[str, Any]:
    """Computes similarity metrics for Siamese pair verification.
    
    Args:
        y_true: Binary ground truth (1 = same brand, 0 = different brand).
        similarities: Cosine similarities in range [-1.0, 1.0].
        threshold: Decision threshold for positive brand match (default: 0.82).
        
    Returns:
        Structured metrics dictionary including ROC-AUC, PR-AUC, Precision, Recall, F1, and CM.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    sims = np.asarray(similarities, dtype=float).ravel()

    # Map cosine similarity [-1.0, 1.0] to [0.0, 1.0] for standard probability metrics
    norm_sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    # Threshold applied directly on cosine similarity
    y_pred = (sims >= threshold).astype(int)

    # Edge cases when single class present in batch/eval
    n_classes = len(np.unique(y_t))
    if n_classes < 2:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_t))
    else:
        try:
            roc_auc = float(roc_auc_score(y_t, norm_sims))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_t, norm_sims))
        except Exception:
            pr_auc = 0.5

    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    acc = float(np.mean(y_t == y_pred))

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "threshold_applied": threshold,
        "total_pairs": len(y_t),
        "positive_pairs": int(np.sum(y_t == 1)),
        "negative_pairs": int(np.sum(y_t == 0)),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "raw": cm.tolist(),
        },
    }


class SiameseTrainer:
    """Trainer for SiameseResNet18 metric learning."""

    def __init__(
        self,
        model: SiameseResNet18,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_type: str = "contrastive",  # "contrastive" or "cosine"
        margin: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        similarity_threshold: float = 0.82,
        early_stopping_patience: int = 5,
        checkpoint_dir: str | Path = "models/checkpoints/siamese",
        report_dir: str | Path = "data/reports",
        device: str | torch.device | None = None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Allow output_dir or config overrides if passed via kwargs
        if "output_dir" in kwargs and kwargs["output_dir"] is not None:
            checkpoint_dir = kwargs["output_dir"]
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            cfg = kwargs["config"]
            train_cfg = cfg.get("training", {})
            loss_type = train_cfg.get("loss_type", loss_type)
            margin = float(train_cfg.get("margin", margin))
            lr = float(train_cfg.get("learning_rate", lr))
            weight_decay = float(train_cfg.get("weight_decay", weight_decay))
            epochs = int(train_cfg.get("epochs", epochs))
            similarity_threshold = float(train_cfg.get("similarity_threshold", similarity_threshold))
            early_stopping_patience = int(train_cfg.get("early_stopping_patience", early_stopping_patience))
            random_seed = int(train_cfg.get("random_seed", random_seed))

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.similarity_threshold = similarity_threshold
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed

        self.checkpoint_dir = Path(checkpoint_dir)
        self.report_dir = Path(report_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.loss_type = loss_type
        self.margin = margin
        if loss_type == "cosine":
            self.criterion: nn.Module = CosineContrastiveLoss(margin=margin)
        else:
            self.criterion = ContrastiveLoss(margin=margin)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )

        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.best_checkpoint_path = self.checkpoint_dir / "best_siamese.pt"
        self.final_checkpoint_path = self.checkpoint_dir / "final_siamese.pt"

    def train_epoch(self) -> float:
        """Runs one training epoch over pairs."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x1, x2, label, _ in self.train_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            z1, z2, _ = self.model(x1, x2)
            loss = self.criterion(z1, z2, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Runs validation evaluation over pairs."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels: list[float] = []
        all_sims: list[float] = []

        for x1, x2, label, _ in self.val_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            label_dev = label.to(self.device)

            z1, z2, sim = self.model(x1, x2)
            loss = self.criterion(z1, z2, label_dev)

            total_loss += loss.item()
            num_batches += 1

            all_labels.extend(label.cpu().numpy().tolist())
            all_sims.extend(sim.cpu().numpy().tolist())

        val_loss = total_loss / max(num_batches, 1)
        metrics = compute_similarity_metrics(
            y_true=all_labels,
            similarities=all_sims,
            threshold=self.similarity_threshold,
        )
        metrics["loss"] = round(val_loss, 4)
        return metrics

    def train(self) -> dict[str, Any]:
        """Executes full Siamese training loop with early stopping."""
        logger.info(f"Starting Siamese training for {self.epochs} epochs on {self.device}...")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            val_f1 = val_metrics["f1"]
            val_loss = val_metrics["loss"]

            logger.info(
                f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}"
            )

            # Checkpoint on best validation F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(self.best_checkpoint_path, epoch, val_metrics)
                logger.info(f"Saved new best checkpoint to {self.best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={self.early_stopping_patience})")
                    break

        self._save_checkpoint(self.final_checkpoint_path, epoch, val_metrics)
        return {
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "best_checkpoint": str(self.best_checkpoint_path),
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_metrics: dict[str, Any]) -> None:
        checkpoint = {
            "model_id": "M-03",
            "architecture": "SiameseResNet18",
            "embedding_dim": self.model.embedding_dim,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "similarity_threshold": self.similarity_threshold,
            "loss_type": self.loss_type,
            "margin": self.margin,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
