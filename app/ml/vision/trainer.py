"""Training and Validation Orchestrator for ResNet-18 (M-02).

Implements AdamW optimization, dynamic class weighting from training data,
Cosine Annealing LR scheduling, early stopping, and best-validation-checkpoint restoration.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.vision.evaluation import compute_vision_metrics, save_evaluation_report
from app.ml.vision.models import CLASS_TO_IDX, IDX_TO_CLASS, ResNet18ScreenshotClassifier

logger = logging.getLogger("phishvision.ml.vision.trainer")


class ResNet18Trainer:
    """Trainer orchestrator for the ResNet-18 binary website screenshot classifier."""

    def __init__(
        self,
        model: ResNet18ScreenshotClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor | None = None,
        epochs: int = 15,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",  # "cosine", "step", or "none"
        early_stopping_patience: int = 5,
        checkpoint_dir: str | Path = "models/checkpoints/resnet18",
        report_dir: str | Path = "data/reports",
        device: str | torch.device | None = None,
        preprocessing_config: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.report_dir = Path(report_dir)
        self.preprocessing_config = preprocessing_config or {}
        self.random_seed = random_seed

        # Setup compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Dynamic class-weighted CrossEntropyLoss based ONLY on training split
        if class_weights is not None:
            weights_tensor = class_weights.to(self.device).float()
            logger.info(f"Applying training dynamic class weights: {weights_tensor.tolist()}")
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning Rate Scheduler
        total_steps = max(len(self.train_loader) * self.epochs, 1)
        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6,
            )
        elif self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(self.epochs // 3, 1),
                gamma=0.5,
            )
        else:
            self.scheduler = None

        self.history: dict[str, list[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_roc_auc": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """Execute one complete training epoch over the training partition."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None and self.scheduler_type == "cosine":
                self.scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

        if self.scheduler is not None and self.scheduler_type == "step":
            self.scheduler.step()

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, Any]]:
        """Evaluate model performance on validation or test DataLoader without gradients."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []

        for images, labels, _ in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            running_loss += float(loss.item())
            n_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        avg_loss = running_loss / max(n_batches, 1)
        metrics = compute_vision_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs,
            loss=avg_loss,
        )
        return avg_loss, metrics

    def save_checkpoint(
        self,
        output_path: Path,
        epoch: int,
        val_macro_f1: float,
        val_metrics: dict[str, Any],
    ) -> None:
        """Save complete training state, model weights, and preprocessing metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_id": "M-02",
            "architecture": "ResNet-18",
            "epoch": epoch,
            "val_macro_f1": val_macro_f1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "model_config": self.model.get_config(),
            "preprocessing_config": self.preprocessing_config,
            "val_metrics": val_metrics,
            "history": self.history,
            "random_seed": self.random_seed,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Checkpoint saved to {output_path} (epoch={epoch}, Val Macro F1={val_macro_f1:.4f})")

    def train(self) -> dict[str, Any]:
        """Execute full training pipeline with early stopping and best-checkpoint restoration."""
        logger.info(f"Starting ResNet-18 training on device '{self.device}' for {self.epochs} epochs...")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        best_val_macro_f1 = -1.0
        best_epoch = 0
        best_val_metrics: dict[str, Any] = {}
        best_model_weights = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        final_checkpoint_path = self.checkpoint_dir / "final_model.pt"

        for epoch in range(1, self.epochs + 1):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)

            val_acc = val_metrics["accuracy"]
            val_macro_f1 = val_metrics["macro_avg"]["f1"]
            val_roc_auc = val_metrics.get("roc_auc")

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)
            self.history["val_roc_auc"].append(val_roc_auc)
            self.history["lr"].append(current_lr)

            auc_str = f"{val_roc_auc:.4f}" if val_roc_auc is not None else "N/A"
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val AUC: {auc_str}"
            )

            # Checkpoint on best validation Macro F1
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_val_metrics = val_metrics
                best_model_weights = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                self.save_checkpoint(best_checkpoint_path, epoch, val_macro_f1, val_metrics)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} (no validation improvement for {self.early_stopping_patience} epochs)."
                    )
                    break

        # Restore best validation weights
        logger.info(f"Restoring best validation model weights from epoch {best_epoch} (Val Macro F1 = {best_val_macro_f1:.4f}).")
        self.model.load_state_dict(best_model_weights)
        self.save_checkpoint(final_checkpoint_path, best_epoch, best_val_macro_f1, best_val_metrics)

        # Export training report
        report_path = self.report_dir / "resnet18_training_history.json"
        save_evaluation_report(
            metrics=best_val_metrics,
            output_path=report_path,
            metadata={
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "history": self.history,
                "preprocessing_config": self.preprocessing_config,
            },
        )

        return {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_metrics": best_val_metrics,
            "history": self.history,
            "best_checkpoint_path": str(best_checkpoint_path),
        }
