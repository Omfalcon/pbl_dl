"""Evaluation and Comprehensive Metrics Computation Module for ResNet-18 (M-02).

Computes accuracy, per-class precision/recall/F1, macro/weighted averages, confusion matrix,
ROC-AUC, and PR-AUC. Safely handles single-class splits and edge cases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("phishvision.ml.vision.evaluation")


def compute_vision_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for binary screenshot classification.
    
    Args:
        y_true: Ground truth binary labels (0 = Legitimate, 1 = Phishing).
        y_pred: Predicted discrete labels (0 or 1).
        y_prob: Predicted probability for the positive class (Phishing, class 1).
        loss: Optional scalar cross-entropy loss value.
        
    Returns:
        Structured metrics dictionary.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    n_samples = int(len(y_true))
    if n_samples == 0:
        return {"total_samples": 0}

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Per-class metrics (zero_division=0 to prevent warnings on 0 support/predictions)
    prec_legit = float(precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    rec_legit = float(recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    f1_legit = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    support_legit = int((y_true == 0).sum())

    prec_phish = float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    rec_phish = float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    f1_phish = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    support_phish = int((y_true == 1).sum())

    # Macro & Weighted metrics
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ROC-AUC & PR-AUC (require probabilities and >= 2 unique classes in y_true)
    roc_auc: float | None = None
    pr_auc: float | None = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Failed to calculate ROC-AUC: {e}")
                roc_auc = None

            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Failed to calculate PR-AUC: {e}")
                pr_auc = None

    metrics: dict[str, Any] = {
        "total_samples": n_samples,
        "loss": float(loss) if loss is not None else None,
        "accuracy": acc,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "raw": [[tn, fp], [fn, tp]],
        },
        "class_0_legitimate": {
            "precision": prec_legit,
            "recall": rec_legit,
            "f1": f1_legit,
            "support": support_legit,
        },
        "class_1_phishing": {
            "precision": prec_phish,
            "recall": rec_phish,
            "f1": f1_phish,
            "support": support_phish,
        },
        "macro_avg": {
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "f1": weighted_f1,
        },
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return metrics


def save_evaluation_report(
    metrics: dict[str, Any],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save evaluation metrics to disk as a JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "metadata": metadata or {},
        "evaluation_metrics": metrics,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Evaluation report successfully saved to {path}")
