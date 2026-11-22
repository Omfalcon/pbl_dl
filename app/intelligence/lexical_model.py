"""Lexical Multi-Layer Perceptron (M-01) Architecture and Inference Wrapper.

Builds on the pure NumPy from-scratch neural network foundations (Sequential, Dense,
ReLU, Sigmoid) with parameter serialization, evaluation metrics, and calibrated thresholding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from app.ml.from_scratch.activations import ReLU, Sigmoid
from app.ml.from_scratch.layers import Dense
from app.ml.from_scratch.network import Sequential


class LexicalMLP:
    """3-Layer Feedforward Multi-Layer Perceptron for lexical phishing risk estimation.
    
    Architecture (M-01):
        Input (40-d) -> Dense(40, 64) -> ReLU -> Dense(64, 32) -> ReLU -> Dense(32, 1) -> Sigmoid
    """

    def __init__(
        self,
        in_features: int = 40,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        random_seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.random_seed = random_seed

        # Construct architecture using pure NumPy layers
        self.dense1 = Dense(in_features, hidden_dim1, init_method="he", random_seed=random_seed)
        self.relu1 = ReLU()
        self.dense2 = Dense(hidden_dim1, hidden_dim2, init_method="he", random_seed=random_seed + 1)
        self.relu2 = ReLU()
        self.dense3 = Dense(hidden_dim2, 1, init_method="xavier", random_seed=random_seed + 2)
        self.sigmoid = Sigmoid()

        self.model = Sequential([
            self.dense1,
            self.relu1,
            self.dense2,
            self.relu2,
            self.dense3,
            self.sigmoid,
        ])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through sequential network."""
        return self.model.forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass propagating error gradients."""
        return self.model.backward(grad_output)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Compute estimated phishing probability p(phishing | url) in [0.0, 1.0]."""
        proba = self.forward(x)
        return proba.reshape(-1, 1)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class: 1 (phishing) if proba >= threshold, else 0 (legitimate)."""
        proba = self.predict_proba(x)
        return (proba >= threshold).astype(np.int64).reshape(-1)

    def get_parameters(self) -> list[np.ndarray]:
        """Collect all model weight and bias arrays."""
        return self.model.get_parameters()

    def get_gradients(self) -> list[np.ndarray]:
        """Collect all model parameter gradient arrays."""
        return self.model.get_gradients()

    def save_weights(self, filepath: str | Path) -> None:
        """Save layer weights and biases to compressed NPZ archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.dense1.weights,
            b1=self.dense1.biases,
            w2=self.dense2.weights,
            b2=self.dense2.biases,
            w3=self.dense3.weights,
            b3=self.dense3.biases,
            in_features=np.array([self.in_features]),
            hidden_dim1=np.array([self.hidden_dim1]),
            hidden_dim2=np.array([self.hidden_dim2]),
        )

    def load_weights(self, filepath: str | Path) -> None:
        """Load layer weights and biases from NPZ archive."""
        data = np.load(filepath)
        self.dense1.weights = data["w1"].copy()
        self.dense1.biases = data["b1"].copy()
        self.dense2.weights = data["w2"].copy()
        self.dense2.biases = data["b2"].copy()
        self.dense3.weights = data["w3"].copy()
        self.dense3.biases = data["b3"].copy()


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics.
    
    Includes Accuracy, Precision, Recall, F1 for both classes, Macro F1,
    ROC-AUC, PR-AUC, and full confusion matrix.
    """
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_prob = np.asarray(y_pred_proba).ravel().astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    total_samples = int(len(y_true))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Per-class metrics
    # Class 1 (Phishing)
    prec_1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_1 = float(2 * prec_1 * rec_1 / (prec_1 + rec_1)) if (prec_1 + rec_1) > 0 else 0.0

    # Class 0 (Legitimate)
    prec_0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_0 = float(2 * prec_0 * rec_0 / (prec_0 + rec_0)) if (prec_0 + rec_0) > 0 else 0.0

    macro_prec = float(0.5 * (prec_0 + prec_1))
    macro_rec = float(0.5 * (rec_0 + rec_1))
    macro_f1 = float(0.5 * (f1_0 + f1_1))
    accuracy = float((tp + tn) / max(total_samples, 1))

    # ROC-AUC computation (trapezoidal rule via sorting)
    roc_auc = _compute_roc_auc(y_true, y_prob)
    pr_auc = _compute_pr_auc(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "total_samples": total_samples,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "accuracy": accuracy,
        "class_0_legitimate": {
            "precision": prec_0,
            "recall": rec_0,
            "f1": f1_0,
            "support": int(np.sum(y_true == 0)),
        },
        "class_1_phishing": {
            "precision": prec_1,
            "recall": rec_1,
            "f1": f1_1,
            "support": int(np.sum(y_true == 1)),
        },
        "macro_avg": {
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": macro_f1,
        },
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compute trapezoidal numerical integration compatible with NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(getattr(np, "trapz")(y, x))
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve via rank sum."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort descending
    desc_order = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_order]

    # Cumulative TPR and FPR
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])

    return _trapz(tpr, fpr)


def _compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision) via trapezoidal rule."""
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0

    desc_order = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    recall = tps / n_pos
    precision = tps / (tps + fps)

    # Prepend (recall=0, precision=precision[0])
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) > 0 else 1.0], precision])

    return _trapz(precision, recall)
