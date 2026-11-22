"""Loss Functions and Their Analytical Gradients from Scratch.

Mathematical Foundations:
1. Mean Squared Error (MSE):
   L(y, y_hat) = (1 / (2 * N)) * sum_{i=1}^N ||y_i - y_hat_i||_2^2
   dL / d y_hat = (1 / N) * (y_hat - y)

2. Binary Cross-Entropy (BCE):
   L(y, y_hat) = -(1 / N) * sum_{i=1}^N [y_i * ln(y_hat_i) + (1 - y_i) * ln(1 - y_hat_i)]
   dL / d y_hat = -(1 / N) * [y / y_hat - (1 - y) / (1 - y_hat)]
   Numerical Safeguard: y_hat clamped to [eps, 1 - eps] to prevent ln(0) = -inf.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Abstract Base Class for loss functions."""

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute scalar loss value."""
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute analytical gradient dL / d y_pred."""
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)


class MeanSquaredError(Loss):
    """Mean Squared Error (L2 loss)."""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape} != y_pred={y_pred.shape}")
        n = y_true.size
        # (1 / (2 * n)) * sum((y_pred - y_true)^2)
        diff = y_pred - y_true
        return float(np.sum(diff ** 2) / (2.0 * max(n, 1)))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape} != y_pred={y_pred.shape}")
        n = y_true.size
        # dL / d y_pred = (y_pred - y_true) / n
        return (y_pred - y_true) / max(n, 1)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy Loss with Numerical Stability Clamping."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape} != y_pred={y_pred.shape}")
        n = y_true.shape[0] if y_true.ndim > 0 else 1

        # Clamp predictions to avoid ln(0) or ln(negative)
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        # -(y * ln(p) + (1 - y) * ln(1 - p))
        bce = -(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        return float(np.mean(bce))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape} != y_pred={y_pred.shape}")
        n = y_true.shape[0] if y_true.ndim > 0 else 1

        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        # dL / d y_hat = (1/n) * [ (y_hat - y) / (y_hat * (1 - y_hat)) ]
        grad = ((y_pred_clipped - y_true) / (y_pred_clipped * (1.0 - y_pred_clipped))) / max(n, 1)
        return grad
