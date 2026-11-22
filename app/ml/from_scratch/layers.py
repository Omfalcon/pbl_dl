"""Fully Connected Dense Layers with Manual Vectorized Backpropagation.

Mathematical Foundations:
For a batch of N input samples X in R^{N x D_{in}}:
1. Forward Affine Transformation:
   Z = X @ W + b
   where:
   - X is (N, D_{in})
   - W is (D_{in}, D_{out})
   - b is (1, D_{out}) broadcast across rows
   - Z is (N, D_{out})

2. Backward Matrix Calculus (Chain Rule):
   Given upstream error gradient dL/dZ in R^{N x D_{out}}:

   a) Gradient with respect to inputs (propagated downstream):
      dL / dX = (dL / dZ) @ W^T
      Dimension: (N, D_{out}) @ (D_{out}, D_{in}) = (N, D_{in})

   b) Gradient with respect to weights:
      dL / dW = X^T @ (dL / dZ)
      Dimension: (D_{in}, N) @ (N, D_{out}) = (D_{in}, D_{out})

   c) Gradient with respect to biases:
      dL / db = sum_{i=1}^N (dL / dZ)_{i, :}
      Dimension: (1, D_{out})
"""

from __future__ import annotations

import numpy as np


class Dense:
    """Fully Connected (Dense / Linear) Neural Network Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = "xavier",
        random_seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng(random_seed)

        # Weight initialization strategy
        if init_method == "xavier":
            # Glorot / Xavier initialization: U(-limit, limit)
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.weights = rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        elif init_method == "he":
            # He / Kaiming initialization: N(0, 2 / in_features)
            std = np.sqrt(2.0 / in_features)
            self.weights = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
        else:
            self.weights = rng.normal(0.0, 0.05, size=(in_features, out_features)).astype(np.float64)

        self.biases = np.zeros((1, out_features), dtype=np.float64)

        # Caches
        self.cached_x: np.ndarray | None = None

        # Parameter Gradients
        self.grad_w: np.ndarray = np.zeros_like(self.weights)
        self.grad_b: np.ndarray = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: Z = X @ W + b."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.in_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.in_features}, got {x.shape[1]}")

        self.cached_x = x.copy()
        z = np.dot(x, self.weights) + self.biases
        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Vectorized backward pass.

        Args:
            grad_output: dL/dZ of shape (N, D_out)

        Returns:
            grad_input: dL/dX of shape (N, D_in)
        """
        if self.cached_x is None:
            raise RuntimeError("backward() called before forward()")

        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        # 1. dL / dW = X^T @ dL/dZ
        self.grad_w = np.dot(self.cached_x.T, grad_output)

        # 2. dL / db = sum over batch rows
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # 3. dL / dX = dL/dZ @ W^T
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update(self, learning_rate: float) -> None:
        """Apply gradient descent step to parameters."""
        self.weights -= learning_rate * self.grad_w
        self.biases -= learning_rate * self.grad_b

    def get_params(self) -> list[np.ndarray]:
        return [self.weights, self.biases]

    def get_grads(self) -> list[np.ndarray]:
        return [self.grad_w, self.grad_b]
