"""Single Artificial Neuron (Perceptron / Sigmoidal Unit) from Scratch.

Mathematical Foundations:
Given an input vector x in R^D:
1. Linear Affine Transformation:
   z = w . x + b = sum_{j=1}^D (w_j * x_j) + b

2. Non-linear Activation:
   a = g(z)

3. Backward Propagation of Errors (Chain Rule):
   Given error signal dL/da from downstream:
   dL/dz = (dL/da) * g'(z)
   dL/dw_j = (dL/dz) * x_j  =>  dL/dw = (dL/dz) * x
   dL/db   = dL/dz
   dL/dx_j = (dL/dz) * w_j  =>  dL/dx = (dL/dz) * w
"""

from __future__ import annotations

import numpy as np

from .activations import Activation, Sigmoid


class Neuron:
    """Represents an isolated single artificial neuron."""

    def __init__(
        self,
        input_dim: int,
        activation: Activation | None = None,
        random_seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.activation = activation or Sigmoid()

        rng = np.random.default_rng(random_seed)
        # Small Gaussian initialization
        self.weights = rng.normal(loc=0.0, scale=0.1, size=(input_dim,)).astype(np.float64)
        self.bias = 0.0

        # Caches for backpropagation
        self.cached_x: np.ndarray | None = None
        self.cached_z: float | None = None
        self.cached_a: float | None = None

        # Gradients
        self.grad_w: np.ndarray = np.zeros_like(self.weights)
        self.grad_b: float = 0.0

    def forward(self, x: np.ndarray) -> float:
        """Compute single forward step: a = g(w^T x + b)."""
        if x.shape != (self.input_dim,):
            raise ValueError(f"Expected input shape ({self.input_dim},); got {x.shape}")

        self.cached_x = x.copy()
        # z = w . x + b
        z = float(np.dot(self.weights, x) + self.bias)
        self.cached_z = z
        # a = g(z)
        a = float(self.activation.forward(np.array([z]))[0])
        self.cached_a = a
        return a

    def backward(self, grad_output: float) -> np.ndarray:
        """Compute backward step via chain rule.

        Returns:
            grad_input: dL/dx in R^{input_dim}
        """
        if self.cached_x is None or self.cached_z is None:
            raise RuntimeError("backward() called before forward()")

        # 1. dL/dz = dL/da * g'(z)
        dz = float(self.activation.backward(np.array([grad_output]))[0])

        # 2. dL/dw = dz * x
        self.grad_w = dz * self.cached_x

        # 3. dL/db = dz
        self.grad_b = dz

        # 4. dL/dx = dz * w
        grad_input = dz * self.weights
        return grad_input

    def update(self, learning_rate: float) -> None:
        """Gradient descent parameter update."""
        self.weights -= learning_rate * self.grad_w
        self.bias -= learning_rate * self.grad_b
