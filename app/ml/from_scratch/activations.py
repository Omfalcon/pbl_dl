"""Activation Functions and Their Analytical Derivatives from Scratch.

Mathematical Foundations:
1. ReLU (Rectified Linear Unit):
   f(z) = max(0, z)
   f'(z) = 1 if z > 0 else 0 (subgradient at z=0 taken as 0)

2. Sigmoid (Logistic Function):
   sigma(z) = 1 / (1 + e^{-z})
   sigma'(z) = sigma(z) * (1 - sigma(z))
   Numerical Safeguard: Clamp z to [-50, 50] to eliminate floating-point overflow.

3. Tanh (Hyperbolic Tangent):
   tanh(z) = (e^z - e^{-z}) / (e^z + e^{-z}) = 2 * sigma(2z) - 1
   tanh'(z) = 1 - tanh^2(z)

4. Softmax (Normalized Exponential Function):
   softmax(z)_i = e^{z_i - max(z)} / sum_{j=1}^C e^{z_j - max(z)}
   Jacobian Matrix:
   d softmax_i / d z_j = softmax_i * (delta_{ij} - softmax_j)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """Abstract Base Class for all activation functions."""

    def __init__(self) -> None:
        self.cached_input: np.ndarray | None = None
        self.cached_output: np.ndarray | None = None

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Compute forward activation."""
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient: dL/dz = dL/da * da/dz."""
        pass

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)


class ReLU(Activation):
    """Rectified Linear Unit Activation."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.cached_input = z.copy()
        out = np.maximum(0.0, z)
        self.cached_output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cached_input is None:
            raise RuntimeError("backward() called before forward()")
        # Indicator function I(z > 0)
        grad_input = grad_output * (self.cached_input > 0.0).astype(np.float64)
        return grad_input


class Sigmoid(Activation):
    """Logistic Sigmoid Activation."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.cached_input = z.copy()
        # Numerical stabilization: avoid exp overflow for large negative z
        z_clamped = np.clip(z, -50.0, 50.0)
        out = 1.0 / (1.0 + np.exp(-z_clamped))
        self.cached_output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cached_output is None:
            raise RuntimeError("backward() called before forward()")
        # d/dz sigma(z) = sigma(z) * (1 - sigma(z))
        s = self.cached_output
        grad_input = grad_output * (s * (1.0 - s))
        return grad_input


class Tanh(Activation):
    """Hyperbolic Tangent Activation."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.cached_input = z.copy()
        z_clamped = np.clip(z, -50.0, 50.0)
        out = np.tanh(z_clamped)
        self.cached_output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cached_output is None:
            raise RuntimeError("backward() called before forward()")
        # d/dz tanh(z) = 1 - tanh^2(z)
        t = self.cached_output
        grad_input = grad_output * (1.0 - t ** 2)
        return grad_input


class Softmax(Activation):
    """Numerically Stable Softmax for Multi-Class Probabilities."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.cached_input = z.copy()
        # Subtract max per row for shift-invariance: e^(z - max) / sum(e^(z - max))
        if z.ndim == 1:
            z_shifted = z - np.max(z)
            exp_z = np.exp(z_shifted)
            out = exp_z / np.sum(exp_z)
        else:
            z_shifted = z - np.max(z, axis=-1, keepdims=True)
            exp_z = np.exp(z_shifted)
            out = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        self.cached_output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Vectorized backward pass through Softmax."""
        if self.cached_output is None:
            raise RuntimeError("backward() called before forward()")
        s = self.cached_output
        # For batch computation: sum(grad_output * s) along class dimension
        # dL/dz_i = s_i * (dL/ds_i - sum_k(dL/ds_k * s_k))
        if s.ndim == 1:
            sum_grad = np.sum(grad_output * s)
            return s * (grad_output - sum_grad)
        else:
            sum_grad = np.sum(grad_output * s, axis=-1, keepdims=True)
            return s * (grad_output - sum_grad)
