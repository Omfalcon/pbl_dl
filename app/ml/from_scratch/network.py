"""Sequential Feedforward Neural Network Pipeline from Scratch.

Orchestrates sequential forward evaluation and backward backpropagation
across arbitrarily deep stacks of Dense layers and Activation modules.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .activations import Activation
from .layers import Dense


class Sequential:
    """Feedforward Multi-Layer Perceptron (MLP) architecture."""

    def __init__(self, layers: list[Dense | Activation] | None = None) -> None:
        self.layers: list[Dense | Activation] = layers or []

    def add(self, layer: Dense | Activation) -> None:
        """Append a layer or activation function to the model."""
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass sequentially through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Execute backward pass in reverse order via chain rule."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate: float) -> None:
        """Perform gradient descent step on all parameterized layers."""
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(learning_rate)

    def get_parameters(self) -> list[np.ndarray]:
        """Collect all model parameters."""
        params: list[np.ndarray] = []
        for layer in self.layers:
            if hasattr(layer, "get_params"):
                params.extend(layer.get_params())
        return params

    def get_gradients(self) -> list[np.ndarray]:
        """Collect all parameter gradients."""
        grads: list[np.ndarray] = []
        for layer in self.layers:
            if hasattr(layer, "get_grads"):
                grads.extend(layer.get_grads())
        return grads
