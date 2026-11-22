"""Phase 3: Neural Network Mathematical Sandbox from Scratch.

Pure NumPy implementations of foundational linear algebra, artificial neurons,
dense layers, activations, loss functions, computational graphs, and numerical
gradient checking. Built without PyTorch autograd to provide rigorous
mathematical clarity and exact analytical derivatives.
"""

from .activations import ReLU, Sigmoid, Softmax, Tanh
from .computational_graph import ComputationalGraph, Node
from .gradient_check import check_gradients, numerical_gradient
from .layers import Dense
from .losses import BinaryCrossEntropy, MeanSquaredError
from .network import Sequential
from .neuron import Neuron
from .optimizers import (
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    Optimizer,
    RMSProp,
    SGD,
    SGDMomentum,
    StepLR,
)
from .primitives import (
    cosine_similarity,
    dot_product,
    matrix_multiply,
    vector_norm,
)

__all__ = [
    "dot_product",
    "vector_norm",
    "matrix_multiply",
    "cosine_similarity",
    "Neuron",
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "Node",
    "ComputationalGraph",
    "Sequential",
    "numerical_gradient",
    "check_gradients",
    "Optimizer",
    "SGD",
    "SGDMomentum",
    "RMSProp",
    "Adam",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
]
