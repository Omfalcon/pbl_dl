"""Explicit Computational Graphs & Reverse-Mode Automatic Differentiation from Scratch.

Mathematical Foundations:
A neural network is a Directed Acyclic Graph (DAG) of mathematical operations.
Given a scalar loss L = f(v_1, v_2, ..., v_m):
1. Forward Pass: Computes values of all intermediate nodes in topological order.
2. Multivariate Chain Rule:
   For any intermediate node v:
   dL / dv = sum_{c in Children(v)} (dL / dc) * (dc / dv)
3. Reverse-Mode Backpropagation: Computes adjoints (dL/dv) in reverse topological order.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np


class Node:
    """A Node in a dynamic computational graph representing a tensor variable or operation."""

    def __init__(
        self,
        data: np.ndarray | float,
        parents: tuple[Node, ...] = (),
        op_name: str = "variable",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.parents: tuple[Node, ...] = parents
        self.op_name: str = op_name
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Node(data={self.data.shape}, op='{self.op_name}')"

    def __add__(self, other: Node | float | np.ndarray) -> Node:
        other_node = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other_node.data, parents=(self, other_node), op_name="add")

        def _backward() -> None:
            # d(x + y)/dx = 1, d(x + y)/dy = 1 (handling broadcasting)
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other_node.grad += _unbroadcast(out.grad, other_node.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: float | np.ndarray) -> Node:
        return self.__add__(other)

    def __sub__(self, other: Node | float | np.ndarray) -> Node:
        other_node = other if isinstance(other, Node) else Node(other)
        out = Node(self.data - other_node.data, parents=(self, other_node), op_name="sub")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other_node.grad -= _unbroadcast(out.grad, other_node.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Node | float | np.ndarray) -> Node:
        other_node = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other_node.data, parents=(self, other_node), op_name="mul")

        def _backward() -> None:
            # Product rule: d(u*v)/du = v, d(u*v)/dv = u
            self.grad += _unbroadcast(out.grad * other_node.data, self.data.shape)
            other_node.grad += _unbroadcast(out.grad * self.data, other_node.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: float | np.ndarray) -> Node:
        return self.__mul__(other)

    def __matmul__(self, other: Node) -> Node:
        if not isinstance(other, Node):
            raise TypeError("MatMul expects another Node")
        out = Node(np.dot(self.data, other.data), parents=(self, other), op_name="matmul")

        def _backward() -> None:
            # d(A @ B)/dA = grad @ B^T, d(A @ B)/dB = A^T @ grad
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        return out

    def relu(self) -> Node:
        out = Node(np.maximum(0.0, self.data), parents=(self,), op_name="relu")

        def _backward() -> None:
            self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> Node:
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -50.0, 50.0)))
        out = Node(s, parents=(self,), op_name="sigmoid")

        def _backward() -> None:
            self.grad += out.grad * (s * (1.0 - s))

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Topological reverse-mode automatic differentiation starting from this scalar root."""
        topo: list[Node] = []
        visited: set[int] = set()

        def _build_topo(v: Node) -> None:
            v_id = id(v)
            if v_id not in visited:
                visited.add(v_id)
                for parent in v.parents:
                    _build_topo(parent)
                topo.append(v)

        _build_topo(self)

        # Initialize root adjoint: dL / dL = 1
        self.grad = np.ones_like(self.data)

        # Traverse in reverse topological order
        for node in reversed(topo):
            node._backward()


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum out dimensions added by broadcasting to match target tensor shape."""
    if grad.shape == target_shape:
        return grad
    if len(target_shape) == 0:
        return np.array(grad.sum(), dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class ComputationalGraph:
    """Helper utility for inspecting execution graphs and gradients."""

    @staticmethod
    def get_topological_sort(root: Node) -> list[Node]:
        topo: list[Node] = []
        visited: set[int] = set()

        def _visit(n: Node) -> None:
            n_id = id(n)
            if n_id not in visited:
                visited.add(n_id)
                for p in n.parents:
                    _visit(p)
                topo.append(n)

        _visit(root)
        return topo
