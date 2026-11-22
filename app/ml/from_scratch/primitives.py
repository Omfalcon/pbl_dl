"""Foundational Linear Algebra Primitives: Vectors, Matrices, and Tensors.

Mathematical Background:
1. Dot Product:
   For vectors u, v in R^n:
   dot(u, v) = sum_{i=1}^n u_i * v_i = ||u|| * ||v|| * cos(theta)

2. Vector p-Norm (Euclidean / L2):
   ||v||_2 = sqrt(sum_{i=1}^n v_i^2)

3. Matrix Multiplication:
   For A in R^{m x k} and B in R^{k x n}:
   C = A @ B in R^{m x n}, where C_{ij} = sum_{p=1}^k A_{ip} * B_{pj}

4. Cosine Similarity:
   cos_sim(u, v) = (u . v) / (||u||_2 * ||v||_2)
"""

from __future__ import annotations

import numpy as np


def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Compute dot product of two 1D vectors u and v.

    Math:
        u . v = sum_{i=1}^n (u_i * v_i)

    Args:
        u: 1D array of shape (N,)
        v: 1D array of shape (N,)

    Returns:
        Scalar inner product value.
    """
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(f"dot_product expects 1D vectors; got u.ndim={u.ndim}, v.ndim={v.ndim}")
    if u.shape[0] != v.shape[0]:
        raise ValueError(f"Vector length mismatch: u={u.shape[0]} != v={v.shape[0]}")

    # Manual summation to demonstrate fundamental mechanics
    total = 0.0
    for i in range(len(u)):
        total += float(u[i] * v[i])
    return total


def vector_norm(v: np.ndarray, p: int = 2) -> float:
    """Compute L_p vector norm.

    Math:
        ||v||_p = (sum_{i=1}^n |v_i|^p)^(1/p)

    Default: L2 Euclidean norm (p=2).
    """
    if v.ndim != 1:
        raise ValueError(f"vector_norm expects 1D vector; got v.ndim={v.ndim}")
    if p < 1:
        raise ValueError(f"Order p must be >= 1; got {p}")

    sum_p = sum(abs(x) ** p for x in v)
    return float(sum_p ** (1.0 / p))


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix multiplication of 2D matrices A (M x K) and B (K x N).

    Math:
        C_{ij} = sum_{p=1}^k A_{ip} * B_{pj}

    Returns:
        C in R^{M x N}
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matrix_multiply expects 2D matrices; got a.shape={a.shape}, b.shape={b.shape}")
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError(f"Inner matrix dimensions must agree: A is ({m}x{k_a}), B is ({k_b}x{n})")

    c = np.zeros((m, n), dtype=np.float64)
    # Fundamental definition through vector dot products of rows and columns
    for i in range(m):
        for j in range(n):
            c[i, j] = sum(a[i, p] * b[p, j] for p in range(k_a))
    return c


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-10) -> float:
    """Compute cosine similarity between two nonzero vectors.

    Math:
        S_C(u, v) = (u . v) / (max(||u||_2 * ||v||_2, eps))
    """
    norm_u = vector_norm(u, p=2)
    norm_v = vector_norm(v, p=2)
    denom = max(norm_u * norm_v, eps)
    return dot_product(u, v) / denom


def validate_tensor_dimensions(
    tensor: np.ndarray,
    expected_rank: int,
    expected_dim: tuple[int | None, ...],
) -> bool:
    """Verify rank and dimensions of input tensor."""
    if tensor.ndim != expected_rank:
        return False
    if len(expected_dim) != expected_rank:
        return False
    for actual, exp in zip(tensor.shape, expected_dim):
        if exp is not None and actual != exp:
            return False
    return True
