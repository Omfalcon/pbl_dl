"""Numerical Gradient Checking Using Two-Sided Finite Differences.

Mathematical Foundations:
For any scalar objective function J(theta), the two-sided central difference approximation is:
dJ / d(theta_i) = [ J(theta_1, ..., theta_i + eps, ...) - J(theta_1, ..., theta_i - eps, ...) ] / (2 * eps) + O(eps^2)

The truncation error is quadratic O(eps^2), which is vastly superior to single-sided forward differences O(eps).

Relative Error Metric:
relative_error = ||grad_analytical - grad_numerical||_2 / max(||grad_analytical||_2 + ||grad_numerical||_2, 1e-8)

Interpretation Guidelines:
- relative_error < 1e-7: Mathematical derivative is exceptionally precise.
- 1e-7 <= relative_error < 1e-5: Derivative is correct (slight floating-point rounding).
- relative_error >= 1e-4: Potential bug in analytical backpropagation or kink in activation (e.g. ReLU at 0).
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def numerical_gradient(
    func: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient of a scalar function func(theta) at theta via central differences."""
    grad = np.zeros_like(theta, dtype=np.float64)
    flat_theta = theta.reshape(-1)
    flat_grad = grad.reshape(-1)

    for i in range(len(flat_theta)):
        orig_val = flat_theta[i]

        # f(theta + eps)
        flat_theta[i] = orig_val + eps
        f_plus = func(theta)

        # f(theta - eps)
        flat_theta[i] = orig_val - eps
        f_minus = func(theta)

        # Restore original value
        flat_theta[i] = orig_val

        # Central difference: (f+ - f-) / (2 * eps)
        flat_grad[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad


def check_gradients(
    grad_analytical: np.ndarray,
    grad_numerical: np.ndarray,
    tolerance: float = 1e-5,
) -> tuple[bool, float]:
    """Compare analytical backpropagation gradients against numerical finite differences.

    Returns:
        (is_valid, relative_error)
    """
    if grad_analytical.shape != grad_numerical.shape:
        raise ValueError(f"Shape mismatch: analytical {grad_analytical.shape} != numerical {grad_numerical.shape}")

    diff_norm = float(np.linalg.norm(grad_analytical - grad_numerical))
    denom = float(np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical))
    rel_error = diff_norm / max(denom, 1e-8)

    is_valid = bool(rel_error <= tolerance)
    return is_valid, rel_error
