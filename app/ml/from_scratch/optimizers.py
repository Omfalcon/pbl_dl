"""Gradient-Based Optimization Algorithms and Learning Rate Schedulers from Scratch.

Mathematical Foundations:
1. Standard Stochastic Gradient Descent (SGD):
   g_t = grad + weight_decay * theta_{t-1}
   theta_t = theta_{t-1} - lr * g_t

2. SGD with Momentum:
   v_t = momentum * v_{t-1} + g_t
   theta_t = theta_{t-1} - lr * v_t
   Accelerates through flat directions and dampens oscillations.

3. RMSProp (Root Mean Square Propagation):
   s_t = decay_rate * s_{t-1} + (1 - decay_rate) * (g_t ** 2)
   theta_t = theta_{t-1} - (lr / sqrt(s_t + eps)) * g_t
   Adapts coordinate-wise step sizes based on recent squared gradient magnitudes.

4. Adam (Adaptive Moment Estimation):
   m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t         (1st moment / mean)
   v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)  (2nd moment / variance)
   m_hat_t = m_t / (1 - beta_1 ** t)                   (bias correction)
   v_hat_t = v_t / (1 - beta_2 ** t)                   (bias correction)
   theta_t = theta_{t-1} - (lr / (sqrt(v_hat_t) + eps)) * m_hat_t

5. Learning Rate Schedulers:
   - Step Decay: lr_t = lr_0 * gamma ** floor(t / step_size)
   - Exponential Decay: lr_t = lr_0 * gamma ** t
   - Cosine Annealing: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t * pi / T_max))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np


class LRScheduler(ABC):
    """Abstract Base Class for Learning Rate Schedulers."""

    def __init__(self, initial_lr: float) -> None:
        if initial_lr <= 0.0 or not math.isfinite(initial_lr):
            raise ValueError(f"Initial learning rate must be positive and finite; got {initial_lr}")
        self.initial_lr = float(initial_lr)
        self.current_lr = float(initial_lr)
        self.step_count = 0

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Calculate learning rate at given step count."""
        pass

    def step(self) -> float:
        """Advance scheduler by one step and return updated learning rate."""
        self.step_count += 1
        self.current_lr = self.get_lr(self.step_count)
        return self.current_lr

    def set_lr(self, new_lr: float) -> None:
        """Manually update learning rate and re-anchor schedule baseline.

        Updates current_lr and initial_lr to new_lr and resets step_count to 0,
        ensuring future scheduler steps decay from the updated baseline rate.
        """
        if new_lr <= 0.0 or not math.isfinite(new_lr):
            raise ValueError(f"Learning rate must be positive and finite; got {new_lr}")
        self.initial_lr = float(new_lr)
        self.current_lr = float(new_lr)
        self.step_count = 0


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs/steps."""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(initial_lr)
        if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)):
            raise TypeError(f"step_size must be an integer; got {type(step_size).__name__}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive integer; got {step_size}")
        if not (0.0 < gamma <= 1.0) or not math.isfinite(gamma):
            raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def get_lr(self, step: int) -> float:
        exponent = step // self.step_size
        return float(self.initial_lr * (self.gamma ** exponent))


class ExponentialLR(LRScheduler):
    """Decays the learning rate exponentially by gamma each step."""

    def __init__(self, initial_lr: float, gamma: float = 0.95) -> None:
        super().__init__(initial_lr)
        if not (0.0 < gamma <= 1.0) or not math.isfinite(gamma):
            raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
        self.gamma = float(gamma)

    def get_lr(self, step: int) -> float:
        return float(self.initial_lr * (self.gamma ** step))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, initial_lr: float, t_max: int, eta_min: float = 0.0) -> None:
        super().__init__(initial_lr)
        if isinstance(t_max, bool) or not isinstance(t_max, (int, np.integer)):
            raise TypeError(f"t_max must be an integer; got {type(t_max).__name__}")
        if t_max <= 0:
            raise ValueError(f"t_max must be positive integer; got {t_max}")
        if eta_min < 0.0 or not math.isfinite(eta_min):
            raise ValueError(f"eta_min must be non-negative and finite; got {eta_min}")
        if eta_min > initial_lr:
            raise ValueError(f"eta_min ({eta_min}) cannot exceed initial_lr ({initial_lr})")
        self.t_max = int(t_max)
        self.eta_min = float(eta_min)

    def set_lr(self, new_lr: float) -> None:
        if new_lr <= 0.0 or not math.isfinite(new_lr):
            raise ValueError(f"Learning rate must be positive and finite; got {new_lr}")
        if self.eta_min > new_lr:
            raise ValueError(
                f"Cannot set learning rate ({new_lr}) below CosineAnnealingLR eta_min ({self.eta_min})"
            )
        super().set_lr(new_lr)

    def get_lr(self, step: int) -> float:
        if step >= self.t_max:
            return float(self.eta_min)
        cos_factor = 1.0 + math.cos(math.pi * step / self.t_max)
        return float(self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * cos_factor)


class Optimizer(ABC):
    """Abstract Base Class for parameter optimizers."""

    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        scheduler: LRScheduler | None = None,
    ) -> None:
        if lr <= 0.0 or not math.isfinite(lr):
            raise ValueError(f"Learning rate must be positive and finite; got {lr}")
        if weight_decay < 0.0 or not math.isfinite(weight_decay):
            raise ValueError(f"Weight decay must be non-negative and finite; got {weight_decay}")

        self._lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.scheduler = scheduler
        self.t = 0  # iteration counter

        # Keep initial optimizer and scheduler learning rates synchronized
        if self.scheduler is not None:
            self._lr = self.scheduler.current_lr

    @property
    def lr(self) -> float:
        """Synchronized learning rate."""
        if self.scheduler is not None:
            return self.scheduler.current_lr
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"Learning rate must be positive and finite; got {value}")
        self._lr = float(value)
        if self.scheduler is not None:
            self.scheduler.set_lr(self._lr)

    def update_lr(self) -> float:
        """Advance scheduler and synchronize learning rate."""
        if self.scheduler is not None:
            self._lr = self.scheduler.step()
        return self.lr

    def validate_inputs(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Validate parameter and gradient list lengths, shapes, and finite values.

        Prevents silent parameter skips through zip truncation and ensures
        all inputs are strictly finite before any mutation occurs.
        """
        if len(params) != len(grads):
            raise ValueError(
                f"Parameter and gradient list lengths must match; got {len(params)} parameters and {len(grads)} gradients"
            )
        for idx, (p, g) in enumerate(zip(params, grads)):
            if not isinstance(p, np.ndarray) or not isinstance(g, np.ndarray):
                raise TypeError(f"Parameters and gradients must be numpy arrays; got {type(p)} and {type(g)} at index {idx}")
            if p.shape != g.shape:
                raise ValueError(
                    f"Shape mismatch at index {idx}: parameter shape {p.shape} != gradient shape {g.shape}"
                )
            if not np.all(np.isfinite(p)):
                raise ValueError(
                    f"Non-finite parameter (NaN or Inf) detected at parameter index {idx}"
                )
            if not np.all(np.isfinite(g)):
                raise ValueError(
                    f"Non-finite gradient (NaN or Inf) detected at parameter index {idx}"
                )

    @abstractmethod
    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply parameter update using gradients in-place."""
        pass


class SGD(Optimizer):
    """Standard Stochastic Gradient Descent with optional Weight Decay."""

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.validate_inputs(params, grads)
        self.t += 1

        for p, g in zip(params, grads):
            grad = g
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p
            p -= self.lr * grad

        self.update_lr()


class SGDMomentum(Optimizer):
    """Stochastic Gradient Descent with Polyak Momentum."""

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: LRScheduler | None = None,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, scheduler=scheduler)
        if not (0.0 <= momentum < 1.0) or not math.isfinite(momentum):
            raise ValueError(f"Momentum must be in [0, 1); got {momentum}")
        self.momentum = float(momentum)
        self.velocities: list[np.ndarray] = []

    def _validate_state(self, params: list[np.ndarray]) -> None:
        """Validate optimizer velocity state consistency when parameter structures change."""
        if not self.velocities:
            self.velocities = [np.zeros_like(p, dtype=np.float64) for p in params]
            return
        if len(self.velocities) != len(params):
            raise ValueError(
                f"Optimizer state inconsistent: expected {len(self.velocities)} velocities, got {len(params)} parameters"
            )
        for idx, (v, p) in enumerate(zip(self.velocities, params)):
            if v.shape != p.shape:
                raise ValueError(
                    f"Optimizer state shape mismatch at index {idx}: velocity shape {v.shape} != param shape {p.shape}"
                )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.validate_inputs(params, grads)
        self._validate_state(params)
        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            grad = g
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p

            # v_t = momentum * v_{t-1} + g_t
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            # p_t = p_{t-1} - lr * v_t
            p -= self.lr * self.velocities[i]

        self.update_lr()


class RMSProp(Optimizer):
    """Root Mean Square Propagation (RMSProp) Optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        decay_rate: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        scheduler: LRScheduler | None = None,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, scheduler=scheduler)
        if not (0.0 <= decay_rate < 1.0) or not math.isfinite(decay_rate):
            raise ValueError(f"Decay rate must be in [0, 1); got {decay_rate}")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError(f"Epsilon must be positive and finite; got {eps}")
        self.decay_rate = float(decay_rate)
        self.eps = float(eps)
        self.squared_grads: list[np.ndarray] = []

    def _validate_state(self, params: list[np.ndarray]) -> None:
        """Validate optimizer squared-gradient state consistency when parameter structures change."""
        if not self.squared_grads:
            self.squared_grads = [np.zeros_like(p, dtype=np.float64) for p in params]
            return
        if len(self.squared_grads) != len(params):
            raise ValueError(
                f"Optimizer state inconsistent: expected {len(self.squared_grads)} states, got {len(params)} parameters"
            )
        for idx, (s, p) in enumerate(zip(self.squared_grads, params)):
            if s.shape != p.shape:
                raise ValueError(
                    f"Optimizer state shape mismatch at index {idx}: squared_grad shape {s.shape} != param shape {p.shape}"
                )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.validate_inputs(params, grads)
        self._validate_state(params)
        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            grad = g
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p

            # s_t = decay * s_{t-1} + (1 - decay) * g_t^2
            self.squared_grads[i] = (
                self.decay_rate * self.squared_grads[i] + (1.0 - self.decay_rate) * (grad ** 2)
            )
            # p_t = p_{t-1} - (lr / sqrt(s_t + eps)) * grad
            p -= (self.lr / np.sqrt(self.squared_grads[i] + self.eps)) * grad

        self.update_lr()


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) Optimizer with Bias Correction."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        scheduler: LRScheduler | None = None,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, scheduler=scheduler)
        if not (0.0 <= beta1 < 1.0) or not math.isfinite(beta1):
            raise ValueError(f"beta1 must be in [0, 1); got {beta1}")
        if not (0.0 <= beta2 < 1.0) or not math.isfinite(beta2):
            raise ValueError(f"beta2 must be in [0, 1); got {beta2}")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError(f"Epsilon must be positive and finite; got {eps}")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: list[np.ndarray] = []  # 1st moment
        self.v: list[np.ndarray] = []  # 2nd moment

    def _validate_state(self, params: list[np.ndarray]) -> None:
        """Validate optimizer moment state consistency when parameter structures change."""
        if not self.m:
            self.m = [np.zeros_like(p, dtype=np.float64) for p in params]
            self.v = [np.zeros_like(p, dtype=np.float64) for p in params]
            return
        if len(self.m) != len(params) or len(self.v) != len(params):
            raise ValueError(
                f"Optimizer state inconsistent: expected {len(self.m)} states, got {len(params)} parameters"
            )
        for idx, (m, v, p) in enumerate(zip(self.m, self.v, params)):
            if m.shape != p.shape or v.shape != p.shape:
                raise ValueError(
                    f"Optimizer state shape mismatch at index {idx}: moment shape {m.shape} != param shape {p.shape}"
                )

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.validate_inputs(params, grads)
        self._validate_state(params)
        self.t += 1

        # Bias correction factors
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for i, (p, g) in enumerate(zip(params, grads)):
            grad = g
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p

            # Update biased 1st moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            # Update biased 2nd raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # Parameter update
            p -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

        self.update_lr()
