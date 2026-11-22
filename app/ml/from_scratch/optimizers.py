"""Gradient-Based Optimization Algorithms and Learning Rate Schedulers from Scratch.

Mathematical Foundations:
1. Standard Stochastic Gradient Descent (SGD):
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
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Calculate learning rate at step."""
        pass

    def step(self) -> float:
        """Advance scheduler by one step and return updated learning rate."""
        self.step_count += 1
        self.current_lr = self.get_lr(self.step_count)
        return self.current_lr


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs/steps."""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: int) -> float:
        exponent = step // self.step_size
        return float(self.initial_lr * (self.gamma ** exponent))


class ExponentialLR(LRScheduler):
    """Decays the learning rate exponentially by gamma each step."""

    def __init__(self, initial_lr: float, gamma: float = 0.95) -> None:
        super().__init__(initial_lr)
        self.gamma = gamma

    def get_lr(self, step: int) -> float:
        return float(self.initial_lr * (self.gamma ** step))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, initial_lr: float, t_max: int, eta_min: float = 0.0) -> None:
        super().__init__(initial_lr)
        self.t_max = t_max
        self.eta_min = eta_min

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
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.t = 0  # iteration counter

    def update_lr(self) -> float:
        if self.scheduler is not None:
            self.lr = self.scheduler.step()
        return self.lr

    @abstractmethod
    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply parameter update using gradients in-place."""
        pass


class SGD(Optimizer):
    """Standard Stochastic Gradient Descent with optional Weight Decay."""

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
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
        self.momentum = momentum
        self.velocities: list[np.ndarray] = []

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        if not self.velocities:
            self.velocities = [np.zeros_like(p) for p in params]

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
        self.decay_rate = decay_rate
        self.eps = eps
        self.squared_grads: list[np.ndarray] = []

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        if not self.squared_grads:
            self.squared_grads = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            grad = g
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p

            # s_t = decay * s_{t-1} + (1 - decay) * g_t^2
            self.squared_grads[i] = (
                self.decay_rate * self.squared_grads[i] + (1.0 - self.decay_rate) * (grad ** 2)
            )
            # p_t = p_{t-1} - (lr / sqrt(s_t + eps)) * g_t
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: list[np.ndarray] = []  # 1st moment
        self.v: list[np.ndarray] = []  # 2nd moment

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

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
