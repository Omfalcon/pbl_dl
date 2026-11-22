"""Unit Tests for Phase 4: Optimizers and Learning Rate Schedulers from Scratch."""

from __future__ import annotations

import math
import numpy as np
import pytest

from app.ml.from_scratch.activations import Sigmoid
from app.ml.from_scratch.layers import Dense
from app.ml.from_scratch.losses import BinaryCrossEntropy, MeanSquaredError
from app.ml.from_scratch.network import Sequential
from app.ml.from_scratch.optimizers import (
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    RMSProp,
    SGD,
    SGDMomentum,
    StepLR,
)


class TestOptimizerHandCalculated:
    """Verifies parameter updates against deterministic hand-calculated closed-form values."""

    def test_sgd_hand_calculated_step(self):
        # theta_0 = [1.0, 2.0], g = [0.5, -0.5], lr = 0.1
        # theta_1 = [1.0 - 0.05, 2.0 - (-0.05)] = [0.95, 2.05]
        param = np.array([1.0, 2.0], dtype=np.float64)
        grad = np.array([0.5, -0.5], dtype=np.float64)

        opt = SGD(lr=0.1)
        opt.step([param], [grad])

        assert np.allclose(param, [0.95, 2.05])

    def test_sgd_momentum_hand_calculated_steps(self):
        # theta_0 = 1.0, v_0 = 0, g_1 = 0.5, momentum = 0.9, lr = 0.1
        # Step 1: v_1 = 0.9 * 0 + 0.5 = 0.5 -> theta_1 = 1.0 - 0.1 * 0.5 = 0.95
        # Step 2: g_2 = 0.5 -> v_2 = 0.9 * 0.5 + 0.5 = 0.95 -> theta_2 = 0.95 - 0.1 * 0.95 = 0.855
        param = np.array([1.0], dtype=np.float64)
        opt = SGDMomentum(lr=0.1, momentum=0.9)

        opt.step([param], [np.array([0.5])])
        assert np.isclose(param[0], 0.95)

        opt.step([param], [np.array([0.5])])
        assert np.isclose(param[0], 0.855)

    def test_rmsprop_hand_calculated_step(self):
        # theta_0 = 1.0, s_0 = 0, g_1 = 2.0, decay = 0.9, lr = 0.01, eps = 1e-8
        # s_1 = 0.9 * 0 + 0.1 * (2.0^2) = 0.4
        # theta_1 = 1.0 - (0.01 / sqrt(0.4 + 1e-8)) * 2.0 = 1.0 - 0.02 / 0.6324555 = 0.96837722
        param = np.array([1.0], dtype=np.float64)
        opt = RMSProp(lr=0.01, decay_rate=0.9, eps=1e-8)

        opt.step([param], [np.array([2.0])])
        expected = 1.0 - (0.01 / math.sqrt(0.4 + 1e-8)) * 2.0
        assert np.isclose(param[0], expected)

    def test_adam_hand_calculated_step(self):
        # theta_0 = 1.0, g_1 = 1.0, beta1 = 0.9, beta2 = 0.999, lr = 0.001, eps = 1e-8
        # Step 1 (t = 1):
        # m_1 = 0.9 * 0 + 0.1 * 1.0 = 0.1
        # v_1 = 0.999 * 0 + 0.001 * 1.0^2 = 0.001
        # m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
        # v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
        # theta_1 = 1.0 - (0.001 / (sqrt(1.0) + 1e-8)) * 1.0 = 1.0 - 0.001 = 0.999
        param = np.array([1.0], dtype=np.float64)
        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

        opt.step([param], [np.array([1.0])])
        assert np.isclose(param[0], 0.999)


class TestLRSchedulers:
    """Verifies StepLR, ExponentialLR, and CosineAnnealingLR scheduling curves."""

    def test_step_lr(self):
        sched = StepLR(initial_lr=0.1, step_size=5, gamma=0.5)
        # steps 1..4: 0.1
        for _ in range(4):
            lr = sched.step()
            assert np.isclose(lr, 0.1)
        # step 5: 0.1 * 0.5 = 0.05
        assert np.isclose(sched.step(), 0.05)

    def test_exponential_lr(self):
        sched = ExponentialLR(initial_lr=0.1, gamma=0.9)
        lr1 = sched.step()
        assert np.isclose(lr1, 0.09)
        lr2 = sched.step()
        assert np.isclose(lr2, 0.081)

    def test_cosine_annealing_lr(self):
        sched = CosineAnnealingLR(initial_lr=0.1, t_max=10, eta_min=0.0)
        # At t=0 -> 0.1
        # At midpoint t=5: 0.5 * 0.1 * (1 + cos(pi/2)) = 0.05
        for _ in range(5):
            lr = sched.step()
        assert np.isclose(lr, 0.05)

        # At t=10: eta_min = 0.0
        for _ in range(5):
            lr = sched.step()
        assert np.isclose(lr, 0.0)


class TestConvergenceSyntheticTasks:
    """Verifies that all 4 custom optimizers successfully minimize loss on synthetic tasks."""

    def test_linear_regression_convergence(self):
        # Target function: y = 3 * x_1 - 2 * x_2 + 1.5
        np.random.seed(42)
        X = np.random.uniform(-2, 2, size=(100, 2))
        y = (3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5).reshape(-1, 1)

        loss_fn = MeanSquaredError()

        # Test each optimizer solves the regression problem
        for opt_class, kwargs in [
            (SGD, {"lr": 0.05}),
            (SGDMomentum, {"lr": 0.02, "momentum": 0.9}),
            (RMSProp, {"lr": 0.05}),
            (Adam, {"lr": 0.1}),
        ]:
            model = Sequential([Dense(in_features=2, out_features=1, random_seed=42)])
            opt = opt_class(**kwargs)

            initial_loss = loss_fn.forward(y, model.forward(X))
            for epoch in range(150):
                preds = model.forward(X)
                grad_loss = loss_fn.backward(y, preds)
                model.backward(grad_loss)
                opt.step(model.get_parameters(), model.get_gradients())

            final_loss = loss_fn.forward(y, model.forward(X))
            # Every optimizer must significantly reduce loss
            assert final_loss < 0.05, f"{opt_class.__name__} failed to converge: final loss {final_loss}"
            assert final_loss < initial_loss * 0.01

    def test_binary_classification_convergence(self):
        # 2D linearly separable blobs
        np.random.seed(42)
        X_neg = np.random.randn(50, 2) + np.array([-2.0, -2.0])
        X_pos = np.random.randn(50, 2) + np.array([2.0, 2.0])
        X = np.vstack([X_neg, X_pos])
        y = np.vstack([np.zeros((50, 1)), np.ones((50, 1))])

        loss_fn = BinaryCrossEntropy()

        for opt_class, kwargs in [
            (SGD, {"lr": 0.1}),
            (SGDMomentum, {"lr": 0.05, "momentum": 0.9}),
            (RMSProp, {"lr": 0.02}),
            (Adam, {"lr": 0.05}),
        ]:
            model = Sequential([
                Dense(in_features=2, out_features=4, random_seed=42),
                Sigmoid(),
                Dense(in_features=4, out_features=1, random_seed=42),
                Sigmoid(),
            ])
            opt = opt_class(**kwargs)

            for epoch in range(100):
                preds = model.forward(X)
                grad_loss = loss_fn.backward(y, preds)
                model.backward(grad_loss)
                opt.step(model.get_parameters(), model.get_gradients())

            final_preds = model.forward(X)
            accuracy = float(np.mean((final_preds > 0.5) == y))
            assert accuracy >= 0.95, f"{opt_class.__name__} classification accuracy too low: {accuracy}"
