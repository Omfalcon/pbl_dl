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


class TestOptimizerHardeningAndEdgeCases:
    """Verifies strict input validation, shape checks, non-finite gradient rejection, and state consistency."""

    def test_invalid_hyperparameters_raise_value_errors(self):
        # Invalid Learning Rates
        with pytest.raises(ValueError, match="positive and finite"):
            SGD(lr=0.0)
        with pytest.raises(ValueError, match="positive and finite"):
            SGD(lr=-0.05)

        # Invalid Weight Decay
        with pytest.raises(ValueError, match="non-negative"):
            SGD(lr=0.01, weight_decay=-0.01)

        # Invalid Momentum
        with pytest.raises(ValueError, match="in \\[0, 1\\)"):
            SGDMomentum(lr=0.01, momentum=1.0)
        with pytest.raises(ValueError, match="in \\[0, 1\\)"):
            SGDMomentum(lr=0.01, momentum=-0.1)

        # Invalid RMSProp decay and eps
        with pytest.raises(ValueError, match="in \\[0, 1\\)"):
            RMSProp(lr=0.01, decay_rate=1.0)
        with pytest.raises(ValueError, match="positive and finite"):
            RMSProp(lr=0.01, eps=0.0)

        # Invalid Adam beta1, beta2 and eps
        with pytest.raises(ValueError, match="beta1 must be in \\[0, 1\\)"):
            Adam(lr=0.001, beta1=1.0)
        with pytest.raises(ValueError, match="beta2 must be in \\[0, 1\\)"):
            Adam(lr=0.001, beta2=-0.01)
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            Adam(lr=0.001, eps=-1e-5)

        # Invalid Schedulers
        with pytest.raises(ValueError, match="step_size must be positive"):
            StepLR(initial_lr=0.1, step_size=0)
        with pytest.raises(ValueError, match="step_size must be positive"):
            StepLR(initial_lr=0.1, step_size=-2)
        with pytest.raises(TypeError, match="step_size must be an integer"):
            StepLR(initial_lr=0.1, step_size=2.5)
        with pytest.raises(TypeError, match="step_size must be an integer"):
            StepLR(initial_lr=0.1, step_size=True)
        with pytest.raises(ValueError, match="gamma must be in \\(0, 1\\]"):
            ExponentialLR(initial_lr=0.1, gamma=0.0)
        with pytest.raises(ValueError, match="t_max must be positive"):
            CosineAnnealingLR(initial_lr=0.1, t_max=0)
        with pytest.raises(ValueError, match="t_max must be positive"):
            CosineAnnealingLR(initial_lr=0.1, t_max=-5)
        with pytest.raises(TypeError, match="t_max must be an integer"):
            CosineAnnealingLR(initial_lr=0.1, t_max=10.5)
        with pytest.raises(TypeError, match="t_max must be an integer"):
            CosineAnnealingLR(initial_lr=0.1, t_max=False)
        with pytest.raises(ValueError, match="eta_min must be non-negative"):
            CosineAnnealingLR(initial_lr=0.1, t_max=10, eta_min=-0.01)

    def test_parameter_gradient_length_mismatch_rejected(self):
        opt = SGD(lr=0.01)
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        g1 = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="Parameter and gradient list lengths must match"):
            opt.step([p1, p2], [g1])

    def test_tensor_shape_mismatch_rejected(self):
        opt = SGD(lr=0.01)
        p = np.zeros((2, 3))
        g = np.zeros((3, 2))  # Transposed shape

        with pytest.raises(ValueError, match="Shape mismatch at index 0"):
            opt.step([p], [g])

    def test_non_finite_gradients_rejected(self):
        opt = SGD(lr=0.01)
        p = np.array([1.0, 2.0])

        # NaN rejection
        with pytest.raises(ValueError, match="Non-finite gradient"):
            opt.step([p], [np.array([np.nan, 0.5])])

        # Inf rejection
        with pytest.raises(ValueError, match="Non-finite gradient"):
            opt.step([p], [np.array([np.inf, 0.5])])

    def test_non_finite_parameters_rejected(self):
        # NaN parameter rejection across all 4 optimizers
        for opt_cls in [SGD, SGDMomentum, RMSProp, Adam]:
            opt = opt_cls()
            with pytest.raises(ValueError, match="Non-finite parameter"):
                opt.step([np.array([1.0, np.nan])], [np.array([0.1, 0.2])])

            with pytest.raises(ValueError, match="Non-finite parameter"):
                opt.step([np.array([np.inf, 2.0])], [np.array([0.1, 0.2])])

            with pytest.raises(ValueError, match="Non-finite parameter"):
                opt.step([np.array([-np.inf, 2.0])], [np.array([0.1, 0.2])])

    def test_non_floating_parameters_rejected(self):
        # Reject integer or non-floating-point parameter arrays across all 4 optimizers
        for opt_cls in [SGD, SGDMomentum, RMSProp, Adam]:
            opt = opt_cls()
            int_param = np.array([1, 2], dtype=np.int64)
            grad = np.array([0.1, 0.2], dtype=np.float64)

            with pytest.raises(TypeError, match="must have floating-point dtype"):
                opt.step([int_param], [grad])

            assert opt.t == 0
            assert np.array_equal(int_param, [1, 2])

            # Also check bool array rejection
            bool_param = np.array([True, False])
            with pytest.raises(TypeError, match="must have floating-point dtype"):
                opt.step([bool_param], [grad])

    def test_read_only_parameters_rejected(self):
        # Reject read-only parameter arrays across all 4 optimizers
        for opt_cls in [SGD, SGDMomentum, RMSProp, Adam]:
            opt = opt_cls()
            ro_param = np.array([1.0, 2.0], dtype=np.float64)
            ro_param.flags.writeable = False
            grad = np.array([0.1, 0.2], dtype=np.float64)

            with pytest.raises(ValueError, match="is read-only and cannot be updated in-place"):
                opt.step([ro_param], [grad])

            assert opt.t == 0

    def test_invalid_inputs_do_not_partially_mutate_state_or_params(self):
        p1 = np.array([10.0, 20.0], dtype=np.float64)
        p1_copy = p1.copy()
        p2_bad = np.array([30.0, np.nan], dtype=np.float64)
        g1 = np.array([1.0, 1.0], dtype=np.float64)
        g2 = np.array([1.0, 1.0], dtype=np.float64)

        opt = SGDMomentum(lr=0.1, momentum=0.9)

        # Finiteness failure on parameter 2 must not touch parameter 1 or state
        with pytest.raises(ValueError, match="Non-finite parameter"):
            opt.step([p1, p2_bad], [g1, g2])

        assert np.array_equal(p1, p1_copy), "Parameter p1 was mutated despite validation error"
        assert opt.t == 0, "Optimizer iteration counter incremented on failed step"
        assert len(opt.velocities) == 0, "Optimizer state created on failed step"

        # Shape mismatch on parameter 2 must also not mutate parameter 1
        p2_wrong_shape = np.array([30.0], dtype=np.float64)
        g2_wrong_shape = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Shape mismatch at index 1"):
            opt.step([p1, p2_wrong_shape], [g1, g2_wrong_shape])

        assert np.array_equal(p1, p1_copy), "Parameter p1 was mutated on shape mismatch"
        assert opt.t == 0

        # Integer parameter at index 1 must also not mutate parameter 1
        p2_int = np.array([30, 40], dtype=np.int64)
        with pytest.raises(TypeError, match="must have floating-point dtype"):
            opt.step([p1, p2_int], [g1, g2])

        assert np.array_equal(p1, p1_copy), "Parameter p1 was mutated on int dtype rejection"
        assert opt.t == 0

        # Read-only parameter at index 1 must also not mutate parameter 1
        p2_ro = np.array([30.0, 40.0], dtype=np.float64)
        p2_ro.flags.writeable = False
        with pytest.raises(ValueError, match="is read-only"):
            opt.step([p1, p2_ro], [g1, g2])

        assert np.array_equal(p1, p1_copy), "Parameter p1 was mutated on read-only rejection"
        assert opt.t == 0

    def test_optimizer_state_consistency_on_param_structure_change(self):
        p1 = np.array([1.0, 2.0])
        g1 = np.array([0.1, 0.2])

        # 1. SGDMomentum state consistency check
        opt_mom = SGDMomentum(lr=0.01)
        opt_mom.step([p1], [g1])
        assert len(opt_mom.velocities) == 1

        # Changing param shape after init must be caught
        p_wrong_shape = np.array([1.0, 2.0, 3.0])
        g_wrong_shape = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Optimizer state shape mismatch"):
            opt_mom.step([p_wrong_shape], [g_wrong_shape])

        # Changing number of params after init must be caught
        with pytest.raises(ValueError, match="Optimizer state inconsistent"):
            opt_mom.step([p1, p1], [g1, g1])

        # 2. Adam state consistency check
        opt_adam = Adam(lr=0.001)
        opt_adam.step([p1], [g1])
        with pytest.raises(ValueError, match="Optimizer state shape mismatch"):
            opt_adam.step([p_wrong_shape], [g_wrong_shape])

    def test_scheduler_synchronization(self):
        sched = StepLR(initial_lr=0.1, step_size=2, gamma=0.5)
        opt = SGD(lr=0.1, scheduler=sched)

        assert opt.lr == 0.1
        p = np.array([1.0])
        g = np.array([0.5])

        # Step 1 -> count=1, lr remains 0.1
        opt.step([p], [g])
        assert opt.lr == 0.1
        assert sched.current_lr == 0.1

        # Step 2 -> count=2, step_size reached -> lr decays to 0.05
        opt.step([p], [g])
        assert opt.lr == 0.05
        assert sched.current_lr == 0.05

        # Manual setter synchronization & re-anchoring regression test
        opt.lr = 0.02
        assert opt.lr == 0.02
        assert sched.current_lr == 0.02
        assert sched.initial_lr == 0.02
        assert sched.step_count == 0

        # Next step must use the new 0.02 rate and NOT revert to old rate (0.1 or 0.05)
        p_val_before = p.copy()
        opt.step([p], [g])
        # param update = p - 0.02 * 0.5 = p - 0.01
        assert np.isclose(p[0], p_val_before[0] - 0.02 * 0.5)
        # Sched step_count is now 1 (step_size=2), so current_lr remains 0.02
        assert opt.lr == 0.02
        assert sched.current_lr == 0.02

        # Step 2 of re-anchored schedule: step_count reaches 2 -> decays to 0.02 * 0.5 = 0.01
        opt.step([p], [g])
        assert opt.lr == 0.01
        assert sched.current_lr == 0.01

    def test_cosine_annealing_manual_lr_rebasing(self):
        sched = CosineAnnealingLR(initial_lr=0.1, t_max=10, eta_min=0.001)
        opt = SGD(lr=0.1, scheduler=sched)

        p = np.array([1.0])
        g = np.array([0.2])
        opt.step([p], [g])

        # Manually reassign lr to 0.04
        opt.lr = 0.04
        assert opt.lr == 0.04
        assert sched.initial_lr == 0.04
        assert sched.current_lr == 0.04
        assert sched.step_count == 0

        # Cannot set lr below eta_min
        with pytest.raises(ValueError, match="Cannot set learning rate .* below .* eta_min"):
            opt.lr = 0.0005
