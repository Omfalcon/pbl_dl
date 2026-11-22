"""Unit Tests for Phase 3: Neural Network Mathematical Sandbox from Scratch."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.from_scratch.activations import ReLU, Sigmoid, Softmax, Tanh
from app.ml.from_scratch.computational_graph import ComputationalGraph, Node
from app.ml.from_scratch.gradient_check import check_gradients, numerical_gradient
from app.ml.from_scratch.layers import Dense
from app.ml.from_scratch.losses import BinaryCrossEntropy, MeanSquaredError
from app.ml.from_scratch.network import Sequential
from app.ml.from_scratch.neuron import Neuron
from app.ml.from_scratch.primitives import (
    cosine_similarity,
    dot_product,
    matrix_multiply,
    vector_norm,
)


class TestLinearAlgebraPrimitives:
    """Verifies fundamental vector and matrix operations."""

    def test_dot_product(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        res = dot_product(u, v)
        assert np.isclose(res, 32.0)

    def test_vector_norm(self):
        v = np.array([3.0, 4.0])
        assert np.isclose(vector_norm(v, p=2), 5.0)
        assert np.isclose(vector_norm(v, p=1), 7.0)

    def test_cosine_similarity(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert np.isclose(cosine_similarity(u, v), 0.0)

        w = np.array([2.0, 0.0])
        assert np.isclose(cosine_similarity(u, w), 1.0)

    def test_matrix_multiply(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        expected = np.dot(A, B)
        res = matrix_multiply(A, B)
        assert np.allclose(res, expected)


class TestActivationsAndDerivatives:
    """Verifies forward values and analytical derivatives for all core activations."""

    def test_relu_forward_and_backward(self):
        relu = ReLU()
        z = np.array([-2.0, 0.0, 3.0, 5.0])
        out = relu.forward(z)
        assert np.allclose(out, [0.0, 0.0, 3.0, 5.0])

        grad_out = np.array([1.0, 1.0, 1.0, 1.0])
        grad_in = relu.backward(grad_out)
        assert np.allclose(grad_in, [0.0, 0.0, 1.0, 1.0])

    def test_sigmoid_forward_and_numerical_stability(self):
        sig = Sigmoid()
        z = np.array([0.0, 100.0, -100.0])
        out = sig.forward(z)
        assert np.isclose(out[0], 0.5)
        assert np.isclose(out[1], 1.0)
        assert np.isclose(out[2], 0.0)

    def test_sigmoid_gradient_check(self):
        sig = Sigmoid()
        z = np.array([0.5, -1.2, 2.0])
        out = sig.forward(z)
        grad_out = np.array([1.0, 1.0, 1.0])
        grad_analytical = sig.backward(grad_out)

        # Numerical gradient
        grad_num = numerical_gradient(lambda z_in: np.sum(sig.forward(z_in)), z)
        valid, err = check_gradients(grad_analytical, grad_num)
        assert valid is True
        assert err < 1e-6

    def test_tanh_gradient_check(self):
        tanh = Tanh()
        z = np.array([0.3, -0.7, 1.5])
        out = tanh.forward(z)
        grad_out = np.array([1.0, 1.0, 1.0])
        grad_analytical = tanh.backward(grad_out)

        grad_num = numerical_gradient(lambda z_in: np.sum(tanh.forward(z_in)), z)
        valid, err = check_gradients(grad_analytical, grad_num)
        assert valid is True
        assert err < 1e-6

    def test_softmax_forward_and_properties(self):
        sm = Softmax()
        z = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        probs = sm.forward(z)
        # Sum of probabilities across classes must equal 1.0
        assert np.allclose(np.sum(probs, axis=-1), [1.0, 1.0])
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


class TestLossFunctions:
    """Verifies Mean Squared Error and Binary Cross Entropy losses and gradients."""

    def test_mse_loss_and_gradient(self):
        mse = MeanSquaredError()
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.8, 0.2, 0.9])
        loss = mse.forward(y_true, y_pred)
        assert loss > 0.0

        grad_analytical = mse.backward(y_true, y_pred)
        grad_num = numerical_gradient(lambda p: mse.forward(y_true, p), y_pred)

        valid, err = check_gradients(grad_analytical, grad_num)
        assert valid is True
        assert err < 1e-6

    def test_bce_loss_and_gradient(self):
        bce = BinaryCrossEntropy()
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.8, 0.2, 0.9])
        loss = bce.forward(y_true, y_pred)
        assert loss > 0.0

        grad_analytical = bce.backward(y_true, y_pred)
        grad_num = numerical_gradient(lambda p: bce.forward(y_true, p), y_pred)

        valid, err = check_gradients(grad_analytical, grad_num)
        assert valid is True
        assert err < 1e-5


class TestSingleNeuron:
    """Verifies isolated single artificial neuron forward and backward logic."""

    def test_neuron_forward_and_gradient(self):
        n = Neuron(input_dim=3, activation=Sigmoid(), random_seed=42)
        x = np.array([0.5, -0.2, 0.8])
        a = n.forward(x)
        assert 0.0 <= a <= 1.0

        grad_in = n.backward(1.0)
        assert grad_in.shape == (3,)

        # Check weights gradient via finite differences
        w_orig = n.weights.copy()

        def loss_func_w(w):
            n.weights = w.copy()
            return n.forward(x)

        grad_num_w = numerical_gradient(loss_func_w, w_orig)
        valid_w, err_w = check_gradients(n.grad_w, grad_num_w)
        assert valid_w is True
        assert err_w < 1e-6


class TestDenseLayerBackpropagation:
    """Verifies matrix calculus and gradients for Dense fully connected layers."""

    def test_dense_gradient_checking(self):
        dense = Dense(in_features=4, out_features=3, init_method="xavier", random_seed=42)
        X = np.random.randn(2, 4)

        # Forward
        Z = dense.forward(X)
        assert Z.shape == (2, 3)

        # Backward with dummy upstream gradient
        dZ = np.random.randn(2, 3)
        dX = dense.backward(dZ)
        assert dX.shape == (2, 4)

        # Check dW
        W_orig = dense.weights.copy()

        def obj_w(W):
            dense.weights = W
            z = dense.forward(X)
            return float(np.sum(z * dZ))

        grad_num_w = numerical_gradient(obj_w, W_orig)
        valid_w, err_w = check_gradients(dense.grad_w, grad_num_w)
        assert valid_w is True
        assert err_w < 1e-6

        # Check db
        b_orig = dense.biases.copy()

        def obj_b(b):
            dense.biases = b
            z = dense.forward(X)
            return float(np.sum(z * dZ))

        grad_num_b = numerical_gradient(obj_b, b_orig)
        valid_b, err_b = check_gradients(dense.grad_b, grad_num_b)
        assert valid_b is True
        assert err_b < 1e-6


class TestSequentialMLPNetwork:
    """Verifies multi-layer forward propagation and end-to-end backpropagation."""

    def test_sequential_gradient_checking(self):
        model = Sequential([
            Dense(in_features=3, out_features=4, random_seed=1),
            Tanh(),
            Dense(in_features=4, out_features=1, random_seed=2),
            Sigmoid(),
        ])
        X = np.array([[0.5, -0.2, 0.1], [0.3, 0.9, -0.4]])
        y_true = np.array([[1.0], [0.0]])
        loss_fn = BinaryCrossEntropy()

        # Forward
        preds = model.forward(X)
        loss = loss_fn.forward(y_true, preds)
        assert loss > 0.0

        # Backward
        d_loss = loss_fn.backward(y_true, preds)
        model.backward(d_loss)

        # Check gradient of first layer weights
        layer0 = model.layers[0]
        assert isinstance(layer0, Dense)
        W0_orig = layer0.weights.copy()

        def loss_wrt_W0(W0):
            layer0.weights = W0
            p = model.forward(X)
            return loss_fn.forward(y_true, p)

        num_grad_w0 = numerical_gradient(loss_wrt_W0, W0_orig)
        valid, err = check_gradients(layer0.grad_w, num_grad_w0, tolerance=1e-5)
        assert valid is True
        assert err < 1e-5


class TestComputationalGraphAutoDiff:
    """Verifies reverse-mode automatic differentiation DAG against finite differences."""

    def test_computational_graph_scalar(self):
        # f(x, y) = (x + y) * (x - y) = x^2 - y^2
        # df/dx = 2x, df/dy = -2y
        x = Node(3.0)
        y = Node(2.0)
        z1 = x + y  # 5.0
        z2 = x - y  # 1.0
        f = z1 * z2  # 5.0

        f.backward()
        assert np.isclose(f.data, 5.0)
        assert np.isclose(x.grad, 6.0)   # 2 * 3 = 6
        assert np.isclose(y.grad, -4.0)  # -2 * 2 = -4

    def test_computational_graph_matmul_and_activation(self):
        # Matrix multiply + Sigmoid DAG
        W = Node(np.array([[0.5, -0.5], [0.8, 0.2]]))
        X = Node(np.array([[1.0, 2.0]]))
        Z = X @ W
        A = Z.sigmoid()
        # Loss: sum(A)
        L = A.data.sum()

        A.backward()

        # Check dL/dW via finite differences
        W_orig = W.data.copy()

        def eval_L(w_in):
            z_in = np.dot(X.data, w_in)
            s_in = 1.0 / (1.0 + np.exp(-z_in))
            return float(s_in.sum())

        grad_num_w = numerical_gradient(eval_L, W_orig)
        valid, err = check_gradients(W.grad, grad_num_w)
        assert valid is True
        assert err < 1e-6
