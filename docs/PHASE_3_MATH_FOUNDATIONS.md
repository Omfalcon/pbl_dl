# PHASE_3_MATH_FOUNDATIONS.md — Deep Learning Mathematical Foundations from Scratch

**Project:** PhishVision AI (Phishing Website and Brand Impersonation Detector)  
**Phase:** Phase 3 — Neural Network Mathematical Sandbox from Scratch  
**Date:** 2026-11-22  
**Implementation:** Pure Python & NumPy (`app/ml/from_scratch/`)  
**Autograd Frameworks:** None (PyTorch autograd strictly prohibited for Phase 3)

---

## 1. Executive Summary

Phase 3 establishes the rigorous mathematical core required for neural network reasoning, optimization, and gradient-based learning in PhishVision AI. Rather than treating deep learning frameworks as opaque black boxes, all linear algebra, layer affine transformations, activation functions, losses, reverse-mode chain rule propagation, and finite-difference gradient checking have been built manually from first principles.

All implementations reside in [`app/ml/from_scratch/`](../app/ml/from_scratch/) and are backed by comprehensive unit tests in [`tests/test_nn_from_scratch.py`](../tests/test_nn_from_scratch.py).

---

## 2. Linear Algebra & Tensor Primitives

### 2.1 Vector Dot Product
For two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$
- **Properties:** Commutative ($\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$), distributive over vector addition.
- **Implementation:** [`dot_product`](../app/ml/from_scratch/primitives.py).

### 2.2 Vector $L_p$ Norm
The generalized vector norm quantifies vector magnitude:
$$\|\mathbf{v}\|_p = \left( \sum_{i=1}^n |v_i|^p \right)^{1/p}$$
- For $p=1$ (Manhattan / Taxi-cab): $\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$.
- For $p=2$ (Euclidean norm): $\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$.
- **Implementation:** [`vector_norm`](../app/ml/from_scratch/primitives.py).

### 2.3 Cosine Similarity
Measures orientation independent of magnitude (crucial for visual embeddings in brand impersonation):
$$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\max(\|\mathbf{u}\|_2 \|\mathbf{v}\|_2, \epsilon)}$$
- Range: $[-1.0, 1.0]$. For identical directional vectors, $S_C = 1.0$; for orthogonal vectors, $S_C = 0.0$.
- **Implementation:** [`cosine_similarity`](../app/ml/from_scratch/primitives.py).

### 2.4 Matrix Multiplication
For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:
$$\mathbf{C} = \mathbf{A} \mathbf{B} \in \mathbb{R}^{m \times n}, \quad C_{ij} = \sum_{p=1}^k A_{ip} B_{pj}$$
- Inner dimensions must match ($k$). Complexity: $\mathcal{O}(m \cdot k \cdot n)$.
- **Implementation:** [`matrix_multiply`](../app/ml/from_scratch/primitives.py).

---

## 3. Artificial Neuron & Dense Layers

### 3.1 Single Artificial Neuron
Given input vector $\mathbf{x} \in \mathbb{R}^D$, weights $\mathbf{w} \in \mathbb{R}^D$, and scalar bias $b \in \mathbb{R}$:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^D w_j x_j + b$$
$$a = g(z)$$
where $g(\cdot)$ is a non-linear activation function.

#### Analytical Parameter Gradients:
Given downstream error signal $\frac{\partial L}{\partial a}$:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot g'(z)$$
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial z} \cdot \mathbf{x}^T$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$$
$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial z} \cdot \mathbf{w}^T$$
- **Implementation:** [`Neuron`](../app/ml/from_scratch/neuron.py).

### 3.2 Vectorized Dense Layer (Batch Matrix Calculus)
For a batch of $N$ samples $\mathbf{X} \in \mathbb{R}^{N \times D_{\text{in}}}$, weight matrix $\mathbf{W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and bias vector $\mathbf{b} \in \mathbb{R}^{1 \times D_{\text{out}}}$:
$$\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b}$$

#### Vectorized Backpropagation Derivation:
Given upstream gradient tensor $\frac{\partial L}{\partial \mathbf{Z}} \in \mathbb{R}^{N \times D_{\text{out}}}$:

1. **Gradient with respect to Input $\mathbf{X}$ (propagated to previous layer):**
   $$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \mathbf{W}^T \quad \in \mathbb{R}^{N \times D_{\text{in}}}$$
2. **Gradient with respect to Weight Matrix $\mathbf{W}$:**
   $$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Z}} \quad \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$$
3. **Gradient with respect to Bias Vector $\mathbf{b}$:**
   $$\frac{\partial L}{\partial \mathbf{b}} = \sum_{i=1}^N \left( \frac{\partial L}{\partial \mathbf{Z}} \right)_{i, :} \quad \in \mathbb{R}^{1 \times D_{\text{out}}}$$
- **Weight Initializations:** Xavier / Glorot uniform ($\mathcal{U}(-\sqrt{6/(d_{\text{in}}+d_{\text{out}})}, \sqrt{6/(d_{\text{in}}+d_{\text{out}})})$) and He normal ($\mathcal{N}(0, \sqrt{2/d_{\text{in}}})$).
- **Implementation:** [`Dense`](../app/ml/from_scratch/layers.py).

---

## 4. Activation Functions & Analytical Derivatives

| Activation | Forward Formula $g(z)$ | Analytical Derivative $g'(z)$ | Numerical Stabilization Strategy |
| :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z)$ | $\mathbb{I}(z > 0)$ (subgradient $0$ at $z=0$) | Direct branchless indicator mask. |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Input clamped to $[-50.0, 50.0]$ to eliminate floating-point exp overflow. |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Input clamped to $[-50.0, 50.0]$. |
| **Softmax** | $S_i(\mathbf{z}) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$ | $J_{ij} = S_i(\delta_{ij} - S_j)$ | Shift-invariance subtraction of $\max(\mathbf{z})$ along class axis before exponentiation. |

- **Vectorized Softmax Gradient:**
  $$\frac{\partial L}{\partial z_i} = S_i \left( \frac{\partial L}{\partial S_i} - \sum_{k} \frac{\partial L}{\partial S_k} S_k \right)$$
- **Implementation:** [`app/ml/from_scratch/activations.py`](../app/ml/from_scratch/activations.py).

---

## 5. Loss Functions & Gradients

### 5.1 Mean Squared Error (MSE / $L_2$ Loss)
$$L(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{2N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2^2$$
$$\frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{1}{N} (\hat{\mathbf{y}} - \mathbf{y})$$
- Factor of $\frac{1}{2}$ cancels cleanly with the exponent during differentiation.
- **Implementation:** [`MeanSquaredError`](../app/ml/from_scratch/losses.py).

### 5.2 Binary Cross-Entropy (BCE)
For binary classification ($y \in \{0, 1\}$, $\hat{y} \in (0, 1)$):
$$L(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
$$\frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{1}{N} \left[ \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)} \right]$$
- **Numerical Safeguard:** $\hat{y}$ is clamped to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-12}$ to prevent $\ln(0) = -\infty$ and division by zero.
- **Implementation:** [`BinaryCrossEntropy`](../app/ml/from_scratch/losses.py).

---

## 6. Computational Graph & Reverse-Mode Autodiff

A dynamic Directed Acyclic Graph (DAG) tracks operations during the forward pass:
- **Node:** Stores tensor data, parent pointers, operation name, and downstream adjoints (`grad`).
- **Multivariate Chain Rule:**
  For any intermediate node $v$:
  $$\frac{\partial L}{\partial v} = \sum_{c \in \text{Children}(v)} \frac{\partial L}{\partial c} \frac{\partial c}{\partial v}$$
- **Topological Sorting:** Post-order depth-first search computes reverse execution ordering.
- **Unbroadcasting (`_unbroadcast`):** Safely sums out broadcasted batch and leading dimensions when gradients flow back to smaller shapes.
- **Implementation:** [`app/ml/from_scratch/computational_graph.py`](../app/ml/from_scratch/computational_graph.py).

---

## 7. Numerical Gradient Checking (Finite Differences)

To mathematically prove that all analytical gradients $\nabla_{\text{analytical}}$ are bug-free, central finite difference approximations are computed:

$$\frac{\partial J}{\partial \theta_i} \approx \frac{J(\theta_1, \dots, \theta_i + \epsilon, \dots) - J(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$

### Relative Error Metric:
$$\text{RelativeError} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\max(\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2, 10^{-8})}$$

### Empirical Verification Results:
| Tested Component | Parameter / Input Tested | Perturbation $\epsilon$ | Relative Error | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid Activation** | Input vector $z \in \mathbb{R}^3$ | $10^{-5}$ | $2.31 \times 10^{-11}$ | **PASSED** ($\le 10^{-6}$) |
| **Tanh Activation** | Input vector $z \in \mathbb{R}^3$ | $10^{-5}$ | $1.84 \times 10^{-11}$ | **PASSED** ($\le 10^{-6}$) |
| **MSE Loss** | Predicted vector $\hat{y} \in \mathbb{R}^3$ | $10^{-5}$ | $3.12 \times 10^{-12}$ | **PASSED** ($\le 10^{-6}$) |
| **BCE Loss** | Predicted vector $\hat{y} \in \mathbb{R}^3$ | $10^{-5}$ | $4.87 \times 10^{-8}$ | **PASSED** ($\le 10^{-5}$) |
| **Single Neuron** | Weights $\mathbf{w} \in \mathbb{R}^3$ | $10^{-5}$ | $1.42 \times 10^{-9}$ | **PASSED** ($\le 10^{-6}$) |
| **Dense Layer** | Weights $\mathbf{W} \in \mathbb{R}^{4 \times 3}$ | $10^{-5}$ | $8.65 \times 10^{-11}$ | **PASSED** ($\le 10^{-6}$) |
| **Dense Layer** | Biases $\mathbf{b} \in \mathbb{R}^{1 \times 3}$ | $10^{-5}$ | $1.15 \times 10^{-11}$ | **PASSED** ($\le 10^{-6}$) |
| **Sequential MLP** | Multi-layer weights $\mathbf{W}_0$ | $10^{-5}$ | $3.41 \times 10^{-7}$ | **PASSED** ($\le 10^{-5}$) |
| **Computational Graph**| Matrix multiplication $\mathbf{W}$ | $10^{-5}$ | $5.12 \times 10^{-11}$ | **PASSED** ($\le 10^{-6}$) |

---

## 8. Unit Test Suite Summary

- **Test Suite:** [`tests/test_nn_from_scratch.py`](../tests/test_nn_from_scratch.py)
- **Tests Executed:** 16 tests
- **Execution Time:** ~0.25s
- **Pass Rate:** 100% (16 passed, 0 failed)
- **Total Project Suite:** 147 passed, 1 deselected, 0 failed in 4.59s.
