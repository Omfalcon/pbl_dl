"""Empirical Comparison of From-Scratch Optimizers on Synthetic Benchmark Tasks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.from_scratch.activations import ReLU, Sigmoid
from app.ml.from_scratch.layers import Dense
from app.ml.from_scratch.losses import BinaryCrossEntropy
from app.ml.from_scratch.network import Sequential
from app.ml.from_scratch.optimizers import Adam, RMSProp, SGD, SGDMomentum, StepLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.compare_optimizers")


def generate_synthetic_data(num_samples: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D binary classification dataset with non-linear decision boundary."""
    rng = np.random.default_rng(seed)
    half = num_samples // 2

    # Class 0: inner distribution
    r0 = rng.uniform(0.0, 1.2, size=(half, 1))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=(half, 1))
    X0 = np.hstack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    y0 = np.zeros((half, 1))

    # Class 1: outer ring distribution
    r1 = rng.uniform(1.8, 3.0, size=(half, 1))
    theta1 = rng.uniform(0.0, 2.0 * np.pi, size=(half, 1))
    X1 = np.hstack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.ones((half, 1))

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    # Shuffle
    indices = rng.permutation(num_samples)
    return X[indices], y[indices]


def build_model(seed: int = 42) -> Sequential:
    """Build standardized MLP: 2 -> 8 (ReLU) -> 4 (ReLU) -> 1 (Sigmoid)."""
    return Sequential([
        Dense(in_features=2, out_features=8, init_method="he", random_seed=seed),
        ReLU(),
        Dense(in_features=8, out_features=4, init_method="he", random_seed=seed + 1),
        ReLU(),
        Dense(in_features=4, out_features=1, init_method="xavier", random_seed=seed + 2),
        Sigmoid(),
    ])


def run_benchmark(epochs: int = 150) -> dict[str, dict]:
    """Train identical models across SGD, SGDMomentum, RMSProp, and Adam."""
    X, y = generate_synthetic_data(num_samples=400, seed=42)
    loss_fn = BinaryCrossEntropy()

    optimizers = {
        "SGD": SGD(lr=0.08),
        "SGD_Momentum": SGDMomentum(lr=0.04, momentum=0.9),
        "RMSProp": RMSProp(lr=0.01, decay_rate=0.9),
        "Adam": Adam(lr=0.02, beta1=0.9, beta2=0.999),
    }

    results: dict[str, dict] = {}

    for name, opt in optimizers.items():
        logger.info(f"Running optimization experiment: {name}")
        model = build_model(seed=42)
        loss_history: list[float] = []

        for epoch in range(epochs):
            preds = model.forward(X)
            loss = loss_fn.forward(y, preds)
            loss_history.append(round(loss, 5))

            grad_loss = loss_fn.backward(y, preds)
            model.backward(grad_loss)
            opt.step(model.get_parameters(), model.get_gradients())

        final_preds = model.forward(X)
        final_acc = float(np.mean((final_preds > 0.5) == y))

        results[name] = {
            "initial_loss": loss_history[0],
            "loss_at_50": loss_history[min(49, epochs - 1)],
            "loss_at_100": loss_history[min(99, epochs - 1)],
            "final_loss": loss_history[-1],
            "final_accuracy": round(final_acc * 100, 2),
            "loss_reduction_pct": round((1.0 - (loss_history[-1] / loss_history[0])) * 100, 2),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizer Benchmark from Scratch")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output", default="data/reports/optimizer_benchmark.json")
    args = parser.parse_args()

    results = run_benchmark(epochs=args.epochs)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("🚀 OPTIMIZER CONVERGENCE BENCHMARK (FROM SCRATCH)")
    print("=" * 80)
    print(f"{'Optimizer':<16} | {'Init Loss':<10} | {'Loss @ 50':<10} | {'Final Loss':<10} | {'Accuracy':<10} | {'Loss Reduction':<14}")
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<16} | {m['initial_loss']:<10.4f} | {m['loss_at_50']:<10.4f} | "
            f"{m['final_loss']:<10.4f} | {m['final_accuracy']:>8.2f}% | {m['loss_reduction_pct']:>12.2f}%"
        )
    print("=" * 80)
    print(f"Results saved to: {out_p}\n")


if __name__ == "__main__":
    main()
