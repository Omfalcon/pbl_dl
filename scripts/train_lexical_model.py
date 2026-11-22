"""Training and Evaluation Pipeline for M-01 Lexical URL MLP.

Executes end-to-end training, validation, calibrated threshold selection, and holdout
test evaluation with strict zero-leakage constraints, class weighting, and model serialization.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root takes precedence over any external site-packages 'app' package
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from app.intelligence.lexical import LexicalFeatureExtractor
from app.intelligence.lexical_model import LexicalMLP, compute_metrics
from app.intelligence.preprocessing import LexicalPreprocessor
from app.ml.from_scratch.losses import BinaryCrossEntropy
from app.ml.from_scratch.optimizers import Adam, CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phishvision.train_lexical")


def load_dataset_splits(splits_dir: str | Path = "data/splits") -> dict[str, pd.DataFrame]:
    """Load pre-split, domain-isolated train/validation/test parquet partitions."""
    splits_path = Path(splits_dir)
    train_file = splits_path / "train" / "urls.parquet"
    val_file = splits_path / "validation" / "urls.parquet"
    test_file = splits_path / "test" / "urls.parquet"

    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required split file: {f}")

    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)
    df_test = pd.read_parquet(test_file)

    # Assert zero domain leakage across splits
    train_domains = set(df_train["registrable_domain"].dropna().unique())
    val_domains = set(df_val["registrable_domain"].dropna().unique())
    test_domains = set(df_test["registrable_domain"].dropna().unique())

    overlap_train_val = train_domains.intersection(val_domains)
    overlap_train_test = train_domains.intersection(test_domains)
    overlap_val_test = val_domains.intersection(test_domains)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Domain leakage detected! "
            f"train/val: {len(overlap_train_val)}, "
            f"train/test: {len(overlap_train_test)}, "
            f"val/test: {len(overlap_val_test)}"
        )

    logger.info(
        f"Loaded dataset splits: Train={len(df_train)} (Class 1={int((df_train['label']==1).sum())}, Class 0={int((df_train['label']==0).sum())}), "
        f"Val={len(df_val)}, Test={len(df_test)}. Zero domain leakage verified."
    )
    return {"train": df_train, "val": df_val, "test": df_test}


def extract_features_for_splits(
    splits: dict[str, pd.DataFrame],
    extractor: LexicalFeatureExtractor,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract 40-dimensional feature matrices and label vectors for all splits."""
    extracted = {}
    for split_name, df in splits.items():
        logger.info(f"Extracting lexical features for '{split_name}' split ({len(df)} records)...")
        # Use canonical_url if available, else original_url
        urls = df["canonical_url"].fillna(df["original_url"]).tolist()
        x_mat = extractor.extract_batch(urls)
        y_vec = df["label"].astype(np.float64).values.reshape(-1, 1)
        extracted[split_name] = (x_mat, y_vec)
        logger.info(f"'{split_name}' feature matrix shape: {x_mat.shape}")
    return extracted


def train_mlp_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 0.005,
    random_seed: int = 42,
) -> tuple[LexicalMLP, dict[str, Any]]:
    """Train LexicalMLP with balanced mini-batch sampling, class-weighted BCE, and CosineAnnealing."""
    np.random.seed(random_seed)
    n_features = x_train.shape[1]
    model = LexicalMLP(in_features=n_features, hidden_dim1=64, hidden_dim2=32, random_seed=random_seed)

    # Compute inverse class frequencies for weighted BCE
    y_flat = y_train.ravel()
    n_total = len(y_flat)
    n_pos = np.sum(y_flat == 1)
    n_neg = np.sum(y_flat == 0)

    # Normalized class weights: w0 and w1
    w0 = float(n_total / (2.0 * max(n_neg, 1)))
    w1 = float(n_total / (2.0 * max(n_pos, 1)))
    logger.info(f"Training dataset imbalance: N_pos={n_pos}, N_neg={n_neg}. Class weights: w0={w0:.3f}, w1={w1:.3f}")

    loss_fn = BinaryCrossEntropy(weight_0=w0, weight_1=w1)

    steps_per_epoch = max(n_total // batch_size, 1)
    total_steps = epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(initial_lr=learning_rate, t_max=total_steps, eta_min=1e-4)
    optimizer = Adam(lr=learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, scheduler=scheduler)

    # Partition indices by class for balanced mini-batch sampling
    idx_pos = np.where(y_flat == 1)[0]
    idx_neg = np.where(y_flat == 0)[0]
    half_batch = batch_size // 2

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}

    logger.info(f"Beginning training for {epochs} epochs ({total_steps} optimizer steps)...")
    for epoch in range(1, epochs + 1):
        # Shuffle indices per epoch
        perm_pos = np.random.permutation(idx_pos)
        perm_neg = np.random.permutation(idx_neg)

        epoch_loss = 0.0
        n_batches = 0

        for step in range(steps_per_epoch):
            # Sample balanced mini-batch (half positive, half negative)
            batch_pos = perm_pos[(step * half_batch) % len(idx_pos) : ((step + 1) * half_batch) % len(idx_pos)]
            if len(batch_pos) < half_batch:
                batch_pos = np.random.choice(idx_pos, size=half_batch, replace=False)

            batch_neg = np.random.choice(idx_neg, size=half_batch, replace=True)
            batch_idx = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch_idx)

            bx = x_train[batch_idx]
            by = y_train[batch_idx]

            # Forward pass
            pred = model.forward(bx)
            loss_val = loss_fn.forward(by, pred)

            # Backward pass
            grad = loss_fn.backward(by, pred)
            model.backward(grad)

            # In-place parameter update with custom Adam optimizer
            optimizer.step(model.get_parameters(), model.get_gradients())

            epoch_loss += loss_val
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation evaluation
        val_pred = model.forward(x_val)
        val_loss = loss_fn.forward(y_val, val_pred)
        val_metrics = compute_metrics(y_val, val_pred, threshold=0.5)
        val_f1 = val_metrics["macro_avg"]["f1"]

        history["epoch"].append(epoch)
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_macro_f1"].append(float(val_f1))

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f} | LR: {optimizer.lr:.6f}"
            )

    return model, history


def calibrate_threshold(
    model: LexicalMLP,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Calibrate optimal decision threshold on validation set maximizing Macro F1."""
    val_probs = model.predict_proba(x_val)
    best_thresh = 0.5
    best_macro_f1 = -1.0
    best_metrics: dict[str, Any] = {}

    for thresh in np.linspace(0.1, 0.9, 81):
        m = compute_metrics(y_val, val_probs, threshold=float(thresh))
        macro_f1 = m["macro_avg"]["f1"]
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh = float(thresh)
            best_metrics = m

    logger.info(
        f"Validation threshold calibration selected tau={best_thresh:.3f} "
        f"achieving Val Macro F1={best_macro_f1:.4f} (Class 0 F1={best_metrics['class_0_legitimate']['f1']:.4f}, Class 1 F1={best_metrics['class_1_phishing']['f1']:.4f})"
    )
    return best_thresh, best_metrics


def main() -> None:
    """Run full lexical pipeline training, threshold calibration, and test evaluation."""
    logger.info("Initializing Lexical URL Pipeline...")
    random_seed = 42

    # 1. Load splits
    splits = load_dataset_splits("data/splits")

    # 2. Extract 40 features
    extractor = LexicalFeatureExtractor()
    feature_data = extract_features_for_splits(splits, extractor)

    x_train_raw, y_train = feature_data["train"]
    x_val_raw, y_val = feature_data["val"]
    x_test_raw, y_test = feature_data["test"]

    # 3. Fit preprocessor on training partition ONLY (Strict zero leakage)
    logger.info("Fitting LexicalPreprocessor on X_train only...")
    preprocessor = LexicalPreprocessor(feature_names=extractor.feature_names)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)
    x_test = preprocessor.transform(x_test_raw)

    # 4. Train MLP model
    model, history = train_mlp_model(
        x_train, y_train, x_val, y_val,
        epochs=25, batch_size=128, learning_rate=0.005, random_seed=random_seed
    )

    # 5. Calibrate threshold on validation set (NOT test set)
    calibrated_threshold, val_calibrated_metrics = calibrate_threshold(model, x_val, y_val)

    # 6. Evaluate on unseen holdout test set using calibrated threshold
    logger.info("Evaluating calibrated model on unseen holdout test partition...")
    test_probs = model.predict_proba(x_test)
    test_metrics = compute_metrics(y_test, test_probs, threshold=calibrated_threshold)
    default_test_metrics = compute_metrics(y_test, test_probs, threshold=0.5)

    logger.info(
        f"Test Partition Evaluation (tau={calibrated_threshold:.3f}):\n"
        f"  Macro F1:          {test_metrics['macro_avg']['f1']:.4f}\n"
        f"  Macro Precision:   {test_metrics['macro_avg']['precision']:.4f}\n"
        f"  Macro Recall:      {test_metrics['macro_avg']['recall']:.4f}\n"
        f"  ROC-AUC:           {test_metrics['roc_auc']:.4f}\n"
        f"  PR-AUC:            {test_metrics['pr_auc']:.4f}\n"
        f"  Class 0 (Legit):   Prec={test_metrics['class_0_legitimate']['precision']:.4f}, Rec={test_metrics['class_0_legitimate']['recall']:.4f}, F1={test_metrics['class_0_legitimate']['f1']:.4f}\n"
        f"  Class 1 (Phish):   Prec={test_metrics['class_1_phishing']['precision']:.4f}, Rec={test_metrics['class_1_phishing']['recall']:.4f}, F1={test_metrics['class_1_phishing']['f1']:.4f}\n"
        f"  Confusion Matrix:  TP={test_metrics['confusion_matrix']['true_positive']}, FP={test_metrics['confusion_matrix']['false_positive']}, TN={test_metrics['confusion_matrix']['true_negative']}, FN={test_metrics['confusion_matrix']['false_negative']}"
    )

    # 7. Save model, preprocessor, and reports
    models_dir = Path("models/lexical")
    models_dir.mkdir(parents=True, exist_ok=True)

    weights_file = models_dir / "model_weights.npz"
    preprocessor_file = models_dir / "preprocessor.json"
    feature_names_file = models_dir / "feature_names.json"
    report_file = Path("data/reports/lexical_mlp_evaluation.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model weights to {weights_file}...")
    model.save_weights(weights_file)

    logger.info(f"Saving preprocessor metadata to {preprocessor_file}...")
    preprocessor.save(preprocessor_file)

    logger.info(f"Saving feature names registry to {feature_names_file}...")
    with open(feature_names_file, "w", encoding="utf-8") as f:
        json.dump(extractor.feature_names, f, indent=2)

    # Assemble comprehensive evaluation report
    report = {
        "model_id": "M-01",
        "model_name": "Lexical URL Multi-Layer Perceptron",
        "random_seed": random_seed,
        "n_features": extractor.num_features,
        "feature_names": extractor.feature_names,
        "calibrated_threshold": calibrated_threshold,
        "architecture": {
            "layer_1": f"Dense({extractor.num_features}, 64) -> ReLU",
            "layer_2": "Dense(64, 32) -> ReLU",
            "layer_3": "Dense(32, 1) -> Sigmoid",
            "total_parameters": 64 * extractor.num_features + 64 + 32 * 64 + 32 + 1 * 32 + 1,
        },
        "dataset_split_sizes": {
            "train": len(splits["train"]),
            "validation": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "training_history": history,
        "validation_calibrated_metrics": val_calibrated_metrics,
        "test_calibrated_metrics": test_metrics,
        "test_default_threshold_metrics": default_test_metrics,
    }

    logger.info(f"Exporting evaluation report to {report_file}...")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
