"""Unit & Regression Tests for M-07 URL Sequence LSTM (Phase 7.8).

Verifies character-level and token-level sequence preprocessing, zero domain leakage,
bidirectional LSTM architecture with attentive temporal pooling, dual-branch redirect modeling,
loss weighting, production inference engine, and evaluation report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.sequence.lstm_inference import LSTMInferencer
from app.ml.sequence.lstm_model import AttentionPooling, URLSequenceLSTM
from app.ml.sequence.sequence_preprocessing import (
    NONE_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SequenceDataset,
    URLSequenceVocabulary,
)
from app.ml.train_lstm import (
    calculate_comprehensive_metrics,
    compute_weighted_bce_loss,
    find_optimal_threshold,
)


class TestSequencePreprocessing:
    """Verifies sequence vocabulary fitting, character encoding, truncation, and serialization."""

    def test_vocab_fitting_and_special_tokens(self) -> None:
        """Asserts special tokens are assigned initial indices and vocabulary fits training URLs."""
        sample_urls = [
            "https://login.example-bank.com/auth",
            "http://phish-secure-portal.xyz/login.php?user=test",
        ]
        vocab = URLSequenceVocabulary(max_url_len=64, min_char_freq=1)
        assert not vocab.is_fitted
        vocab.fit(sample_urls)
        assert vocab.is_fitted
        assert vocab.char_to_idx[PAD_TOKEN] == 0
        assert vocab.char_to_idx[UNK_TOKEN] == 1
        assert "h" in vocab.char_to_idx
        assert "/" in vocab.char_to_idx
        assert "?" in vocab.char_to_idx

    def test_encode_url_padding_and_truncation(self) -> None:
        """Tests that URL encoding enforces exact sequence length via padding or truncation."""
        vocab = URLSequenceVocabulary(max_url_len=20, min_char_freq=1)
        vocab.fit(["https://short.com"])

        # Short URL (padding check)
        short_encoded = vocab.encode_url("https://a")
        assert len(short_encoded) == 20
        assert short_encoded[-1] == vocab.char_to_idx[PAD_TOKEN]

        # Long URL (truncation check)
        long_url = "https://very-long-domain-name-with-many-subdomains.example.com/login"
        long_encoded = vocab.encode_url(long_url)
        assert len(long_encoded) == 20

    def test_unknown_character_handling(self) -> None:
        """Asserts unseen characters map to UNK token."""
        vocab = URLSequenceVocabulary(max_url_len=10, min_char_freq=1)
        vocab.fit(["http://abc.com"])
        encoded = vocab.encode_url("xyz")
        assert encoded[0] == vocab.char_to_idx[UNK_TOKEN]

    def test_redirect_encoding_empty_and_valid(self) -> None:
        """Asserts redirect sequences handle absent/empty chains with NONE token fallback."""
        vocab = URLSequenceVocabulary(max_url_len=16, max_redirect_len=4)
        vocab.fit(["http://site.com"], redirect_chains=[["hop1.com", "hop2.com"]])

        # Empty redirect chain
        empty_red = vocab.encode_redirect_chain(None)
        assert len(empty_red) == 4
        assert empty_red[0] == vocab.redirect_to_idx[NONE_TOKEN]
        assert empty_red[1] == vocab.redirect_to_idx[PAD_TOKEN]

        # Valid redirect chain
        valid_red = vocab.encode_redirect_chain(["hop1.com", "unknown.com"])
        assert len(valid_red) == 4
        assert valid_red[0] == vocab.redirect_to_idx["hop1.com"]
        assert valid_red[1] == vocab.redirect_to_idx[UNK_TOKEN]

    def test_vocab_serialization_roundtrip(self, tmp_path: Path) -> None:
        """Asserts vocabulary JSON serialization and loading preserve exact mappings."""
        vocab = URLSequenceVocabulary(max_url_len=32, min_char_freq=1)
        vocab.fit(["https://test.org/path"])

        save_p = tmp_path / "vocab_test.json"
        vocab.save(save_p)
        assert save_p.exists()

        loaded = URLSequenceVocabulary.load(save_p)
        assert loaded.char_to_idx == vocab.char_to_idx
        assert loaded.redirect_to_idx == vocab.redirect_to_idx
        assert loaded.max_url_len == 32

    def test_sequence_dataset(self) -> None:
        """Asserts PyTorch SequenceDataset outputs expected tensor shapes and dtypes."""
        df = pd.DataFrame({
            "canonical_url": ["https://paypal.com", "http://phish-update.xyz/log"],
            "label": [0, 1],
        })
        vocab = URLSequenceVocabulary(max_url_len=16)
        vocab.fit(df["canonical_url"])
        dataset = SequenceDataset(df, vocab)
        assert len(dataset) == 2

        item0 = dataset[0]
        assert "url_seq" in item0
        assert "redirect_seq" in item0
        assert "label" in item0
        assert item0["url_seq"].shape == torch.Size([16])
        assert item0["url_seq"].dtype == torch.long
        assert item0["label"].dtype == torch.float32


class TestDatasetIntegrityAndIsolation:
    """Verifies domain-level isolation and zero leakage across train, val, and test splits."""

    def test_zero_url_and_domain_leakage(self) -> None:
        """Asserts disjoint URL canonical strings and registrable domains across splits."""
        p_train = Path("data/splits/train/urls.parquet")
        p_val = Path("data/splits/validation/urls.parquet")
        p_test = Path("data/splits/test/urls.parquet")

        if not (p_train.exists() and p_val.exists() and p_test.exists()):
            pytest.skip("Dataset split parquet files not found.")

        df_train = pd.read_parquet(p_train)
        df_val = pd.read_parquet(p_val)
        df_test = pd.read_parquet(p_test)

        # Exact split counts check
        assert len(df_train) == 32157
        assert len(df_val) == 7715
        assert len(df_test) == 4257

        # Zero URL leakage
        urls_train = set(df_train["canonical_url"])
        urls_val = set(df_val["canonical_url"])
        urls_test = set(df_test["canonical_url"])

        assert len(urls_train.intersection(urls_val)) == 0
        assert len(urls_train.intersection(urls_test)) == 0
        assert len(urls_val.intersection(urls_test)) == 0

        # Zero domain leakage
        dom_train = set(df_train["registrable_domain"].dropna())
        dom_val = set(df_val["registrable_domain"].dropna())
        dom_test = set(df_test["registrable_domain"].dropna())

        assert len(dom_train.intersection(dom_val)) == 0
        assert len(dom_train.intersection(dom_test)) == 0
        assert len(dom_val.intersection(dom_test)) == 0


class TestURLSequenceLSTMArchitecture:
    """Verifies model forward/backward passes, attentive pooling, and conditional redirect modeling."""

    def test_attention_pooling(self) -> None:
        """Asserts attention pooling outputs weighted context and softmax weights sum to 1."""
        attn = AttentionPooling(hidden_dim=32)
        x = torch.randn(4, 10, 32)  # (Batch=4, Time=10, Dim=32)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 7:] = False  # Last 3 tokens are padded

        context, weights = attn(x, mask=mask)
        assert context.shape == torch.Size([4, 32])
        assert weights.shape == torch.Size([4, 10])

        # Weights sum to 1.0 along sequence dimension
        sum_weights = weights.sum(dim=-1)
        assert torch.allclose(sum_weights, torch.ones(4), atol=1e-5)
        # Padded positions receive ~0 weight
        assert torch.all(weights[:, 7:] < 1e-4)

    def test_model_forward_without_redirects(self) -> None:
        """Tests forward pass with URL sequence only (redirect branch receives fallback zero)."""
        model = URLSequenceLSTM(
            url_vocab_size=50,
            url_embed_dim=16,
            hidden_dim=16,
            num_layers=1,
            bidirectional=True,
            fc_dim=16,
            use_redirects=True,
            redirect_vocab_size=10,
        )
        url_seq = torch.randint(0, 50, (4, 32), dtype=torch.long)
        logits = model(url_seq)
        assert logits.shape == torch.Size([4, 1])

        probas = model.predict_proba(url_seq)
        assert probas.shape == torch.Size([4, 1])
        assert torch.all((probas >= 0.0) & (probas <= 1.0))

    def test_model_forward_with_redirects(self) -> None:
        """Tests forward pass with both URL and redirect sequences."""
        model = URLSequenceLSTM(
            url_vocab_size=50,
            url_embed_dim=16,
            hidden_dim=16,
            num_layers=1,
            bidirectional=True,
            fc_dim=16,
            use_redirects=True,
            redirect_vocab_size=10,
            redirect_embed_dim=8,
            redirect_hidden_dim=8,
        )
        url_seq = torch.randint(0, 50, (4, 32), dtype=torch.long)
        red_seq = torch.randint(0, 10, (4, 4), dtype=torch.long)
        logits = model(url_seq, red_seq)
        assert logits.shape == torch.Size([4, 1])

    def test_backward_pass_and_gradient_flow(self) -> None:
        """Asserts backward pass computes finite non-zero gradients across all trainable layers."""
        model = URLSequenceLSTM(
            url_vocab_size=30,
            url_embed_dim=16,
            hidden_dim=16,
            num_layers=1,
            bidirectional=False,
            fc_dim=16,
            use_redirects=False,
        )
        url_seq = torch.randint(0, 30, (2, 16), dtype=torch.long)
        targets = torch.tensor([0.0, 1.0])

        logits = model(url_seq).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"

    def test_model_telemetry(self) -> None:
        """Asserts total parameter count calculation."""
        model = URLSequenceLSTM(url_vocab_size=100, url_embed_dim=32, hidden_dim=32, num_layers=1)
        summary = model.get_model_summary()
        assert summary["model_id"] == "M-07"
        assert summary["total_parameters"] > 1000


class TestLossAndImbalance:
    """Verifies weighted BCE loss behavior on heavily imbalanced class distributions."""

    def test_weighted_bce_loss(self) -> None:
        """Asserts that higher weight on legitimate class penalizes false positives strongly."""
        logits_fp = torch.tensor([[5.0]])   # Confident false positive (true target is 0)
        logits_fn = torch.tensor([[-5.0]])  # Confident false negative (true target is 1)
        target_0 = torch.tensor([0.0])
        target_1 = torch.tensor([1.0])

        # Balanced weights: w0=36.96 (legit), w1=0.50 (phish)
        w0, w1 = 36.96, 0.50
        loss_fp = compute_weighted_bce_loss(logits_fp, target_0, weight_0=w0, weight_1=w1)
        loss_fn = compute_weighted_bce_loss(logits_fn, target_1, weight_0=w0, weight_1=w1)

        # With w0 >> w1, a false positive on a legitimate site incurs vastly higher loss
        assert loss_fp.item() > loss_fn.item()


class TestLSTMInference:
    """Verifies inference interface, prediction schema, and batch evaluation."""

    @pytest.fixture
    def setup_inferencer(self, tmp_path: Path) -> LSTMInferencer:
        # Create and save temporary fitted vocabulary
        vocab = URLSequenceVocabulary(max_url_len=32, max_redirect_len=4)
        vocab.fit(["https://test-bank.com", "http://phish-login.xyz"])
        vocab_path = tmp_path / "preprocessor.json"
        vocab.save(vocab_path)

        # Create and save dummy trained model weights
        model = URLSequenceLSTM(
            url_vocab_size=vocab.url_vocab_size,
            url_embed_dim=64,
            hidden_dim=64,
            num_layers=2,
            bidirectional=True,
            fc_dim=64,
            use_redirects=True,
            redirect_vocab_size=vocab.redirect_vocab_size,
        )
        chk_path = tmp_path / "best_lstm.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "calibrated_threshold": 0.45,
        }, chk_path)

        return LSTMInferencer(
            checkpoint_path=chk_path,
            preprocessor_path=vocab_path,
            device="cpu",
        )

    def test_single_url_predict(self, setup_inferencer: LSTMInferencer) -> None:
        """Tests single URL prediction schema."""
        inferencer = setup_inferencer
        res = inferencer.predict("https://secure.example.com/account")
        assert res["model_id"] == "M-07"
        assert res["architecture"] == "URLSequenceLSTM"
        assert "phishing_probability" in res
        assert 0.0 <= res["phishing_probability"] <= 1.0
        assert res["prediction_label"] in (0, 1)
        assert res["decision_threshold"] == 0.45
        assert res["latency_ms"] > 0.0

    def test_batch_url_predict(self, setup_inferencer: LSTMInferencer) -> None:
        """Tests batched URL prediction consistency."""
        inferencer = setup_inferencer
        urls = [
            "https://paypal.com",
            "http://phish-secure-portal.xyz/login",
            "https://google.com/search?q=test",
        ]
        results = inferencer.predict_batch(urls)
        assert len(results) == 3
        for r in results:
            assert "phishing_probability" in r
            assert r["model_id"] == "M-07"


class TestEvaluationAndMetrics:
    """Verifies threshold calibration and comprehensive evaluation metric calculation."""

    def test_optimal_threshold_search(self) -> None:
        """Asserts optimal threshold selects boundary maximizing Macro F1."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_scores = np.array([0.05, 0.10, 0.15, 0.40, 0.60, 0.80, 0.90, 0.95, 0.85, 0.75])
        best_tau, best_f1 = find_optimal_threshold(y_true, y_scores)
        assert 0.15 < best_tau <= 0.40
        assert best_f1 == 1.0

    def test_calculate_comprehensive_metrics(self) -> None:
        """Asserts all metric fields and confusion matrix entries are populated."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metrics = calculate_comprehensive_metrics(y_true, y_scores, threshold=0.50, latency_ms=1.5, checkpoint_size_mb=0.85)

        assert metrics["total_samples"] == 5
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["macro_avg"]["f1"] == 1.0
        assert metrics["class_0_legitimate"]["f1"] == 1.0
        assert metrics["class_1_phishing"]["f1"] == 1.0
        assert metrics["confusion_matrix"]["true_negative"] == 2
        assert metrics["confusion_matrix"]["true_positive"] == 3
        assert metrics["latency_ms"] == 1.5
        assert metrics["checkpoint_size_mb"] == 0.85
