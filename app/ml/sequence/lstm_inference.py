"""Production Inference Engine for M-07 URL Sequence LSTM (Phase 7.8).

Provides single-URL and batched sequential inference with latency measurement,
calibrated decision thresholding, and clean interfaces for multimodal risk fusion.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ml.sequence.lstm_model import URLSequenceLSTM
from app.ml.sequence.sequence_preprocessing import URLSequenceVocabulary

logger = logging.getLogger("phishvision.lstm_inference")


class LSTMInferencer:
    """Production inference interface for M-07 LSTM sequential classifier."""

    def __init__(
        self,
        checkpoint_path: str | Path = "models/checkpoints/lstm/best_lstm.pt",
        preprocessor_path: str | Path = "models/checkpoints/lstm/preprocessor.json",
        device: str | int | None = None,
        threshold: float = 0.50,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.threshold = threshold

        # Determine compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        # Load vocabulary preprocessor
        if self.preprocessor_path.exists():
            self.vocab = URLSequenceVocabulary.load(self.preprocessor_path)
            logger.info(f"Loaded vocabulary from: {self.preprocessor_path}")
        else:
            raise FileNotFoundError(
                f"Vocabulary preprocessor not found at {self.preprocessor_path}. "
                "Fit vocabulary on training split first."
            )

        # Instantiate model architecture
        self.model = URLSequenceLSTM(
            url_vocab_size=self.vocab.url_vocab_size,
            url_embed_dim=64,
            hidden_dim=64,
            num_layers=2,
            bidirectional=True,
            dropout=0.0,  # Zero dropout during production inference
            fc_dim=64,
            use_redirects=True,
            redirect_vocab_size=self.vocab.redirect_vocab_size,
            redirect_embed_dim=32,
            redirect_hidden_dim=32,
        )

        # Load trained weights
        if self.checkpoint_path.exists():
            chk = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = chk.get("model_state_dict", chk)
            self.model.load_state_dict(state_dict)
            if "calibrated_threshold" in chk:
                self.threshold = float(chk["calibrated_threshold"])
            logger.info(
                f"Loaded M-07 LSTM checkpoint from: {self.checkpoint_path} "
                f"(Calibrated threshold: {self.threshold:.4f})"
            )
        else:
            logger.warning(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Inferencer initialized with raw initialized weights."
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        url: str,
        redirect_chain: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform sequential phishing prediction on a single URL string.
        
        Args:
            url: Raw or canonical target URL string.
            redirect_chain: Optional list of intermediate redirect hostnames.
            threshold: Optional decision boundary override (defaults to calibrated threshold).
        
        Returns:
            Dictionary containing binary prediction, probability, confidence, and telemetry.
        """
        tau = threshold if threshold is not None else self.threshold
        t0 = time.perf_counter()

        url_seq = self.vocab.encode_url(url)
        red_seq = self.vocab.encode_redirect_chain(redirect_chain)

        t_url = torch.tensor([url_seq], dtype=torch.long, device=self.device)
        t_red = torch.tensor([red_seq], dtype=torch.long, device=self.device)

        with torch.no_grad():
            proba_tensor = self.model.predict_proba(t_url, t_red)
            proba = float(proba_tensor.item())

        is_phishing = bool(proba >= tau)
        confidence = proba if is_phishing else (1.0 - proba)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "url": url,
            "model_id": "M-07",
            "architecture": "URLSequenceLSTM",
            "prediction_label": 1 if is_phishing else 0,
            "prediction_name": "PHISHING" if is_phishing else "LEGITIMATE",
            "phishing_probability": round(proba, 5),
            "confidence": round(confidence, 5),
            "decision_threshold": round(tau, 4),
            "redirect_chain_analyzed": bool(redirect_chain and len(redirect_chain) > 0),
            "num_redirect_hops": len(redirect_chain) if redirect_chain else 0,
            "device": str(self.device),
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(
        self,
        urls: list[str],
        redirect_chains: list[list[str]] | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Perform sequential predictions on a batch of URLs."""
        if not urls:
            return []

        tau = threshold if threshold is not None else self.threshold
        t0 = time.perf_counter()

        url_indices = [self.vocab.encode_url(u) for u in urls]
        t_urls = torch.tensor(url_indices, dtype=torch.long, device=self.device)

        if redirect_chains:
            red_indices = [self.vocab.encode_redirect_chain(r) for r in redirect_chains]
            t_reds = torch.tensor(red_indices, dtype=torch.long, device=self.device)
        else:
            red_indices = [self.vocab.encode_redirect_chain(None) for _ in urls]
            t_reds = torch.tensor(red_indices, dtype=torch.long, device=self.device)

        with torch.no_grad():
            probas = self.model.predict_proba(t_urls, t_reds).squeeze(-1).cpu().numpy()

        total_latency = (time.perf_counter() - t0) * 1000.0
        avg_latency = total_latency / max(1, len(urls))

        results = []
        for idx, (url, proba) in enumerate(zip(urls, probas)):
            p = float(proba)
            is_phish = bool(p >= tau)
            conf = p if is_phish else (1.0 - p)
            red_chain = redirect_chains[idx] if redirect_chains and idx < len(redirect_chains) else None

            results.append({
                "url": url,
                "model_id": "M-07",
                "architecture": "URLSequenceLSTM",
                "prediction_label": 1 if is_phish else 0,
                "prediction_name": "PHISHING" if is_phish else "LEGITIMATE",
                "phishing_probability": round(p, 5),
                "confidence": round(conf, 5),
                "decision_threshold": round(tau, 4),
                "redirect_chain_analyzed": bool(red_chain and len(red_chain) > 0),
                "num_redirect_hops": len(red_chain) if red_chain else 0,
                "latency_ms": round(avg_latency, 3),
            })

        return results
