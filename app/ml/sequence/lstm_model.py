"""M-07 URL Sequence LSTM Architecture for Sequential Phishing Detection (Phase 7.8).

Implements a bidirectional LSTM neural network with attentive temporal pooling,
character-level embedding of raw and canonical URLs, and conditional redirect-sequence modeling.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("phishvision.lstm_model")


class AttentionPooling(nn.Module):
    """Self-attentive temporal pooling layer.
    
    Computes a learned weighted sum across sequential hidden states:
        score_t = v^T tanh(W h_t + b)
        alpha_t = softmax(score_t)
        context = sum_t (alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attentive context vector.
        
        Args:
            hidden_states: Tensor of shape (B, T, D).
            mask: Optional boolean mask of shape (B, T) where True indicates valid tokens.
        
        Returns:
            context: Attended representation of shape (B, D).
            weights: Attention weights of shape (B, T).
        """
        # (B, T, D) -> (B, T, D)
        energy = torch.tanh(self.projection(hidden_states))
        # (B, T, D) -> (B, T, 1) -> (B, T)
        scores = self.context_vector(energy).squeeze(-1)

        if mask is not None:
            # Mask out padding tokens with a large negative value before softmax
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        # (B, 1, T) @ (B, T, D) -> (B, 1, D) -> (B, D)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class URLSequenceLSTM(nn.Module):
    """M-07 Bidirectional LSTM Phishing Website Classifier.
    
    Processes sequential URL character patterns with attentive pooling and optional
    redirect-sequence embeddings.
    """

    def __init__(
        self,
        url_vocab_size: int = 128,
        url_embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        fc_dim: int = 64,
        use_redirects: bool = True,
        redirect_vocab_size: int = 32,
        redirect_embed_dim: int = 32,
        redirect_hidden_dim: int = 32,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.url_vocab_size = url_vocab_size
        self.url_embed_dim = url_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.use_redirects = use_redirects
        self.padding_idx = padding_idx

        # 1. URL Character Embedding & LSTM Branch
        self.url_embedding = nn.Embedding(
            num_embeddings=url_vocab_size,
            embedding_dim=url_embed_dim,
            padding_idx=padding_idx,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.url_lstm = nn.LSTM(
            input_size=url_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        url_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.url_attention = AttentionPooling(hidden_dim=url_out_dim)

        # 2. Redirect Sequence Branch (Conditional dynamic scanner integration)
        if use_redirects:
            self.redirect_embedding = nn.Embedding(
                num_embeddings=redirect_vocab_size,
                embedding_dim=redirect_embed_dim,
                padding_idx=padding_idx,
            )
            self.redirect_lstm = nn.LSTM(
                input_size=redirect_embed_dim,
                hidden_size=redirect_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            combined_dim = url_out_dim + redirect_hidden_dim
        else:
            self.redirect_embedding = None
            self.redirect_lstm = None
            combined_dim = url_out_dim

        # 3. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(
        self,
        url_seq: torch.Tensor,
        redirect_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute forward pass logits.
        
        Args:
            url_seq: LongTensor of shape (B, T_url) with character indices.
            redirect_seq: Optional LongTensor of shape (B, T_red) with redirect tokens.
        
        Returns:
            logits: Unnormalized logits of shape (B, 1).
        """
        batch_size = url_seq.size(0)

        # Create padding mask for attention pooling (True for non-pad tokens)
        url_mask = url_seq != self.padding_idx

        # Embed and process URL sequence
        # (B, T) -> (B, T, D_emb)
        embedded_url = self.url_embedding(url_seq)
        # (B, T, D_emb) -> (B, T, D_lstm)
        url_lstm_out, _ = self.url_lstm(embedded_url)
        # Attentive pooling: (B, T, D_lstm) -> (B, D_lstm)
        url_features, _ = self.url_attention(url_lstm_out, mask=url_mask)

        # Process redirect sequence if enabled
        if self.use_redirects and self.redirect_embedding is not None and self.redirect_lstm is not None:
            if redirect_seq is not None:
                embedded_red = self.redirect_embedding(redirect_seq)
                red_lstm_out, (h_n, _) = self.redirect_lstm(embedded_red)
                # Use final hidden state from 1-layer LSTM: (1, B, D_red) -> (B, D_red)
                red_features = h_n.squeeze(0)
            else:
                # Fallback: zero representation when redirect sequence is omitted
                red_features = torch.zeros(
                    batch_size,
                    self.redirect_lstm.hidden_size,
                    device=url_seq.device,
                    dtype=url_features.dtype,
                )
            combined_features = torch.cat([url_features, red_features], dim=-1)
        else:
            combined_features = url_features

        # Classification logits: (B, 1)
        logits = self.classifier(combined_features)
        return logits

    def predict_proba(
        self,
        url_seq: torch.Tensor,
        redirect_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibrated phishing probabilities in [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(url_seq, redirect_seq)
            return torch.sigmoid(logits)

    @property
    def total_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> dict[str, Any]:
        """Return structured architectural telemetry."""
        return {
            "model_id": "M-07",
            "architecture": "URLSequenceLSTM",
            "url_vocab_size": self.url_vocab_size,
            "url_embed_dim": self.url_embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "use_redirects": self.use_redirects,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.total_parameters,
        }
