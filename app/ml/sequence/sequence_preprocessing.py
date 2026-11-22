"""Sequence Preprocessing and Tokenization for M-07 LSTM (Phase 7.8).

Implements strict zero-leakage vocabulary fitting, character and token-level encoding,
padding, truncation, and dataset packaging for URL sequences and redirect chains.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("phishvision.sequence_preprocessing")

# Special token constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
NONE_TOKEN = "<NONE>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class URLSequenceVocabulary:
    """Manages character-level and token-level vocabularies for URL sequences.
    
    CRITICAL SCIENTIFIC INTEGRITY RULE:
    Vocabulary MUST be fitted strictly on training data records only.
    Validation and test data must NEVER be used during vocabulary fitting.
    """

    def __init__(
        self,
        max_url_len: int = 256,
        max_redirect_len: int = 8,
        min_char_freq: int = 1,
        min_token_freq: int = 1,
    ) -> None:
        self.max_url_len = max_url_len
        self.max_redirect_len = max_redirect_len
        self.min_char_freq = min_char_freq
        self.min_token_freq = min_token_freq

        # URL character mapping
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}

        # Redirect hostname / domain token mapping
        self.redirect_to_idx: dict[str, int] = {}
        self.idx_to_redirect: dict[int, str] = {}

        self.is_fitted: bool = False

    def fit(
        self,
        urls: list[str] | pd.Series,
        redirect_chains: list[list[str]] | None = None,
    ) -> URLSequenceVocabulary:
        """Fit vocabulary strictly on training URLs and optional redirect observations.
        
        Args:
            urls: Collection of training URL strings.
            redirect_chains: Optional collection of training redirect hostname lists.
        """
        # 1. Build character vocabulary
        char_counts: dict[str, int] = {}
        for url in urls:
            if not isinstance(url, str):
                continue
            for ch in url:
                char_counts[ch] = char_counts.get(ch, 0) + 1

        self.char_to_idx = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.idx_to_char = {idx: tok for tok, idx in self.char_to_idx.items()}

        next_idx = len(SPECIAL_TOKENS)
        for ch, count in sorted(char_counts.items(), key=lambda x: (-x[1], x[0])):
            if count >= self.min_char_freq and ch not in self.char_to_idx:
                self.char_to_idx[ch] = next_idx
                self.idx_to_char[next_idx] = ch
                next_idx += 1

        # 2. Build redirect token vocabulary (handling empty/absent redirect data gracefully)
        redirect_special = [PAD_TOKEN, UNK_TOKEN, NONE_TOKEN]
        self.redirect_to_idx = {tok: idx for idx, tok in enumerate(redirect_special)}
        self.idx_to_redirect = {idx: tok for tok, idx in self.redirect_to_idx.items()}

        if redirect_chains:
            token_counts: dict[str, int] = {}
            for chain in redirect_chains:
                if not chain:
                    continue
                for hop in chain:
                    if isinstance(hop, str) and hop.strip():
                        # Normalize hostname tokens
                        norm_hop = hop.lower().strip()
                        token_counts[norm_hop] = token_counts.get(norm_hop, 0) + 1

            next_red_idx = len(redirect_special)
            for tok, count in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
                if count >= self.min_token_freq and tok not in self.redirect_to_idx:
                    self.redirect_to_idx[tok] = next_red_idx
                    self.idx_to_redirect[next_red_idx] = tok
                    next_red_idx += 1

        self.is_fitted = True
        logger.info(
            f"Fitted URL vocabulary: {len(self.char_to_idx)} characters, "
            f"{len(self.redirect_to_idx)} redirect tokens."
        )
        return self

    def encode_url(self, url: str) -> list[int]:
        """Encode a single URL into character index sequence with truncation/padding."""
        if not self.is_fitted:
            raise RuntimeError("Vocabulary must be fitted before encoding.")

        pad_id = self.char_to_idx[PAD_TOKEN]
        unk_id = self.char_to_idx[UNK_TOKEN]

        if not isinstance(url, str) or not url:
            return [pad_id] * self.max_url_len

        # Convert characters to indices
        indices = [self.char_to_idx.get(ch, unk_id) for ch in url]

        # Truncate if longer than max_len
        if len(indices) > self.max_url_len:
            indices = indices[: self.max_url_len]
        else:
            # Pad to max_len
            indices = indices + [pad_id] * (self.max_url_len - len(indices))

        return indices

    def encode_redirect_chain(self, redirect_chain: list[str] | None) -> list[int]:
        """Encode a redirect hostname sequence with truncation/padding."""
        if not self.is_fitted:
            raise RuntimeError("Vocabulary must be fitted before encoding.")

        pad_id = self.redirect_to_idx[PAD_TOKEN]
        unk_id = self.redirect_to_idx[UNK_TOKEN]
        none_id = self.redirect_to_idx[NONE_TOKEN]

        if not redirect_chain:
            # Empty / no redirect chain: fill with NONE_TOKEN followed by PAD
            indices = [none_id]
        else:
            indices = [
                self.redirect_to_idx.get(h.lower().strip(), unk_id)
                for h in redirect_chain
                if isinstance(h, str) and h.strip()
            ]
            if not indices:
                indices = [none_id]

        if len(indices) > self.max_redirect_len:
            indices = indices[: self.max_redirect_len]
        else:
            indices = indices + [pad_id] * (self.max_redirect_len - len(indices))

        return indices

    @property
    def url_vocab_size(self) -> int:
        return len(self.char_to_idx)

    @property
    def redirect_vocab_size(self) -> int:
        return len(self.redirect_to_idx)

    def to_dict(self) -> dict[str, Any]:
        """Serialize vocabulary mappings to JSON-compatible dictionary."""
        return {
            "max_url_len": self.max_url_len,
            "max_redirect_len": self.max_redirect_len,
            "min_char_freq": self.min_char_freq,
            "min_token_freq": self.min_token_freq,
            "char_to_idx": self.char_to_idx,
            "redirect_to_idx": self.redirect_to_idx,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> URLSequenceVocabulary:
        """Instantiate vocabulary from serialized dictionary."""
        vocab = cls(
            max_url_len=data["max_url_len"],
            max_redirect_len=data["max_redirect_len"],
            min_char_freq=data.get("min_char_freq", 1),
            min_token_freq=data.get("min_token_freq", 2),
        )
        vocab.char_to_idx = {k: int(v) for k, v in data["char_to_idx"].items()}
        vocab.idx_to_char = {int(v): k for k, v in vocab.char_to_idx.items()}
        vocab.redirect_to_idx = {k: int(v) for k, v in data["redirect_to_idx"].items()}
        vocab.idx_to_redirect = {int(v): k for k, v in vocab.redirect_to_idx.items()}
        vocab.is_fitted = data.get("is_fitted", True)
        return vocab

    def save(self, filepath: str | Path) -> None:
        """Save vocabulary to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved vocabulary to: {p}")

    @classmethod
    def load(cls, filepath: str | Path) -> URLSequenceVocabulary:
        """Load vocabulary from JSON file."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class SequenceDataset(Dataset):
    """PyTorch Dataset yielding character sequences and binary labels.
    
    Supports static URLs and dynamic redirect chain sequences.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: URLSequenceVocabulary,
        url_column: str = "canonical_url",
        label_column: str = "label",
        redirect_column: str | None = None,
    ) -> None:
        self.vocab = vocab
        self.url_column = url_column if url_column in df.columns else "original_url"
        self.label_column = label_column
        self.redirect_column = redirect_column

        # Pre-extract values for fast indexing
        self.urls: list[str] = df[self.url_column].astype(str).tolist()
        self.labels: list[int] = (
            df[self.label_column].astype(int).tolist()
            if self.label_column in df.columns
            else [0] * len(self.urls)
        )

        self.has_redirects = (
            redirect_column is not None
            and redirect_column in df.columns
            and df[redirect_column].notna().any()
        )
        if self.has_redirects:
            self.redirects = df[redirect_column].tolist()
        else:
            self.redirects = [None] * len(self.urls)

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        url = self.urls[idx]
        label = self.labels[idx]
        redirect = self.redirects[idx]

        url_seq = self.vocab.encode_url(url)
        redirect_seq = self.vocab.encode_redirect_chain(redirect)

        return {
            "url_seq": torch.tensor(url_seq, dtype=torch.long),
            "redirect_seq": torch.tensor(redirect_seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }
