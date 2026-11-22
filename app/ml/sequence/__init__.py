"""Sequence Learning Subpackage for PhishVision AI (Phase 7.8).

Provides sequential neural architectures (M-07 LSTM) for URL character and token
pattern analysis, redirect sequence modeling, and sequential phishing detection.
"""

from app.ml.sequence.lstm_model import URLSequenceLSTM
from app.ml.sequence.lstm_inference import LSTMInferencer
from app.ml.sequence.sequence_preprocessing import URLSequenceVocabulary, SequenceDataset

__all__ = [
    "URLSequenceLSTM",
    "LSTMInferencer",
    "URLSequenceVocabulary",
    "SequenceDataset",
]
