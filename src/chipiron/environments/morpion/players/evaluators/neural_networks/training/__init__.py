"""Public Morpion neural-network training API."""

from __future__ import annotations

from .args import (
    InvalidValidationFractionError,
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    MorpionTrainingProgressCallback,
)
from .diagnostics import (
    MorpionDiagnosticPredictionResult,
    UnsupportedMorpionDiagnosticInputFormatError,
    predict_morpion_rows_for_diagnostics,
    try_predict_morpion_rows_for_diagnostics,
)
from .flat_tensor_cache import is_flat_morpion_training_model_kind
from .service import train_morpion_regressor, train_morpion_regressor_streaming
from .streaming import morpion_streaming_split_policy

__all__ = [
    "InvalidValidationFractionError",
    "MorpionDiagnosticPredictionResult",
    "MorpionStreamingTrainingArgs",
    "MorpionTrainingArgs",
    "MorpionTrainingProgressCallback",
    "UnsupportedMorpionDiagnosticInputFormatError",
    "is_flat_morpion_training_model_kind",
    "morpion_streaming_split_policy",
    "predict_morpion_rows_for_diagnostics",
    "train_morpion_regressor",
    "train_morpion_regressor_streaming",
    "try_predict_morpion_rows_for_diagnostics",
]
