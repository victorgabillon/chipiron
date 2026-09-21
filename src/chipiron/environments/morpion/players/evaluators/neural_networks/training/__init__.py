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
from .relational_entity_token_cache import (
    MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT,
    InvalidMorpionRelationalEntityTokenCacheError,
    MorpionRelationalEntityTokenCache,
    MorpionRelationalEntityTokenCacheManifest,
    MorpionRelationalEntityTokenCachePaths,
    default_relational_entity_token_cache_paths,
    load_or_materialize_relational_entity_token_cache,
    load_relational_entity_token_cache,
    relational_entity_token_cache_batch,
    relational_entity_token_cache_is_valid,
)
from .service import train_morpion_regressor, train_morpion_regressor_streaming
from .streaming import morpion_streaming_split_policy

__all__ = [
    "MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT",
    "InvalidMorpionRelationalEntityTokenCacheError",
    "InvalidValidationFractionError",
    "MorpionDiagnosticPredictionResult",
    "MorpionRelationalEntityTokenCache",
    "MorpionRelationalEntityTokenCacheManifest",
    "MorpionRelationalEntityTokenCachePaths",
    "MorpionStreamingTrainingArgs",
    "MorpionTrainingArgs",
    "MorpionTrainingProgressCallback",
    "UnsupportedMorpionDiagnosticInputFormatError",
    "default_relational_entity_token_cache_paths",
    "is_flat_morpion_training_model_kind",
    "load_or_materialize_relational_entity_token_cache",
    "load_relational_entity_token_cache",
    "morpion_streaming_split_policy",
    "predict_morpion_rows_for_diagnostics",
    "relational_entity_token_cache_batch",
    "relational_entity_token_cache_is_valid",
    "train_morpion_regressor",
    "train_morpion_regressor_streaming",
    "try_predict_morpion_rows_for_diagnostics",
]
