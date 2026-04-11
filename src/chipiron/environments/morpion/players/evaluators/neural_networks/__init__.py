"""Morpion neural-network evaluator helpers."""

from typing import TYPE_CHECKING, Any

from .bundle import (
    MORPION_MANIFEST_FILE_NAME,
    MORPION_MODEL_ARGS_FILE_NAME,
    MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME,
    MORPION_MODEL_WEIGHTS_FILE_NAME,
    IncompatibleMorpionModelBundleError,
    InvalidMorpionModelBundleError,
    MorpionModelManifest,
    load_morpion_model_bundle,
    load_morpion_regressor_for_inference,
    save_morpion_model_bundle,
)
from .feature_extractor import (
    CandidateSegment,
    extract_morpion_features,
    morpion_feature_names,
)
from .model import (
    MORPION_FEATURE_SCHEMA,
    MORPION_INPUT_DIM,
    MissingMorpionHiddenDimError,
    MorpionRegressor,
    MorpionRegressorArgs,
    UnsupportedMorpionModelKindError,
    build_morpion_regressor,
)
from .morpion_nn_input import MorpionNNInput, build_morpion_nn_input
from .state_to_tensor import (
    MorpionFeatureTensorConverter,
    morpion_input_dim,
    morpion_state_to_tensor,
)

if TYPE_CHECKING:
    from .train import MorpionTrainingArgs

__all__ = [
    "MORPION_FEATURE_SCHEMA",
    "MORPION_INPUT_DIM",
    "MORPION_MANIFEST_FILE_NAME",
    "MORPION_MODEL_ARGS_FILE_NAME",
    "MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME",
    "MORPION_MODEL_WEIGHTS_FILE_NAME",
    "CandidateSegment",
    "IncompatibleMorpionModelBundleError",
    "InvalidMorpionModelBundleError",
    "MissingMorpionHiddenDimError",
    "MorpionFeatureTensorConverter",
    "MorpionModelManifest",
    "MorpionNNInput",
    "MorpionRegressor",
    "MorpionRegressorArgs",
    "MorpionTrainingArgs",
    "UnsupportedMorpionModelKindError",
    "build_morpion_nn_input",
    "build_morpion_regressor",
    "extract_morpion_features",
    "load_morpion_model_bundle",
    "load_morpion_regressor_for_inference",
    "morpion_feature_names",
    "morpion_input_dim",
    "morpion_state_to_tensor",
    "save_morpion_model_bundle",
    "train_morpion_regressor",
]


def __getattr__(name: str) -> Any:
    """Lazily import training helpers to avoid dataset import cycles."""
    if name in {"MorpionTrainingArgs", "train_morpion_regressor"}:
        from .train import MorpionTrainingArgs, train_morpion_regressor

        exports = {
            "MorpionTrainingArgs": MorpionTrainingArgs,
            "train_morpion_regressor": train_morpion_regressor,
        }
        return exports[name]
    raise AttributeError(name)
