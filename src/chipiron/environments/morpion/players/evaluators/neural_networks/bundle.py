"""Save/load helpers for Morpion regressor bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from chipiron.learning.torch_runtime import state_dict_on_cpu

from .feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MORPION_CANONICAL_FEATURE_NAMES,
    MORPION_FEATURE_SCHEMA,
    MorpionFeatureSubset,
    full_morpion_feature_subset,
    resolve_morpion_feature_subset,
)
from .graph_tokens import (
    MORPION_GRAPH_INPUT_REPRESENTATION,
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MORPION_GRAPH_TOKEN_FEATURE_NAMES,
    is_morpion_entity_token_transformer_model_kind,
)
from .model import (
    MORPION_INPUT_DIM,
    MorpionRegressor,
    MorpionRegressorArgs,
    build_morpion_regressor,
)

MORPION_MODEL_ARGS_FILE_NAME = "morpion_regressor_args.json"
MORPION_MANIFEST_FILE_NAME = "morpion_manifest.json"
MORPION_MODEL_WEIGHTS_FILE_NAME = "param.pt"
MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME = "param.json"


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionModelManifest:
    """Sidecar manifest protecting Morpion model-bundle compatibility."""

    game_kind: str = "morpion"
    feature_schema: str = MORPION_FEATURE_SCHEMA
    input_dim: int = MORPION_INPUT_DIM
    input_representation: str = "handcrafted_features"
    target_kind: str = "backup_value"
    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(
        default_factory=lambda: MORPION_CANONICAL_FEATURE_NAMES
    )
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


class InvalidMorpionModelBundleError(ValueError):
    """Raised when a Morpion model bundle is structurally malformed."""

    @classmethod
    def invalid_model_args_mapping(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-model-args-mapping error."""
        return cls(
            f"Invalid Morpion model args in {path!s}: expected mapping with string keys."
        )

    @classmethod
    def invalid_model_kind(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-model-kind error."""
        return cls(
            f"Invalid Morpion model args in {path!s}: `model_kind` must be a string."
        )

    @classmethod
    def invalid_hidden_sizes(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-hidden-sizes error."""
        return cls(
            f"Invalid Morpion model args in {path!s}: `hidden_sizes` must be a list "
            "or tuple of integer-like values."
        )

    @classmethod
    def invalid_feature_subset_name(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-feature-subset-name error."""
        return cls(
            f"Invalid Morpion model bundle metadata in {path!s}: `feature_subset_name` must be a string."
        )

    @classmethod
    def invalid_feature_names(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-feature-names error."""
        return cls(
            f"Invalid Morpion model bundle metadata in {path!s}: `feature_names` must be a list or tuple of strings."
        )

    @classmethod
    def missing_feature_subset_metadata(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the missing-feature-subset-metadata error."""
        return cls(
            "Invalid Morpion model bundle metadata in "
            f"{path!s}: bundles whose input_dim differs from the canonical full "
            "feature width must persist explicit feature subset metadata."
        )

    @classmethod
    def inconsistent_input_dim(
        cls,
        path: Path,
        *,
        input_dim: int,
        expected_input_dim: int,
    ) -> InvalidMorpionModelBundleError:
        """Return the inconsistent-input-dimension error."""
        return cls(
            f"Invalid Morpion model bundle metadata in {path!s}: input_dim={input_dim} "
            f"does not match the resolved feature subset width {expected_input_dim}."
        )

    @classmethod
    def invalid_manifest_mapping(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-manifest-mapping error."""
        return cls(
            f"Invalid Morpion model manifest in {path!s}: expected mapping with string keys."
        )

    @classmethod
    def invalid_manifest_metadata(
        cls,
        path: Path,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-manifest-metadata error."""
        return cls(
            f"Invalid Morpion model manifest in {path!s}: `metadata` must be a mapping."
        )

    @classmethod
    def invalid_integer_like_value(
        cls,
        value: object,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-integer-like-value error."""
        return cls(f"Expected an integer-like value, got {type(value).__name__}.")

    @classmethod
    def invalid_float_like_value(
        cls,
        value: object,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-float-like-value error."""
        return cls(f"Expected a float-like value, got {type(value).__name__}.")

    @classmethod
    def invalid_bool_value(
        cls,
        value: object,
    ) -> InvalidMorpionModelBundleError:
        """Return the invalid-bool-value error."""
        return cls(f"Expected a bool value, got {type(value).__name__}.")


class IncompatibleMorpionModelBundleError(ValueError):
    """Raised when a Morpion model bundle is incompatible with current code."""

    @classmethod
    def wrong_game_kind(
        cls,
        game_kind: str,
    ) -> IncompatibleMorpionModelBundleError:
        """Return the incompatible-game-kind error."""
        return cls(f"Expected a Morpion model bundle, got game_kind={game_kind!r}.")

    @classmethod
    def wrong_feature_schema(
        cls,
        feature_schema: str,
    ) -> IncompatibleMorpionModelBundleError:
        """Return the incompatible-feature-schema error."""
        return cls(
            f"Expected feature_schema={MORPION_FEATURE_SCHEMA!r}, got "
            f"{feature_schema!r}."
        )

    @classmethod
    def wrong_input_dim(
        cls,
        *,
        expected_input_dim: int,
        actual_input_dim: int,
    ) -> IncompatibleMorpionModelBundleError:
        """Return the incompatible-input-dimension error."""
        return cls(f"Expected input_dim={expected_input_dim}, got {actual_input_dim}.")

    @classmethod
    def wrong_feature_names(
        cls,
        *,
        expected_feature_names: tuple[str, ...],
        actual_feature_names: tuple[str, ...],
    ) -> IncompatibleMorpionModelBundleError:
        """Return the incompatible-feature-names error."""
        return cls(
            "Expected feature_names="
            f"{expected_feature_names!r}, got {actual_feature_names!r}."
        )


def save_morpion_model_bundle(
    model: MorpionRegressor,
    output_dir: str | Path,
    *,
    model_args: MorpionRegressorArgs,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save one Morpion model bundle with weights, args, and manifest."""
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    weights_path = bundle_dir / MORPION_MODEL_WEIGHTS_FILE_NAME
    args_path = bundle_dir / MORPION_MODEL_ARGS_FILE_NAME
    manifest_path = bundle_dir / MORPION_MANIFEST_FILE_NAME
    readable_weights_path = bundle_dir / MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME

    torch.save(state_dict_on_cpu(model), weights_path)
    model.log_readable_model_weights_to_file(str(readable_weights_path))

    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(_model_args_to_dict(model_args), handle, indent=2, sort_keys=True)

    manifest = MorpionModelManifest(
        input_dim=model_args.input_dim,
        input_representation=(
            MORPION_GRAPH_INPUT_REPRESENTATION
            if is_morpion_entity_token_transformer_model_kind(model_args.model_kind)
            else "handcrafted_features"
        ),
        model_kind=model_args.model_kind,
        feature_subset_name=model_args.feature_subset_name,
        feature_names=model_args.feature_names,
        metadata=_bundle_metadata(model_args, metadata),
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(_manifest_to_dict(manifest), handle, indent=2, sort_keys=True)


def load_morpion_model_bundle(
    input_dir: str | Path,
) -> tuple[MorpionRegressor, MorpionRegressorArgs, MorpionModelManifest]:
    """Load one Morpion model bundle and validate manifest compatibility."""
    bundle_dir = Path(input_dir)
    args_path = bundle_dir / MORPION_MODEL_ARGS_FILE_NAME
    manifest_path = bundle_dir / MORPION_MANIFEST_FILE_NAME
    weights_path = bundle_dir / MORPION_MODEL_WEIGHTS_FILE_NAME

    model_args = _load_model_args(args_path)
    manifest = _load_manifest(manifest_path)
    _validate_manifest_compatibility(manifest, model_args)

    model = build_morpion_regressor(model_args)
    model.load_weights_from_file(str(weights_path))
    return model, model_args, manifest


def load_morpion_regressor_for_inference(
    input_dir: str | Path,
) -> MorpionRegressor:
    """Load one Morpion regressor bundle and switch the model to eval mode."""
    model, _, _ = load_morpion_model_bundle(input_dir)
    model.eval()
    return model


def _bundle_metadata(
    model_args: MorpionRegressorArgs,
    metadata: dict[str, object] | None,
) -> dict[str, object]:
    """Return manifest metadata augmented with graph-token schema details."""
    bundle_metadata = dict(metadata) if metadata is not None else {}
    if is_morpion_entity_token_transformer_model_kind(model_args.model_kind):
        bundle_metadata.update(
            {
                "graph_token_feature_names": list(MORPION_GRAPH_TOKEN_FEATURE_NAMES),
                "graph_max_tokens": model_args.graph_max_tokens,
                "graph_d_model": model_args.graph_d_model,
                "graph_n_head": model_args.graph_n_head,
                "graph_n_layer": model_args.graph_n_layer,
            }
        )
    return bundle_metadata


def _load_model_args(path: Path) -> MorpionRegressorArgs:
    """Load Morpion regressor args from one JSON file."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not _is_str_key_mapping(raw):
        raise InvalidMorpionModelBundleError.invalid_model_args_mapping(path)
    data = cast("Mapping[str, object]", raw)
    model_kind = data.get("model_kind", "linear")
    input_dim = data.get("input_dim", MORPION_INPUT_DIM)
    if not isinstance(model_kind, str):
        raise InvalidMorpionModelBundleError.invalid_model_kind(path)
    feature_subset = (
        full_morpion_feature_subset()
        if is_morpion_entity_token_transformer_model_kind(model_kind)
        else _load_feature_subset(
            data,
            path,
            input_dim=_coerce_int(input_dim),
        )
    )
    return MorpionRegressorArgs(
        model_kind=model_kind,
        feature_subset_name=feature_subset.name,
        feature_names=feature_subset.feature_names,
        hidden_sizes=_load_hidden_sizes(data, path),
        graph_max_tokens=_coerce_int(data.get("graph_max_tokens", 1536)),
        graph_input_feature_dim=_coerce_int(
            data.get("graph_input_feature_dim", MORPION_GRAPH_TOKEN_FEATURE_DIM)
        ),
        graph_d_model=_coerce_int(data.get("graph_d_model", 64)),
        graph_n_head=_coerce_int(data.get("graph_n_head", 4)),
        graph_n_layer=_coerce_int(data.get("graph_n_layer", 2)),
        graph_dim_feedforward=_coerce_int(data.get("graph_dim_feedforward", 256)),
        graph_dropout_ratio=_coerce_float(data.get("graph_dropout_ratio", 0.0)),
        graph_pooling=str(data.get("graph_pooling", "value_token")),
        graph_output_tanh=_coerce_bool(data.get("graph_output_tanh", False)),
    )


def _load_manifest(path: Path) -> MorpionModelManifest:
    """Load the Morpion model manifest from one JSON file."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not _is_str_key_mapping(raw):
        raise InvalidMorpionModelBundleError.invalid_manifest_mapping(path)
    data = cast("Mapping[str, object]", raw)
    metadata = data.get("metadata")
    if metadata is None:
        metadata_dict: dict[str, Any] = {}
    elif isinstance(metadata, dict):
        metadata_dict = dict(cast("dict[str, Any]", metadata))
    else:
        raise InvalidMorpionModelBundleError.invalid_manifest_metadata(path)
    input_dim = _coerce_int(data.get("input_dim", MORPION_INPUT_DIM))
    model_kind = str(data.get("model_kind", "linear"))
    input_representation = str(data.get("input_representation", "handcrafted_features"))
    feature_subset = (
        full_morpion_feature_subset()
        if is_morpion_entity_token_transformer_model_kind(model_kind)
        else _load_feature_subset(data, path, input_dim=input_dim)
    )
    return MorpionModelManifest(
        game_kind=str(data.get("game_kind", "morpion")),
        feature_schema=str(data.get("feature_schema", MORPION_FEATURE_SCHEMA)),
        input_dim=input_dim,
        input_representation=input_representation,
        target_kind=str(data.get("target_kind", "backup_value")),
        model_kind=model_kind,
        feature_subset_name=feature_subset.name,
        feature_names=feature_subset.feature_names,
        metadata=metadata_dict,
    )


def _manifest_to_dict(manifest: MorpionModelManifest) -> dict[str, object]:
    """Serialize one Morpion model manifest into JSON-friendly data."""
    return {
        "game_kind": manifest.game_kind,
        "feature_schema": manifest.feature_schema,
        "input_dim": manifest.input_dim,
        "input_representation": manifest.input_representation,
        "target_kind": manifest.target_kind,
        "model_kind": manifest.model_kind,
        "feature_subset_name": manifest.feature_subset_name,
        "feature_names": list(manifest.feature_names),
        "metadata": dict(manifest.metadata),
    }


def _validate_manifest_compatibility(
    manifest: MorpionModelManifest,
    model_args: MorpionRegressorArgs,
) -> None:
    """Validate that the loaded Morpion manifest matches current code."""
    if manifest.game_kind != "morpion":
        raise IncompatibleMorpionModelBundleError.wrong_game_kind(manifest.game_kind)
    if is_morpion_entity_token_transformer_model_kind(model_args.model_kind):
        if not is_morpion_entity_token_transformer_model_kind(manifest.model_kind):
            raise IncompatibleMorpionModelBundleError.wrong_input_dim(
                expected_input_dim=model_args.input_dim,
                actual_input_dim=manifest.input_dim,
            )
        if manifest.input_representation != MORPION_GRAPH_INPUT_REPRESENTATION:
            raise IncompatibleMorpionModelBundleError.wrong_feature_schema(
                manifest.input_representation
            )
        if manifest.input_dim != model_args.graph_input_feature_dim:
            raise IncompatibleMorpionModelBundleError.wrong_input_dim(
                expected_input_dim=model_args.graph_input_feature_dim,
                actual_input_dim=manifest.input_dim,
            )
        return
    if manifest.feature_schema != MORPION_FEATURE_SCHEMA:
        raise IncompatibleMorpionModelBundleError.wrong_feature_schema(
            manifest.feature_schema
        )
    if manifest.input_dim != model_args.input_dim:
        raise IncompatibleMorpionModelBundleError.wrong_input_dim(
            expected_input_dim=model_args.input_dim,
            actual_input_dim=manifest.input_dim,
        )
    if manifest.feature_names != model_args.feature_names:
        raise IncompatibleMorpionModelBundleError.wrong_feature_names(
            expected_feature_names=model_args.feature_names,
            actual_feature_names=manifest.feature_names,
        )


def _model_args_to_dict(model_args: MorpionRegressorArgs) -> dict[str, object]:
    """Serialize Morpion regressor args into JSON-friendly data."""
    data: dict[str, object] = {
        "model_kind": model_args.model_kind,
        "input_dim": model_args.input_dim,
        "feature_subset_name": model_args.feature_subset_name,
        "feature_names": list(model_args.feature_names),
        "hidden_sizes": None
        if model_args.hidden_sizes is None
        else list(model_args.hidden_sizes),
    }
    if is_morpion_entity_token_transformer_model_kind(model_args.model_kind):
        data.update(
            {
                "graph_max_tokens": model_args.graph_max_tokens,
                "graph_input_feature_dim": model_args.graph_input_feature_dim,
                "graph_d_model": model_args.graph_d_model,
                "graph_n_head": model_args.graph_n_head,
                "graph_n_layer": model_args.graph_n_layer,
                "graph_dim_feedforward": model_args.graph_dim_feedforward,
                "graph_dropout_ratio": model_args.graph_dropout_ratio,
                "graph_pooling": model_args.graph_pooling,
                "graph_output_tanh": model_args.graph_output_tanh,
            }
        )
    return data


def _load_feature_subset(
    data: Mapping[str, object],
    path: Path,
    *,
    input_dim: int,
) -> MorpionFeatureSubset:
    """Load one explicit or legacy Morpion feature subset payload."""
    raw_feature_subset_name = data.get("feature_subset_name")
    if raw_feature_subset_name is not None and not isinstance(
        raw_feature_subset_name, str
    ):
        raise InvalidMorpionModelBundleError.invalid_feature_subset_name(path)

    raw_feature_names = data.get("feature_names")
    if raw_feature_names is None:
        feature_names: tuple[str, ...] | None = None
    elif not isinstance(raw_feature_names, list | tuple):
        raise InvalidMorpionModelBundleError.invalid_feature_names(path)
    else:
        typed_feature_names = cast(
            "list[object] | tuple[object, ...]",
            raw_feature_names,
        )
        if not all(isinstance(item, str) for item in typed_feature_names):
            raise InvalidMorpionModelBundleError.invalid_feature_names(path)
        feature_names = tuple(cast("str", item) for item in typed_feature_names)

    if raw_feature_subset_name is None and feature_names is None:
        if input_dim != MORPION_INPUT_DIM:
            raise InvalidMorpionModelBundleError.missing_feature_subset_metadata(path)
        return full_morpion_feature_subset()

    subset = resolve_morpion_feature_subset(
        feature_subset_name=raw_feature_subset_name,
        feature_names=feature_names,
    )
    if subset.dimension != input_dim:
        raise InvalidMorpionModelBundleError.inconsistent_input_dim(
            path,
            input_dim=input_dim,
            expected_input_dim=subset.dimension,
        )
    return subset


def _is_str_key_mapping(obj: object) -> bool:
    """Return whether ``obj`` is a mapping with string keys."""
    if not isinstance(obj, Mapping):
        return False
    mapping = cast("Mapping[object, object]", obj)
    return all(isinstance(key, str) for key in mapping)


def _coerce_int(value: object) -> int:
    """Return one JSON-loaded integer-like payload as ``int``."""
    if isinstance(value, bool):
        raise InvalidMorpionModelBundleError.invalid_integer_like_value(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise InvalidMorpionModelBundleError.invalid_integer_like_value(value)
    if isinstance(value, str):
        return int(value)
    raise InvalidMorpionModelBundleError.invalid_integer_like_value(value)


def _coerce_float(value: object) -> float:
    """Return one JSON-loaded float-like payload as ``float``."""
    if isinstance(value, bool):
        raise InvalidMorpionModelBundleError.invalid_float_like_value(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise InvalidMorpionModelBundleError.invalid_float_like_value(value)


def _coerce_bool(value: object) -> bool:
    """Return one JSON-loaded bool payload as ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    raise InvalidMorpionModelBundleError.invalid_bool_value(value)


def _load_hidden_sizes(
    data: Mapping[str, object],
    path: Path,
) -> tuple[int, ...] | None:
    """Load current or legacy hidden-layer settings from one args payload."""
    hidden_sizes = data.get("hidden_sizes")
    if hidden_sizes is None and "hidden_dim" in data:
        hidden_dim = data.get("hidden_dim")
        return None if hidden_dim is None else (_coerce_int(hidden_dim),)
    if hidden_sizes is None:
        return None
    if not isinstance(hidden_sizes, list | tuple):
        raise InvalidMorpionModelBundleError.invalid_hidden_sizes(path)
    typed_hidden_sizes = cast("list[object] | tuple[object, ...]", hidden_sizes)
    try:
        return tuple(_coerce_int(item) for item in typed_hidden_sizes)
    except ValueError as exc:
        raise InvalidMorpionModelBundleError.invalid_hidden_sizes(path) from exc


__all__ = [
    "MORPION_MANIFEST_FILE_NAME",
    "MORPION_MODEL_ARGS_FILE_NAME",
    "MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME",
    "MORPION_MODEL_WEIGHTS_FILE_NAME",
    "IncompatibleMorpionModelBundleError",
    "InvalidMorpionModelBundleError",
    "MorpionModelManifest",
    "load_morpion_model_bundle",
    "load_morpion_regressor_for_inference",
    "save_morpion_model_bundle",
]
