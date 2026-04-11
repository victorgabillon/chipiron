"""Save/load helpers for Morpion regressor bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from .model import (
    MORPION_FEATURE_SCHEMA,
    MORPION_INPUT_DIM,
    MorpionRegressor,
    MorpionRegressorArgs,
    build_morpion_regressor,
)

MORPION_MODEL_ARGS_FILE_NAME = "morpion_regressor_args.json"
MORPION_MANIFEST_FILE_NAME = "morpion_manifest.json"
MORPION_MODEL_WEIGHTS_FILE_NAME = "param.pt"
MORPION_MODEL_READABLE_WEIGHTS_FILE_NAME = "param.yaml"


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionModelManifest:
    """Sidecar manifest protecting Morpion model-bundle compatibility."""

    game_kind: str = "morpion"
    feature_schema: str = MORPION_FEATURE_SCHEMA
    input_dim: int = MORPION_INPUT_DIM
    target_kind: str = "backup_value"
    model_kind: str = "linear"
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
        return cls(
            f"Expected an integer-like value, got {type(value).__name__}."
        )


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
        input_dim: int,
    ) -> IncompatibleMorpionModelBundleError:
        """Return the incompatible-input-dimension error."""
        return cls(
            f"Expected input_dim={MORPION_INPUT_DIM}, got {input_dim}."
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

    torch.save(model.state_dict(), weights_path)
    model.log_readable_model_weights_to_file(str(readable_weights_path))

    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(model_args), handle, indent=2, sort_keys=True)

    manifest = MorpionModelManifest(
        model_kind=model_args.model_kind,
        metadata=dict(metadata) if metadata is not None else {},
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
    _validate_manifest_compatibility(manifest)

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


def _load_model_args(path: Path) -> MorpionRegressorArgs:
    """Load Morpion regressor args from one JSON file."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not _is_str_key_mapping(raw):
        raise InvalidMorpionModelBundleError.invalid_model_args_mapping(path)
    data = cast("Mapping[str, object]", raw)
    model_kind = data.get("model_kind", "linear")
    input_dim = data.get("input_dim", MORPION_INPUT_DIM)
    hidden_dim = data.get("hidden_dim")
    if not isinstance(model_kind, str):
        raise InvalidMorpionModelBundleError.invalid_model_kind(path)
    return MorpionRegressorArgs(
        model_kind=model_kind,
        input_dim=_coerce_int(input_dim),
        hidden_dim=None if hidden_dim is None else _coerce_int(hidden_dim),
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
    return MorpionModelManifest(
        game_kind=str(data.get("game_kind", "morpion")),
        feature_schema=str(data.get("feature_schema", MORPION_FEATURE_SCHEMA)),
        input_dim=_coerce_int(data.get("input_dim", MORPION_INPUT_DIM)),
        target_kind=str(data.get("target_kind", "backup_value")),
        model_kind=str(data.get("model_kind", "linear")),
        metadata=metadata_dict,
    )


def _manifest_to_dict(manifest: MorpionModelManifest) -> dict[str, object]:
    """Serialize one Morpion model manifest into JSON-friendly data."""
    return {
        "game_kind": manifest.game_kind,
        "feature_schema": manifest.feature_schema,
        "input_dim": manifest.input_dim,
        "target_kind": manifest.target_kind,
        "model_kind": manifest.model_kind,
        "metadata": dict(manifest.metadata),
    }


def _validate_manifest_compatibility(manifest: MorpionModelManifest) -> None:
    """Validate that the loaded Morpion manifest matches current code."""
    if manifest.game_kind != "morpion":
        raise IncompatibleMorpionModelBundleError.wrong_game_kind(manifest.game_kind)
    if manifest.feature_schema != MORPION_FEATURE_SCHEMA:
        raise IncompatibleMorpionModelBundleError.wrong_feature_schema(
            manifest.feature_schema
        )
    if manifest.input_dim != MORPION_INPUT_DIM:
        raise IncompatibleMorpionModelBundleError.wrong_input_dim(manifest.input_dim)


def _is_str_key_mapping(obj: object) -> bool:
    """Return whether ``obj`` is a mapping with string keys."""
    if not isinstance(obj, Mapping):
        return False
    mapping = cast("Mapping[object, object]", obj)
    return all(isinstance(key, str) for key in mapping)


def _coerce_int(value: object) -> int:
    """Return one JSON-loaded integer-like payload as ``int``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    raise InvalidMorpionModelBundleError.invalid_integer_like_value(value)


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
