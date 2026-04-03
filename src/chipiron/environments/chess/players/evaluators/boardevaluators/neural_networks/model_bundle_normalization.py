"""Normalize legacy NN config paths into model bundle references."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol

from chipiron.models.model_bundle import (
    ModelBundleRef,
    ResolvedModelBundle,
    resolve_model_bundle,
)
from chipiron.utils.small_tools import MyPath


class LegacyModelBundleNormalizationError(ValueError):
    """Raised when a legacy weights path cannot be normalized into a bundle."""


class LegacyNeuralNetModelConfig(Protocol):
    """Structural type for legacy neural-network config objects."""

    @property
    def model_weights_file_name(self) -> MyPath:
        """Return the legacy weights-path field."""
        ...


def _assert_file_like_weights_path(weights_path: str) -> None:
    """Reject paths that clearly point to bundle directories instead of weight files."""
    if not weights_path:
        raise LegacyModelBundleNormalizationError(
            "Cannot normalize an empty model_weights_file_name into a model bundle."
        )
    if weights_path.endswith(("/", "\\")):
        raise LegacyModelBundleNormalizationError(
            f"Legacy model_weights_file_name {weights_path!r} must point to a file, not a directory."
        )


def _build_hf_bundle_ref(model_weights_file_name: str) -> ModelBundleRef:
    """Convert an hf:// weights URI into a folder-level bundle reference."""
    payload = model_weights_file_name[len("hf://") :]
    repo_and_path, revision = (
        payload.rsplit("@", 1) if "@" in payload else (payload, "main")
    )
    parts = [part for part in repo_and_path.split("/") if part]
    if len(parts) < 4:
        raise LegacyModelBundleNormalizationError(
            "Legacy HF model_weights_file_name must point to a file inside a bundle, "
            f"got {model_weights_file_name!r}."
        )

    weights_file = parts[-1]
    bundle_path = PurePosixPath(*parts[2:-1])
    if str(bundle_path) in {"", "."}:
        raise LegacyModelBundleNormalizationError(
            "Legacy HF model_weights_file_name must include a bundle folder before the weights file, "
            f"got {model_weights_file_name!r}."
        )

    return ModelBundleRef(
        uri=f"hf://{parts[0]}/{parts[1]}/{bundle_path.as_posix()}@{revision or 'main'}",
        weights_file=weights_file,
    )


def _build_package_bundle_ref(model_weights_file_name: str) -> ModelBundleRef:
    """Convert a package:// weights URI into a folder-level bundle reference."""
    relative_path = model_weights_file_name[len("package://") :]
    file_path = PurePosixPath(relative_path)
    if str(file_path.parent) in {"", "."}:
        raise LegacyModelBundleNormalizationError(
            "Legacy package model_weights_file_name must point to a file inside a bundle, "
            f"got {model_weights_file_name!r}."
        )

    return ModelBundleRef(
        uri=f"package://{file_path.parent.as_posix()}",
        weights_file=file_path.name,
    )


def _build_local_bundle_ref(model_weights_file_name: str) -> ModelBundleRef:
    """Convert a local weights path into a folder-level bundle reference."""
    file_path = Path(model_weights_file_name).expanduser()
    if file_path.name == "":
        raise LegacyModelBundleNormalizationError(
            "Legacy local model_weights_file_name must point to a file, "
            f"got {model_weights_file_name!r}."
        )

    return ModelBundleRef(
        uri=str(file_path.parent),
        weights_file=file_path.name,
    )


def model_bundle_ref_from_model_weights_path(
    model_weights_file_name: MyPath,
) -> ModelBundleRef:
    """Normalize a legacy weights path into a folder-level model bundle reference."""
    weights_path = str(model_weights_file_name)
    _assert_file_like_weights_path(weights_path)

    if weights_path.startswith("hf://"):
        return _build_hf_bundle_ref(weights_path)
    if weights_path.startswith("package://"):
        return _build_package_bundle_ref(weights_path)
    return _build_local_bundle_ref(weights_path)


def model_bundle_ref_from_legacy_nn_config(
    legacy_nn_config: LegacyNeuralNetModelConfig,
) -> ModelBundleRef:
    """Normalize the legacy NN config shape into a model bundle reference."""
    return model_bundle_ref_from_model_weights_path(
        legacy_nn_config.model_weights_file_name
    )


def resolve_model_bundle_from_model_weights_path(
    model_weights_file_name: MyPath,
) -> ResolvedModelBundle:
    """Resolve a legacy weights path into a concrete local model bundle."""
    bundle_ref = model_bundle_ref_from_model_weights_path(model_weights_file_name)
    return resolve_model_bundle(bundle_ref)


def resolve_model_bundle_from_legacy_nn_config(
    legacy_nn_config: LegacyNeuralNetModelConfig,
) -> ResolvedModelBundle:
    """Resolve the legacy NN config shape into a concrete local model bundle."""
    bundle_ref = model_bundle_ref_from_legacy_nn_config(legacy_nn_config)
    return resolve_model_bundle(bundle_ref)


__all__ = [
    "LegacyModelBundleNormalizationError",
    "LegacyNeuralNetModelConfig",
    "model_bundle_ref_from_legacy_nn_config",
    "model_bundle_ref_from_model_weights_path",
    "resolve_model_bundle_from_legacy_nn_config",
    "resolve_model_bundle_from_model_weights_path",
]
