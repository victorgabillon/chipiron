"""Flat handcrafted-feature tensor cache for Morpion streaming training."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Final, cast

import torch

from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_row_chunks_from_path,
)
from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    process_morpion_supervised_row_to_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
    full_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch

FLAT_TENSOR_CACHE_DIR_NAME: Final[str] = "tensor_cache"
FLAT_TENSOR_CACHE_FORMAT: Final[str] = "morpion_flat_features_v1"
FLAT_TENSOR_CACHE_INPUT_KEY: Final[str] = "input_tensor"
FLAT_TENSOR_CACHE_TARGET_KEY: Final[str] = "target_tensor"


class InvalidMorpionFlatTensorCacheError(ValueError):
    """Raised when one flat tensor cache cannot be used safely."""

    @classmethod
    def missing_or_stale(
        cls,
        rows_path: str | os.PathLike[str],
    ) -> InvalidMorpionFlatTensorCacheError:
        """Return the missing or stale cache error."""
        return cls(
            f"Flat tensor cache is missing or stale for rows artifact: {rows_path!r}."
        )

    @classmethod
    def unknown_feature(cls, feature_name: str) -> InvalidMorpionFlatTensorCacheError:
        """Return the unknown feature-name error."""
        return cls(f"Unknown Morpion flat feature name: {feature_name!r}.")

    @classmethod
    def invalid_feature_order(cls) -> InvalidMorpionFlatTensorCacheError:
        """Return the invalid canonical feature-order error."""
        return cls("Flat feature subsets must follow canonical feature order.")

    @classmethod
    def empty_feature_subset(cls) -> InvalidMorpionFlatTensorCacheError:
        """Return the empty feature-subset error."""
        return cls("Flat feature subsets must contain at least one feature.")


@dataclass(frozen=True, slots=True)
class FlatTensorCachePaths:
    """Paths for one persisted Morpion flat tensor cache."""

    tensor_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class FlatTensorCacheManifest:
    """Manifest describing one persisted Morpion flat tensor cache."""

    format: str
    source_rows_path: str
    source_rows_size: int
    source_rows_mtime: float
    row_count: int
    feature_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    dtype: str
    created_at_unix_s: float

    @classmethod
    def from_json_payload(
        cls,
        payload: dict[str, object],
    ) -> FlatTensorCacheManifest:
        """Build one cache manifest from a decoded JSON payload."""
        return cls(
            format=_required_str(payload, "format"),
            source_rows_path=_required_str(payload, "source_rows_path"),
            source_rows_size=_required_int(payload, "source_rows_size"),
            source_rows_mtime=_required_float(payload, "source_rows_mtime"),
            row_count=_required_int(payload, "row_count"),
            feature_names=tuple(_required_str_sequence(payload, "feature_names")),
            input_shape=tuple(_required_int_sequence(payload, "input_shape")),
            target_shape=tuple(_required_int_sequence(payload, "target_shape")),
            dtype=_required_str(payload, "dtype"),
            created_at_unix_s=_required_float(payload, "created_at_unix_s"),
        )

    def to_json_payload(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload."""
        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True, slots=True)
class FlatTensorCache:
    """Loaded Morpion flat tensor cache and cache timing metadata."""

    paths: FlatTensorCachePaths
    manifest: FlatTensorCacheManifest
    input_tensor: torch.Tensor
    target_tensor: torch.Tensor
    rebuilt: bool
    materialize_seconds: float
    load_seconds: float


def is_flat_morpion_training_model_kind(model_kind: str) -> bool:
    """Return whether a Morpion model kind consumes flat handcrafted features."""
    return model_kind in {"linear", "mlp"}


def default_flat_tensor_cache_paths(
    rows_path: str | os.PathLike[str],
    *,
    max_rows: int | None = None,
) -> FlatTensorCachePaths:
    """Return default cache paths beside one Morpion rows artifact."""
    source = Path(rows_path)
    row_limit_tag = "all" if max_rows is None else f"max_rows_{max_rows}"
    cache_stem = f"{source.stem}.flat_features.{row_limit_tag}"
    cache_dir = source.parent / FLAT_TENSOR_CACHE_DIR_NAME
    return FlatTensorCachePaths(
        tensor_path=cache_dir / f"{cache_stem}.pt",
        manifest_path=cache_dir / f"{cache_stem}.manifest.json",
    )


def load_or_materialize_flat_tensor_cache(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
) -> FlatTensorCache:
    """Load a valid flat tensor cache or rebuild it from Morpion rows."""
    paths = default_flat_tensor_cache_paths(rows_path, max_rows=max_rows)
    started_at = perf_counter()
    loaded = _try_load_valid_flat_tensor_cache(
        paths=paths,
        rows_path=rows_path,
    )
    if loaded is not None:
        return FlatTensorCache(
            paths=paths,
            manifest=loaded.manifest,
            input_tensor=loaded.input_tensor,
            target_tensor=loaded.target_tensor,
            rebuilt=False,
            materialize_seconds=0.0,
            load_seconds=perf_counter() - started_at,
        )
    materialize_started_at = perf_counter()
    input_tensor, target_tensor = _materialize_flat_tensors(
        rows_path=rows_path,
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
    )
    manifest = _flat_tensor_cache_manifest(
        rows_path=rows_path,
        input_tensor=input_tensor,
        target_tensor=target_tensor,
    )
    paths.tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            FLAT_TENSOR_CACHE_INPUT_KEY: input_tensor,
            FLAT_TENSOR_CACHE_TARGET_KEY: target_tensor,
        },
        paths.tensor_path,
    )
    with open(paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_json_payload(), handle, indent=2, sort_keys=True)
    return FlatTensorCache(
        paths=paths,
        manifest=manifest,
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        rebuilt=True,
        materialize_seconds=perf_counter() - materialize_started_at,
        load_seconds=0.0,
    )


def flat_tensor_cache_is_valid(
    *,
    rows_path: str | os.PathLike[str],
    max_rows: int | None = None,
) -> bool:
    """Return whether the default cache exists and matches the source rows."""
    paths = default_flat_tensor_cache_paths(rows_path, max_rows=max_rows)
    return (
        _try_load_valid_flat_tensor_cache(paths=paths, rows_path=rows_path) is not None
    )


def load_flat_tensor_cache(
    *,
    rows_path: str | os.PathLike[str],
    max_rows: int | None = None,
) -> FlatTensorCache:
    """Load one existing valid Morpion flat tensor cache."""
    paths = default_flat_tensor_cache_paths(rows_path, max_rows=max_rows)
    started_at = perf_counter()
    loaded = _try_load_valid_flat_tensor_cache(paths=paths, rows_path=rows_path)
    if loaded is None:
        raise InvalidMorpionFlatTensorCacheError.missing_or_stale(rows_path)
    return FlatTensorCache(
        paths=paths,
        manifest=loaded.manifest,
        input_tensor=loaded.input_tensor,
        target_tensor=loaded.target_tensor,
        rebuilt=False,
        materialize_seconds=0.0,
        load_seconds=perf_counter() - started_at,
    )


def feature_indices_for_subset(
    requested_feature_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Return canonical feature indices for one ordered flat-feature subset."""
    canonical_index = {
        feature_name: index
        for index, feature_name in enumerate(MORPION_CANONICAL_FEATURE_NAMES)
    }
    indices: list[int] = []
    previous_index = -1
    for feature_name in requested_feature_names:
        try:
            index = canonical_index[feature_name]
        except KeyError as exc:
            raise InvalidMorpionFlatTensorCacheError.unknown_feature(
                feature_name
            ) from exc
        if index <= previous_index:
            raise InvalidMorpionFlatTensorCacheError.invalid_feature_order()
        indices.append(index)
        previous_index = index
    if not indices:
        raise InvalidMorpionFlatTensorCacheError.empty_feature_subset()
    return tuple(indices)


def select_flat_features(
    input_tensor: torch.Tensor,
    requested_feature_names: tuple[str, ...],
) -> torch.Tensor:
    """Select one ordered feature subset from canonical cached flat inputs."""
    if requested_feature_names == MORPION_CANONICAL_FEATURE_NAMES:
        return input_tensor
    indices = feature_indices_for_subset(requested_feature_names)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return input_tensor.index_select(1, index_tensor)


def flat_cache_batch(
    *,
    cache: FlatTensorCache,
    row_indices: tuple[int, ...],
    requested_feature_names: tuple[str, ...],
) -> TensorSupervisedBatch:
    """Build one supervised tensor batch from cached canonical flat tensors."""
    index_tensor = torch.tensor(row_indices, dtype=torch.long)
    input_tensor = cache.input_tensor.index_select(0, index_tensor)
    target_tensor = cache.target_tensor.index_select(0, index_tensor)
    return TensorSupervisedBatch(
        input_tensor=select_flat_features(input_tensor, requested_feature_names),
        target_tensor=target_tensor,
        is_batch=True,
    )


@dataclass(frozen=True, slots=True)
class _LoadedFlatTensorCache:
    manifest: FlatTensorCacheManifest
    input_tensor: torch.Tensor
    target_tensor: torch.Tensor


def _try_load_valid_flat_tensor_cache(
    *,
    paths: FlatTensorCachePaths,
    rows_path: str | os.PathLike[str],
) -> _LoadedFlatTensorCache | None:
    """Load a cache when its manifest and tensors match the source rows."""
    manifest = _read_flat_tensor_cache_manifest(paths.manifest_path)
    if manifest is None or not _manifest_matches_rows(manifest, rows_path=rows_path):
        return None
    payload = _read_flat_tensor_cache_payload(paths.tensor_path)
    if payload is None:
        return None
    input_tensor = payload[FLAT_TENSOR_CACHE_INPUT_KEY]
    target_tensor = payload[FLAT_TENSOR_CACHE_TARGET_KEY]
    if not _tensors_match_manifest(
        manifest=manifest,
        input_tensor=input_tensor,
        target_tensor=target_tensor,
    ):
        return None
    return _LoadedFlatTensorCache(
        manifest=manifest,
        input_tensor=input_tensor,
        target_tensor=target_tensor,
    )


def _read_flat_tensor_cache_manifest(
    manifest_path: Path,
) -> FlatTensorCacheManifest | None:
    """Read a cache manifest, returning ``None`` when it is unusable."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return FlatTensorCacheManifest.from_json_payload(
            cast("dict[str, object]", payload)
        )
    except (
        InvalidMorpionFlatTensorCacheError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _read_flat_tensor_cache_payload(
    tensor_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Read cache tensors, returning ``None`` when the payload is unusable."""
    try:
        payload = _torch_load_cpu(tensor_path)
    except (OSError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    input_tensor = payload.get(FLAT_TENSOR_CACHE_INPUT_KEY)
    target_tensor = payload.get(FLAT_TENSOR_CACHE_TARGET_KEY)
    if not isinstance(input_tensor, torch.Tensor) or not isinstance(
        target_tensor, torch.Tensor
    ):
        return None
    return {
        FLAT_TENSOR_CACHE_INPUT_KEY: input_tensor,
        FLAT_TENSOR_CACHE_TARGET_KEY: target_tensor,
    }


def _torch_load_cpu(path: Path) -> object:
    """Load one torch payload onto CPU without requiring CUDA."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _materialize_flat_tensors(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize canonical flat feature and target tensors from row chunks."""
    dynamics = MorpionDynamics()
    converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=full_morpion_feature_subset(),
    )
    input_tensors: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []
    for rows in iter_morpion_supervised_row_chunks_from_path(
        os.fspath(rows_path),
        chunk_size=row_chunk_size,
        max_rows=max_rows,
    ):
        for row in rows:
            sample = process_morpion_supervised_row_to_tensors(
                row,
                dynamics=dynamics,
                converter=converter,
            )
            input_tensors.append(sample.input_tensor.detach().cpu())
            target_tensors.append(sample.target_tensor.detach().cpu())
    if not input_tensors:
        return (
            torch.empty(
                (0, len(MORPION_CANONICAL_FEATURE_NAMES)),
                dtype=torch.float32,
            ),
            torch.empty((0, 1), dtype=torch.float32),
        )
    return (
        torch.stack(input_tensors).cpu(),
        torch.stack(target_tensors).cpu(),
    )


def _flat_tensor_cache_manifest(
    *,
    rows_path: str | os.PathLike[str],
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> FlatTensorCacheManifest:
    """Build one manifest for cached flat tensors."""
    source = Path(rows_path).resolve()
    source_stat = source.stat()
    return FlatTensorCacheManifest(
        format=FLAT_TENSOR_CACHE_FORMAT,
        source_rows_path=str(source),
        source_rows_size=source_stat.st_size,
        source_rows_mtime=source_stat.st_mtime,
        row_count=int(input_tensor.shape[0]),
        feature_names=MORPION_CANONICAL_FEATURE_NAMES,
        input_shape=tuple(int(item) for item in input_tensor.shape),
        target_shape=tuple(int(item) for item in target_tensor.shape),
        dtype=str(input_tensor.dtype),
        created_at_unix_s=time(),
    )


def _manifest_matches_rows(
    manifest: FlatTensorCacheManifest,
    *,
    rows_path: str | os.PathLike[str],
) -> bool:
    """Return whether one manifest still matches its source rows artifact."""
    source = Path(rows_path).resolve()
    try:
        source_stat = source.stat()
    except OSError:
        return False
    return (
        manifest.format == FLAT_TENSOR_CACHE_FORMAT
        and manifest.source_rows_path == str(source)
        and manifest.source_rows_size == source_stat.st_size
        and manifest.source_rows_mtime == source_stat.st_mtime
        and manifest.feature_names == MORPION_CANONICAL_FEATURE_NAMES
        and manifest.input_shape
        == (
            manifest.row_count,
            len(MORPION_CANONICAL_FEATURE_NAMES),
        )
        and manifest.target_shape == (manifest.row_count, 1)
        and manifest.dtype == str(torch.float32)
    )


def _tensors_match_manifest(
    *,
    manifest: FlatTensorCacheManifest,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> bool:
    """Return whether cached tensors match manifest shape and CPU constraints."""
    return (
        input_tensor.device.type == "cpu"
        and target_tensor.device.type == "cpu"
        and str(input_tensor.dtype) == manifest.dtype
        and str(target_tensor.dtype) == manifest.dtype
        and tuple(int(item) for item in input_tensor.shape) == manifest.input_shape
        and tuple(int(item) for item in target_tensor.shape) == manifest.target_shape
    )


def _required_str(payload: dict[str, object], key: str) -> str:
    """Return one required JSON string."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidMorpionFlatTensorCacheError
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    """Return one required JSON integer."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMorpionFlatTensorCacheError
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    """Return one required JSON finite number as float."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidMorpionFlatTensorCacheError
    return float(value)


def _required_str_sequence(payload: dict[str, object], key: str) -> tuple[str, ...]:
    """Return one required JSON string tuple."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidMorpionFlatTensorCacheError
    return tuple(value)


def _required_int_sequence(payload: dict[str, object], key: str) -> tuple[int, ...]:
    """Return one required JSON integer tuple."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise InvalidMorpionFlatTensorCacheError
    return tuple(value)


__all__ = [
    "FLAT_TENSOR_CACHE_DIR_NAME",
    "FLAT_TENSOR_CACHE_FORMAT",
    "FlatTensorCache",
    "FlatTensorCacheManifest",
    "FlatTensorCachePaths",
    "InvalidMorpionFlatTensorCacheError",
    "default_flat_tensor_cache_paths",
    "feature_indices_for_subset",
    "flat_cache_batch",
    "flat_tensor_cache_is_valid",
    "is_flat_morpion_training_model_kind",
    "load_flat_tensor_cache",
    "load_or_materialize_flat_tensor_cache",
    "select_flat_features",
]
