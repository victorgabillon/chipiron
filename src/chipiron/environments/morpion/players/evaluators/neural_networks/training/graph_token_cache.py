"""Packed graph-token tensor cache for Morpion streaming training."""

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
    process_morpion_supervised_row_to_graph_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MorpionGraphTokenConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch

GRAPH_TOKEN_CACHE_DIR_NAME: Final[str] = "tensor_cache"
GRAPH_TOKEN_CACHE_FORMAT: Final[str] = "morpion_graph_tokens_v1"
GRAPH_TOKEN_CACHE_PACKED_INPUT_KEY: Final[str] = "packed_input_tensor"
GRAPH_TOKEN_CACHE_LENGTHS_KEY: Final[str] = "token_lengths"
GRAPH_TOKEN_CACHE_TARGET_KEY: Final[str] = "target_tensor"


class InvalidMorpionGraphTokenCacheError(ValueError):
    """Raised when one graph-token cache cannot be used safely."""

    @classmethod
    def missing_or_stale(
        cls,
        rows_path: str | os.PathLike[str],
    ) -> InvalidMorpionGraphTokenCacheError:
        """Return the missing or stale cache error."""
        return cls(
            f"Graph-token cache is missing or stale for rows artifact: {rows_path!r}."
        )


@dataclass(frozen=True, slots=True)
class GraphTokenCachePaths:
    """Paths for one persisted Morpion graph-token cache."""

    tensor_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class GraphTokenCacheManifest:
    """Manifest describing one persisted Morpion graph-token cache."""

    format: str
    source_rows_path: str
    source_rows_size: int
    source_rows_mtime_ns: int
    row_count: int
    graph_max_tokens: int
    graph_token_feature_dim: int
    packed_input_shape: tuple[int, int]
    token_lengths_shape: tuple[int]
    target_shape: tuple[int, int]
    input_dtype: str
    lengths_dtype: str
    target_dtype: str
    created_at_unix_s: float

    @classmethod
    def from_json_payload(
        cls,
        payload: dict[str, object],
    ) -> GraphTokenCacheManifest:
        """Build one cache manifest from a decoded JSON payload."""
        return cls(
            format=_required_str(payload, "format"),
            source_rows_path=_required_str(payload, "source_rows_path"),
            source_rows_size=_required_int(payload, "source_rows_size"),
            source_rows_mtime_ns=_required_int(payload, "source_rows_mtime_ns"),
            row_count=_required_int(payload, "row_count"),
            graph_max_tokens=_required_int(payload, "graph_max_tokens"),
            graph_token_feature_dim=_required_int(
                payload,
                "graph_token_feature_dim",
            ),
            packed_input_shape=_required_shape_2(payload, "packed_input_shape"),
            token_lengths_shape=_required_shape_1(payload, "token_lengths_shape"),
            target_shape=_required_shape_2(payload, "target_shape"),
            input_dtype=_required_str(payload, "input_dtype"),
            lengths_dtype=_required_str(payload, "lengths_dtype"),
            target_dtype=_required_str(payload, "target_dtype"),
            created_at_unix_s=_required_float(payload, "created_at_unix_s"),
        )

    def to_json_payload(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload."""
        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True, slots=True)
class GraphTokenCache:
    """Loaded Morpion graph-token cache and cache timing metadata."""

    paths: GraphTokenCachePaths
    manifest: GraphTokenCacheManifest
    packed_input_tensor: torch.Tensor
    token_lengths: torch.Tensor
    token_offsets: torch.Tensor
    target_tensor: torch.Tensor
    rebuilt: bool
    materialize_seconds: float
    load_seconds: float


def default_graph_token_cache_paths(
    rows_path: str | os.PathLike[str],
    *,
    graph_max_tokens: int,
    max_rows: int | None = None,
) -> GraphTokenCachePaths:
    """Return default graph-token cache paths beside one rows artifact."""
    source = Path(rows_path)
    row_limit_tag = "all" if max_rows is None else f"max_rows_{max_rows}"
    cache_stem = (
        f"{source.stem}.graph_tokens.max_tokens_{graph_max_tokens}.{row_limit_tag}"
    )
    cache_dir = source.parent / GRAPH_TOKEN_CACHE_DIR_NAME
    return GraphTokenCachePaths(
        tensor_path=cache_dir / f"{cache_stem}.pt",
        manifest_path=cache_dir / f"{cache_stem}.manifest.json",
    )


def load_or_materialize_graph_token_cache(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
    graph_max_tokens: int,
) -> GraphTokenCache:
    """Load a valid graph-token cache or rebuild it from Morpion rows."""
    paths = default_graph_token_cache_paths(
        rows_path,
        graph_max_tokens=graph_max_tokens,
        max_rows=max_rows,
    )
    started_at = perf_counter()
    loaded = _try_load_valid_graph_token_cache(
        paths=paths,
        rows_path=rows_path,
        graph_max_tokens=graph_max_tokens,
    )
    if loaded is not None:
        return GraphTokenCache(
            paths=paths,
            manifest=loaded.manifest,
            packed_input_tensor=loaded.packed_input_tensor,
            token_lengths=loaded.token_lengths,
            token_offsets=_token_offsets(loaded.token_lengths),
            target_tensor=loaded.target_tensor,
            rebuilt=False,
            materialize_seconds=0.0,
            load_seconds=perf_counter() - started_at,
        )
    materialize_started_at = perf_counter()
    packed_input_tensor, token_lengths, target_tensor = _materialize_graph_tokens(
        rows_path=rows_path,
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        graph_max_tokens=graph_max_tokens,
    )
    manifest = _graph_token_cache_manifest(
        rows_path=rows_path,
        graph_max_tokens=graph_max_tokens,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        target_tensor=target_tensor,
    )
    paths.tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            GRAPH_TOKEN_CACHE_PACKED_INPUT_KEY: packed_input_tensor,
            GRAPH_TOKEN_CACHE_LENGTHS_KEY: token_lengths,
            GRAPH_TOKEN_CACHE_TARGET_KEY: target_tensor,
        },
        paths.tensor_path,
    )
    with open(paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_json_payload(), handle, indent=2, sort_keys=True)
    return GraphTokenCache(
        paths=paths,
        manifest=manifest,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        token_offsets=_token_offsets(token_lengths),
        target_tensor=target_tensor,
        rebuilt=True,
        materialize_seconds=perf_counter() - materialize_started_at,
        load_seconds=0.0,
    )


def graph_token_cache_is_valid(
    *,
    rows_path: str | os.PathLike[str],
    graph_max_tokens: int,
    max_rows: int | None = None,
) -> bool:
    """Return whether the default graph-token cache matches the rows source."""
    paths = default_graph_token_cache_paths(
        rows_path,
        graph_max_tokens=graph_max_tokens,
        max_rows=max_rows,
    )
    return (
        _try_load_valid_graph_token_cache(
            paths=paths,
            rows_path=rows_path,
            graph_max_tokens=graph_max_tokens,
        )
        is not None
    )


def load_graph_token_cache(
    *,
    rows_path: str | os.PathLike[str],
    graph_max_tokens: int,
    max_rows: int | None = None,
) -> GraphTokenCache:
    """Load one existing valid Morpion graph-token cache."""
    paths = default_graph_token_cache_paths(
        rows_path,
        graph_max_tokens=graph_max_tokens,
        max_rows=max_rows,
    )
    started_at = perf_counter()
    loaded = _try_load_valid_graph_token_cache(
        paths=paths,
        rows_path=rows_path,
        graph_max_tokens=graph_max_tokens,
    )
    if loaded is None:
        raise InvalidMorpionGraphTokenCacheError.missing_or_stale(rows_path)
    return GraphTokenCache(
        paths=paths,
        manifest=loaded.manifest,
        packed_input_tensor=loaded.packed_input_tensor,
        token_lengths=loaded.token_lengths,
        token_offsets=_token_offsets(loaded.token_lengths),
        target_tensor=loaded.target_tensor,
        rebuilt=False,
        materialize_seconds=0.0,
        load_seconds=perf_counter() - started_at,
    )


def graph_cache_batch(
    *,
    cache: GraphTokenCache,
    row_indices: tuple[int, ...],
) -> TensorSupervisedBatch:
    """Build one padded graph-token batch from packed cached tensors."""
    if not row_indices:
        return TensorSupervisedBatch(
            input_tensor=torch.empty(
                (0, 0, cache.manifest.graph_token_feature_dim),
                dtype=torch.float32,
            ),
            target_tensor=torch.empty((0, 1), dtype=torch.float32),
            is_batch=True,
        )
    lengths = [int(cache.token_lengths[row_index].item()) for row_index in row_indices]
    max_token_count = max(lengths)
    input_tensor = torch.zeros(
        (len(row_indices), max_token_count, cache.manifest.graph_token_feature_dim),
        dtype=cache.packed_input_tensor.dtype,
    )
    for batch_index, row_index in enumerate(row_indices):
        token_count = lengths[batch_index]
        token_start = int(cache.token_offsets[row_index].item())
        token_end = token_start + token_count
        input_tensor[batch_index, :token_count, :] = cache.packed_input_tensor[
            token_start:token_end,
            :,
        ]
    index_tensor = torch.tensor(row_indices, dtype=torch.long)
    return TensorSupervisedBatch(
        input_tensor=input_tensor,
        target_tensor=cache.target_tensor.index_select(0, index_tensor),
        is_batch=True,
    )


@dataclass(frozen=True, slots=True)
class _LoadedGraphTokenCache:
    manifest: GraphTokenCacheManifest
    packed_input_tensor: torch.Tensor
    token_lengths: torch.Tensor
    target_tensor: torch.Tensor


def _try_load_valid_graph_token_cache(
    *,
    paths: GraphTokenCachePaths,
    rows_path: str | os.PathLike[str],
    graph_max_tokens: int,
) -> _LoadedGraphTokenCache | None:
    """Load a graph-token cache when it matches its source rows and args."""
    manifest = _read_graph_token_cache_manifest(paths.manifest_path)
    if manifest is None or not _manifest_matches_rows(
        manifest,
        rows_path=rows_path,
        graph_max_tokens=graph_max_tokens,
    ):
        return None
    payload = _read_graph_token_cache_payload(paths.tensor_path)
    if payload is None:
        return None
    packed_input_tensor = payload[GRAPH_TOKEN_CACHE_PACKED_INPUT_KEY]
    token_lengths = payload[GRAPH_TOKEN_CACHE_LENGTHS_KEY]
    target_tensor = payload[GRAPH_TOKEN_CACHE_TARGET_KEY]
    if not _tensors_match_manifest(
        manifest=manifest,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        target_tensor=target_tensor,
    ):
        return None
    return _LoadedGraphTokenCache(
        manifest=manifest,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        target_tensor=target_tensor,
    )


def _read_graph_token_cache_manifest(
    manifest_path: Path,
) -> GraphTokenCacheManifest | None:
    """Read a cache manifest, returning ``None`` when it is unusable."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return GraphTokenCacheManifest.from_json_payload(
            cast("dict[str, object]", payload)
        )
    except (
        InvalidMorpionGraphTokenCacheError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _read_graph_token_cache_payload(
    tensor_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Read cache tensors, returning ``None`` when the payload is unusable."""
    try:
        payload = _torch_load_cpu(tensor_path)
    except (OSError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    packed_input_tensor = payload.get(GRAPH_TOKEN_CACHE_PACKED_INPUT_KEY)
    token_lengths = payload.get(GRAPH_TOKEN_CACHE_LENGTHS_KEY)
    target_tensor = payload.get(GRAPH_TOKEN_CACHE_TARGET_KEY)
    if (
        not isinstance(packed_input_tensor, torch.Tensor)
        or not isinstance(token_lengths, torch.Tensor)
        or not isinstance(target_tensor, torch.Tensor)
    ):
        return None
    return {
        GRAPH_TOKEN_CACHE_PACKED_INPUT_KEY: packed_input_tensor,
        GRAPH_TOKEN_CACHE_LENGTHS_KEY: token_lengths,
        GRAPH_TOKEN_CACHE_TARGET_KEY: target_tensor,
    }


def _torch_load_cpu(path: Path) -> object:
    """Load one torch payload onto CPU without requiring CUDA."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _materialize_graph_tokens(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
    graph_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize packed graph tokens and targets from row chunks."""
    dynamics = MorpionDynamics()
    graph_converter = MorpionGraphTokenConverter(
        dynamics=dynamics,
        max_tokens=graph_max_tokens,
    )
    token_tensors: list[torch.Tensor] = []
    token_lengths: list[int] = []
    target_tensors: list[torch.Tensor] = []
    for rows in iter_morpion_supervised_row_chunks_from_path(
        os.fspath(rows_path),
        chunk_size=row_chunk_size,
        max_rows=max_rows,
    ):
        for row in rows:
            sample = process_morpion_supervised_row_to_graph_tensors(
                row,
                dynamics=dynamics,
                converter=graph_converter,
            )
            input_tensor = sample.input_tensor.detach().cpu()
            target_tensor = sample.target_tensor.detach().cpu()
            token_tensors.append(input_tensor)
            token_lengths.append(int(input_tensor.shape[0]))
            target_tensors.append(target_tensor)
    if not token_tensors:
        return (
            torch.empty((0, MORPION_GRAPH_TOKEN_FEATURE_DIM), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )
    return (
        torch.cat(token_tensors, dim=0).cpu(),
        torch.tensor(token_lengths, dtype=torch.long),
        torch.stack(target_tensors).cpu(),
    )


def _graph_token_cache_manifest(
    *,
    rows_path: str | os.PathLike[str],
    graph_max_tokens: int,
    packed_input_tensor: torch.Tensor,
    token_lengths: torch.Tensor,
    target_tensor: torch.Tensor,
) -> GraphTokenCacheManifest:
    """Build one manifest for cached graph-token tensors."""
    source = Path(rows_path).resolve()
    source_stat = source.stat()
    return GraphTokenCacheManifest(
        format=GRAPH_TOKEN_CACHE_FORMAT,
        source_rows_path=str(source),
        source_rows_size=source_stat.st_size,
        source_rows_mtime_ns=source_stat.st_mtime_ns,
        row_count=int(token_lengths.shape[0]),
        graph_max_tokens=graph_max_tokens,
        graph_token_feature_dim=MORPION_GRAPH_TOKEN_FEATURE_DIM,
        packed_input_shape=(
            int(packed_input_tensor.shape[0]),
            int(packed_input_tensor.shape[1]),
        ),
        token_lengths_shape=(int(token_lengths.shape[0]),),
        target_shape=(int(target_tensor.shape[0]), int(target_tensor.shape[1])),
        input_dtype=str(packed_input_tensor.dtype),
        lengths_dtype=str(token_lengths.dtype),
        target_dtype=str(target_tensor.dtype),
        created_at_unix_s=time(),
    )


def _manifest_matches_rows(
    manifest: GraphTokenCacheManifest,
    *,
    rows_path: str | os.PathLike[str],
    graph_max_tokens: int,
) -> bool:
    """Return whether one manifest still matches its source and graph args."""
    source = Path(rows_path).resolve()
    try:
        source_stat = source.stat()
    except OSError:
        return False
    return (
        manifest.format == GRAPH_TOKEN_CACHE_FORMAT
        and manifest.source_rows_path == str(source)
        and manifest.source_rows_size == source_stat.st_size
        and manifest.source_rows_mtime_ns == source_stat.st_mtime_ns
        and manifest.graph_max_tokens == graph_max_tokens
        and manifest.graph_token_feature_dim == MORPION_GRAPH_TOKEN_FEATURE_DIM
        and manifest.packed_input_shape[1] == MORPION_GRAPH_TOKEN_FEATURE_DIM
        and manifest.token_lengths_shape == (manifest.row_count,)
        and manifest.target_shape == (manifest.row_count, 1)
        and manifest.input_dtype == str(torch.float32)
        and manifest.lengths_dtype == str(torch.int64)
        and manifest.target_dtype == str(torch.float32)
    )


def _tensors_match_manifest(
    *,
    manifest: GraphTokenCacheManifest,
    packed_input_tensor: torch.Tensor,
    token_lengths: torch.Tensor,
    target_tensor: torch.Tensor,
) -> bool:
    """Return whether cached tensors match manifest shape and CPU constraints."""
    if (
        packed_input_tensor.device.type != "cpu"
        or token_lengths.device.type != "cpu"
        or target_tensor.device.type != "cpu"
        or str(packed_input_tensor.dtype) != manifest.input_dtype
        or str(token_lengths.dtype) != manifest.lengths_dtype
        or str(target_tensor.dtype) != manifest.target_dtype
        or tuple(int(item) for item in packed_input_tensor.shape)
        != manifest.packed_input_shape
        or tuple(int(item) for item in token_lengths.shape)
        != manifest.token_lengths_shape
        or tuple(int(item) for item in target_tensor.shape) != manifest.target_shape
    ):
        return False
    if token_lengths.numel() != manifest.row_count:
        return False
    if token_lengths.numel() > 0 and bool(torch.any(token_lengths < 0).item()):
        return False
    return int(token_lengths.sum().item()) == manifest.packed_input_shape[0]


def _token_offsets(token_lengths: torch.Tensor) -> torch.Tensor:
    """Return start offsets into the packed token tensor for each row."""
    if token_lengths.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(
        (
            torch.zeros((1,), dtype=torch.long),
            torch.cumsum(token_lengths[:-1], dim=0),
        )
    )


def _required_str(payload: dict[str, object], key: str) -> str:
    """Return one required JSON string."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidMorpionGraphTokenCacheError
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    """Return one required JSON integer."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMorpionGraphTokenCacheError
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    """Return one required JSON finite number as float."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidMorpionGraphTokenCacheError
    return float(value)


def _required_int_sequence(payload: dict[str, object], key: str) -> tuple[int, ...]:
    """Return one required JSON integer tuple."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise InvalidMorpionGraphTokenCacheError
    return tuple(value)


def _required_shape_1(payload: dict[str, object], key: str) -> tuple[int]:
    """Return one required length-one JSON shape tuple."""
    value = _required_int_sequence(payload, key)
    if len(value) != 1:
        raise InvalidMorpionGraphTokenCacheError
    return (value[0],)


def _required_shape_2(payload: dict[str, object], key: str) -> tuple[int, int]:
    """Return one required length-two JSON shape tuple."""
    value = _required_int_sequence(payload, key)
    if len(value) != 2:
        raise InvalidMorpionGraphTokenCacheError
    return (value[0], value[1])


__all__ = [
    "GRAPH_TOKEN_CACHE_DIR_NAME",
    "GRAPH_TOKEN_CACHE_FORMAT",
    "GraphTokenCache",
    "GraphTokenCacheManifest",
    "GraphTokenCachePaths",
    "InvalidMorpionGraphTokenCacheError",
    "default_graph_token_cache_paths",
    "graph_cache_batch",
    "graph_token_cache_is_valid",
    "load_graph_token_cache",
    "load_or_materialize_graph_token_cache",
]
