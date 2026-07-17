"""Packed relational entity-token cache for Morpion supervised rows."""

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
    process_morpion_supervised_row_to_relational_entity_token_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch

RELATIONAL_ENTITY_TOKEN_CACHE_DIR_NAME: Final[str] = "tensor_cache"
MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT: Final[str] = (
    "morpion_relational_entity_token_cache_v1"
)
RELATIONAL_CACHE_PACKED_INPUT_KEY: Final[str] = "packed_input_tensor"
RELATIONAL_CACHE_TOKEN_LENGTHS_KEY: Final[str] = "token_lengths"
RELATIONAL_CACHE_PACKED_RELATIONS_KEY: Final[str] = "packed_relation_triples"
RELATIONAL_CACHE_RELATION_LENGTHS_KEY: Final[str] = "relation_lengths"
RELATIONAL_CACHE_TARGET_KEY: Final[str] = "target_tensor"


class InvalidMorpionRelationalEntityTokenCacheError(ValueError):
    """Raised when a relational entity-token cache cannot be used safely."""

    @classmethod
    def missing_or_stale(
        cls,
        rows_path: str | os.PathLike[str],
    ) -> InvalidMorpionRelationalEntityTokenCacheError:
        """Return the missing or stale relational cache error."""
        return cls(
            "Relational entity-token cache is missing or stale for rows artifact: "
            f"{rows_path!r}."
        )


@dataclass(frozen=True, slots=True)
class MorpionRelationalEntityTokenCachePaths:
    """Paths for one persisted relational entity-token cache."""

    tensor_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class MorpionRelationalEntityTokenCacheManifest:
    """Manifest describing one persisted relational entity-token cache."""

    format: str
    input_representation: str
    relation_schema: str
    relation_type_count: int
    source_rows_path: str
    source_rows_size: int
    source_rows_mtime_ns: int
    row_count: int
    entity_max_tokens: int
    input_feature_dim: int
    packed_input_shape: tuple[int, int]
    token_lengths_shape: tuple[int]
    packed_relation_shape: tuple[int, int]
    relation_lengths_shape: tuple[int]
    target_shape: tuple[int, int]
    input_dtype: str
    token_lengths_dtype: str
    relation_dtype: str
    relation_lengths_dtype: str
    target_dtype: str
    created_at_unix_s: float

    @classmethod
    def from_json_payload(
        cls,
        payload: dict[str, object],
    ) -> MorpionRelationalEntityTokenCacheManifest:
        """Build one relational cache manifest from decoded JSON."""
        return cls(
            format=_required_str(payload, "format"),
            input_representation=_required_str(payload, "input_representation"),
            relation_schema=_required_str(payload, "relation_schema"),
            relation_type_count=_required_int(payload, "relation_type_count"),
            source_rows_path=_required_str(payload, "source_rows_path"),
            source_rows_size=_required_int(payload, "source_rows_size"),
            source_rows_mtime_ns=_required_int(payload, "source_rows_mtime_ns"),
            row_count=_required_int(payload, "row_count"),
            entity_max_tokens=_required_int(payload, "entity_max_tokens"),
            input_feature_dim=_required_int(payload, "input_feature_dim"),
            packed_input_shape=_required_shape_2(payload, "packed_input_shape"),
            token_lengths_shape=_required_shape_1(payload, "token_lengths_shape"),
            packed_relation_shape=_required_shape_2(
                payload, "packed_relation_shape"
            ),
            relation_lengths_shape=_required_shape_1(
                payload, "relation_lengths_shape"
            ),
            target_shape=_required_shape_2(payload, "target_shape"),
            input_dtype=_required_str(payload, "input_dtype"),
            token_lengths_dtype=_required_str(payload, "token_lengths_dtype"),
            relation_dtype=_required_str(payload, "relation_dtype"),
            relation_lengths_dtype=_required_str(
                payload, "relation_lengths_dtype"
            ),
            target_dtype=_required_str(payload, "target_dtype"),
            created_at_unix_s=_required_float(payload, "created_at_unix_s"),
        )

    def to_json_payload(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload."""
        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True, slots=True)
class MorpionRelationalEntityTokenCache:
    """Loaded packed relational cache and timing metadata."""

    paths: MorpionRelationalEntityTokenCachePaths
    manifest: MorpionRelationalEntityTokenCacheManifest
    packed_input_tensor: torch.Tensor
    token_lengths: torch.Tensor
    token_offsets: torch.Tensor
    packed_relation_triples: torch.Tensor
    relation_lengths: torch.Tensor
    relation_offsets: torch.Tensor
    target_tensor: torch.Tensor
    rebuilt: bool
    materialize_seconds: float
    load_seconds: float


@dataclass(frozen=True, slots=True)
class _LoadedMorpionRelationalEntityTokenCache:
    """Validated persisted relational cache tensors."""

    manifest: MorpionRelationalEntityTokenCacheManifest
    packed_input_tensor: torch.Tensor
    token_lengths: torch.Tensor
    packed_relation_triples: torch.Tensor
    relation_lengths: torch.Tensor
    target_tensor: torch.Tensor


def default_relational_entity_token_cache_paths(
    rows_path: str | os.PathLike[str],
    *,
    entity_max_tokens: int,
    max_rows: int | None = None,
) -> MorpionRelationalEntityTokenCachePaths:
    """Return default relational cache paths beside one rows artifact."""
    source = Path(rows_path)
    row_limit_tag = "all" if max_rows is None else f"max_rows_{max_rows}"
    cache_stem = (
        f"{source.stem}.relational_entity_tokens."
        f"max_tokens_{entity_max_tokens}.{row_limit_tag}"
    )
    cache_dir = source.parent / RELATIONAL_ENTITY_TOKEN_CACHE_DIR_NAME
    return MorpionRelationalEntityTokenCachePaths(
        tensor_path=cache_dir / f"{cache_stem}.pt",
        manifest_path=cache_dir / f"{cache_stem}.manifest.json",
    )


def load_or_materialize_relational_entity_token_cache(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
    entity_max_tokens: int,
) -> MorpionRelationalEntityTokenCache:
    """Load a valid relational cache or rebuild it from raw rows."""
    paths = default_relational_entity_token_cache_paths(
        rows_path,
        entity_max_tokens=entity_max_tokens,
        max_rows=max_rows,
    )
    load_started_at = perf_counter()
    loaded = _try_load_valid_relational_entity_token_cache(
        paths=paths,
        rows_path=rows_path,
        entity_max_tokens=entity_max_tokens,
    )
    if loaded is not None:
        return _cache_from_loaded(
            paths=paths,
            loaded=loaded,
            load_seconds=perf_counter() - load_started_at,
        )

    materialize_started_at = perf_counter()
    (
        packed_input_tensor,
        token_lengths,
        packed_relation_triples,
        relation_lengths,
        target_tensor,
    ) = _materialize_relational_entity_tokens(
        rows_path=rows_path,
        row_chunk_size=row_chunk_size,
        max_rows=max_rows,
        entity_max_tokens=entity_max_tokens,
    )
    manifest = _relational_entity_token_cache_manifest(
        rows_path=rows_path,
        entity_max_tokens=entity_max_tokens,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        packed_relation_triples=packed_relation_triples,
        relation_lengths=relation_lengths,
        target_tensor=target_tensor,
    )
    paths.tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            RELATIONAL_CACHE_PACKED_INPUT_KEY: packed_input_tensor,
            RELATIONAL_CACHE_TOKEN_LENGTHS_KEY: token_lengths,
            RELATIONAL_CACHE_PACKED_RELATIONS_KEY: packed_relation_triples,
            RELATIONAL_CACHE_RELATION_LENGTHS_KEY: relation_lengths,
            RELATIONAL_CACHE_TARGET_KEY: target_tensor,
        },
        paths.tensor_path,
    )
    with open(paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_json_payload(), handle, indent=2, sort_keys=True)
    return MorpionRelationalEntityTokenCache(
        paths=paths,
        manifest=manifest,
        packed_input_tensor=packed_input_tensor,
        token_lengths=token_lengths,
        token_offsets=_packed_offsets(token_lengths),
        packed_relation_triples=packed_relation_triples,
        relation_lengths=relation_lengths,
        relation_offsets=_packed_offsets(relation_lengths),
        target_tensor=target_tensor,
        rebuilt=True,
        materialize_seconds=perf_counter() - materialize_started_at,
        load_seconds=0.0,
    )


def relational_entity_token_cache_is_valid(
    *,
    rows_path: str | os.PathLike[str],
    entity_max_tokens: int,
    max_rows: int | None = None,
) -> bool:
    """Return whether the default relational cache is current and compatible."""
    paths = default_relational_entity_token_cache_paths(
        rows_path,
        entity_max_tokens=entity_max_tokens,
        max_rows=max_rows,
    )
    return (
        _try_load_valid_relational_entity_token_cache(
            paths=paths,
            rows_path=rows_path,
            entity_max_tokens=entity_max_tokens,
        )
        is not None
    )


def load_relational_entity_token_cache(
    *,
    rows_path: str | os.PathLike[str],
    entity_max_tokens: int,
    max_rows: int | None = None,
) -> MorpionRelationalEntityTokenCache:
    """Load one existing valid relational entity-token cache."""
    paths = default_relational_entity_token_cache_paths(
        rows_path,
        entity_max_tokens=entity_max_tokens,
        max_rows=max_rows,
    )
    started_at = perf_counter()
    loaded = _try_load_valid_relational_entity_token_cache(
        paths=paths,
        rows_path=rows_path,
        entity_max_tokens=entity_max_tokens,
    )
    if loaded is None:
        raise InvalidMorpionRelationalEntityTokenCacheError.missing_or_stale(
            rows_path
        )
    return _cache_from_loaded(
        paths=paths,
        loaded=loaded,
        load_seconds=perf_counter() - started_at,
    )


def relational_entity_token_cache_batch(
    *,
    cache: MorpionRelationalEntityTokenCache,
    row_indices: tuple[int, ...],
) -> TensorSupervisedBatch:
    """Reconstruct one padded two-input batch from packed cache tensors."""
    if not row_indices:
        return TensorSupervisedBatch(
            input_tensor=torch.empty(
                (0, 0, cache.manifest.input_feature_dim),
                dtype=torch.float32,
            ),
            auxiliary_input_tensors=(
                torch.empty((0, 0, 3), dtype=torch.long),
            ),
            target_tensor=torch.empty((0, 1), dtype=torch.float32),
            is_batch=True,
        )

    token_lengths = [
        int(cache.token_lengths[row_index].item()) for row_index in row_indices
    ]
    relation_lengths = [
        int(cache.relation_lengths[row_index].item()) for row_index in row_indices
    ]
    input_tensor = torch.zeros(
        (
            len(row_indices),
            max(token_lengths),
            cache.manifest.input_feature_dim,
        ),
        dtype=torch.float32,
    )
    relation_triples = torch.zeros(
        (len(row_indices), max(relation_lengths), 3),
        dtype=torch.long,
    )
    for batch_index, row_index in enumerate(row_indices):
        token_count = token_lengths[batch_index]
        token_start = int(cache.token_offsets[row_index].item())
        input_tensor[batch_index, :token_count, :] = cache.packed_input_tensor[
            token_start : token_start + token_count
        ]
        relation_count = relation_lengths[batch_index]
        relation_start = int(cache.relation_offsets[row_index].item())
        relation_triples[batch_index, :relation_count, :] = (
            cache.packed_relation_triples[
                relation_start : relation_start + relation_count
            ].to(dtype=torch.long)
        )
    index_tensor = torch.tensor(row_indices, dtype=torch.long)
    return TensorSupervisedBatch(
        input_tensor=input_tensor,
        auxiliary_input_tensors=(relation_triples,),
        target_tensor=cache.target_tensor.index_select(0, index_tensor),
        is_batch=True,
    )


def _cache_from_loaded(
    *,
    paths: MorpionRelationalEntityTokenCachePaths,
    loaded: _LoadedMorpionRelationalEntityTokenCache,
    load_seconds: float,
) -> MorpionRelationalEntityTokenCache:
    """Build the public loaded cache with derived offsets."""
    return MorpionRelationalEntityTokenCache(
        paths=paths,
        manifest=loaded.manifest,
        packed_input_tensor=loaded.packed_input_tensor,
        token_lengths=loaded.token_lengths,
        token_offsets=_packed_offsets(loaded.token_lengths),
        packed_relation_triples=loaded.packed_relation_triples,
        relation_lengths=loaded.relation_lengths,
        relation_offsets=_packed_offsets(loaded.relation_lengths),
        target_tensor=loaded.target_tensor,
        rebuilt=False,
        materialize_seconds=0.0,
        load_seconds=load_seconds,
    )


def _try_load_valid_relational_entity_token_cache(
    *,
    paths: MorpionRelationalEntityTokenCachePaths,
    rows_path: str | os.PathLike[str],
    entity_max_tokens: int,
) -> _LoadedMorpionRelationalEntityTokenCache | None:
    """Load a relational cache only when source, schema, and tensors match."""
    manifest = _read_relational_entity_token_cache_manifest(paths.manifest_path)
    if manifest is None or not _manifest_matches_rows(
        manifest,
        rows_path=rows_path,
        entity_max_tokens=entity_max_tokens,
    ):
        return None
    payload = _read_relational_entity_token_cache_payload(paths.tensor_path)
    if payload is None:
        return None
    loaded = _LoadedMorpionRelationalEntityTokenCache(
        manifest=manifest,
        packed_input_tensor=payload[RELATIONAL_CACHE_PACKED_INPUT_KEY],
        token_lengths=payload[RELATIONAL_CACHE_TOKEN_LENGTHS_KEY],
        packed_relation_triples=payload[RELATIONAL_CACHE_PACKED_RELATIONS_KEY],
        relation_lengths=payload[RELATIONAL_CACHE_RELATION_LENGTHS_KEY],
        target_tensor=payload[RELATIONAL_CACHE_TARGET_KEY],
    )
    if not _tensors_match_manifest(loaded):
        return None
    return loaded


def _read_relational_entity_token_cache_manifest(
    manifest_path: Path,
) -> MorpionRelationalEntityTokenCacheManifest | None:
    """Read a relational manifest, returning ``None`` when unusable."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return MorpionRelationalEntityTokenCacheManifest.from_json_payload(
            cast("dict[str, object]", payload)
        )
    except (
        InvalidMorpionRelationalEntityTokenCacheError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _read_relational_entity_token_cache_payload(
    tensor_path: Path,
) -> dict[str, torch.Tensor] | None:
    """Read relational cache tensors, returning ``None`` when unusable."""
    try:
        payload = _torch_load_cpu(tensor_path)
    except (OSError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    keys = (
        RELATIONAL_CACHE_PACKED_INPUT_KEY,
        RELATIONAL_CACHE_TOKEN_LENGTHS_KEY,
        RELATIONAL_CACHE_PACKED_RELATIONS_KEY,
        RELATIONAL_CACHE_RELATION_LENGTHS_KEY,
        RELATIONAL_CACHE_TARGET_KEY,
    )
    tensors: dict[str, torch.Tensor] = {}
    for key in keys:
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            return None
        tensors[key] = tensor
    return tensors


def _torch_load_cpu(path: Path) -> object:
    """Load one torch payload onto CPU without requiring CUDA."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _materialize_relational_entity_tokens(
    *,
    rows_path: str | os.PathLike[str],
    row_chunk_size: int,
    max_rows: int | None,
    entity_max_tokens: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Materialize packed relational inputs and targets from row chunks."""
    dynamics = MorpionDynamics()
    converter = MorpionRelationalEntityTokenConverter(
        dynamics=dynamics,
        max_tokens=entity_max_tokens,
    )
    token_tensors: list[torch.Tensor] = []
    token_lengths: list[int] = []
    relation_tensors: list[torch.Tensor] = []
    relation_lengths: list[int] = []
    target_tensors: list[torch.Tensor] = []
    for rows in iter_morpion_supervised_row_chunks_from_path(
        os.fspath(rows_path),
        chunk_size=row_chunk_size,
        max_rows=max_rows,
    ):
        for row in rows:
            sample = process_morpion_supervised_row_to_relational_entity_token_tensors(
                row,
                dynamics=dynamics,
                converter=converter,
            )
            tokens = sample.input_tensor.detach().cpu()
            relations = sample.auxiliary_input_tensors[0].detach().to(
                device="cpu",
                dtype=torch.int32,
            )
            token_tensors.append(tokens)
            token_lengths.append(int(tokens.shape[0]))
            relation_tensors.append(relations)
            relation_lengths.append(int(relations.shape[0]))
            target_tensors.append(sample.target_tensor.detach().cpu())
    if not token_tensors:
        return (
            torch.empty((0, MORPION_ENTITY_TOKEN_FEATURE_DIM), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )
    return (
        torch.cat(token_tensors, dim=0).cpu(),
        torch.tensor(token_lengths, dtype=torch.long),
        torch.cat(relation_tensors, dim=0).to(dtype=torch.int32, device="cpu"),
        torch.tensor(relation_lengths, dtype=torch.long),
        torch.stack(target_tensors).cpu(),
    )


def _relational_entity_token_cache_manifest(
    *,
    rows_path: str | os.PathLike[str],
    entity_max_tokens: int,
    packed_input_tensor: torch.Tensor,
    token_lengths: torch.Tensor,
    packed_relation_triples: torch.Tensor,
    relation_lengths: torch.Tensor,
    target_tensor: torch.Tensor,
) -> MorpionRelationalEntityTokenCacheManifest:
    """Build one manifest for packed relational tensors."""
    source = Path(rows_path).resolve()
    source_stat = source.stat()
    return MorpionRelationalEntityTokenCacheManifest(
        format=MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT,
        input_representation=MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
        relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
        relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
        source_rows_path=str(source),
        source_rows_size=source_stat.st_size,
        source_rows_mtime_ns=source_stat.st_mtime_ns,
        row_count=int(token_lengths.shape[0]),
        entity_max_tokens=entity_max_tokens,
        input_feature_dim=MORPION_ENTITY_TOKEN_FEATURE_DIM,
        packed_input_shape=_shape_2(packed_input_tensor),
        token_lengths_shape=_shape_1(token_lengths),
        packed_relation_shape=_shape_2(packed_relation_triples),
        relation_lengths_shape=_shape_1(relation_lengths),
        target_shape=_shape_2(target_tensor),
        input_dtype=str(packed_input_tensor.dtype),
        token_lengths_dtype=str(token_lengths.dtype),
        relation_dtype=str(packed_relation_triples.dtype),
        relation_lengths_dtype=str(relation_lengths.dtype),
        target_dtype=str(target_tensor.dtype),
        created_at_unix_s=time(),
    )


def _manifest_matches_rows(
    manifest: MorpionRelationalEntityTokenCacheManifest,
    *,
    rows_path: str | os.PathLike[str],
    entity_max_tokens: int,
) -> bool:
    """Return whether a manifest matches its source and relational schema."""
    source = Path(rows_path).resolve()
    try:
        source_stat = source.stat()
    except OSError:
        return False
    return (
        manifest.format == MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT
        and manifest.input_representation
        == MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION
        and manifest.relation_schema == MORPION_ENTITY_RELATION_SCHEMA
        and manifest.relation_type_count == MORPION_ENTITY_RELATION_TYPE_COUNT
        and manifest.source_rows_path == str(source)
        and manifest.source_rows_size == source_stat.st_size
        and manifest.source_rows_mtime_ns == source_stat.st_mtime_ns
        and manifest.entity_max_tokens == entity_max_tokens
        and manifest.input_feature_dim == MORPION_ENTITY_TOKEN_FEATURE_DIM
        and manifest.packed_input_shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
        and manifest.packed_relation_shape[1] == 3
        and manifest.token_lengths_shape == (manifest.row_count,)
        and manifest.relation_lengths_shape == (manifest.row_count,)
        and manifest.target_shape == (manifest.row_count, 1)
        and manifest.input_dtype == str(torch.float32)
        and manifest.token_lengths_dtype == str(torch.int64)
        and manifest.relation_dtype == str(torch.int32)
        and manifest.relation_lengths_dtype == str(torch.int64)
        and manifest.target_dtype == str(torch.float32)
    )


def _tensors_match_manifest(
    loaded: _LoadedMorpionRelationalEntityTokenCache,
) -> bool:
    """Return whether loaded tensors match manifest and packed invariants."""
    manifest = loaded.manifest
    tensors = (
        loaded.packed_input_tensor,
        loaded.token_lengths,
        loaded.packed_relation_triples,
        loaded.relation_lengths,
        loaded.target_tensor,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        return False
    if (
        loaded.packed_input_tensor.ndim != 2
        or loaded.token_lengths.ndim != 1
        or loaded.packed_relation_triples.ndim != 2
        or loaded.relation_lengths.ndim != 1
        or loaded.target_tensor.ndim != 2
    ):
        return False
    if (
        str(loaded.packed_input_tensor.dtype) != manifest.input_dtype
        or str(loaded.token_lengths.dtype) != manifest.token_lengths_dtype
        or str(loaded.packed_relation_triples.dtype) != manifest.relation_dtype
        or str(loaded.relation_lengths.dtype) != manifest.relation_lengths_dtype
        or str(loaded.target_tensor.dtype) != manifest.target_dtype
        or _shape_2(loaded.packed_input_tensor) != manifest.packed_input_shape
        or _shape_1(loaded.token_lengths) != manifest.token_lengths_shape
        or _shape_2(loaded.packed_relation_triples)
        != manifest.packed_relation_shape
        or _shape_1(loaded.relation_lengths) != manifest.relation_lengths_shape
        or _shape_2(loaded.target_tensor) != manifest.target_shape
    ):
        return False
    if loaded.token_lengths.numel() != manifest.row_count:
        return False
    if loaded.relation_lengths.numel() != manifest.row_count:
        return False
    if (
        loaded.token_lengths.numel() > 0
        and bool(torch.any(loaded.token_lengths < 0).item())
    ):
        return False
    if (
        loaded.relation_lengths.numel() > 0
        and bool(torch.any(loaded.relation_lengths < 0).item())
    ):
        return False
    return (
        int(loaded.token_lengths.sum().item()) == manifest.packed_input_shape[0]
        and int(loaded.relation_lengths.sum().item())
        == manifest.packed_relation_shape[0]
    )


def _packed_offsets(lengths: torch.Tensor) -> torch.Tensor:
    """Return packed tensor start offsets for every row length."""
    if lengths.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(
        (
            torch.zeros((1,), dtype=torch.long),
            torch.cumsum(lengths[:-1], dim=0),
        )
    )


def _shape_1(tensor: torch.Tensor) -> tuple[int]:
    """Return one tensor shape known to have one dimension."""
    return (int(tensor.shape[0]),)


def _shape_2(tensor: torch.Tensor) -> tuple[int, int]:
    """Return one tensor shape known to have two dimensions."""
    return int(tensor.shape[0]), int(tensor.shape[1])


def _required_str(payload: dict[str, object], key: str) -> str:
    """Return one required JSON string."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidMorpionRelationalEntityTokenCacheError
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    """Return one required JSON integer."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMorpionRelationalEntityTokenCacheError
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    """Return one required JSON finite number."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidMorpionRelationalEntityTokenCacheError
    return float(value)


def _required_int_sequence(
    payload: dict[str, object],
    key: str,
) -> tuple[int, ...]:
    """Return one required JSON integer tuple."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise InvalidMorpionRelationalEntityTokenCacheError
    return tuple(value)


def _required_shape_1(payload: dict[str, object], key: str) -> tuple[int]:
    """Return one required length-one JSON shape tuple."""
    value = _required_int_sequence(payload, key)
    if len(value) != 1:
        raise InvalidMorpionRelationalEntityTokenCacheError
    return (value[0],)


def _required_shape_2(
    payload: dict[str, object],
    key: str,
) -> tuple[int, int]:
    """Return one required length-two JSON shape tuple."""
    value = _required_int_sequence(payload, key)
    if len(value) != 2:
        raise InvalidMorpionRelationalEntityTokenCacheError
    return value[0], value[1]


__all__ = [
    "MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT",
    "RELATIONAL_ENTITY_TOKEN_CACHE_DIR_NAME",
    "InvalidMorpionRelationalEntityTokenCacheError",
    "MorpionRelationalEntityTokenCache",
    "MorpionRelationalEntityTokenCacheManifest",
    "MorpionRelationalEntityTokenCachePaths",
    "default_relational_entity_token_cache_paths",
    "load_or_materialize_relational_entity_token_cache",
    "load_relational_entity_token_cache",
    "relational_entity_token_cache_batch",
    "relational_entity_token_cache_is_valid",
]
