"""Serial validated-row conversion and bounded staging to existing cache formats."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
    require,
    sync_directory,
)
from chipiron.environments.morpion.learning.tree_to_dataset import (
    _iter_validated_morpion_jsonl_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
    full_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

from . import flat_tensor_cache as flat
from . import relational_entity_token_cache as relational
from .flat_tensor_cache import _flat_tensor_cache_manifest
from .relational_entity_token_cache import _relational_entity_token_cache_manifest
from .state_split import _canonical_identity_from_state, _split_identified_rows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from chipiron.environments.morpion.learning import MorpionSupervisedRow

    from .shared_preparation import _Budget, _Recipe
    from .state_split import StateSplit


@dataclass(frozen=True)
class _Chunk:
    path: Path
    sha256: str
    rows: int
    tokens: int
    relations: int


def _save(path: Path, payload: object) -> None:
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _save_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        stream.flush()
        os.fsync(stream.fileno())


def _empty(rows: int, tokens: int, relations: int) -> dict[str, torch.Tensor]:
    return {
        "flat": torch.empty(
            (rows, len(MORPION_CANONICAL_FEATURE_NAMES)), dtype=torch.float32
        ),
        "tokens": torch.empty(
            (tokens, MORPION_ENTITY_TOKEN_FEATURE_DIM), dtype=torch.float32
        ),
        "relations": torch.empty((relations, 3), dtype=torch.int32),
        "token_lengths": torch.empty((rows,), dtype=torch.long),
        "relation_lengths": torch.empty((rows,), dtype=torch.long),
        "targets": torch.empty((rows, 1), dtype=torch.float32),
    }


def _pack(
    chunks: list[_Chunk],
    rows: Path,
    directory: Path,
    recipe: _Recipe,
    budget: _Budget,
    packed_bytes: int,
) -> dict[str, float]:
    """Allocate one final packed payload, never a list of all decoded states."""
    started = perf_counter()
    budget.check_export(directory, packed_bytes)
    counts = (
        sum(c.rows for c in chunks),
        sum(c.tokens for c in chunks),
        sum(c.relations for c in chunks),
    )
    packed = _empty(*counts)
    offset = token_offset = relation_offset = 0
    for chunk in chunks:
        require(file_digest(chunk.path) == chunk.sha256, "Prepared chunk changed")
        tensors = torch.load(
            chunk.path, map_location="cpu", weights_only=True, mmap=True
        )
        for name in ("flat", "token_lengths", "relation_lengths", "targets"):
            packed[name][offset : offset + chunk.rows].copy_(tensors[name])
        packed["tokens"][token_offset : token_offset + chunk.tokens].copy_(
            tensors["tokens"]
        )
        packed["relations"][relation_offset : relation_offset + chunk.relations].copy_(
            tensors["relations"]
        )
        offset += chunk.rows
        token_offset += chunk.tokens
        relation_offset += chunk.relations
        del tensors
    require(
        (offset, token_offset, relation_offset) == counts,
        "Prepared chunk counts differ",
    )
    packed_s = perf_counter() - started
    started = perf_counter()
    flat_manifest = _flat_tensor_cache_manifest(
        rows_path=rows, input_tensor=packed["flat"], target_tensor=packed["targets"]
    )
    _save(
        directory / "flat.pt",
        {
            flat.FLAT_TENSOR_CACHE_INPUT_KEY: packed["flat"],
            flat.FLAT_TENSOR_CACHE_TARGET_KEY: packed["targets"],
        },
    )
    _save_json(directory / "flat.manifest.json", flat_manifest.to_json_payload())
    relational_manifest = _relational_entity_token_cache_manifest(
        rows_path=rows,
        entity_max_tokens=recipe.entity_max_tokens,
        packed_input_tensor=packed["tokens"],
        token_lengths=packed["token_lengths"],
        packed_relation_triples=packed["relations"],
        relation_lengths=packed["relation_lengths"],
        target_tensor=packed["targets"],
    )
    _save(
        directory / "relational.pt",
        {
            relational.RELATIONAL_CACHE_PACKED_INPUT_KEY: packed["tokens"],
            relational.RELATIONAL_CACHE_TOKEN_LENGTHS_KEY: packed["token_lengths"],
            relational.RELATIONAL_CACHE_PACKED_RELATIONS_KEY: packed["relations"],
            relational.RELATIONAL_CACHE_RELATION_LENGTHS_KEY: packed[
                "relation_lengths"
            ],
            relational.RELATIONAL_CACHE_TARGET_KEY: packed["targets"],
        },
    )
    _save_json(
        directory / "relational.manifest.json", relational_manifest.to_json_payload()
    )
    # Array ownership ends here, before the ordinary readers validate the export.
    return {"packing_s": packed_s, "native_export_s": perf_counter() - started}


def build_shared_bundle(
    rows: Path,
    directory: Path,
    recipe: _Recipe,
    budget: _Budget,
    row_chunk_size: int,
) -> tuple[StateSplit, dict[str, float | int]]:
    """Use one certified state per row, existing converters and the native split fold."""
    budget.check_memory(3 * budget.chunk_bytes)
    dynamics = MorpionDynamics()
    flat_converter = MorpionFeatureTensorConverter(
        dynamics, full_morpion_feature_subset()
    )
    relational_converter = MorpionRelationalEntityTokenConverter(
        dynamics=dynamics, max_tokens=recipe.entity_max_tokens
    )
    chunk_size = min(row_chunk_size, 128)
    timings: dict[str, float | int] = {
        "row_validation_s": 0.0,
        "identity_s": 0.0,
        "flat_encode_s": 0.0,
        "relational_encode_s": 0.0,
        "chunk_write_s": 0.0,
        "resolved_chunk_rows": chunk_size,
    }
    chunk_directory = directory / "chunks"
    chunk_directory.mkdir()
    chunks: list[_Chunk] = []
    buffers: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("flat", "tokens", "relations", "targets")
    }
    buffered_bytes = packed_bytes = staged_bytes = 0

    def flush() -> None:
        nonlocal buffered_bytes, staged_bytes
        if not buffers["flat"]:
            return
        tick = perf_counter()
        tensors = {
            "flat": torch.stack(buffers["flat"]),
            "tokens": torch.cat(buffers["tokens"]),
            "relations": torch.cat(buffers["relations"]),
            "targets": torch.stack(buffers["targets"]),
            "token_lengths": torch.tensor(
                [len(t) for t in buffers["tokens"]], dtype=torch.long
            ),
            "relation_lengths": torch.tensor(
                [len(t) for t in buffers["relations"]], dtype=torch.long
            ),
        }
        budget.check_stage(directory, packed_bytes, staged_bytes, buffered_bytes)
        path = chunk_directory / f"{len(chunks):08d}.pt"
        _save(path, tensors)
        staged_bytes += path.stat().st_size
        chunks.append(
            _Chunk(
                path,
                file_digest(path),
                len(tensors["flat"]),
                len(tensors["tokens"]),
                len(tensors["relations"]),
            )
        )
        for values in buffers.values():
            values.clear()
        buffered_bytes = 0
        timings["chunk_write_s"] += perf_counter() - tick

    def identified_rows() -> Iterator[tuple[MorpionSupervisedRow, str]]:
        nonlocal buffered_bytes, packed_bytes
        # The native split validates the first excluded row, unlike feature-only
        # materialization. Retain that error boundary without encoding that row.
        limit = None if recipe.max_rows is None else recipe.max_rows + 1
        iterator = iter(_iter_validated_morpion_jsonl_rows(rows, max_rows=limit))
        index = 0
        while True:
            tick = perf_counter()
            validated = next(iterator, None)
            timings["row_validation_s"] += perf_counter() - tick
            if validated is None or (
                recipe.max_rows is not None and index >= recipe.max_rows
            ):
                break
            tick = perf_counter()
            identity = _canonical_identity_from_state(validated.state)
            timings["identity_s"] += perf_counter() - tick
            # Frozen wrapper: sets/tuples are immutable and dir_usage is copied.
            # Both converters are unchanged and see this one consistent state.
            state = dynamics.wrap_atomheart_state(validated.state)
            tick = perf_counter()
            flat_tensor = flat_converter.state_to_tensor(state)
            timings["flat_encode_s"] += perf_counter() - tick
            tick = perf_counter()
            converted = relational_converter.state_to_tensors(state)
            timings["relational_encode_s"] += perf_counter() - tick
            values = {
                "flat": flat_tensor,
                "tokens": converted.token_tensor,
                "relations": converted.relation_triples.to(
                    dtype=torch.int32, device="cpu"
                ),
                "targets": torch.tensor(
                    [validated.row.target_value], dtype=torch.float32
                ),
            }
            size = sum(t.numel() * t.element_size() for t in values.values()) + 16
            require(
                size <= budget.chunk_bytes,
                "One prepared row exceeds chunk memory budget",
            )
            if buffered_bytes + size > budget.chunk_bytes:
                flush()
            packed_bytes += size
            require(
                packed_bytes <= budget.packed_bytes,
                "Prepared tensors exceed packing memory budget",
            )
            for name, tensor in values.items():
                buffers[name].append(tensor)
            buffered_bytes += size
            if len(buffers["flat"]) >= chunk_size:
                flush()
            yield validated.row, identity
            index += 1
        flush()

    started = perf_counter()
    split = _split_identified_rows(
        identified_rows(), fraction=recipe.fraction, seed=recipe.seed
    )
    timings["conversion_and_split_s"] = perf_counter() - started
    timings.update(_pack(chunks, rows, directory, recipe, budget, packed_bytes))
    for chunk in chunks:
        chunk.path.unlink()
    chunk_directory.rmdir()
    _save_json(
        directory / "split.json",
        {
            "indices": sorted(split.validation_indices),
            "rows": split.row_count,
            "population": split.validation_population_sha256,
        },
    )
    sync_directory(directory)
    timings.update(
        packed_bytes=packed_bytes,
        staging_bytes=staged_bytes,
        chunks=len(chunks),
        rows=split.row_count,
    )
    return split, timings
