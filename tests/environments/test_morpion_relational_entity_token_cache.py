"""Tests for the packed Morpion relational entity-token cache."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.datasets import (
    collate_morpion_relational_entity_token_supervised_samples,
    process_morpion_supervised_row_to_relational_entity_token_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    InvalidMorpionRelationalEntityTokenCacheError,
    default_relational_entity_token_cache_paths,
    load_or_materialize_relational_entity_token_cache,
    load_relational_entity_token_cache,
    relational_entity_token_cache_batch,
    relational_entity_token_cache_is_valid,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_entity_token_cache import (
    _build_jsonl_rows_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_default_relational_cache_paths_include_token_and_row_limits(
    tmp_path: Path,
) -> None:
    """Relational cache paths should be distinct and encode both limits."""
    rows_path = tmp_path / "rows" / "generation_000035.jsonl"

    all_paths = default_relational_entity_token_cache_paths(
        rows_path,
        entity_max_tokens=1536,
    )
    limited_paths = default_relational_entity_token_cache_paths(
        rows_path,
        entity_max_tokens=1536,
        max_rows=10_000,
    )

    assert all_paths.tensor_path == (
        tmp_path
        / "rows"
        / "tensor_cache"
        / "generation_000035.relational_entity_tokens.max_tokens_1536.all.pt"
    )
    assert all_paths.manifest_path.name == (
        "generation_000035.relational_entity_tokens."
        "max_tokens_1536.all.manifest.json"
    )
    assert limited_paths.tensor_path.name == (
        "generation_000035.relational_entity_tokens."
        "max_tokens_1536.max_rows_10000.pt"
    )


def test_relational_cache_materializes_reuses_and_packs(tmp_path: Path) -> None:
    """The cache should persist canonical packed CPU tensors and metadata."""
    rows_path, _rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.0, 0.5),
    )

    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
        entity_max_tokens=128,
    )
    cached_again = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
        entity_max_tokens=128,
    )

    assert cache.rebuilt is True
    assert cached_again.rebuilt is False
    assert cached_again.manifest.row_count == 3
    assert cached_again.manifest.relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert (
        cached_again.manifest.relation_type_count
        == MORPION_ENTITY_RELATION_TYPE_COUNT
    )
    assert relational_entity_token_cache_is_valid(
        rows_path=rows_path,
        entity_max_tokens=128,
        max_rows=3,
    )
    assert cached_again.packed_input_tensor.device.type == "cpu"
    assert cached_again.token_lengths.device.type == "cpu"
    assert cached_again.packed_relation_triples.device.type == "cpu"
    assert cached_again.relation_lengths.device.type == "cpu"
    assert cached_again.target_tensor.device.type == "cpu"
    assert cached_again.packed_input_tensor.dtype == torch.float32
    assert cached_again.token_lengths.dtype == torch.long
    assert cached_again.packed_relation_triples.dtype == torch.int32
    assert cached_again.relation_lengths.dtype == torch.long
    assert cached_again.target_tensor.dtype == torch.float32
    assert cached_again.token_lengths.shape == (3,)
    assert cached_again.relation_lengths.shape == (3,)
    assert cached_again.packed_input_tensor.shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert cached_again.packed_relation_triples.shape[1] == 3
    assert cached_again.target_tensor.shape == (3, 1)
    assert int(cached_again.token_lengths.sum().item()) == int(
        cached_again.packed_input_tensor.shape[0]
    )
    assert int(cached_again.relation_lengths.sum().item()) == int(
        cached_again.packed_relation_triples.shape[0]
    )

    explicitly_loaded = load_relational_entity_token_cache(
        rows_path=rows_path,
        entity_max_tokens=128,
        max_rows=3,
    )
    assert explicitly_loaded.rebuilt is False


def test_relational_cache_invalidates_on_source_change_and_token_cap(
    tmp_path: Path,
) -> None:
    """Source fingerprints and token caps should select compatible caches only."""
    rows_path, _rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(0.25, 0.5),
    )
    first = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        entity_max_tokens=64,
    )
    other_cap = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        entity_max_tokens=128,
    )

    assert first.paths != other_cap.paths
    assert first.paths.tensor_path.exists()
    assert other_cap.paths.tensor_path.exists()
    with open(rows_path, "a", encoding="utf-8") as handle:
        handle.write("\n")

    assert not relational_entity_token_cache_is_valid(
        rows_path=rows_path,
        entity_max_tokens=64,
    )
    with pytest.raises(InvalidMorpionRelationalEntityTokenCacheError):
        load_relational_entity_token_cache(
            rows_path=rows_path,
            entity_max_tokens=64,
        )


def test_cached_relational_batch_exactly_matches_direct_collation(
    tmp_path: Path,
) -> None:
    """Packed reconstruction should preserve both inputs and requested order."""
    rows_path, rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.0, 0.5),
    )
    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        entity_max_tokens=128,
    )
    row_indices = (3, 0, 2)
    dynamics = MorpionDynamics()
    converter = MorpionRelationalEntityTokenConverter(
        dynamics=dynamics,
        max_tokens=128,
    )
    direct_batch = collate_morpion_relational_entity_token_supervised_samples(
        tuple(
            process_morpion_supervised_row_to_relational_entity_token_tensors(
                rows[row_index],
                dynamics=dynamics,
                converter=converter,
            )
            for row_index in row_indices
        )
    )
    cached_batch = relational_entity_token_cache_batch(
        cache=cache,
        row_indices=row_indices,
    )

    assert cached_batch.is_batch is True
    assert len(cached_batch.auxiliary_input_tensors) == 1
    assert cached_batch.auxiliary_input_tensors[0].dtype == torch.long
    assert torch.equal(cached_batch.input_tensor, direct_batch.input_tensor)
    assert torch.equal(
        cached_batch.auxiliary_input_tensors[0],
        direct_batch.auxiliary_input_tensors[0],
    )
    assert torch.equal(cached_batch.target_tensor, direct_batch.target_tensor)


def test_relational_cache_and_batch_handle_empty_rows(tmp_path: Path) -> None:
    """Empty artifacts and selections should retain exact rank and dtype."""
    rows_path, _rows = _build_jsonl_rows_file(tmp_path, target_values=())

    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        entity_max_tokens=128,
    )
    batch = relational_entity_token_cache_batch(cache=cache, row_indices=())

    assert cache.manifest.row_count == 0
    assert cache.packed_input_tensor.shape == (0, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert cache.token_lengths.shape == (0,)
    assert cache.token_offsets.shape == (0,)
    assert cache.packed_relation_triples.shape == (0, 3)
    assert cache.packed_relation_triples.dtype == torch.int32
    assert cache.relation_lengths.shape == (0,)
    assert cache.relation_offsets.shape == (0,)
    assert cache.target_tensor.shape == (0, 1)
    assert relational_entity_token_cache_is_valid(
        rows_path=rows_path,
        entity_max_tokens=128,
    )
    assert batch.input_tensor.shape == (0, 0, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert batch.input_tensor.dtype == torch.float32
    assert batch.auxiliary_input_tensors[0].shape == (0, 0, 3)
    assert batch.auxiliary_input_tensors[0].dtype == torch.long
    assert batch.target_tensor.shape == (0, 1)
    assert batch.target_tensor.dtype == torch.float32


@pytest.mark.parametrize(
    ("manifest_key", "invalid_value"),
    (
        ("format", "wrong_cache_v1"),
        ("input_representation", "wrong_entity_tokens_v1"),
        ("relation_schema", "wrong_relations_v1"),
        ("relation_type_count", 17),
        ("input_feature_dim", 24),
        ("packed_relation_shape", [1, 4]),
        ("relation_dtype", "torch.int64"),
    ),
)
def test_relational_cache_rejects_incompatible_manifest_schema(
    tmp_path: Path,
    manifest_key: str,
    invalid_value: object,
) -> None:
    """Every semantic and packed-layout discriminator should be validated."""
    rows_path, _rows = _build_jsonl_rows_file(tmp_path, target_values=(0.25,))
    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=1,
        max_rows=None,
        entity_max_tokens=128,
    )
    with open(cache.paths.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest[manifest_key] = invalid_value
    with open(cache.paths.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    assert not relational_entity_token_cache_is_valid(
        rows_path=rows_path,
        entity_max_tokens=128,
    )
    rebuilt = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=1,
        max_rows=None,
        entity_max_tokens=128,
    )
    assert rebuilt.rebuilt is True
    assert rebuilt.manifest.relation_schema == MORPION_ENTITY_RELATION_SCHEMA

