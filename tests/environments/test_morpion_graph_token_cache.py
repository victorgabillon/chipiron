"""Tests for Morpion graph-token tensor cache streaming support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.datasets import (
    collate_morpion_graph_supervised_samples,
    process_morpion_supervised_row_to_graph_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MorpionGraphTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.graph_token_cache import (
    default_graph_token_cache_paths,
    graph_cache_batch,
    graph_token_cache_is_valid,
    load_or_materialize_graph_token_cache,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_model_bundle import _build_rows_file

if TYPE_CHECKING:
    from pathlib import Path


def test_default_graph_token_cache_paths_include_token_and_row_limits(
    tmp_path: Path,
) -> None:
    """Default graph cache paths should include max tokens and row limit."""
    rows_path = tmp_path / "rows" / "generation_000035.jsonl"

    all_paths = default_graph_token_cache_paths(rows_path, graph_max_tokens=1536)
    limited_paths = default_graph_token_cache_paths(
        rows_path,
        graph_max_tokens=1536,
        max_rows=10_000,
    )

    assert all_paths.tensor_path == (
        tmp_path
        / "rows"
        / "tensor_cache"
        / "generation_000035.graph_tokens.max_tokens_1536.all.pt"
    )
    assert all_paths.manifest_path == (
        tmp_path
        / "rows"
        / "tensor_cache"
        / "generation_000035.graph_tokens.max_tokens_1536.all.manifest.json"
    )
    assert limited_paths.tensor_path.name == (
        "generation_000035.graph_tokens.max_tokens_1536.max_rows_10000.pt"
    )


def test_graph_token_cache_materializes_reuses_and_packs(
    tmp_path: Path,
) -> None:
    """A graph-token cache should build canonical packed CPU tensors."""
    rows_path, _rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.0, 0.5),
    )

    cache = load_or_materialize_graph_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
        graph_max_tokens=128,
    )
    cached_again = load_or_materialize_graph_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
        graph_max_tokens=128,
    )

    assert cache.rebuilt is True
    assert cached_again.rebuilt is False
    assert cached_again.load_seconds >= 0.0
    assert graph_token_cache_is_valid(
        rows_path=rows_path,
        graph_max_tokens=128,
        max_rows=3,
    )
    assert cached_again.manifest.row_count == 3
    assert cached_again.manifest.graph_max_tokens == 128
    assert cached_again.packed_input_tensor.device.type == "cpu"
    assert cached_again.token_lengths.device.type == "cpu"
    assert cached_again.target_tensor.device.type == "cpu"
    assert cached_again.token_lengths.shape == (3,)
    assert cached_again.target_tensor.shape == (3, 1)
    assert cached_again.packed_input_tensor.shape[1] == MORPION_GRAPH_TOKEN_FEATURE_DIM
    assert int(cached_again.token_lengths.sum().item()) == int(
        cached_again.packed_input_tensor.shape[0]
    )


def test_graph_token_cache_invalidates_when_source_changes(tmp_path: Path) -> None:
    """Cache validity should depend on the source rows artifact fingerprint."""
    rows_path, _rows = _build_jsonl_rows_file(tmp_path, target_values=(0.25, 0.5))

    load_or_materialize_graph_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        graph_max_tokens=128,
    )
    assert graph_token_cache_is_valid(rows_path=rows_path, graph_max_tokens=128)

    with open(rows_path, "a", encoding="utf-8") as handle:
        handle.write("\n")

    assert not graph_token_cache_is_valid(rows_path=rows_path, graph_max_tokens=128)


def test_graph_cache_batch_matches_direct_graph_collate(tmp_path: Path) -> None:
    """Packed cache batch reconstruction should match direct graph collate."""
    rows_path, rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.0, 0.5),
    )
    cache = load_or_materialize_graph_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        graph_max_tokens=128,
    )
    row_indices = (0, 2, 3)
    dynamics = MorpionDynamics()
    converter = MorpionGraphTokenConverter(dynamics=dynamics, max_tokens=128)
    direct_batch = collate_morpion_graph_supervised_samples(
        tuple(
            process_morpion_supervised_row_to_graph_tensors(
                rows[row_index],
                dynamics=dynamics,
                converter=converter,
            )
            for row_index in row_indices
        )
    )
    cached_batch = graph_cache_batch(cache=cache, row_indices=row_indices)

    torch.testing.assert_close(
        cached_batch.get_input_layer(),
        direct_batch.get_input_layer(),
    )
    torch.testing.assert_close(
        cached_batch.get_target_value(),
        direct_batch.get_target_value(),
    )


def test_graph_token_cache_handles_empty_rows(tmp_path: Path) -> None:
    """An empty rows artifact should materialize a valid empty graph cache."""
    rows_path, _rows = _build_jsonl_rows_file(tmp_path, target_values=())

    cache = load_or_materialize_graph_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        graph_max_tokens=128,
    )

    assert cache.manifest.row_count == 0
    assert cache.packed_input_tensor.shape == (0, MORPION_GRAPH_TOKEN_FEATURE_DIM)
    assert cache.token_lengths.shape == (0,)
    assert cache.target_tensor.shape == (0, 1)
    assert graph_token_cache_is_valid(rows_path=rows_path, graph_max_tokens=128)


def _build_jsonl_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...],
) -> tuple[Path, tuple[MorpionSupervisedRow, ...]]:
    """Build one Morpion supervised JSONL rows artifact and return its rows."""
    json_rows_path = _build_rows_file(tmp_path, target_values=target_values)
    rows = load_morpion_supervised_rows(json_rows_path)
    jsonl_rows_path = tmp_path / "morpion_supervised_rows.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=rows.rows,
        metadata=rows.metadata,
        path=jsonl_rows_path,
    )
    return jsonl_rows_path, rows.rows
