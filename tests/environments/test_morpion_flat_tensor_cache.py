"""Tests for Morpion flat tensor cache streaming support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_CANONICAL_FEATURE_NAMES,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_MODEL_KIND,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    is_flat_morpion_training_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
    default_flat_tensor_cache_paths,
    feature_indices_for_subset,
    flat_cache_batch,
    flat_tensor_cache_is_valid,
    load_or_materialize_flat_tensor_cache,
    select_flat_features,
)
from tests.environments.test_morpion_model_bundle import _build_rows_file

if TYPE_CHECKING:
    from pathlib import Path


def test_default_flat_tensor_cache_paths_include_row_limit(tmp_path: Path) -> None:
    """Default cache paths should be stable and adjacent to the rows artifact."""
    rows_path = tmp_path / "rows" / "generation_000035.jsonl"

    all_paths = default_flat_tensor_cache_paths(rows_path)
    limited_paths = default_flat_tensor_cache_paths(rows_path, max_rows=10_000)

    assert all_paths.tensor_path == (
        tmp_path / "rows" / "tensor_cache" / "generation_000035.flat_features.all.pt"
    )
    assert all_paths.manifest_path == (
        tmp_path
        / "rows"
        / "tensor_cache"
        / "generation_000035.flat_features.all.manifest.json"
    )
    assert limited_paths.tensor_path.name == (
        "generation_000035.flat_features.max_rows_10000.pt"
    )


def test_flat_tensor_cache_materializes_reuses_and_batches(
    tmp_path: Path,
) -> None:
    """A flat cache should build canonical CPU tensors and load them on reuse."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.0, 0.5),
    )

    cache = load_or_materialize_flat_tensor_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
    )
    cached_again = load_or_materialize_flat_tensor_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=3,
    )
    requested_features = MORPION_CANONICAL_FEATURE_NAMES[:5]
    batch = flat_cache_batch(
        cache=cached_again,
        row_indices=(0, 2),
        requested_feature_names=requested_features,
    )

    assert cache.rebuilt is True
    assert cached_again.rebuilt is False
    assert cached_again.load_seconds >= 0.0
    assert flat_tensor_cache_is_valid(rows_path=rows_path, max_rows=3)
    assert cached_again.manifest.row_count == 3
    assert cached_again.manifest.feature_names == MORPION_CANONICAL_FEATURE_NAMES
    assert cached_again.input_tensor.device.type == "cpu"
    assert cached_again.target_tensor.device.type == "cpu"
    assert cached_again.input_tensor.shape == (3, len(MORPION_CANONICAL_FEATURE_NAMES))
    assert cached_again.target_tensor.shape == (3, 1)
    assert batch.get_input_layer().shape == (2, len(requested_features))
    assert batch.get_target_value().shape == (2, 1)
    torch.testing.assert_close(
        batch.get_input_layer(),
        cached_again.input_tensor.index_select(
            0,
            torch.tensor((0, 2), dtype=torch.long),
        )[:, : len(requested_features)],
    )


def test_flat_tensor_cache_invalidates_when_source_changes(tmp_path: Path) -> None:
    """Cache validity should depend on the source rows artifact fingerprint."""
    rows_path = _build_jsonl_rows_file(tmp_path, target_values=(0.25, 0.5))

    load_or_materialize_flat_tensor_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
    )
    assert flat_tensor_cache_is_valid(rows_path=rows_path)

    with open(rows_path, "a", encoding="utf-8") as handle:
        handle.write("\n")

    assert not flat_tensor_cache_is_valid(rows_path=rows_path)


def test_flat_feature_subset_selection_uses_canonical_indices() -> None:
    """Feature selection should preserve canonical feature ordering."""
    input_tensor = torch.arange(
        2 * len(MORPION_CANONICAL_FEATURE_NAMES),
        dtype=torch.float32,
    ).reshape(2, len(MORPION_CANONICAL_FEATURE_NAMES))
    requested_features = (
        MORPION_CANONICAL_FEATURE_NAMES[0],
        MORPION_CANONICAL_FEATURE_NAMES[3],
        MORPION_CANONICAL_FEATURE_NAMES[4],
    )

    selected = select_flat_features(input_tensor, requested_features)

    assert feature_indices_for_subset(requested_features) == (0, 3, 4)
    torch.testing.assert_close(selected, input_tensor[:, (0, 3, 4)])


def test_flat_cache_model_kind_predicate_keeps_graph_path_uncached() -> None:
    """Only flat handcrafted-feature model kinds should use the tensor cache."""
    assert is_flat_morpion_training_model_kind("linear")
    assert is_flat_morpion_training_model_kind("mlp")
    assert not is_flat_morpion_training_model_kind(MORPION_GRAPH_MODEL_KIND)


def _build_jsonl_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...],
) -> Path:
    """Build one Morpion supervised JSONL rows artifact."""
    json_rows_path = _build_rows_file(tmp_path, target_values=target_values)
    rows = load_morpion_supervised_rows(json_rows_path)
    jsonl_rows_path = tmp_path / "morpion_supervised_rows.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=rows.rows,
        metadata=rows.metadata,
        path=jsonl_rows_path,
    )
    return jsonl_rows_path
