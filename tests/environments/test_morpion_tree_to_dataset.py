"""Tests for Morpion raw supervised-row extraction from training snapshots."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"

if "chipiron" not in sys.modules:
    _chipiron_stub = ModuleType("chipiron")
    _chipiron_stub.__path__ = [str(_CHIPIRON_PACKAGE_ROOT)]
    sys.modules["chipiron"] = _chipiron_stub

if "atomheart" not in sys.modules:
    _atomheart_stub = ModuleType("atomheart")
    _atomheart_stub.__path__ = [str(_ATOMHEART_PACKAGE_ROOT)]
    sys.modules["atomheart"] = _atomheart_stub

if "anemone" not in sys.modules:
    _anemone_stub = ModuleType("anemone")
    _anemone_stub.__path__ = [str(_ANEMONE_PACKAGE_ROOT)]
    sys.modules["anemone"] = _anemone_stub

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
)
from atomheart.games.morpion import MorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.learning.tree_to_dataset import (
    InvalidMorpionStateRefPayloadError,
    MalformedMorpionSupervisedRowsError,
    decode_morpion_state_ref_payload,
    is_morpion_state_ref_payload,
    load_morpion_supervised_rows,
    morpion_supervised_rows_from_dict,
    save_morpion_supervised_rows,
    training_node_to_morpion_supervised_row,
    training_tree_snapshot_to_morpion_supervised_rows,
)


def _make_morpion_payload() -> dict[str, object]:
    """Build one real Morpion checkpoint payload from a one-step state."""
    dynamics = MorpionDynamics()
    start_state = morpion_initial_state()
    first_action = dynamics.all_legal_actions(start_state)[0]
    next_state = dynamics.step(start_state, first_action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(next_state))


def _make_training_node(
    *,
    node_id: str = "node",
    state_ref_payload: object | None = None,
    direct_value_scalar: float | None = 0.5,
    backed_up_value_scalar: float | None = 1.25,
    is_terminal: bool = False,
    is_exact: bool = True,
    depth: int = 3,
    visit_count: int | None = 7,
) -> TrainingNodeSnapshot:
    """Build one training node snapshot for row-extraction tests."""
    return TrainingNodeSnapshot(
        node_id=node_id,
        parent_ids=(),
        child_ids=(),
        depth=depth,
        state_ref_payload=state_ref_payload,
        direct_value_scalar=direct_value_scalar,
        backed_up_value_scalar=backed_up_value_scalar,
        is_terminal=is_terminal,
        is_exact=is_exact,
        over_event_label=None,
        visit_count=visit_count,
        metadata={"source": "test"},
    )


def test_morpion_payload_validation_and_decode_round_trip() -> None:
    """Real Morpion checkpoint payloads should validate and round-trip."""
    payload = _make_morpion_payload()

    assert is_morpion_state_ref_payload(payload) is True

    decoded_state = decode_morpion_state_ref_payload(payload)
    codec = MorpionStateCheckpointCodec()

    assert codec.dump_state_ref(decoded_state) == payload


def test_single_training_node_converts_to_row() -> None:
    """A valid training node should convert to one raw supervised row."""
    node = _make_training_node(state_ref_payload=_make_morpion_payload())

    row = training_node_to_morpion_supervised_row(node)

    assert row is not None
    assert row.target_value == 1.25
    assert row.node_id == "node"
    assert row.depth == 3
    assert row.is_terminal is False
    assert row.is_exact is True
    assert row.visit_count == 7
    assert row.direct_value == 0.5


def test_training_node_can_prefer_direct_value() -> None:
    """Direct-value extraction should be configurable per call."""
    node = _make_training_node(state_ref_payload=_make_morpion_payload())

    row = training_node_to_morpion_supervised_row(
        node,
        use_backed_up_value=False,
    )

    assert row is not None
    assert row.target_value == 0.5


def test_training_node_filters_work() -> None:
    """Extraction filters should reject nodes that do not meet requirements."""
    payload = _make_morpion_payload()

    assert (
        training_node_to_morpion_supervised_row(
            _make_training_node(state_ref_payload=None)
        )
        is None
    )
    assert (
        training_node_to_morpion_supervised_row(
            _make_training_node(
                state_ref_payload=payload,
                direct_value_scalar=0.5,
                backed_up_value_scalar=None,
            )
        )
        is None
    )
    assert (
        training_node_to_morpion_supervised_row(
            _make_training_node(state_ref_payload=payload, depth=1),
            min_depth=2,
        )
        is None
    )
    assert (
        training_node_to_morpion_supervised_row(
            _make_training_node(state_ref_payload=payload, visit_count=1),
            min_visit_count=2,
        )
        is None
    )
    assert (
        training_node_to_morpion_supervised_row(
            _make_training_node(
                state_ref_payload=payload,
                is_terminal=False,
                is_exact=False,
            ),
            require_exact_or_terminal=True,
        )
        is None
    )


def test_invalid_payload_raises_clearly() -> None:
    """Malformed Morpion payloads should raise instead of being silently kept."""
    node = _make_training_node(
        state_ref_payload={"variant": "5T", "played_moves": "not-a-sequence"}
    )

    with pytest.raises(InvalidMorpionStateRefPayloadError):
        training_node_to_morpion_supervised_row(node)


def test_filtered_out_node_skips_payload_validation() -> None:
    """Filter rejection should return ``None`` before malformed payload validation."""
    node = _make_training_node(
        state_ref_payload={"variant": "5T", "played_moves": "not-a-sequence"},
        depth=1,
    )

    row = training_node_to_morpion_supervised_row(node, min_depth=2)

    assert row is None


def test_full_snapshot_extracts_ordered_rows() -> None:
    """Tree extraction should preserve surviving node order and settings metadata."""
    payload = _make_morpion_payload()
    snapshot = TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(
            _make_training_node(node_id="keep-1", state_ref_payload=payload, depth=2),
            _make_training_node(
                node_id="drop-depth",
                state_ref_payload=payload,
                depth=1,
            ),
            _make_training_node(node_id="keep-2", state_ref_payload=payload, depth=4),
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )

    rows = training_tree_snapshot_to_morpion_supervised_rows(
        snapshot,
        min_depth=2,
        use_backed_up_value=True,
    )

    assert tuple(row.node_id for row in rows.rows) == ("keep-1", "keep-2")
    assert rows.metadata["num_rows"] == 2
    assert rows.metadata["source_root_node_id"] == "root"
    assert rows.metadata["min_depth"] == 2
    assert rows.metadata["use_backed_up_value"] is True


def test_persistence_round_trip_for_morpion_supervised_rows(
    tmp_path: Path,
) -> None:
    """Persisted Morpion supervised rows should round-trip through disk."""
    payload = _make_morpion_payload()
    snapshot = TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(_make_training_node(state_ref_payload=payload),),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )
    rows = training_tree_snapshot_to_morpion_supervised_rows(snapshot)
    path = tmp_path / "morpion_supervised_rows.json"

    save_morpion_supervised_rows(rows, path)
    restored = load_morpion_supervised_rows(path)

    assert restored == rows


def test_malformed_row_payload_fails_loudly() -> None:
    """Malformed raw-row payloads should raise instead of being silently dropped."""
    with pytest.raises(MalformedMorpionSupervisedRowsError):
        morpion_supervised_rows_from_dict({"rows": {}})
