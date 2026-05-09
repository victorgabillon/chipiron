"""Tests for additive sharded Morpion training export persistence."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
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

from anemone.training_export import TrainingNodeSnapshot, TrainingTreeSnapshot
from atomheart.games.morpion import MorpionDynamics
from atomheart.games.morpion import initial_state as morpion_initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.learning import (
    training_tree_snapshot_to_morpion_supervised_rows,
)


def _payload_after_n_moves(move_count: int) -> dict[str, object]:
    """Build one real Morpion payload after ``move_count`` legal moves."""
    dynamics = MorpionDynamics()
    state = morpion_initial_state()
    for action in dynamics.all_legal_actions(state)[:move_count]:
        state = dynamics.step(state, action).next_state
    codec = MorpionStateCheckpointCodec()
    return cast("dict[str, object]", codec.dump_state_ref(state))


@dataclass(slots=True)
class _LiveNode:
    """Small live-node stub exposing the fields used by export builders."""

    id: str
    depth: int
    state_payload: dict[str, object]
    direct_value: float | None
    backed_up_value: float | None
    is_terminal: bool
    is_exact: bool
    visit_count: int | None
    metadata: dict[str, object] = field(default_factory=dict)
    parent_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    over_event_label: str | None = None
    allow_state_access: bool = True
    state_access_count: int = 0

    @property
    def state(self) -> dict[str, object]:
        """Return the stored payload or fail when old-node access regresses."""
        if not self.allow_state_access:
            raise AssertionError(f"state accessed for old node {self.id}")
        self.state_access_count += 1
        return self.state_payload


def _value_to_scalar(value: object | None) -> float | None:
    """Return float scalars for the live-node stubs used in these tests."""
    if value is None:
        return None
    return float(cast("int | float", value))


def _expected_snapshot(nodes: tuple[_LiveNode, ...], *, root_node_id: str) -> TrainingTreeSnapshot:
    """Build the flat training snapshot that the sharded reader should match."""
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=tuple(
            TrainingNodeSnapshot(
                node_id=node.id,
                parent_ids=node.parent_ids,
                child_ids=node.child_ids,
                depth=node.depth,
                state_ref_payload=dict(node.state_payload),
                direct_value_scalar=node.direct_value,
                backed_up_value_scalar=node.backed_up_value,
                is_terminal=node.is_terminal,
                is_exact=node.is_exact,
                over_event_label=node.over_event_label,
                visit_count=node.visit_count,
                metadata=dict(node.metadata),
            )
            for node in nodes
        ),
        metadata={"format_kind": "training_tree_snapshot", "format_version": 1},
    )


def test_sharded_generation_one_round_trips_rows_equivalently(tmp_path: Path) -> None:
    """Generation one should write manifests/shards and load equivalent rows."""
    output_dir = tmp_path / "tree_exports_sharded"
    root_node = _LiveNode(
        id="root",
        depth=0,
        state_payload=_payload_after_n_moves(0),
        direct_value=0.25,
        backed_up_value=0.5,
        is_terminal=False,
        is_exact=False,
        visit_count=11,
        child_ids=("leaf",),
        metadata={"source": "sharded-test"},
    )
    leaf_node = _LiveNode(
        id="leaf",
        depth=1,
        state_payload=_payload_after_n_moves(1),
        direct_value=0.75,
        backed_up_value=1.0,
        is_terminal=True,
        is_exact=True,
        visit_count=5,
        parent_ids=("root",),
        metadata={"source": "sharded-test"},
    )
    nodes = (root_node, leaf_node)

    generation_manifest_path = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=nodes,
        root_node_id="root",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: dict(cast("dict[str, object]", state)),
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(generation_manifest_path)
    expected_snapshot = _expected_snapshot(nodes, root_node_id="root")

    manifest_payload = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    root_manifest_payload = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest_payload["new_node_count"] == 2
    assert manifest_payload["node_count"] == 2
    assert root_manifest_payload["latest_generation"] == 1
    assert loaded_snapshot.root_node_id == expected_snapshot.root_node_id
    assert loaded_snapshot.nodes == expected_snapshot.nodes
    assert training_tree_snapshot_to_morpion_supervised_rows(loaded_snapshot) == (
        training_tree_snapshot_to_morpion_supervised_rows(expected_snapshot)
    )


def test_sharded_generation_two_reuses_old_nodes_without_state_access(tmp_path: Path) -> None:
    """Generation two should only serialize new-node state payloads."""
    output_dir = tmp_path / "tree_exports_sharded"
    generation_one_nodes = (
        _LiveNode(
            id="a",
            depth=0,
            state_payload=_payload_after_n_moves(0),
            direct_value=0.1,
            backed_up_value=0.2,
            is_terminal=False,
            is_exact=False,
            visit_count=10,
            child_ids=("b",),
            metadata={"tag": "gen1-a"},
        ),
        _LiveNode(
            id="b",
            depth=1,
            state_payload=_payload_after_n_moves(1),
            direct_value=0.9,
            backed_up_value=1.1,
            is_terminal=True,
            is_exact=True,
            visit_count=7,
            parent_ids=("a",),
            metadata={"tag": "gen1-b"},
        ),
    )
    save_morpion_sharded_training_tree_from_live_nodes(
        nodes=generation_one_nodes,
        root_node_id="a",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: dict(cast("dict[str, object]", state)),
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )

    old_a = _LiveNode(
        id="a",
        depth=0,
        state_payload=_payload_after_n_moves(0),
        direct_value=0.3,
        backed_up_value=0.4,
        is_terminal=False,
        is_exact=False,
        visit_count=15,
        child_ids=("c", "b"),
        metadata={"tag": "gen2-a"},
        allow_state_access=False,
    )
    new_c = _LiveNode(
        id="c",
        depth=1,
        state_payload=_payload_after_n_moves(2),
        direct_value=0.5,
        backed_up_value=0.6,
        is_terminal=False,
        is_exact=True,
        visit_count=4,
        parent_ids=("a",),
        metadata={"tag": "gen2-c"},
    )
    old_b = _LiveNode(
        id="b",
        depth=1,
        state_payload=_payload_after_n_moves(1),
        direct_value=1.2,
        backed_up_value=1.4,
        is_terminal=True,
        is_exact=True,
        visit_count=9,
        parent_ids=("a",),
        metadata={"tag": "gen2-b"},
        allow_state_access=False,
    )
    generation_two_nodes = (old_a, new_c, old_b)

    generation_manifest_path = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=generation_two_nodes,
        root_node_id="a",
        output_dir=output_dir,
        generation=2,
        state_ref_dumper=lambda state: dict(cast("dict[str, object]", state)),
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(generation_manifest_path)
    expected_snapshot = _expected_snapshot(generation_two_nodes, root_node_id="a")

    generation_two_node_shard = json.loads(
        (output_dir / "node_shards" / "generation_000002.json").read_text(
            encoding="utf-8"
        )
    )
    generation_two_update_shard = json.loads(
        (output_dir / "update_shards" / "generation_000002.json").read_text(
            encoding="utf-8"
        )
    )

    assert old_a.state_access_count == 0
    assert old_b.state_access_count == 0
    assert new_c.state_access_count == 1
    assert [record["node_id"] for record in generation_two_node_shard["nodes"]] == ["c"]
    assert [update["node_id"] for update in generation_two_update_shard["updates"]] == [
        "a",
        "c",
        "b",
    ]
    assert generation_two_update_shard["updates"][0]["backed_up_value_scalar"] == 0.4
    assert [node.node_id for node in loaded_snapshot.nodes] == ["a", "c", "b"]
    assert loaded_snapshot.nodes[0].backed_up_value_scalar == 0.4
    assert loaded_snapshot.nodes[1].backed_up_value_scalar == 0.6
    assert loaded_snapshot.nodes[2].backed_up_value_scalar == 1.4
    assert loaded_snapshot.nodes[0].metadata == {"tag": "gen2-a"}
    assert loaded_snapshot.nodes[1].metadata == {"tag": "gen2-c"}
    assert loaded_snapshot.nodes[2].metadata == {"tag": "gen2-b"}
    assert training_tree_snapshot_to_morpion_supervised_rows(loaded_snapshot) == (
        training_tree_snapshot_to_morpion_supervised_rows(expected_snapshot)
    )