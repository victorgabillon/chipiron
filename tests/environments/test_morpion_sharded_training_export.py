"""Tests for additive sharded Morpion training export persistence."""
# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

_REPO_ROOT = Path(__file__).resolve().parents[2]
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    checkpoint_payload_to_jsonable,
)

_CHIPIRON_PACKAGE_ROOT = _REPO_ROOT / "src" / "chipiron"
_ATOMHEART_PACKAGE_ROOT = _REPO_ROOT.parent / "atomheart" / "src" / "atomheart"
_ANEMONE_PACKAGE_ROOT = _REPO_ROOT.parent / "anemone" / "src" / "anemone"


def _state_access_regression_error(node_id: str) -> AssertionError:
    """Return the stable old-node state-access regression error."""
    return AssertionError(f"state accessed for old node {node_id}")


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

from anemone.checkpoints.state_handles import (
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
)
from anemone.training_export import TrainingTreeSnapshot
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
from tests.environments.morpion_training_snapshot_helpers import (
    make_training_node_snapshot,
)


def _payload_after_n_moves(move_count: int) -> object:
    """Build one real Morpion payload after ``move_count`` legal moves."""
    dynamics = MorpionDynamics()
    state = morpion_initial_state()
    for action in dynamics.all_legal_actions(state)[:move_count]:
        state = dynamics.step(state, action).next_state
    codec = MorpionStateCheckpointCodec()
    return codec.dump_state_ref(state)


def _compact_payload_after_n_moves(
    move_count: int,
) -> tuple[int, tuple[int, ...]]:
    """Build one tuple-shaped Morpion checkpoint payload."""
    payload = _payload_after_n_moves(move_count)
    return cast("tuple[int, tuple[int, ...]]", payload)


@dataclass(slots=True)
class _LiveNode:
    """Small live-node stub exposing the fields used by export builders."""

    id: str
    depth: int
    state_payload: object
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
    state_handle: object | None = None

    @property
    def state(self) -> object:
        """Return the stored payload or fail when old-node access regresses."""
        if not self.allow_state_access:
            raise _state_access_regression_error(self.id)
        self.state_access_count += 1
        return self.state_payload


@dataclass(slots=True)
class _LiveResolverShape:
    """Resolver stub matching live compact resolver payload storage shape."""

    state_payloads_by_node_id: dict[int, object]


def _value_to_scalar(value: object | None) -> float | None:
    """Return float scalars for the live-node stubs used in these tests."""
    if value is None:
        return None
    return float(cast("int | float", value))


def _expected_snapshot(
    nodes: tuple[_LiveNode, ...], *, root_node_id: str
) -> TrainingTreeSnapshot:
    """Build the flat training snapshot that the sharded reader should match."""
    return TrainingTreeSnapshot(
        root_node_id=root_node_id,
        nodes=tuple(
            make_training_node_snapshot(
                node_id=node.id,
                parent_ids=node.parent_ids,
                child_ids=node.child_ids,
                depth=node.depth,
                state_ref_payload=checkpoint_payload_to_jsonable(node.state_payload),
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


def test_sharded_generation_one_round_trips_rows_equivalently(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Generation one should write manifests/shards and load equivalent rows."""
    caplog.set_level(logging.INFO)
    output_dir = tmp_path / "tree_exports_sharded"
    root_node = _LiveNode(
        id="root",
        depth=0,
        state_payload=_compact_payload_after_n_moves(0),
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
        state_payload=_compact_payload_after_n_moves(1),
        direct_value=0.75,
        backed_up_value=1.0,
        is_terminal=True,
        is_exact=True,
        visit_count=5,
        parent_ids=("root",),
        metadata={"source": "sharded-test"},
    )
    nodes = (root_node, leaf_node)

    generation_manifest_path, stats = (
        save_morpion_sharded_training_tree_from_live_nodes(
            nodes=nodes,
            root_node_id="root",
            output_dir=output_dir,
            generation=1,
            state_ref_dumper=lambda state: state,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
        )
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )
    expected_snapshot = _expected_snapshot(nodes, root_node_id="root")

    manifest_payload = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    root_manifest_payload = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest_payload["new_node_count"] == 2
    assert manifest_payload["node_count"] == 2
    assert root_manifest_payload["latest_generation"] == 1
    assert stats.node_count == 2
    assert stats.new_node_count == 2
    assert stats.reused_node_count == 0
    assert stats.rows_written == 4
    assert stats.shards_written == 5
    assert stats.bytes_written > 0
    assert stats.row_build_s >= 0.0
    assert stats.json_encode_s >= 0.0
    assert stats.write_s >= 0.0
    assert stats.total_s >= 0.0
    assert "[tree-export-profile]" in caplog.text
    assert "[tree-export-timing]" in caplog.text
    assert "[tree-export-memory]" in caplog.text
    assert root_node.state_access_count == 1
    assert leaf_node.state_access_count == 1
    assert loaded_snapshot.root_node_id == expected_snapshot.root_node_id
    assert loaded_snapshot.nodes == expected_snapshot.nodes
    loaded_rows = training_tree_snapshot_to_morpion_supervised_rows(loaded_snapshot)
    expected_rows = training_tree_snapshot_to_morpion_supervised_rows(expected_snapshot)

    assert loaded_rows == expected_rows
    assert tuple(row.node_id for row in loaded_rows.rows) == ("root", "leaf")
    assert tuple(row.target_value for row in loaded_rows.rows) == (0.5, 1.0)
    assert tuple(row.metadata["target_source"] for row in loaded_rows.rows) == (
        "backed_up_value",
        "backed_up_value",
    )


def test_sharded_generation_two_reuses_old_nodes_without_state_access(
    tmp_path: Path,
) -> None:
    """Generation two should only serialize new-node state payloads."""
    output_dir = tmp_path / "tree_exports_sharded"
    generation_one_nodes = (
        _LiveNode(
            id="a",
            depth=0,
            state_payload=_compact_payload_after_n_moves(0),
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
            state_payload=_compact_payload_after_n_moves(1),
            direct_value=0.9,
            backed_up_value=1.1,
            is_terminal=True,
            is_exact=True,
            visit_count=7,
            parent_ids=("a",),
            metadata={"tag": "gen1-b"},
        ),
    )
    _generation_one_manifest_path, generation_one_stats = (
        save_morpion_sharded_training_tree_from_live_nodes(
            nodes=generation_one_nodes,
            root_node_id="a",
            output_dir=output_dir,
            generation=1,
            state_ref_dumper=lambda state: state,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
        )
    )

    old_a = _LiveNode(
        id="a",
        depth=0,
        state_payload=_compact_payload_after_n_moves(0),
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
        state_payload=_compact_payload_after_n_moves(2),
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
        state_payload=_compact_payload_after_n_moves(1),
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

    generation_manifest_path, generation_two_stats = (
        save_morpion_sharded_training_tree_from_live_nodes(
            nodes=generation_two_nodes,
            root_node_id="a",
            output_dir=output_dir,
            generation=2,
            state_ref_dumper=lambda state: state,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
        )
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )
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

    assert generation_one_stats.reused_node_count == 0
    assert generation_two_stats.node_count == 3
    assert generation_two_stats.new_node_count == 1
    assert generation_two_stats.reused_node_count == 2
    assert generation_two_stats.rows_written == 4
    assert generation_two_stats.bytes_written > 0
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
    assert generation_two_update_shard["updates"][0]["child_ids"] == ["c", "b"]
    assert loaded_snapshot.nodes == expected_snapshot.nodes
    loaded_rows = training_tree_snapshot_to_morpion_supervised_rows(loaded_snapshot)
    expected_rows = training_tree_snapshot_to_morpion_supervised_rows(expected_snapshot)

    assert loaded_rows == expected_rows
    assert tuple(row.node_id for row in loaded_rows.rows) == ("a", "c", "b")
    assert tuple(row.metadata["target_source"] for row in loaded_rows.rows) == (
        "backed_up_value",
        "backed_up_value",
        "backed_up_value",
    )


def test_new_checkpoint_backed_delta_node_exports_without_state_access(
    tmp_path: Path,
) -> None:
    """New checkpoint-backed nodes should reuse payloads without resolving state."""
    output_dir = tmp_path / "tree_exports_sharded"
    base_payload = _compact_payload_after_n_moves(0)
    expected_payload = _compact_payload_after_n_moves(1)
    delta_move = expected_payload[1][0]
    resolver = CheckpointStateResolver(
        state_codec=cast("object", object()),
        state_payloads_by_node_id={
            0: AnchorCheckpointStatePayload(anchor_ref=base_payload),
            1: DeltaCheckpointStatePayload(
                state_parent_node_id=0,
                state_parent_branch=None,
                delta_ref=delta_move,
            ),
        },
    )
    node = _LiveNode(
        id="checkpoint-backed",
        depth=1,
        state_payload=expected_payload,
        direct_value=0.5,
        backed_up_value=0.75,
        is_terminal=False,
        is_exact=True,
        visit_count=4,
        allow_state_access=False,
        state_handle=CheckpointBackedStateHandle(resolver=resolver, node_id=1),
    )

    generation_manifest_path, stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=(node,),
        root_node_id="checkpoint-backed",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: state,
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )
    node_shard_payload = json.loads(
        (output_dir / "node_shards" / "generation_000001.json").read_text(
            encoding="utf-8"
        )
    )

    assert stats.new_node_count == 1
    assert node.state_access_count == 0
    assert node_shard_payload["nodes"][0]["state_ref_payload"] == [
        expected_payload[0],
        list(expected_payload[1]),
    ]
    assert loaded_snapshot.nodes[0].state_ref_payload == [
        expected_payload[0],
        list(expected_payload[1]),
    ]


def test_new_checkpoint_backed_delta_node_with_live_resolver_exports_without_state_access(
    tmp_path: Path,
) -> None:
    """Live resolver payload maps should support the same no-state fast path."""
    output_dir = tmp_path / "tree_exports_sharded"
    base_payload = _compact_payload_after_n_moves(0)
    expected_payload = _compact_payload_after_n_moves(1)
    delta_move = expected_payload[1][0]
    resolver = _LiveResolverShape(
        state_payloads_by_node_id={
            0: AnchorCheckpointStatePayload(anchor_ref=base_payload),
            1: DeltaCheckpointStatePayload(
                state_parent_node_id=0,
                state_parent_branch=None,
                delta_ref=delta_move,
            ),
        }
    )
    node = _LiveNode(
        id="live-checkpoint-backed",
        depth=1,
        state_payload=expected_payload,
        direct_value=0.5,
        backed_up_value=0.75,
        is_terminal=False,
        is_exact=True,
        visit_count=4,
        allow_state_access=False,
        state_handle=CheckpointBackedStateHandle(
            resolver=cast("object", resolver),
            node_id=1,
        ),
    )

    generation_manifest_path, _stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=(node,),
        root_node_id="live-checkpoint-backed",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: state,
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )

    assert node.state_access_count == 0
    assert loaded_snapshot.nodes[0].state_ref_payload == [
        expected_payload[0],
        list(expected_payload[1]),
    ]


def test_checkpoint_backed_delta_missing_parent_falls_back_to_state_access(
    tmp_path: Path,
) -> None:
    """Missing raw parent payloads should not crash the sharded export."""
    output_dir = tmp_path / "tree_exports_sharded"
    expected_payload = _compact_payload_after_n_moves(1)
    resolver = _LiveResolverShape(
        state_payloads_by_node_id={
            1: DeltaCheckpointStatePayload(
                state_parent_node_id=0,
                state_parent_branch=None,
                delta_ref=expected_payload[1][0],
            ),
        }
    )
    node = _LiveNode(
        id="missing-parent",
        depth=1,
        state_payload=expected_payload,
        direct_value=0.5,
        backed_up_value=0.75,
        is_terminal=False,
        is_exact=True,
        visit_count=4,
        state_handle=CheckpointBackedStateHandle(
            resolver=cast("object", resolver),
            node_id=1,
        ),
    )

    generation_manifest_path, _stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=(node,),
        root_node_id="missing-parent",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: state,
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )

    assert node.state_access_count == 1
    assert loaded_snapshot.nodes[0].state_ref_payload == [
        expected_payload[0],
        list(expected_payload[1]),
    ]


def test_sharded_export_serializes_compact_tuple_payloads(tmp_path: Path) -> None:
    """Tuple-shaped checkpoint payloads should export without dict assumptions."""
    output_dir = tmp_path / "tree_exports_sharded"
    compact_payload = _compact_payload_after_n_moves(2)
    nodes = (
        _LiveNode(
            id="root",
            depth=0,
            state_payload=compact_payload,
            direct_value=0.2,
            backed_up_value=0.4,
            is_terminal=False,
            is_exact=False,
            visit_count=3,
            metadata={"source": "compact"},
        ),
    )

    generation_manifest_path, stats = (
        save_morpion_sharded_training_tree_from_live_nodes(
            nodes=nodes,
            root_node_id="root",
            output_dir=output_dir,
            generation=1,
            state_ref_dumper=lambda state: state,
            direct_value_extractor=_value_to_scalar,
            backed_up_value_extractor=_value_to_scalar,
        )
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )
    node_shard_payload = json.loads(
        (output_dir / "node_shards" / "generation_000001.json").read_text(
            encoding="utf-8"
        )
    )

    assert stats.new_node_count == 1
    assert node_shard_payload["nodes"][0]["state_ref_payload"] == [
        compact_payload[0],
        list(compact_payload[1]),
    ]
    assert loaded_snapshot.nodes[0].state_ref_payload == [
        compact_payload[0],
        list(compact_payload[1]),
    ]


def test_sharded_export_preserves_opaque_mapping_payloads(tmp_path: Path) -> None:
    """Arbitrary mapping payloads should still round-trip as opaque export data."""
    output_dir = tmp_path / "tree_exports_sharded"
    payload = {"domain": "generic", "payload": {"values": [0, 1, 2]}}
    nodes = (
        _LiveNode(
            id="root",
            depth=0,
            state_payload=payload,
            direct_value=0.2,
            backed_up_value=0.4,
            is_terminal=False,
            is_exact=False,
            visit_count=3,
        ),
    )

    generation_manifest_path, _stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=nodes,
        root_node_id="root",
        output_dir=output_dir,
        generation=1,
        state_ref_dumper=lambda state: state,
        direct_value_extractor=_value_to_scalar,
        backed_up_value_extractor=_value_to_scalar,
    )
    loaded_snapshot = load_morpion_sharded_training_tree_snapshot(
        generation_manifest_path
    )

    assert loaded_snapshot.nodes[0].state_ref_payload == payload
