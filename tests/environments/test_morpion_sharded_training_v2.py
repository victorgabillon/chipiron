"""Scientific contracts, historical compatibility and actual row parity for v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, get_args

import pytest
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
    checkpoint_payload_to_jsonable,
    serialize_value,
)
from anemone.checkpoints.state_handles import (
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
)
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.common.value_snapshot import NodeTargetSource
from anemone.objectives.single_agent_max import SingleAgentMaxObjective
from anemone.training_export import build_training_tree_snapshot
from anemone.training_export.serialization import (
    training_tree_snapshot_from_dict,
    training_tree_snapshot_to_dict,
)
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from atomheart.games.morpion.dynamics import MorpionDynamics
from valanga import Outcome
from valanga.over_event import OverEvent

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    ScientificExportError,
    load_scientific_training_generation,
    save_scientific_training_generation,
    value_score,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)

if TYPE_CHECKING:
    from pathlib import Path

    from valanga.evaluations import Value

CODEC = MorpionStateCheckpointCodec()
PRODUCER = {
    "revision": "synthetic-contract-fixture",
    "evaluator": "controlled-values",
    "config_sha256": "0" * 64,
}
A, B = 9222105397312061435, 9222668330085744635


@dataclass
class Evaluation:
    """Controlled values, using authoritative candidate and objective APIs."""

    direct_value: Value | None
    tree_value: Value | None
    all_branches_generated: bool = False

    def get_effective_value_candidate(self) -> Any:
        """Ask Anemone to compare the actual candidates; do not reimplement it."""
        objective = SingleAgentMaxObjective()
        return canonical_value.get_effective_value_candidate(
            tree_value=self.tree_value,
            direct_value=self.direct_value,
            all_branches_generated=self.all_branches_generated,
            semantic_compare=lambda a, b: objective.semantic_compare(a, b, None),
        )

    def has_exact_value(self) -> bool:
        """Return effective exactness independently of tree certainty."""
        return canonical_value.is_exact_value(
            self.get_effective_value_candidate().value
        )

    def is_terminal(self) -> bool:
        """Return terminality of this candidate only."""
        return canonical_value.is_terminal_value(
            self.get_effective_value_candidate().value
        )


@dataclass
class Node:
    """Tiny real-codec DAG; declared value cases do not claim these boards solved."""

    id: int
    depth: int
    state_handle: CheckpointBackedStateHandle
    tree_evaluation: Evaluation
    parent_ids: tuple[int, ...] = ()
    child_ids: tuple[int, ...] = ()
    visit_count: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    accesses: int = 0

    @property
    def state(self) -> Any:
        """Resolve a real checkpoint; count actual explicit state requests."""
        self.accesses += 1
        return self.state_handle.get()


def nodes_fixture() -> tuple[Node, ...]:
    """Two legal commuting paths join after replay B,A instead of A,B."""
    resolver = CheckpointStateResolver(
        CODEC,
        {
            0: AnchorCheckpointStatePayload((0, ())),
            1: DeltaCheckpointStatePayload(0, None, B),
            2: DeltaCheckpointStatePayload(0, None, A),
            3: DeltaCheckpointStatePayload(1, None, A),
        },
    )
    estimate = canonical_value.make_estimate_value
    return (
        Node(
            0,
            0,
            CheckpointBackedStateHandle(resolver, 0),
            Evaluation(estimate(score=1), estimate(score=6)),
            child_ids=(1, 2),
        ),
        Node(
            1,
            1,
            CheckpointBackedStateHandle(resolver, 1),
            Evaluation(estimate(score=2), estimate(score=4)),
            parent_ids=(0,),
            child_ids=(3,),
        ),
        Node(
            2,
            1,
            CheckpointBackedStateHandle(resolver, 2),
            Evaluation(estimate(score=9), None),
            parent_ids=(0,),
        ),
        Node(
            3,
            2,
            CheckpointBackedStateHandle(resolver, 3),
            Evaluation(canonical_value.make_forced_value(score=4), None),
            parent_ids=(1,),
        ),
    )


def advance(nodes: tuple[Node, ...]) -> None:
    """Add an incoming edge and a tree value on previously exported nodes."""
    nodes[2].child_ids = (3,)
    nodes[3].parent_ids = (1, 2)
    nodes[2].tree_evaluation.tree_value = canonical_value.make_estimate_value(score=3)


def flat(
    nodes: tuple[object, ...], source: NodeTargetSource = NodeTargetSource.TREE_VALUE
) -> Any:
    """Independent authoritative flat oracle, including every scientific field."""
    snapshot = build_training_tree_snapshot(
        nodes,
        root_node_id="0",
        state_ref_dumper=CODEC.dump_state_ref,
        direct_value_extractor=value_score,
        tree_value_extractor=value_score,
        effective_value_extractor=value_score,
        target_value_extractor=value_score,
        target_source=source,
        metadata={"default_target_source": source.value},
    )
    return training_tree_snapshot_from_dict(
        json.loads(json.dumps(training_tree_snapshot_to_dict(snapshot)))
    )


@pytest.mark.parametrize("source", list(NodeTargetSource))
def test_three_generations_exact_full_snapshot_parity(
    tmp_path: Path, source: NodeTargetSource
) -> None:
    """Current parents, depth, absent targets, provenance and canonical refs survive."""
    nodes = nodes_fixture()
    old = None
    for generation in (1, 2, 3):
        if generation == 2:
            advance(nodes)
        if generation == 3:
            # Depth is current, not irreversibly frozen at creation.
            nodes[3].depth = 7
            nodes[3].visit_count = 19
            nodes[3].metadata["generation"] = 3
        expected = flat(nodes, source)
        before = sum(node.accesses for node in nodes)
        path, stats = save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=tmp_path / "v2",
            generation=generation,
            producer=PRODUCER,
            target_source=source,
        )
        actual = load_scientific_training_generation(path)
        assert training_tree_snapshot_to_dict(
            actual.snapshot
        ) == training_tree_snapshot_to_dict(expected)
        assert sum(node.accesses for node in nodes) - before == (
            4 if generation == 1 else 0
        )
        assert stats.new_node_count == (4 if generation == 1 else 0)
        assert actual.provenance["source_generation"] == generation
        if generation == 1:
            old = path.read_bytes()
        else:
            old_path = path.parent.parent / "generation_000001/generation.json"
            assert old_path.read_bytes() == old
            assert load_scientific_training_generation(old_path).snapshot.nodes[
                3
            ].parent_ids == ("1",)
        if generation == 2:
            join = actual.snapshot.nodes[3]
            assert join.parent_ids == ("1", "2")
            partial = actual.snapshot.nodes[2]
            assert (partial.effective_value_scalar, partial.effective_value_source) == (
                9,
                "direct_self",
            )
            assert join.state_ref_payload == [0, [A, B]]


@pytest.mark.parametrize(
    "direct,tree,complete,expected,provenance",
    [
        (None, None, False, None, "none"),
        (5, None, False, 5, "direct_self"),
        (None, 3, False, 3, "tree_child"),
        (9, 3, False, 9, "direct_self"),
        (9, 3, True, 3, "tree_child"),
        (3, 9, False, 9, "tree_child"),
        (3, 3, False, 3, "tree_child"),
    ],
)
@pytest.mark.parametrize("source", list(NodeTargetSource))
def test_value_selection_contract(
    tmp_path: Path,
    direct: float | None,
    tree: float | None,
    complete: bool,
    expected: float | None,
    provenance: str,
    source: NodeTargetSource,
) -> None:
    """Independent explicit expected values, including null tree targets."""
    node = nodes_fixture()[0]
    node.child_ids = ()
    estimate = canonical_value.make_estimate_value
    node.tree_evaluation = Evaluation(
        None if direct is None else estimate(score=direct),
        None if tree is None else estimate(score=tree),
        complete,
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
        target_source=source,
    )
    loaded = load_scientific_training_generation(path)
    row = loaded.snapshot.nodes[0]
    assert (row.effective_value_scalar, row.effective_value_source) == (
        expected,
        provenance,
    )
    assert (
        row.target_value_scalar
        == {
            NodeTargetSource.TREE_VALUE: tree,
            NodeTargetSource.DIRECT_VALUE: direct,
            NodeTargetSource.EFFECTIVE_VALUE: expected,
        }[source]
    )
    assert training_tree_snapshot_to_dict(
        loaded.snapshot
    ) == training_tree_snapshot_to_dict(flat((node,), source))


@pytest.mark.parametrize("complete", [False, True])
def test_certainty_score_association_and_ties(tmp_path: Path, complete: bool) -> None:
    """An exact direct candidate does not certify an estimated tree scalar."""
    node = nodes_fixture()[0]
    node.child_ids = ()
    node.tree_evaluation = Evaluation(
        canonical_value.make_forced_value(score=9),
        canonical_value.make_estimate_value(score=3),
        complete,
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    loaded = load_scientific_training_generation(path)
    cert = loaded.value_contracts["0"]
    assert cert["certainties"] == {
        "direct": "FORCED",
        "tree": "ESTIMATE",
        "effective": "ESTIMATE" if complete else "FORCED",
        "target": "ESTIMATE",
    }
    assert cert["backup_exactness_compatible"] is complete
    assert loaded.snapshot.nodes[0].is_exact is (not complete)
    rows = extract_rows_from_training_snapshot(
        args=MorpionBootstrapArgs(
            work_dir=tmp_path, dataset_family_target_policy="pv_mean_prediction"
        ),
        snapshot=loaded.snapshot,
        generation=1,
    )
    assert rows.rows[0].target_value == (9 if complete else 3)
    assert rows.rows[0].is_exact is (not complete)
    node.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=9)
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=2,
        producer=PRODUCER,
    )
    row = load_scientific_training_generation(path).snapshot.nodes[0]
    assert row.effective_value_source == ("tree_child" if complete else "direct_self")


@pytest.mark.parametrize("direct_exact", [False, True])
@pytest.mark.parametrize("tree_exact", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_equal_score_certainty_ordering(
    tmp_path: Path, direct_exact: bool, tree_exact: bool, complete: bool
) -> None:
    """Keep source and certainty on every direction of an equal-score tie."""
    node = nodes_fixture()[0]
    node.child_ids = ()
    direct_factory = (
        canonical_value.make_forced_value
        if direct_exact
        else canonical_value.make_estimate_value
    )
    tree_factory = (
        canonical_value.make_forced_value
        if tree_exact
        else canonical_value.make_estimate_value
    )
    node.tree_evaluation = Evaluation(
        direct_factory(score=9), tree_factory(score=9), complete
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    actual = load_scientific_training_generation(path)
    direct_selected = direct_exact and not tree_exact and not complete
    assert actual.snapshot.nodes[0].effective_value_source == (
        "direct_self" if direct_selected else "tree_child"
    )
    assert actual.value_contracts["0"]["certainties"]["effective"] == (
        "FORCED" if direct_selected or tree_exact else "ESTIMATE"
    )


def test_terminal_is_separate_from_forced(tmp_path: Path) -> None:
    """Declared Value semantics only: the initial board is not claimed terminal."""
    node = nodes_fixture()[0]
    node.child_ids = ()
    node.tree_evaluation = Evaluation(
        canonical_value.make_terminal_value(
            score=0, over_event=OverEvent(outcome=Outcome.DRAW)
        ),
        None,
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    result = load_scientific_training_generation(path)
    assert result.snapshot.nodes[0].is_terminal and result.snapshot.nodes[0].is_exact
    assert result.value_contracts["0"]["certainties"]["direct"] == "TERMINAL"
    parent = canonical_value.make_backed_up_value(
        score=0,
        exact=True,
        node_is_terminal=False,
        over_event=OverEvent(outcome=Outcome.DRAW),
    )
    assert parent.certainty.name == "FORCED"


def test_complete_value_payload_keeps_event_and_line(tmp_path: Path) -> None:
    """Preserve the actual certified Value payload, including tagged line atoms."""
    node = nodes_fixture()[0]
    node.child_ids = ()
    value = replace(
        canonical_value.make_terminal_value(
            score=-0.0, over_event=OverEvent(outcome=Outcome.DRAW)
        ),
        line=[("branch", 7)],
    )
    node.tree_evaluation = Evaluation(value, None)
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
        target_source=NodeTargetSource.EFFECTIVE_VALUE,
    )
    loaded = load_scientific_training_generation(path)
    payload = checkpoint_payload_to_jsonable(serialize_value(value))
    assert loaded.value_contracts["0"]["values"] == {
        "direct": payload,
        "tree": None,
        "effective": payload,
        "target": payload,
    }
    assert loaded.snapshot.nodes[0].target_value_scalar.hex() == "-0x0.0p+0"
    assert loaded.provenance["producer"]["api_sha256"]


def test_real_terminal_board_direct_only_target(tmp_path: Path) -> None:
    """Reach an actual terminal board legally; exact direct does not fill T."""
    dynamics = MorpionDynamics()
    state = CODEC.load_state_ref((0, ()))
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        if not actions:
            break
        state = dynamics.step(state, actions[0]).next_state
    assert not dynamics.all_legal_actions(state)
    terminal = canonical_value.make_terminal_value(
        score=float(state.moves), over_event=OverEvent(outcome=Outcome.DRAW)
    )
    resolver = CheckpointStateResolver(
        CODEC, {0: AnchorCheckpointStatePayload(CODEC.dump_state_ref(state))}
    )
    node = Node(
        0,
        state.moves,
        CheckpointBackedStateHandle(resolver, 0),
        Evaluation(terminal, None, True),
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    actual = load_scientific_training_generation(path)
    assert actual.snapshot == flat((node,))
    row = actual.snapshot.nodes[0]
    assert row.is_terminal and row.is_exact
    assert row.direct_value_scalar == state.moves and row.target_value_scalar is None


@pytest.mark.parametrize("variant_code", [0, 1])
def test_canonical_reference_both_morpion_variants(
    tmp_path: Path, variant_code: int
) -> None:
    """Use the authoritative replay codec for real legal 5T and 5D histories."""
    dynamics = MorpionDynamics()
    state = CODEC.load_state_ref((variant_code, ()))
    for _ in range(3):
        state = dynamics.step(state, dynamics.all_legal_actions(state)[0]).next_state
    resolver = CheckpointStateResolver(
        CODEC, {0: AnchorCheckpointStatePayload(CODEC.dump_state_ref(state))}
    )
    node = Node(
        0,
        3,
        CheckpointBackedStateHandle(resolver, 0),
        Evaluation(canonical_value.make_estimate_value(score=3), None),
    )
    path, _ = save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    result = load_scientific_training_generation(path)
    assert result.snapshot == flat((node,))
    assert CODEC.load_state_ref(result.snapshot.nodes[0].state_ref_payload) == state


def test_actual_rows_nine_policies_exact_jsonl(tmp_path: Path) -> None:
    """648 cases compare every row, summary and complete serialized output."""
    nodes = nodes_fixture()
    cases = 0
    for generation in (1, 2):
        if generation == 2:
            advance(nodes)
        path, _ = save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=tmp_path / "v2",
            generation=generation,
            producer=PRODUCER,
        )
        left, right = flat(nodes), load_scientific_training_generation(path).snapshot
        for policy in get_args(PvFamilyTargetPolicy):
            for backups in (True, False):
                for blend in (0.0, 0.25, 1.0):
                    for filters in (
                        {},
                        {"require_exact_or_terminal": True},
                        {"max_rows": 0},
                        {"max_rows": 2},
                        {"min_depth": 1},
                        {"min_visit_count": 4},
                    ):
                        args = MorpionBootstrapArgs(
                            work_dir=tmp_path,
                            dataset_family_target_policy=policy,
                            use_backed_up_value=backups,
                            dataset_family_prediction_blend=blend,
                            **filters,
                        )
                        a = extract_rows_from_training_snapshot(
                            args=args, snapshot=left, generation=generation
                        )
                        b = extract_rows_from_training_snapshot(
                            args=args, snapshot=right, generation=generation
                        )
                        assert a == b
                        for name, rows in (("flat", a), ("v2", b)):
                            save_morpion_supervised_rows_streaming(
                                rows=rows.rows,
                                metadata=rows.metadata,
                                path=tmp_path / f"{name}.jsonl",
                            )
                        assert (tmp_path / "flat.jsonl").read_bytes() == (
                            tmp_path / "v2.jsonl"
                        ).read_bytes()
                        cases += 1
    assert cases == 648


def test_v1_interpretation_frozen_and_old_reader_refuses_v2(tmp_path: Path) -> None:
    """The old scientific failures remain exactly observable in explicit v1."""
    nodes = nodes_fixture()
    kwargs = {
        "nodes": nodes,
        "root_node_id": "0",
        "output_dir": tmp_path / "v1",
        "state_ref_dumper": CODEC.dump_state_ref,
        "direct_value_extractor": value_score,
        "backed_up_value_extractor": value_score,
    }
    save_morpion_sharded_training_tree_from_live_nodes(generation=1, **kwargs)
    advance(nodes)
    path, _ = save_morpion_sharded_training_tree_from_live_nodes(generation=2, **kwargs)
    legacy = load_morpion_sharded_training_tree_snapshot(path)
    assert legacy.nodes[3].parent_ids == ("1",)
    assert legacy.nodes[3].state_ref_payload == [0, [B, A]]
    assert (
        legacy.nodes[2].effective_value_scalar,
        legacy.nodes[2].effective_value_source,
    ) == (3, "tree_value")
    assert legacy.nodes[3].target_value_scalar == 4  # v1 direct fallback, frozen
    before = {p: p.read_bytes() for p in (tmp_path / "v1").rglob("*.json")}
    with pytest.raises(ScientificExportError):
        save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=tmp_path / "v1",
            generation=1,
            producer=PRODUCER,
        )
    assert all(p.read_bytes() == content for p, content in before.items())
    v2, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    with pytest.raises((KeyError, TypeError, ValueError)):
        load_morpion_sharded_training_tree_snapshot(v2)
    with pytest.raises(ScientificExportError):
        load_scientific_training_generation(path)


@pytest.mark.parametrize(
    "mutation", ["absent", "null", "source", "certainty", "parents", "version", "order"]
)
def test_malformed_and_mixed_sources_fail_closed(tmp_path: Path, mutation: str) -> None:
    """Even rehashed malformed updates cannot become plausible scientific rows."""
    nodes = nodes_fixture()
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    shard = path.parent / "updates.jsonl"
    lines = [json.loads(line) for line in shard.read_text().splitlines()]
    if mutation == "absent":
        del lines[1]["target_value_scalar"]
    elif mutation == "null":
        lines[1]["tree_value_scalar"] = None
    elif mutation == "source":
        lines[1]["effective_value_source"] = "tree_value"
    elif mutation == "certainty":
        lines[1]["value_contract"]["certainties"]["target"] = "FORCED"
    elif mutation == "parents":
        lines[4]["parent_ids"] = ["2"]
    elif mutation == "version":
        lines[0]["format_version"] = 99
    else:
        lines[2]["order_index"] = 0
    shard.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    manifest = json.loads(path.read_text())
    manifest["update_shard"]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ScientificExportError):
        load_scientific_training_generation(path)


def test_interruption_and_immutable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted rows do not publish a generation or damage the last one."""
    nodes = nodes_fixture()
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    catalog = path.parent.parent / "manifest.json"
    saved = catalog.read_bytes()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "chipiron.environments.morpion.bootstrap.sharded_training_v2._validate_update",
            lambda _row: (_ for _ in ()).throw(OSError("interrupted")),
        )
        with pytest.raises(OSError):
            save_scientific_training_generation(
                nodes=nodes,
                root_node_id="0",
                output_dir=tmp_path / "v2",
                generation=2,
                producer=PRODUCER,
            )
    assert catalog.read_bytes() == saved
    assert not list(catalog.parent.glob(".v2-pending-*"))
    assert load_scientific_training_generation(path).snapshot == flat(nodes)
    with pytest.raises(ScientificExportError):
        save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=tmp_path / "v2",
            generation=1,
            producer=PRODUCER,
        )


def test_real_search_and_saved_checkpoint_parity(tmp_path: Path) -> None:
    """Small actual search, saved/reloaded runtime, current graph and next steps."""
    from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs

    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        AnemoneMorpionSearchRunner,
        AnemoneMorpionSearchRunnerArgs,
        default_search_args,
    )

    args = AnemoneMorpionSearchRunnerArgs(
        search_args=replace(
            default_search_args(),
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit", tree_branch_limit=512
            ),
        ),
        random_seed=4,
    )
    runner = AnemoneMorpionSearchRunner(args)
    runner.load_or_create(None, None)
    runner.grow(2)
    checkpoint = tmp_path / "checkpoint.json"
    runner.save_checkpoint(checkpoint)
    restored = AnemoneMorpionSearchRunner(args)
    restored.load_or_create(checkpoint, None)
    for generation in (1, 2):
        assert (
            runner._random_generator.getstate() == restored._random_generator.getstate()
        )
        for name, instance in (("original", runner), ("restored", restored)):
            nodes = tuple(instance.iter_profile_nodes())
            expected, _ = instance.build_training_tree_snapshot_payload()
            path, _ = save_scientific_training_generation(
                nodes=nodes,
                root_node_id=expected.root_node_id,
                output_dir=tmp_path / name,
                generation=generation,
                producer=PRODUCER,
            )
            actual = load_scientific_training_generation(path)
            expected = training_tree_snapshot_from_dict(
                json.loads(json.dumps(training_tree_snapshot_to_dict(expected)))
            )
            assert training_tree_snapshot_to_dict(
                actual.snapshot
            ) == training_tree_snapshot_to_dict(expected)
            # This verifies current runtime scores, without assuming the
            # controlled direct-exact/tree-estimate case is reachable here.
            assert all(
                c["backup_exactness_compatible"]
                for c in actual.value_contracts.values()
            )
            for policy in get_args(PvFamilyTargetPolicy):
                dataset_args = MorpionBootstrapArgs(
                    work_dir=tmp_path, dataset_family_target_policy=policy
                )
                assert extract_rows_from_training_snapshot(
                    args=dataset_args, snapshot=expected, generation=generation
                ) == extract_rows_from_training_snapshot(
                    args=dataset_args, snapshot=actual.snapshot, generation=generation
                )
        runner.grow(1)
        restored.grow(1)
        assert runner.current_tree_size() == restored.current_tree_size()


def test_deterministic_exports_and_parallel_edges(tmp_path: Path) -> None:
    """Repeat every byte across fresh runs, keeping parallel branch link order."""
    for name in ("a", "b"):
        nodes = nodes_fixture()
        nodes[0].child_ids = (1, 2, 1)
        for generation in (1, 2):
            if generation == 2:
                advance(nodes)
            path, _ = save_scientific_training_generation(
                nodes=nodes,
                root_node_id="0",
                output_dir=tmp_path / name,
                generation=generation,
                producer=PRODUCER,
            )
            assert load_scientific_training_generation(path).snapshot.nodes[
                0
            ].child_ids == ("1", "2", "1")
    left = {
        p.relative_to(tmp_path / "a"): p.read_bytes()
        for p in (tmp_path / "a").rglob("*")
        if p.is_file()
    }
    right = {
        p.relative_to(tmp_path / "b"): p.read_bytes()
        for p in (tmp_path / "b").rglob("*")
        if p.is_file()
    }
    assert left == right


@pytest.mark.parametrize("role", ["catalog", "generation"])
@pytest.mark.parametrize("version", [1, 3, "2", True])
def test_explicit_unsupported_versions_rejected(
    tmp_path: Path, role: str, version: object
) -> None:
    """A v2 envelope is never guessed from old/future/coerced version labels."""
    path, _ = save_scientific_training_generation(
        nodes=nodes_fixture(),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    changed = path if role == "generation" else path.parent.parent / "manifest.json"
    data = json.loads(changed.read_text())
    data["format_version"] = version
    changed.write_text(json.dumps(data))
    with pytest.raises(ScientificExportError):
        load_scientific_training_generation(path)


def test_source_hash_and_noncanonical_refs_rejected(tmp_path: Path) -> None:
    """Byte-equivalent claims require the declared canonical reference encoding."""
    path, _ = save_scientific_training_generation(
        nodes=nodes_fixture(),
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    shard = path.parent / "records.jsonl"
    original = shard.read_bytes()
    shard.write_bytes(original + b" ")
    with pytest.raises(ScientificExportError, match="hash"):
        load_scientific_training_generation(path)
    rows = [json.loads(line) for line in original.splitlines()]
    rows[4]["state_ref_payload"] = [0, [B, A]]
    shard.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = json.loads(path.read_text())
    manifest["record_shards"][0]["sha256"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ScientificExportError, match="noncanonical"):
        load_scientific_training_generation(path)


def test_published_orphan_requires_explicit_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After directory rename, failed catalog publication remains invisible."""
    nodes = nodes_fixture()
    first, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    with monkeypatch.context() as scoped:

        def interrupted(*_args: object) -> None:
            message = "catalog publication interrupted"
            raise OSError(message)

        scoped.setattr(
            "chipiron.environments.morpion.bootstrap.sharded_training_v2.os.replace",
            interrupted,
        )
        with pytest.raises(OSError):
            save_scientific_training_generation(
                nodes=nodes,
                root_node_id="0",
                output_dir=tmp_path / "v2",
                generation=2,
                producer=PRODUCER,
            )
    assert load_scientific_training_generation(first).snapshot == flat(nodes)
    with pytest.raises(ScientificExportError, match="unpublished"):
        load_scientific_training_generation(
            first.parent.parent / "generation_000002/generation.json"
        )
    with pytest.raises(ScientificExportError, match="recovery"):
        save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=tmp_path / "v2",
            generation=2,
            producer=PRODUCER,
        )
