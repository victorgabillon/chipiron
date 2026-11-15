"""Canonical values survive legacy sharding without changing Dataset targets."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from atomheart.games.morpion import MorpionDynamics, initial_state
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
    save_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
    save_morpion_sharded_training_tree_from_live_nodes,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from chipiron.environments.morpion.learning import (
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionStateEvaluator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from anemone.training_export import TrainingTreeSnapshot

    from chipiron.environments.morpion.types import MorpionState


def _decreasing_score(_self: MorpionStateEvaluator, state: MorpionState) -> float:
    """Make a direct-winning partial root through evaluation, not value injection."""
    return 100.0 - state.moves


def _runner(
    monkeypatch: pytest.MonkeyPatch, *, complete: bool
) -> runner_module.AnemoneMorpionSearchRunner:
    """Grow two legal nodes with the production factory, backup and proof policy."""
    monkeypatch.setattr(MorpionStateEvaluator, "score", _decreasing_score)
    if complete:
        dynamics = MorpionDynamics()
        state = initial_state()
        for _ in range(200):
            actions = dynamics.all_legal_actions(state)
            child = dynamics.step(state, actions[0]).next_state
            if not dynamics.all_legal_actions(child):
                assert len(actions) == 1
                monkeypatch.setattr(
                    runner_module, "initial_state", lambda state=state: state
                )
                break
            state = child
        else:
            pytest.fail("deterministic legal endgame did not terminate")
    runner = runner_module.AnemoneMorpionSearchRunner(
        runner_module.AnemoneMorpionSearchRunnerArgs(
            search_args=replace(
                runner_module.default_search_args(),
                stopping_criterion=TreeBranchLimitArgs(
                    type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                    tree_branch_limit=128 if complete else 1,
                ),
            ),
            random_seed=4,
        )
    )
    runner.load_or_create(None, None)
    runner.grow(1)
    assert runner.current_tree_size() == 2
    return runner


def _value_score(value: object | None) -> float | None:
    """Export untransformed typed scores, as the production runner does."""
    if value is None:
        return None
    assert isinstance(value, Value)
    return float(value.score)


def _export(
    runner: runner_module.AnemoneMorpionSearchRunner, path: Path, generation: int = 1
) -> tuple[Path, TrainingTreeSnapshot]:
    """Export live nodes with the same reversible state codec as normal search."""
    codec = MorpionStateCheckpointCodec()
    manifest, _stats = save_morpion_sharded_training_tree_from_live_nodes(
        nodes=tuple(runner.iter_profile_nodes()),
        root_node_id="0",
        output_dir=path,
        generation=generation,
        direct_value_extractor=_value_score,
        backed_up_value_extractor=_value_score,
        state_ref_dumper=lambda state: codec.dump_state_ref(state.to_atomheart_state()),
    )
    return manifest, load_morpion_sharded_training_tree_snapshot(manifest)


@pytest.mark.parametrize("complete", [False, True])
def test_effective_value_and_certainty_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Partial direct estimates and fully opened child proofs retain provenance."""
    runner = _runner(monkeypatch, complete=complete)
    flat, _stats = runner.build_training_tree_snapshot_payload()
    manifest, sharded = _export(runner, tmp_path / "sharded")
    for reused in (False, True):
        with ShardedTrainingGenerationView(
            manifest, cache_dir=tmp_path / "reader-cache"
        ) as view:
            assert view.index_reused is reused
            assert tuple(view.iter_nodes_in_export_order()) == sharded.nodes
    root = next(iter(runner.iter_profile_nodes()))
    assert isinstance(root, AlgorithmNode)
    evaluation = root.tree_evaluation
    assert isinstance(evaluation, NodeMaxEvaluation)
    candidate = evaluation.get_effective_value_candidate()
    assert root.all_branches_generated is complete
    assert candidate.value is not None
    assert candidate.source.value == ("tree_child" if complete else "direct_self")
    assert candidate.value.certainty == (
        Certainty.FORCED if complete else Certainty.ESTIMATE
    )
    assert evaluation.direct_value is not None
    assert evaluation.tree_value is not None
    assert evaluation.direct_value.score > evaluation.tree_value.score
    for expected, actual in zip(flat.nodes, sharded.nodes, strict=True):
        assert actual.effective_value_scalar == expected.effective_value_scalar
        assert actual.effective_value_source == expected.effective_value_source
        assert (actual.is_exact, actual.is_terminal) == (
            expected.is_exact,
            expected.is_terminal,
        )
    assert sharded.nodes[0].is_exact is complete
    assert sharded.nodes[0].is_terminal is False
    assert training_tree_snapshot_to_morpion_supervised_rows(
        sharded
    ) == training_tree_snapshot_to_morpion_supervised_rows(flat)


@pytest.mark.parametrize("complete", [False, True])
def test_scientific_formats_keep_canonical_value_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Existing v2 and closed exports already retain value-associated certainty."""
    runner = _runner(monkeypatch, complete=complete)
    nodes = tuple(runner.iter_profile_nodes())
    producer = {
        "revision": "effective-provenance-regression",
        "evaluator": "deterministic-decreasing-score",
        "config_sha256": "0" * 64,
    }
    v2_path, _stats = save_scientific_training_generation(
        output_dir=tmp_path / "v2",
        nodes=nodes,
        root_node_id="0",
        generation=1,
        producer=producer,
    )
    closed_path, _timings = save_closed_scientific_snapshot(
        destination=tmp_path / "closed",
        nodes=nodes,
        root_node_id="0",
        generation=1,
        producer=producer,
    )
    for path, exported in (
        (v2_path, load_scientific_training_generation(v2_path)),
        (closed_path, load_closed_scientific_snapshot(closed_path)),
    ):
        for node, snapshot in zip(nodes, exported.snapshot.nodes, strict=True):
            assert isinstance(node, AlgorithmNode)
            evaluation = node.tree_evaluation
            assert isinstance(evaluation, NodeMaxEvaluation)
            candidate = evaluation.get_effective_value_candidate()
            assert candidate.value is not None
            assert snapshot.effective_value_scalar == candidate.value.score
            assert snapshot.effective_value_source == candidate.source.value
            contract = exported.value_contracts[snapshot.node_id]
            assert (
                contract["certainties"]["effective"] == candidate.value.certainty.name
            )
            assert (
                contract["values"]["effective"]
                == contract["values"][
                    "tree" if candidate.source.value == "tree_child" else "direct"
                ]
            )
        with ShardedTrainingGenerationView(
            path,
            cache_dir=tmp_path / "reader-cache" / path.parent.name,
            allow_scientific_v2=True,
            allow_closed_snapshot=True,
        ) as view:
            assert tuple(view.iter_nodes_in_export_order()) == exported.snapshot.nodes
            assert view.load_value_contracts(
                tuple(n.node_id for n in exported.snapshot.nodes)
            ) == tuple(
                exported.value_contracts[n.node_id] for n in exported.snapshot.nodes
            )


@pytest.mark.parametrize("missing", ["both", "scalar", "source"])
def test_legacy_missing_effective_fields_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Old rows keep their targets and flags but do not acquire canonical values."""
    runner = _runner(monkeypatch, complete=False)
    manifest, fresh = _export(runner, tmp_path / "legacy")
    update_path = (
        manifest.parent / json.loads(manifest.read_text())["update_shard_path"]
    )
    payload = json.loads(update_path.read_text())
    for update in payload["updates"]:
        if missing in ("both", "scalar"):
            update.pop("effective_value_scalar", None)
        if missing in ("both", "source"):
            update.pop("effective_value_source", None)
    update_path.write_text(json.dumps(payload))
    legacy = load_morpion_sharded_training_tree_snapshot(manifest)
    with ShardedTrainingGenerationView(
        manifest, cache_dir=tmp_path / "legacy-reader-cache"
    ) as view:
        assert tuple(view.iter_nodes_in_export_order()) == legacy.nodes
    for old, new in zip(legacy.nodes, fresh.nodes, strict=True):
        assert old.effective_value_scalar is None
        assert old.effective_value_source == "unavailable_legacy"
        assert (
            replace(
                old,
                effective_value_scalar=new.effective_value_scalar,
                effective_value_source=new.effective_value_source,
            )
            == new
        )
    assert training_tree_snapshot_to_morpion_supervised_rows(
        legacy
    ) == training_tree_snapshot_to_morpion_supervised_rows(fresh)
    # Appending a new generation must not upgrade or rewrite historical updates.
    historical_bytes = update_path.read_bytes()
    _next_manifest, next_snapshot = _export(runner, manifest.parent, generation=2)
    assert next_snapshot.nodes == fresh.nodes
    assert update_path.read_bytes() == historical_bytes
