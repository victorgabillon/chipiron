"""Exercise the pinned Anemone API, never the previous Chipiron refresh kernel."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from anemone.node_evaluation.common import canonical_value
from atomheart.games.morpion import MorpionDynamics as AtomDynamics
from valanga.evaluations import Certainty

from docs.research.model_activation import test_science as dataset_oracle
from docs.research.model_activation.probe import (
    differences,
    exact,
    graph,
    nodes,
    reference,
    state,
)
from docs.research.model_activation.test_contract import OneStateChange, partial_graph
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    model_patch,
    neural_evaluator,
)

if TYPE_CHECKING:
    from pathlib import Path


def refresh(instance: Any, evaluator: Any, batch_size: int = 64) -> Any:
    """Delegate the entire operation to the dependency; no backup algorithm here."""
    return instance._runtime.refresh_with_evaluator(
        evaluator, scope="all", batch_size=batch_size
    )


def assert_oracle(instance: Any, expected: dict) -> None:
    """Compare the existing independent oracle and independently derived ordering."""
    assert differences(instance, expected) == []
    for node in nodes(instance):
        ordered = sorted(
            node.iter_child_links(),
            key=lambda bc: (
                -expected[bc[1].id]["effective"].score,
                -int(exact(expected[bc[1].id]["effective"])),
                len(expected[bc[1].id]["pv"]),
                bc[1].id,
                str(bc[0]),
            ),
        )
        assert tuple(node.tree_evaluation.decision_ordered_branches()) == tuple(
            branch for branch, _ in ordered
        )


def test_real_seven_node_witness() -> None:
    """The actual failed root now equals the independent reference exactly."""
    instance = partial_graph()
    evaluator = OneStateChange(
        instance._runtime.tree_manager.node_evaluator.master_state_value_evaluator, 2
    )
    expected = reference(instance, evaluator)
    refresh(instance, evaluator, 2)
    assert_oracle(instance, expected)
    assert instance._runtime.tree.root_node.tree_evaluation.tree_value.score == (
        100.00000100000001
    )


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_real_dag_repeated_refresh(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    """Verify every window, Value, certainty, PV/order, edge and version."""
    instance = graph()
    assert sum(len(n.parent_nodes) > 1 for n in nodes(instance)) >= 7
    topology = [
        (n.id, tuple(p.id for p in n.parent_nodes), tuple(n.iter_child_links()))
        for n in nodes(instance)
    ]
    rng = instance._random_generator.getstate()
    selection_count = instance._runtime.node_selector.base._selection_step_count
    evaluator_type = type(instance._runtime.tree_manager.node_evaluator)
    original = evaluator_type.evaluate_nodes
    windows = []

    def observe(self: Any, batch: Any, **kwargs: Any) -> Any:
        windows.append(len(batch))
        return original(self, batch, **kwargs)

    monkeypatch.setattr(evaluator_type, "evaluate_nodes", observe)
    for prediction in (42, 7, 42):
        evaluator = neural_evaluator(prediction)
        expected = reference(instance, evaluator)
        old_version = instance._runtime.evaluator_version
        report = refresh(instance, evaluator, batch_size)
        assert max(windows) <= batch_size
        assert report.requested_count == instance.current_tree_size()
        assert_oracle(instance, expected)
        assert instance._runtime.evaluator_version == old_version + 1
        assert all(
            n.tree_evaluation.direct_evaluation_version == old_version + 1
            for n in nodes(instance)
            if n.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        )
        assert instance._random_generator.getstate() == rng
        assert (
            instance._runtime.node_selector.base._selection_step_count
            == selection_count
        )
        assert topology == [
            (n.id, tuple(p.id for p in n.parent_nodes), tuple(n.iter_child_links()))
            for n in nodes(instance)
        ]


def test_real_endgame_certificate_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhaust the actual one-action game, then bind that proof to this state/Value."""
    source, _ = endgame.__wrapped__()
    dynamics = AtomDynamics()
    actions = dynamics.all_legal_actions(source)
    assert len(actions) == 1
    terminal = dynamics.step(source, actions[0]).next_state
    assert not dynamics.all_legal_actions(terminal)
    proven_score = float(len(terminal.played_moves))
    assert proven_score == 35
    instance = make_runner(monkeypatch, state=source)
    root = next(nodes(instance))
    certificate = replace(root.tree_evaluation.tree_value, score=proven_score)
    assert certificate.certainty == Certainty.FORCED
    root.tree_evaluation.direct_value = certificate
    version = root.tree_evaluation.direct_evaluation_version
    objective = root.tree_evaluation.required_objective

    def validate(node: Any) -> bool:
        return (
            node is root
            and node.state.to_atomheart_state() == source
            and node.tree_evaluation.required_objective == objective
            and node.tree_evaluation.direct_value is certificate
            and node.tree_evaluation.direct_value.score == proven_score
        )

    expected = reference(instance, neural_evaluator(42))
    report = instance._runtime.refresh_with_evaluator(
        neural_evaluator(42),
        scope="all",
        batch_size=1,
        validate_direct_certificate=validate,
    )
    assert report.skipped_certified_count == 1
    assert report.skipped_terminal_count == 1
    assert root.tree_evaluation.direct_value is certificate
    assert root.tree_evaluation.direct_evaluation_version == version
    assert_oracle(instance, expected)


@pytest.mark.parametrize("kind", ["direct", "partial_tree", "hidden_direct"])
def test_unknown_or_inconsistent_proof_refused_before_mutation(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A certainty enum cannot supply the missing scientific proof authority."""
    instance = make_runner(monkeypatch)
    root = next(nodes(instance))
    if kind == "partial_tree":
        root.tree_evaluation.tree_value = canonical_value.make_forced_value(score=35)
    else:
        root.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
        if kind == "hidden_direct":
            root.tree_node.all_branches_generated = True
    before = state(instance)
    version = instance._runtime.evaluator_version
    with pytest.raises(ValueError, match="Cannot refresh node"):
        refresh(instance, neural_evaluator(42))
    assert state(instance) == before
    assert instance._runtime.evaluator_version == version


@pytest.mark.parametrize("complete", [False, True])
def test_exact_estimate_equal_score_order(
    monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Use a genuine terminal child for an exact/estimated sibling tie."""
    instance = make_runner(monkeypatch, state=endgame.__wrapped__()[0])
    root, terminal = tuple(nodes(instance))
    root.tree_node.all_branches_generated = complete
    root.tree_evaluation.tree_value = None
    evaluator = neural_evaluator(35)
    expected = reference(instance, evaluator)
    refresh(instance, evaluator, 1)
    assert_oracle(instance, expected)
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL


def test_growth_and_patch_numerical_policies_differ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Record the actual built-in patch's raw score, without normalizing it away."""
    instance = graph()
    refresh(instance, neural_evaluator(42))
    patch = model_patch(instance, monkeypatch, 42, tmp_path)
    rows = {row.node_id: row for row in patch.rows}
    deeper = next(
        n for n in nodes(instance) if n.tree_depth > 0 and not n.state.is_game_over()
    )
    assert rows[str(deeper.id)].direct_value == 42
    assert deeper.tree_evaluation.direct_value.score == (
        (1 / 0.99999999) ** deeper.tree_depth * 42
    )
    assert (
        deeper.tree_evaluation.direct_value.score != rows[str(deeper.id)].direct_value
    )


@pytest.mark.parametrize("confidence,policy,backup,exact", dataset_oracle.CASES)
def test_dataset_policies_after_anemone_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    confidence: str,
    policy: str,
    backup: bool,
    exact: bool,
) -> None:
    """Reuse every complete-file oracle/gate assertion with only the refresh replaced."""
    monkeypatch.setattr(dataset_oracle, "bounded_probe", refresh)
    dataset_oracle.test_refreshed_endgame_full_and_bounded_rows(
        tmp_path, monkeypatch, confidence, policy, backup, exact
    )
