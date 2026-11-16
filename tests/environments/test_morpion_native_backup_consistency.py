"""The published Anemone backup stays consistent through Chipiron operations."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from anemone.updates.value_propagator import ValuePropagator
from valanga.evaluations import Certainty

from tests.environments.test_morpion_committed_evaluator_restore import committed_neural
from tests.environments.test_morpion_effective_provenance import _runner
from tests.environments.test_morpion_value_certainty import (
    model_patch,
    neural_evaluator,
)

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(nodes: tuple[Any, ...]) -> dict[int, Any]:
    """Include cached ordering keys as well as visible values and decisions."""
    return {
        node.id: (
            node.tree_evaluation.direct_value,
            node.tree_evaluation.tree_value,
            node.tree_evaluation.effective_value,
            node.tree_evaluation.effective_value_source,
            tuple(node.tree_evaluation.best_branch_sequence),
            tuple(node.tree_evaluation.decision_ordered_branches()),
            deepcopy(node.tree_evaluation.decision_ordering.branch_ordering_keys),
            node.tree_evaluation.is_terminal(),
            node.tree_evaluation.has_exact_value(),
        )
        for node in nodes
    }


def _assert_native_consistent(runner: Any) -> None:
    """Recompute only a disposable graph copy using the existing native backup."""
    live = tuple(runner.iter_profile_nodes())
    before = _snapshot(live)
    copied = deepcopy(live)
    propagator = ValuePropagator()
    for _ in range(2):
        for node in sorted(copied, key=lambda node: node.tree_depth, reverse=True):
            propagator._recompute_node_value_from_full_child_snapshot(node)
        assert _snapshot(copied) == before
    assert _snapshot(live) == before


@pytest.mark.parametrize("layout", ["json-zst", "sharded"])
@pytest.mark.parametrize("rollout", [False, True])
def test_native_backups_after_refresh_and_growth(
    tmp_path: Path, layout: str, rollout: bool
) -> None:
    """Ordinary and rollout Growth settle before and after an explicit full refresh."""
    _, _, runner, _ = committed_neural(tmp_path, layout, rollout)
    _assert_native_consistent(runner)
    runner.validate_model_refresh_certificates()
    runner.refresh_model_estimates(tmp_path / "model")
    _assert_native_consistent(runner)
    size = runner.current_tree_size()
    runner.grow(2)
    assert runner.current_tree_size() > size
    _assert_native_consistent(runner)


@pytest.mark.parametrize("complete", [False, True])
def test_native_backups_preserve_masking_and_certificates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Partial direct masks and complete terminal proofs survive refresh and patches."""
    runner = _runner(monkeypatch, complete=complete)
    _assert_native_consistent(runner)
    nodes = tuple(runner.iter_profile_nodes())
    root = nodes[0]
    assert root.all_branches_generated is complete
    assert (
        root.tree_evaluation.direct_value.score > root.tree_evaluation.tree_value.score
    )
    proofs = {
        node.id: (node.tree_evaluation.direct_value, node.tree_evaluation.tree_value)
        for node in nodes
        if node.tree_evaluation.has_exact_value()
    }
    runner._runtime.refresh_with_evaluator(neural_evaluator(42.0), scope="all")
    _assert_native_consistent(runner)
    runner.apply_reevaluation_patch(model_patch(runner, monkeypatch, 43.0, tmp_path))
    _assert_native_consistent(runner)
    if complete:
        assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
        assert root.tree_evaluation.tree_value == proofs[root.id][1]
        terminal = nodes[1]
        assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
        assert terminal.tree_evaluation.direct_value == proofs[terminal.id][0]
