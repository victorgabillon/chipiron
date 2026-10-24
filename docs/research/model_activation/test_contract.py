"""Executable scientific evidence; stock defects are asserted, not hidden by xfail."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from anemone.node_evaluation.common import canonical_value
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapEffectiveRuntimeConfig,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    AnemoneMorpionSearchRunner,
)
from docs.research.model_activation.probe import (
    bounded_probe,
    differences,
    graph,
    nodes,
    reference,
    state,
)
from docs.research.rolling_snapshot.measure import runner
from tests.environments.test_morpion_checkpoint_parity import _selection_signature
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    neural_evaluator,
)

if TYPE_CHECKING:
    from pathlib import Path


class OneStateChange:
    """Controlled model changes ONLY a real partially expanded state's estimate."""

    def __init__(self, previous: Any, selected: int) -> None:
        """Bind the old evaluator and one controlled state-specific change."""
        self.previous, self.selected, self.over = previous, selected, previous.over

    def evaluate_batch_items(self, items: Any) -> list:
        """Return the controlled model output independently of tree backup."""
        return [
            canonical_value.make_estimate_value(score=100)
            if n.id == self.selected
            else self.previous.evaluate_batch_items([n])[0]
            for n in items
        ]


def partial_graph() -> Any:
    """Check partial graph."""
    args = runner()._args
    args = replace(
        args,
        search_args=replace(
            args.search_args,
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit", tree_branch_limit=6
            ),
        ),
    )
    r = AnemoneMorpionSearchRunner(args)
    r.load_or_create(None, None)
    r.grow(2)
    return r


def test_stock_full_refresh_loses_direct_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check stock full refresh loses direct certificate."""
    r = make_runner(monkeypatch)
    root = next(nodes(r))
    root.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
    r._runtime.refresh_with_evaluator(neural_evaluator(42), scope="all")
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.direct_value.score == 42


def test_stock_full_refresh_loses_partial_tree_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check stock full refresh loses partial tree certificate."""
    r = make_runner(monkeypatch)
    root = next(nodes(r))
    root.tree_evaluation.tree_value = canonical_value.make_forced_value(score=35)
    r._runtime.refresh_with_evaluator(neural_evaluator(42), scope="all")
    assert root.tree_evaluation.tree_value.certainty == Certainty.ESTIMATE


def test_real_partial_change_is_lost_by_stock_propagator() -> None:
    """Check real partial change is lost by stock propagator."""
    r = partial_graph()
    assert r.current_tree_size() == 7
    changed = next(n for n in nodes(r) if n.id == 2)
    assert not changed.tree_node.all_branches_generated and changed.branches_children
    evaluator = OneStateChange(
        r._runtime.tree_manager.node_evaluator.master_state_value_evaluator, 2
    )
    expected = reference(r, evaluator)
    r._runtime.refresh_with_evaluator(evaluator, scope="all")
    assert differences(r, expected) == [dict(node_id=0, fields=["tree", "effective"])]
    assert r._runtime.tree.root_node.tree_evaluation.tree_value.score < 3
    assert expected[0]["tree"].score > 100


def test_bounded_sweep_repairs_real_partial_change() -> None:
    """Check bounded sweep repairs real partial change."""
    r = partial_graph()
    evaluator = OneStateChange(
        r._runtime.tree_manager.node_evaluator.master_state_value_evaluator, 2
    )
    expected = reference(r, evaluator)
    bounded_probe(r, evaluator, 2)
    assert differences(r, expected) == []


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_real_dag_repeated_models_exact_reference(batch_size: int) -> None:
    """Check real dag repeated models exact reference."""
    r = graph()
    assert sum(len(n.parent_nodes) > 1 for n in nodes(r)) >= 7
    topology = [
        (
            n.id,
            tuple(p.id for p in n.parent_nodes),
            tuple((str(b), c.id) for b, c in n.iter_child_links()),
        )
        for n in nodes(r)
    ]
    rng = r._random_generator.getstate()
    selector_count = r._runtime.node_selector.base._selection_step_count
    for prediction in (42, 7, 42):
        evaluator = neural_evaluator(prediction)
        expected = reference(r, evaluator)
        previous_version = r._runtime.evaluator_version
        observed = bounded_probe(r, evaluator, batch_size)
        assert observed["max_window"] <= batch_size
        assert r._runtime.evaluator_version == previous_version + 1
        assert all(
            n.tree_evaluation.direct_evaluation_version == r._runtime.evaluator_version
            for n in nodes(r)
            if n.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        )
        assert differences(r, expected) == []
        assert r._random_generator.getstate() == rng
        assert r._runtime.node_selector.base._selection_step_count == selector_count
        assert topology == [
            (
                n.id,
                tuple(p.id for p in n.parent_nodes),
                tuple((str(b), c.id) for b, c in n.iter_child_links()),
            )
            for n in nodes(r)
        ]


def test_real_endgame_forced_35_estimated_42(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check real endgame forced 35 estimated 42."""
    r = make_runner(monkeypatch, state=endgame.__wrapped__()[0])
    expected = reference(r, neural_evaluator(42))
    bounded_probe(r, neural_evaluator(42), 1)
    assert differences(r, expected) == []
    root, terminal = tuple(nodes(r))
    assert root.tree_evaluation.direct_value.score == 42
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.tree_value.score == 35
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL


def test_probe_preserves_direct_proof_and_rejects_partial_tree_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check probe preserves direct proof and rejects partial tree proof."""
    r = make_runner(monkeypatch)
    root = next(nodes(r))
    root.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
    certificate_version = root.tree_evaluation.direct_evaluation_version
    expected = reference(r, neural_evaluator(42))
    bounded_probe(r, neural_evaluator(42))
    assert differences(r, expected) == []
    assert root.tree_evaluation.direct_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.direct_evaluation_version == certificate_version
    assert certificate_version < r._runtime.evaluator_version
    root.tree_evaluation.tree_value = canonical_value.make_forced_value(score=35)
    before = state(r)
    with pytest.raises(ValueError, match="partial tree certificate"):
        bounded_probe(r, neural_evaluator(7))
    assert state(r) == before


def test_lazy_attach_has_stale_values_and_no_read_fence() -> None:
    """Check lazy attach has stale values and no read fence."""
    r = graph()
    before = state(r)
    versions = {n.id: n.tree_evaluation.direct_evaluation_version for n in nodes(r)}
    r._runtime.set_evaluator(neural_evaluator(42))
    assert state(r) == before
    assert all(
        n.tree_evaluation.direct_evaluation_version < r._runtime.evaluator_version
        for n in nodes(r)
    )
    # The actual selector accepts this mixed-version tree; it does not demand refresh.
    r._runtime._select_node_for_expansion()
    assert {
        n.id: n.tree_evaluation.direct_evaluation_version for n in nodes(r)
    } == versions


def test_runner_refresh_defaults_to_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The existing runner reevaluation boolean does not promise an all-node refresh."""
    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module

    instance = graph()
    root = next(nodes(instance))
    before = root.tree_evaluation.direct_value
    monkeypatch.setattr(
        runner_module,
        "load_morpion_evaluator_from_model_bundle",
        lambda _: neural_evaluator(42),
    )
    instance._set_runtime_evaluator_from_bundle(
        tmp_path / "model", reevaluate_tree=True
    )
    assert root.tree_evaluation.direct_value == before
    assert (
        root.tree_evaluation.direct_evaluation_version
        < instance._runtime.evaluator_version
    )


@pytest.mark.parametrize("rollout", [False, True])
def test_refresh_checkpoint_rng_and_next_selections(
    tmp_path: Path, rollout: bool
) -> None:
    """Check refresh checkpoint rng and next selections."""
    from anemone.checkpoints.rng_state import rollout_random_generator

    from chipiron.environments.morpion.bootstrap.config import (
        MorpionBootstrapRolloutConfig,
    )
    from chipiron.environments.morpion.bootstrap.runtime.runner import (
        default_search_args,
    )

    if rollout:
        args = replace(
            runner()._args,
            search_args=default_search_args(
                rollout=MorpionBootstrapRolloutConfig(enabled=True, max_extra_steps=1)
            ),
        )
        continuous = AnemoneMorpionSearchRunner(args)
        continuous.load_or_create(
            None, None, MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000)
        )
        continuous.grow(3)
    else:
        continuous = graph()
    rollout_rng = rollout_random_generator(continuous._runtime)
    before_rollout = None if rollout_rng is None else rollout_rng.getstate()
    bounded_probe(continuous, neural_evaluator(42))
    assert (None if rollout_rng is None else rollout_rng.getstate()) == before_rollout
    checkpoint = tmp_path / "checkpoint"
    continuous.save_checkpoint(checkpoint)
    restored = AnemoneMorpionSearchRunner(continuous._args)
    restored.load_or_create(
        checkpoint,
        None,
        MorpionBootstrapEffectiveRuntimeConfig(tree_branch_limit=100000),
    )
    assert state(restored) == state(continuous)
    assert (
        restored._random_generator.getstate() == continuous._random_generator.getstate()
    )
    # Checkpoint does not serialize model weights. Install identical evaluator WITHOUT
    # incrementing the saved version; demonstrate the exact attach interface needed.
    restored._runtime.tree_manager.node_evaluator.master_state_value_evaluator = (
        neural_evaluator(42)
    )
    restored_rollout_rng = rollout_random_generator(restored._runtime)
    assert (
        None if restored_rollout_rng is None else restored_rollout_rng.getstate()
    ) == before_rollout
    for _ in range(3):
        continuous.grow(1)
        restored.grow(1)
        assert state(restored) == state(continuous)
        assert _selection_signature(restored) == _selection_signature(continuous)
        assert (
            restored._random_generator.getstate()
            == continuous._random_generator.getstate()
        )
        if rollout:
            assert restored_rollout_rng.getstate() == rollout_rng.getstate()


def test_public_attach_increments_version_even_for_same_model(tmp_path: Path) -> None:
    """Restoring identical weights needs an attach operation distinct from a switch."""
    instance = graph()
    bounded_probe(instance, neural_evaluator(42))
    path = tmp_path / "checkpoint"
    instance.save_checkpoint(path)
    restored = runner()
    restored.load_or_create(path, None)
    before = restored._runtime.evaluator_version
    restored._runtime.set_evaluator(neural_evaluator(42))
    assert restored._runtime.evaluator_version == before + 1
    assert all(
        n.tree_evaluation.direct_evaluation_version == before for n in nodes(restored)
    )


def test_restore_resolver_materializes_all_states_despite_bounded_windows(
    tmp_path: Path,
) -> None:
    """A bounded inference list alone does not bound the existing decode cache."""
    instance = graph()
    path = tmp_path / "checkpoint"
    instance.save_checkpoint(path)
    restored = runner()
    restored.load_or_create(path, None)
    first = next(nodes(restored))
    resolver = first.tree_node.state_handle.resolver
    before = len(resolver._resolved_states)
    bounded_probe(restored, neural_evaluator(42), 1)
    assert len(resolver._resolved_states) == restored.current_tree_size()
    assert before < len(resolver._resolved_states)


@pytest.mark.parametrize("complete", [False, True])
def test_equal_scalar_exact_estimate_ties(
    monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Preserve existing effective-value ordering without certifying equal estimates."""
    instance = make_runner(monkeypatch)
    root, child = tuple(nodes(instance))
    root.tree_node.all_branches_generated = complete
    root.tree_evaluation.direct_value = canonical_value.make_forced_value(score=9)
    child.tree_evaluation.direct_value = canonical_value.make_estimate_value(score=9)
    from docs.research.model_activation.probe import backup_phase

    backup_phase(instance)
    assert root.tree_evaluation.direct_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.tree_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.effective_value.certainty == (
        Certainty.ESTIMATE if complete else Certainty.FORCED
    )
