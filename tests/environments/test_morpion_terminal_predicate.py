"""Exact terminality without changing symmetry-reduced counts or scores."""

from dataclasses import replace
from random import Random
from typing import Any, cast

import pytest
from atomheart.games.morpion import initial_state
from atomheart.games.morpion.state import Variant
from valanga.evaluations import Certainty

from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState


def assert_terminal_contract(state: MorpionState) -> None:
    """Compare against the unchanged public count and heuristic evaluation."""
    dynamics = MorpionDynamics()
    atom = state.to_atomheart_state()
    count = len(dynamics.inner.legal_actions(atom).get_all())
    assert dynamics.legal_action_count(state) == count
    assert dynamics.legal_action_count(atom) == count
    assert dynamics.is_terminal_state(atom) == (count == 0)
    detector = MorpionOverEventDetector()
    evaluator = MorpionStateEvaluator()
    for cached_terminal in (False, True):
        wrapped = replace(state, is_terminal=cached_terminal)
        assert dynamics.is_terminal_state(wrapped) == (count == 0)
        reference = evaluator.evaluate(wrapped)
        over, score = detector.check_obvious_over_events(wrapped)
        if count == 0:
            assert reference.certainty is Certainty.TERMINAL
            assert score == reference.score == float(state.moves)
            assert over == reference.over_event
        else:
            assert reference.certainty is Certainty.ESTIMATE
            assert reference.score == float(state.moves) + 0.01 * float(count)
            assert (over, score) == (None, None)


@pytest.mark.fast
@pytest.mark.parametrize("variant", list(Variant))
@pytest.mark.parametrize("seed", [0, 1, 1729])
def test_terminal_predicate_matches_complete_legal_histories(
    variant: Variant, seed: int
) -> None:
    """Both rule variants retain exact terminal events and numeric heuristics."""
    dynamics = MorpionDynamics()
    atom = initial_state(variant)
    random = Random(seed)
    for _ in range(160):
        state = MorpionState.from_atomheart_state(atom)
        assert_terminal_contract(state)
        actions = dynamics.inner.all_legal_actions(atom)
        if not actions:
            return
        atom = dynamics.inner.step(atom, random.choice(actions)).next_state
    pytest.fail("Bounded fixture must include an actual terminal position.")


@pytest.mark.fast
@pytest.mark.parametrize("variant", list(Variant))
def test_terminal_predicate_handles_sparse_and_translated_states(
    variant: Variant,
) -> None:
    """Empty boards and translated open/blocked lines use actual legality."""
    for points in (
        frozenset(),
        frozenset({(100, -200)}),
        frozenset((100 + index, -200) for index in range(4)),
    ):
        state = MorpionState(
            points=points,
            used_unit_segments=frozenset(),
            dir_usage_entries=(),
            moves=17,
            variant=variant,
        )
        assert_terminal_contract(state)
        blocked = replace(
            state,
            used_unit_segments=frozenset({((100, -200), (101, -200))}),
        )
        assert_terminal_contract(blocked)


@pytest.mark.fast
def test_terminal_optimization_preserves_symmetry_reduced_count() -> None:
    """Terminality is boolean; raw move counts must not replace heuristic counts."""
    dynamics = MorpionDynamics()
    state = MorpionState.from_atomheart_state(initial_state())
    assert len(dynamics.all_legal_actions(state)) > dynamics.legal_action_count(state)
    assert_terminal_contract(state)


@pytest.mark.fast
def test_malformed_state_errors_are_not_terminal_results() -> None:
    """Neither an invalid board nor a cached terminal flag masks rule errors."""
    state = replace(
        MorpionState.from_atomheart_state(initial_state()),
        used_unit_segments=cast("Any", None),
        is_terminal=True,
    )
    dynamics = MorpionDynamics()
    with pytest.raises(TypeError):
        dynamics.legal_action_count(state)
    with pytest.raises(TypeError):
        dynamics.is_terminal_state(state)
    with pytest.raises(TypeError):
        MorpionOverEventDetector().check_obvious_over_events(state)
