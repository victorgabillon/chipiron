"""Targeted cross-repo shape checks for the PR1 safety net."""

from __future__ import annotations

from valanga import Color

from atomheart.games.checkers.state import CheckersState
from atomheart.games.integer_reduction.state import IntegerReductionState
def test_local_atomheart_contains_both_turnless_and_turn_based_states() -> None:
    """Make the current single-player versus two-player state mismatch explicit."""
    solo_state = IntegerReductionState(3)
    alternating_state = CheckersState.standard(turn=Color.BLACK)

    assert hasattr(solo_state, "turn") is False
    assert hasattr(alternating_state, "turn") is True
    assert alternating_state.turn is Color.BLACK
