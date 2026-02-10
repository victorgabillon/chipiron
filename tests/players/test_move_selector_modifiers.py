"""Unit tests for move selector recommendation modifiers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from valanga import BranchKey, BranchName, Color
from valanga.evaluations import FloatyStateEvaluation
from valanga.policy import Recommendation

from chipiron.players.move_selector.modifiers import (
    AccelerateWhenWinning,
    ComposedBranchSelector,
    chess_progress_gain_zeroing,
)


class DummySelector:
    """Base selector stub that always returns a fixed recommendation."""

    def __init__(self, rec: Recommendation) -> None:
        """Store a fixed recommendation to return."""
        self._rec = rec

    def recommend(
        self,
        state: Any,
        seed: int,
        notify_progress: Any = None,
    ) -> Recommendation:
        """Return the preconfigured recommendation."""
        _ = (state, seed, notify_progress)
        return self._rec


@dataclass
class FakeTurnState:
    """Minimal TurnState-like object for testing modifiers."""

    turn: Color = Color.WHITE
    is_over: bool = False
    name_to_key: dict[BranchName, BranchKey] = field(default_factory=dict)
    stepped: list[BranchKey] = field(default_factory=list)

    def is_game_over(self) -> bool:
        """Return whether this fake state is terminal."""
        return self.is_over

    def branch_key_from_name(self, name: BranchName) -> BranchKey:
        """Map branch name to branch key."""
        return self.name_to_key[name]

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> FakeTurnState:
        """Return a lightweight copy compatible with modifier expectations."""
        _ = (stack, deep_copy_legal_moves)
        new = copy.copy(self)
        new.stepped = []
        return new

    def step(self, branch_key: BranchKey) -> object:
        """Record a step and return a non-None placeholder modification."""
        self.stepped.append(branch_key)
        return object()


@dataclass
class FakeChessState(FakeTurnState):
    """Adds the chess capability needed by chess_progress_gain_zeroing."""

    zeroing_keys: set[BranchKey] = field(default_factory=set)

    def is_zeroing(self, move: BranchKey) -> bool:
        """Return whether a branch key is considered zeroing in this fake state."""
        return move in self.zeroing_keys


def test_accelerate_when_not_winning_does_not_override() -> None:
    """Modifier should not alter recommendation when root eval is not winning."""
    state = FakeChessState(
        turn=Color.WHITE,
        name_to_key={"a": 1, "b": 2},
        zeroing_keys={2},
    )
    rec = Recommendation(
        recommended_name="a",
        evaluation=FloatyStateEvaluation(value_white=0.2),
        branch_evals={
            "a": FloatyStateEvaluation(value_white=0.2),
            "b": FloatyStateEvaluation(value_white=0.99),
        },
    )
    selector = ComposedBranchSelector(
        base=DummySelector(rec),
        modifiers=(AccelerateWhenWinning(progress_gain_fn=chess_progress_gain_zeroing),),
    )

    out = selector.recommend(state=state, seed=0)

    assert out.recommended_name == "a"


def test_accelerate_when_winning_prefers_zeroing_move_that_stays_winning() -> None:
    """When winning, modifier should prefer a winning child with higher progress gain."""
    state = FakeChessState(
        turn=Color.WHITE,
        name_to_key={"a": 1, "b": 2},
        zeroing_keys={2},
    )
    rec = Recommendation(
        recommended_name="a",
        evaluation=FloatyStateEvaluation(value_white=0.99),
        branch_evals={
            "a": FloatyStateEvaluation(value_white=0.99),
            "b": FloatyStateEvaluation(value_white=0.99),
        },
    )
    selector = ComposedBranchSelector(
        base=DummySelector(rec),
        modifiers=(AccelerateWhenWinning(progress_gain_fn=chess_progress_gain_zeroing),),
    )

    out = selector.recommend(state=state, seed=0)

    assert out.recommended_name == "b"


def test_accelerate_when_winning_does_not_choose_non_winning_child() -> None:
    """Modifier should not pick a progress move that loses the winning eval filter."""
    state = FakeChessState(
        turn=Color.WHITE,
        name_to_key={"a": 1, "b": 2},
        zeroing_keys={2},
    )
    rec = Recommendation(
        recommended_name="a",
        evaluation=FloatyStateEvaluation(value_white=0.99),
        branch_evals={
            "a": FloatyStateEvaluation(value_white=0.99),
            "b": FloatyStateEvaluation(value_white=0.0),
        },
    )
    selector = ComposedBranchSelector(
        base=DummySelector(rec),
        modifiers=(AccelerateWhenWinning(progress_gain_fn=chess_progress_gain_zeroing),),
    )

    out = selector.recommend(state=state, seed=0)

    assert out.recommended_name == "a"


def test_accelerate_when_winning_no_override_when_best_gain_is_zero() -> None:
    """Modifier should keep base recommendation when no positive gain exists."""
    state = FakeChessState(
        turn=Color.WHITE,
        name_to_key={"a": 1, "b": 2},
        zeroing_keys=set(),
    )
    rec = Recommendation(
        recommended_name="a",
        evaluation=FloatyStateEvaluation(value_white=0.99),
        branch_evals={
            "a": FloatyStateEvaluation(value_white=0.99),
            "b": FloatyStateEvaluation(value_white=0.99),
        },
    )
    selector = ComposedBranchSelector(
        base=DummySelector(rec),
        modifiers=(AccelerateWhenWinning(progress_gain_fn=chess_progress_gain_zeroing),),
    )

    out = selector.recommend(state=state, seed=0)

    assert out.recommended_name == "a"
