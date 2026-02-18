from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from valanga import Color
from chipiron.environments.chess.search_dynamics import ChessSearchDynamics


class FakeBoard:
    def __init__(self, history: list[str], next_turn: Color) -> None:
        self.move_history_stack = list(history)
        self._next_turn = next_turn
        self.copy_calls: list[bool] = []

    def copy(self, *, stack: bool, deep_copy_legal_moves: bool = True) -> "FakeBoard":
        _ = deep_copy_legal_moves
        self.copy_calls.append(stack)

        history = list(self.move_history_stack) if stack else []
        copied = FakeBoard(history=history, next_turn=self._next_turn)
        copied.copy_calls = self.copy_calls
        return copied

    def play_move_key(self, *, move: Any) -> None:
        self.move_history_stack.append(move)
        self._next_turn = Color.BLACK if self._next_turn is Color.WHITE else Color.WHITE

    def is_game_over(self) -> bool:
        return False

    @property
    def turn(self) -> Color:
        return self._next_turn


@dataclass(frozen=True)
class FakeState:
    board: FakeBoard
    turn: Color = Color.WHITE
    half_move: int = 0
    full_move_number: int = 1

    # ChessSearchDynamics.step uses state._replace(...)
    def _replace(self, **kwargs: Any) -> "FakeState":
        return replace(self, **kwargs)


def test_chess_search_dynamics_step_uses_depth_to_control_stack_copy() -> None:
    dyn = ChessSearchDynamics(copy_stack_until_depth=2)

    base_history = ["m1", "m2", "m3"]
    state = FakeState(
        board=FakeBoard(history=base_history, next_turn=Color.WHITE),
        turn=Color.WHITE,
    )

    # depth < threshold => stack copied
    out0 = dyn.step(state, action="m4", depth=0).next_state
    assert state.board.copy_calls == [True]
    assert out0.board.move_history_stack == base_history + ["m4"]

    # depth >= threshold => stack not copied (optimization)
    out2 = dyn.step(state, action="mX", depth=2).next_state
    assert state.board.copy_calls == [True, False]
    assert out2.board.move_history_stack == ["mX"]


def test_chess_search_dynamics_step_default_depth_copies_stack() -> None:
    dyn = ChessSearchDynamics(copy_stack_until_depth=2)

    base_history = ["m1", "m2"]
    state = FakeState(
        board=FakeBoard(history=base_history, next_turn=Color.WHITE),
        turn=Color.WHITE,
    )

    out = dyn.step(state, action="m3").next_state  # depth omitted => default 0
    assert state.board.copy_calls == [True]
    assert out.board.move_history_stack == ["m1", "m2", "m3"]
