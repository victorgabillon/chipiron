from dataclasses import dataclass, field
from typing import Any

from anemone.dynamics import SearchDynamics

from chipiron.environments.chess.search_dynamics import ChessCopyStackSearchDynamics
from chipiron.environments.chess.types import ChessState


@dataclass
class FakeBoard:
    history: list[str] = field(default_factory=list)
    copy_calls: list[tuple[bool, bool]] = field(default_factory=list)
    game_over: bool = False

    def copy(self, *, stack: bool, deep_copy_legal_moves: bool) -> "FakeBoard":
        self.copy_calls.append((stack, deep_copy_legal_moves))
        history = list(self.history) if stack else []
        return FakeBoard(
            history=history,
            copy_calls=self.copy_calls,
            game_over=self.game_over,
        )

    def play_move_key(self, move: str) -> None:
        self.history.append(move)

    def is_game_over(self) -> bool:
        return self.game_over


# NOTE: wrapper.step() intentionally does not call base.step();
# this dummy exists only for delegated non-step methods.
class DummyBaseSearchDynamics(SearchDynamics[ChessState, Any]):
    __anemone_search_dynamics__ = True

    def legal_actions(self, state: ChessState) -> Any:
        return ()

    def action_name(self, state: ChessState, action: Any) -> str:
        return str(action)

    def action_from_name(self, state: ChessState, name: str) -> Any:
        return name

    def step(self, state: ChessState, action: Any, *, depth: int = 0) -> Any:
        raise NotImplementedError


def _state_with_history(history: list[str]) -> ChessState:
    return ChessState(FakeBoard(history=history))


def test_step_copies_stack_below_threshold() -> None:
    state = _state_with_history(["e2e4"])
    dynamics = ChessCopyStackSearchDynamics(
        base=DummyBaseSearchDynamics(),
        copy_stack_until_depth=2,
        deep_copy_legal_moves=True,
    )

    transition = dynamics.step(state, "e7e5", depth=1)

    assert state.board.copy_calls == [(True, True)]
    assert transition.next_state.board.history == ["e2e4", "e7e5"]


def test_step_does_not_copy_stack_at_or_above_threshold() -> None:
    state = _state_with_history(["e2e4"])
    dynamics = ChessCopyStackSearchDynamics(
        base=DummyBaseSearchDynamics(),
        copy_stack_until_depth=2,
        deep_copy_legal_moves=True,
    )

    transition = dynamics.step(state, "e7e5", depth=2)

    assert state.board.copy_calls == [(False, True)]
    assert transition.next_state.board.history == ["e7e5"]


def test_step_default_depth_copies_stack() -> None:
    state = _state_with_history(["e2e4"])
    dynamics = ChessCopyStackSearchDynamics(
        base=DummyBaseSearchDynamics(),
        copy_stack_until_depth=2,
        deep_copy_legal_moves=False,
    )

    transition = dynamics.step(state, "e7e5")

    assert state.board.copy_calls == [(True, False)]
    assert transition.next_state.board.history == ["e2e4", "e7e5"]
