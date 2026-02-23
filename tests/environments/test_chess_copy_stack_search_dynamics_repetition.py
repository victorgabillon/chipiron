from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from anemone.dynamics import normalize_search_dynamics
from atomheart import ChessDynamics
from atomheart.games.chess.board import BoardFactory, IBoard, create_board_factory
from atomheart.games.chess.board.utils import FenPlusHistory

from chipiron.environments.chess.search_dynamics import ChessCopyStackSearchDynamics
from chipiron.environments.chess.types import ChessState

if TYPE_CHECKING:
    from valanga import Transition


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_threefold_repetition_detected_with_copy_stack_until_depth_2(
    use_rusty_board: bool,
) -> None:
    """
    Build a position that is exactly 2 ply away from a 3-fold repetition.

    Then compare:
      - copy_stack_until_depth=2 -> preserves stack at depths 0 and 1 -> repetition detected
      - copy_stack_until_depth=0 -> never preserves stack -> repetition NOT detected
    """
    board_factory: BoardFactory = create_board_factory(use_rust_boards=use_rusty_board)
    board: IBoard = board_factory(
        fen_with_history=FenPlusHistory(
            current_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
    )

    prefix_moves = ["b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6"]
    for uci in prefix_moves:
        move = board.get_move_key_from_uci(move_uci=uci)
        board.play_move_key(move=move)

    assert not board.is_game_over()

    root_state = ChessState(board)
    base = normalize_search_dynamics(ChessDynamics())

    move1 = board.get_move_key_from_uci(move_uci="c3b1")

    dyn_ok = ChessCopyStackSearchDynamics(
        base=base,
        copy_stack_until_depth=2,
        deep_copy_legal_moves=True,
    )

    t1: Transition[ChessState] = dyn_ok.step(root_state, move1, depth=0)
    move2 = t1.next_state.board.get_move_key_from_uci(move_uci="c6b8")

    t2: Transition[ChessState] = dyn_ok.step(t1.next_state, move2, depth=1)

    assert t2.next_state.board.is_game_over(), "3-fold repetition should be detected"

    dyn_bad = ChessCopyStackSearchDynamics(
        base=base,
        copy_stack_until_depth=0,
        deep_copy_legal_moves=True,
    )

    t1b: Transition[ChessState] = dyn_bad.step(root_state, move1, depth=0)
    move2b = t1b.next_state.board.get_move_key_from_uci(move_uci="c6b8")
    t2b: Transition[ChessState] = dyn_bad.step(t1b.next_state, move2b, depth=1)

    assert not t2b.next_state.board.is_game_over(), (
        "Without copying stack history, repetition should not be detected"
    )


if __name__ == "__main__":
    test_threefold_repetition_detected_with_copy_stack_until_depth_2(
        use_rusty_board=True
    )
    test_threefold_repetition_detected_with_copy_stack_until_depth_2(
        use_rusty_board=False
    )
