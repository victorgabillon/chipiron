"""Module for test stalemate."""

from typing import TYPE_CHECKING

import pytest
from atomheart.games.chess.board import (
    BoardFactory,
    IBoard,
    create_board_factory,
)
from atomheart.games.chess.board.utils import FenPlusHistory
from atomheart.games.chess.move.move_factory import (
    MoveFactory,
    create_move_factory,
)

if TYPE_CHECKING:
    from atomheart.games.chess.move import IMove


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_three_fold_repetition(use_rusty_board: bool) -> None:
    # TODO: maybe this sis more a unit test for the is_game_over method atm
    """Test three fold repetition."""
    board_factory: BoardFactory = create_board_factory(use_rust_boards=use_rusty_board)
    board: IBoard = board_factory(
        fen_with_history=FenPlusHistory(
            current_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
    )

    move_factory: MoveFactory = create_move_factory(use_rust_boards=use_rusty_board)
    moves_uci = ["b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8"]

    for move_uci in moves_uci:
        move: IMove = move_factory(move_uci=move_uci, board=board)
        print(f"Playing move {move_uci}, game over: {board.is_game_over()}")
        board.play_move(move=move)

    assert board.is_game_over()


if __name__ == "__main__":
    print("Testing three fold repetition with RustyBoard...")

    print("Testing three fold repetition with RustyBoard...")
    test_three_fold_repetition(use_rusty_board=True)

    print("Testing three fold repetition with PythonBoard...")
    test_three_fold_repetition(use_rusty_board=False)
