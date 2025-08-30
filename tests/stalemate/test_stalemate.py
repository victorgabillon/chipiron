import pytest

from chipiron.environments.chess_env.board import (
    BoardFactory,
    IBoard,
    create_board_factory,
)
from chipiron.environments.chess_env.board.utils import FenPlusHistory
from chipiron.environments.chess_env.move import IMove
from chipiron.environments.chess_env.move_factory import (
    MoveFactory,
    create_move_factory,
)


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_three_fold_repetition(use_rusty_board: bool):
    # todo maybe this sis more a unit test for the is_game_over method atm

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

        board.play_move(move=move)

    assert board.is_game_over()


test_three_fold_repetition(use_rusty_board=True)
test_three_fold_repetition(use_rusty_board=False)
