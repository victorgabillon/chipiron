from sunau import Au_write
from typing import Protocol, Counter

import chess
import shakmaty_python_binding
from wtforms.validators import AnyOf

from chipiron.environments.chess.board.iboard import IBoard
from chipiron.environments.chess.move.imove import IMove
from . import BoardChi
from .board import RustyBoardChi
from .move import moveUci

from collections import Counter

from typing import Any

class MoveFactory(Protocol):
    def __call__(
            self,
            move_uci: moveUci,
            board: IBoard[Any] | None = None
    ) -> IMove:
        ...


def create_move_factory(
        use_rust_boards: bool,
) -> MoveFactory:
    move_factory: MoveFactory

    chess_rust_binding = shakmaty_python_binding.MyChess(_fen_start=chess.STARTING_FEN  )
    a_test_debug : IBoard = RustyBoardChi(chess_=chess_rust_binding,
                                          compute_board_modification=True,
                                          rep_to_count=Counter())



    if use_rust_boards:
        move_factory = create_rust_move_test_2
    else:
        move_factory = create_move
    return move_factory


def create_rust_move(
        move_uci: moveUci,
        board: RustyBoardChi | None = None
) -> shakmaty_python_binding.MyMove:
    assert board is not None
    return shakmaty_python_binding.MyMove(
        move_uci,
        board.chess_
    )


def create_rust_move_test_2(
        move_uci: moveUci,
        board: IBoard | None = None
) -> shakmaty_python_binding.MyMove:
    assert isinstance(board,RustyBoardChi)
    return shakmaty_python_binding.MyMove(
        move_uci,
        board.chess_
    )


def create_rust_move_test(
        move_uci: moveUci,
        board: IBoard | None = None
) -> chess.Move:
    assert board is not None
    return chess.Move.from_uci(move_uci)


def create_move(
        move_uci: moveUci,
        board: BoardChi | None = None
) -> chess.Move:
    return chess.Move.from_uci(move_uci)
