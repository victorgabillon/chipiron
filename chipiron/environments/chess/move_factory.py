from typing import Protocol

import chess
import shakmaty_python_binding

from chipiron.environments.chess.board.iboard import IBoard
from chipiron.environments.chess.move.imove import IMove

from .board import RustyBoardChi
from .move import moveUci


class MoveFactory(Protocol):
    def __call__(self, move_uci: moveUci, board: IBoard | None = None) -> IMove: ...


def create_move_factory(
    use_rust_boards: bool,
) -> MoveFactory:
    move_factory: MoveFactory

    if use_rust_boards:
        # todo can we go back to the not test version without the assert? generics or just typos?
        move_factory = create_rust_move_test_2
    else:
        move_factory = create_move
    return move_factory


def create_rust_move(
    move_uci: moveUci, board: RustyBoardChi | None = None
) -> shakmaty_python_binding.MyMove:
    assert board is not None
    return shakmaty_python_binding.MyMove(move_uci, board.chess_)


def create_rust_move_test_2(
    move_uci: moveUci, board: IBoard | None = None
) -> shakmaty_python_binding.MyMove:
    assert isinstance(board, RustyBoardChi)
    binding_move = shakmaty_python_binding.MyMove(move_uci, board.chess_)
    # rust_move = RustMove(move=binding_move, uci=move_uci)
    # return rust_move
    return binding_move


def create_rust_move_test(move_uci: moveUci, board: IBoard | None = None) -> chess.Move:
    assert board is not None
    return chess.Move.from_uci(move_uci)


def create_move(move_uci: moveUci, board: IBoard | None = None) -> chess.Move:
    return chess.Move.from_uci(move_uci)
