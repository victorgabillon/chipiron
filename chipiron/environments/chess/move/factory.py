from chipiron.environments.chess.board.iboard import IBoard
from .imove import IMove
moveUci = str

from typing import Protocol

import chess
import shakmaty_python_binding


class MoveFactory(Protocol):
    def __call__(self, move_uci: moveUci, board: IBoard | None = None) -> IMove:
        ...


def create_move_factory(
        use_rust_boards: bool,
) -> MoveFactory:
    move_factory: MoveFactory
    if use_rust_boards:
        move_factory = create_rust_move
    else:
        move_factory = create_move
    return move_factory


def create_rust_move(
        move_uci: moveUci,
        board: IBoard | None = None
):
    return shakmaty_python_binding.MyMove(move_uci, board)


def create_move(
        move_uci: moveUci,
        board: IBoard | None = None
):
    return chess.Move.from_uci(move_uci)
