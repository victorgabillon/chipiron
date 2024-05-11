"""
Module to create a chess board.
"""
from functools import partial
from typing import Protocol

import chess
import shakmaty_python_binding

from chipiron.environments.chess.board.board import BoardChi
from chipiron.environments.chess.board.rusty_board import RustyBoardChi
from .iboard import IBoard


class BoardFactory(Protocol):
    def __call__(self, fen: str | None = None) -> IBoard:
        ...


def create_board_factory(
        use_rust_boards: bool,
        use_board_modification: bool
) -> BoardFactory:
    board_factory: BoardFactory
    if use_rust_boards:
        print('RUSToooooooooooooo', use_board_modification)
        board_factory = partial(create_rust_board, use_board_modification=use_board_modification)
    else:
        print('oooooooooooooo', use_board_modification)
        board_factory = partial(create_board, use_board_modification=use_board_modification)
    return board_factory


def create_board(
        fen: str | None = None,
        use_board_modification: bool = False
) -> BoardChi:
    """
    Create a chess board.

    Args:
        use_board_modification (bool): whether to use the board modification
        fen (str | None): The FEN (Forsyth-Edwards Notation) string representing the board position.
                          If None, the starting position is used.

    Returns:
        BoardChi: The created chess board.

    """
    chess_board: chess.Board = chess.Board(fen=fen)
    board: BoardChi = BoardChi(
        board=chess_board,
        compute_board_modification=use_board_modification
    )
    return board


def create_rust_board(
        fen: str | None = None,
        use_board_modification: bool = False
) -> RustyBoardChi:
    """
    Create a rust chess board.

    Args:
        use_board_modification (bool): whether to use the board modification
        fen (str | None): The FEN (Forsyth-Edwards Notation) string representing the board position.
                          If None, the starting position is used.

    Returns:
        BoardChi: The created chess board.

    """
    chess_rust_binding = shakmaty_python_binding.MyChess(_fen_start=fen)
    board: RustyBoardChi = RustyBoardChi(
        chess_=chess_rust_binding,
    )

    return board
