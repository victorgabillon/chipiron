"""
Module to create a chess board.
"""
from functools import partial
from typing import Callable

import chess

from chipiron.environments.chess.board.board import BoardChi

BoardFactory = Callable[[str], BoardChi]


def create_board_factory(
        use_rust_boards: bool,
        use_board_modification: bool
) -> BoardFactory:
    board_factory: BoardFactory
    if use_rust_boards:
        board_factory = partial(create_board, use_board_modification=use_board_modification)
    else:
        print('oooooooooooooo',use_board_modification)

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
    chess_board: chess.Board = chess.Board()
    board: BoardChi = BoardChi(
        board=chess_board,
        compute_board_modification=use_board_modification
    )
    if fen is not None:
        board.set_starting_position(fen=fen)
    return board
