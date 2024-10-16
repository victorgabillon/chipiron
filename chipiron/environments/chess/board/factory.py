"""
Module to create a chess board.
"""
from functools import partial
from typing import Protocol

import chess
import shakmaty_python_binding

from .board_chi import BoardChi
from .iboard import IBoard
from .rusty_board import RustyBoardChi
from .utils import fen, FenPlusMoveHistory


class BoardFactory(Protocol):
    def __call__(self, fen_with_history: FenPlusMoveHistory | None = None) -> IBoard:
        ...


def create_board_factory(
        use_rust_boards: bool,
        use_board_modification: bool
) -> BoardFactory:
    board_factory: BoardFactory
    if use_rust_boards:
        board_factory = partial(create_rust_board, use_board_modification=use_board_modification)
    else:
        board_factory = partial(create_board, use_board_modification=use_board_modification)
    return board_factory


def create_board(
        fen_with_history: FenPlusMoveHistory | None = None,
        use_board_modification: bool = False
) -> BoardChi:
    """
    Create a chess board.

    Args:
        use_board_modification (bool): whether to use the board modification
        fen_with_history (FenPlusMoves | None): The BoardWithHistory that contains a fen and the subsequent moves.
            The FEN (Forsyth-Edwards Notation) string representing the board position. If None, the starting position
            is used.

    Returns:
        BoardChi: The created chess board.

    """
    chess_board: chess.Board
    current_fen: fen

    if fen_with_history is not None:
        current_fen: fen = fen_with_history.current_fen
        chess_board = chess.Board(fen=current_fen)
        chess_board.move_stack = fen_with_history.historical_moves

    else:
        chess_board = chess.Board()

    board: BoardChi = BoardChi(
        board=chess_board,
        compute_board_modification=use_board_modification
    )
    return board


def create_rust_board(
        fen_with_history: FenPlusMoveHistory | None = None,
        use_board_modification: bool = False
) -> RustyBoardChi:
    """
    Create a rust chess board.

    Args:
        use_board_modification (bool): whether to use the board modification
        board_with_history (FenPlusMoves | None): The BoardWithHistory that contains a fen and the subsequent moves.
            The FEN (Forsyth-Edwards Notation) string representing the board position. If None, the starting position
            is used.

    Returns:
        RustyBoardChi: The created chess board.

    """
    current_fen: fen

    if fen_with_history is not None:
        current_fen: fen = fen_with_history.current_fen
        chess_rust_binding = shakmaty_python_binding.MyChess(_fen_start=current_fen)

    else:
        chess_rust_binding = shakmaty_python_binding.MyChess()

    rusty_board_chi: RustyBoardChi = RustyBoardChi(
        chess_=chess_rust_binding,
        compute_board_modification=use_board_modification
    )
    rusty_board_chi.move_stack = fen_with_history.historical_moves

    return rusty_board_chi
