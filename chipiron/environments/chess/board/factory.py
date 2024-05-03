"""
Module to create a chess board.
"""
import chess

from chipiron.environments.chess.board.board import BoardChi

def create_board_factory():
   ...

def create_board(
        fen: str | None = None
) -> BoardChi:
    """
    Create a chess board.

    Args:
        fen (str | None): The FEN (Forsyth-Edwards Notation) string representing the board position.
                          If None, the starting position is used.

    Returns:
        BoardChi: The created chess board.

    """
    chess_board: chess.Board = chess.Board()
    board: BoardChi = BoardChi(board=chess_board)
    if fen is not None:
        board.set_starting_position(fen=fen)
    return board
