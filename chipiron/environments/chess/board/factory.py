import chess

from chipiron.environments.chess.board.board import BoardChi


def create_board(
        fen=None
) -> BoardChi:
    chess_board: chess.Board = chess.Board()
    board: BoardChi = BoardChi(board=chess_board)
    if fen is not None:
        board.set_starting_position(fen=fen)
    return board
