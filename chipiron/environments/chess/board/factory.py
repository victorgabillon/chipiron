from chipiron.environments.chess.board.board import BoardChi
import chess


def create_board(
) -> BoardChi:
    chess_board: chess.Board = chess.Board()
    board: BoardChi = BoardChi(board=chess_board)
    return board
