import math

import chess

from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.boardevaluators.board_evaluator import BoardEvaluator


def value_base(
        board: BoardChi,
        color: chess.Color
) -> int:
    value_white_: int = bin(board.board.pawns & board.board.occupied_co[color]).count('1') \
                        + bin(board.board.knights & board.board.occupied_co[color]).count('1') * 3 \
                        + bin(board.board.bishops & board.board.occupied_co[color]).count('1') * 3 \
                        + bin(board.board.rooks & board.board.occupied_co[color]).count('1') * 5 \
                        + bin(board.board.queens & board.board.occupied_co[color]).count('1') * 9
    return value_white_


def add_pawns_value_white(board):
    # code to push the pawns to advance by giving more value to the  pawns that are advanced
    add_value: float = 0
    for pawn in list(board.board.pieces(chess.PAWN, chess.WHITE)):
        add_value += int((pawn - 8) / 8) / 50. * 1
    return add_value


def add_pawns_value_black(board):
    # code to push the pawns to advance by giving more value to the  pawns that are advanced

    add_value: float = 0
    for pawn in list(board.board.pieces(chess.PAWN, chess.BLACK)):
        add_value += int((63 - pawn - 8) / 8) / 50. * 1
    return add_value


def value_white(board):
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    return value_white_pieces - value_black_pieces


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def value_player_to_move(board):
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    if board.chess_board.turn == chess.WHITE:
        return sigmoid((value_white_pieces - value_black_pieces) * .2)
    else:
        return sigmoid((value_black_pieces - value_white_pieces) * .2)


class BasicEvaluation(BoardEvaluator):

    def __init__(self):
        pass

    def value_white(self, board):
        value_white_pieces = value_base(board, chess.WHITE)
        value_black_pieces = value_base(board, chess.BLACK)
        value_white_pieces += add_pawns_value_white(board)
        value_black_pieces += add_pawns_value_black(board)

        return value_white_pieces - value_black_pieces  # + 100 * board.chess_board.is_check() - 200 * queen_atta
