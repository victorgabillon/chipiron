import chess
import torch
def value_base(board, color):
    value_white = bin(board.chess_board.pawns & board.chess_board.occupied_co[color]).count('1') \
                  + bin(board.chess_board.knights & board.chess_board.occupied_co[color]).count('1') * 3 \
                  + bin(board.chess_board.bishops & board.chess_board.occupied_co[color]).count('1') * 3 \
                  + bin(board.chess_board.rooks & board.chess_board.occupied_co[color]).count('1') * 5 \
                  + bin(board.chess_board.queens & board.chess_board.occupied_co[color]).count('1') * 9
    return value_white


def add_pawns_value_white(board):
    add_value = 0
    for pawn in list(board.chess_board.pieces(chess.PAWN, chess.WHITE)):
        add_value += int((pawn - 8) / 8) / 50. * 1
    return add_value


def add_pawns_value_black(board):
    add_value = 0
    for pawn in list(board.chess_board.pieces(chess.PAWN, chess.BLACK)):
        add_value += int((63 - pawn - 8) / 8) / 50. * 1
    return add_value


def value_white(board):
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    return value_white_pieces - value_black_pieces



import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def value_player_to_move(board):
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    if board.chess_board.turn == chess.WHITE:
        return sigmoid((value_white_pieces - value_black_pieces)*.2)
    else:
        return sigmoid((value_black_pieces - value_white_pieces)*.2)


class BasicEvaluation:

    def __init__(self):
        pass

    def value_white(self, board):
        value_white_pieces = value_base(board, chess.WHITE)
        value_black_pieces = value_base(board, chess.BLACK)
        value_white_pieces += add_pawns_value_white(board)
        value_black_pieces += add_pawns_value_black(board)

        return value_white_pieces - value_black_pieces
