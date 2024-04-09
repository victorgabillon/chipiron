"""
Module for the basic evaluation of a chess board.
"""
import math

import chess

from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.boardevaluators.board_evaluator import BoardEvaluator


def value_base(
        board: BoardChi,
        color: chess.Color
) -> int:
    """Calculate the base value of the given board for the specified color.

    Args:
        board (BoardChi): The chess board.
        color (chess.Color): The color for which to calculate the value.

    Returns:
        int: The base value of the board for the specified color.
    """
    value_white_: int = bin(board.board.pawns & board.board.occupied_co[color]).count('1') \
                        + bin(board.board.knights & board.board.occupied_co[color]).count('1') * 3 \
                        + bin(board.board.bishops & board.board.occupied_co[color]).count('1') * 3 \
                        + bin(board.board.rooks & board.board.occupied_co[color]).count('1') * 5 \
                        + bin(board.board.queens & board.board.occupied_co[color]).count('1') * 9
    return value_white_


def add_pawns_value_white(
        board: BoardChi
) -> float:
    """Calculate the additional value for white pawns based on their advancement.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The additional value for white pawns.
    """
    add_value: float = 0
    for pawn in list(board.board.pieces(chess.PAWN, chess.WHITE)):
        add_value += int((pawn - 8) / 8) / 50. * 1
    return add_value


def add_pawns_value_black(
        board: BoardChi
) -> float:
    """Calculate the value to be added for black pawns based on their position.

    This function calculates the value to be added for black pawns based on their position on the board.
    The value is determined by giving more value to the pawns that are advanced.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The value to be added for black pawns.
    """
    add_value: float = 0
    for pawn in list(board.board.pieces(chess.PAWN, chess.BLACK)):
        add_value += int((63 - pawn - 8) / 8) / 50. * 1
    return add_value


def value_white(board: BoardChi) -> float:
    """Calculate the value of the white pieces on the board.

    This function calculates the value of the white pieces on the board by subtracting the value of the black pieces from the value of the white pieces.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The value of the white pieces on the board.
    """
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    return value_white_pieces - value_black_pieces


import math

def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function of a given input.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 1 / (1 + math.exp(-x))


def value_player_to_move(board: BoardChi) -> float:
    """Calculate the value of the player to move.

    This function calculates the value of the player to move based on the difference in piece values
    between the two players on the board.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The value of the player to move.
    """
    value_white_pieces = value_base(board, chess.WHITE)
    value_black_pieces = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    if board.board.turn == chess.WHITE:
        return sigmoid((value_white_pieces - value_black_pieces) * .2)
    else:
        return sigmoid((value_black_pieces - value_white_pieces) * .2)


class BasicEvaluation(BoardEvaluator):
    """A basic board evaluator that calculates the value of the board for the white player.

    Args:
        BoardEvaluator (type): The base class for board evaluators.

    Attributes:
        None

    Methods:
        value_white: Calculates the value of the board for the white player.

    """

    def __init__(self) -> None:
        """Initialize the BasicEvaluation object.

        This method initializes the BasicEvaluation object.

        Parameters:
            None

        Returns:
            None
        """
        pass

    def value_white(self, board: BoardChi) -> float:
        """Calculates the value of the board for the white player.

        Args:
            board (BoardChi): The chess board.

        Returns:
            float: The value of the board for the white player.

        """
        value_white_pieces: float = float(value_base(board, chess.WHITE))
        value_black_pieces: float = float(value_base(board, chess.BLACK))
        value_white_pieces += add_pawns_value_white(board)
        value_black_pieces += add_pawns_value_black(board)

        return value_white_pieces - value_black_pieces  # + 100 * board.chess_board.is_check() - 200 * queen_atta
