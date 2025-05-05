"""
Module for the basic evaluation of a chess board.
"""

import math

import chess
from chess import Square

from chipiron.environments.chess.board import IBoard
from chipiron.players.boardevaluators.board_evaluator import BoardEvaluator


def value_base(board: IBoard, color: chess.Color) -> int:
    """Calculate the base value of the given board for the specified color.

    Args:
        board (BoardChi): The chess board.
        color (chess.Color): The color for which to calculate the value.

    Returns:
        int: The base value of the board for the specified color.
    """
    value_white_: int = (
        bin(board.pawns & board.occupied_color(color)).count("1")
        + bin(board.knights & board.occupied_color(color)).count("1") * 3
        + bin(board.bishops & board.occupied_color(color)).count("1") * 3
        + bin(board.rooks & board.occupied_color(color)).count("1") * 5
        + bin(board.queens & board.occupied_color(color)).count("1") * 9
    )
    return value_white_


def add_pawns_value_white(board: IBoard) -> float:
    """Calculate the additional value for white pawns based on their advancement.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The additional value for white pawns.
    """
    add_value: float = 0
    pawn: Square
    for pawn in chess.scan_forward(board.pawns & board.white):
        add_value += int((pawn - 8) / 8) / 50.0 * 1
    return add_value


def add_pawns_value_black(board: IBoard) -> float:
    """Calculate the value to be added for black pawns based on their position.

    This function calculates the value to be added for black pawns based on their position on the board.
    The value is determined by giving more value to the pawns that are advanced.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The value to be added for black pawns.
    """
    add_value: float = 0
    pawn: Square
    for pawn in chess.scan_forward(board.pawns & board.black):
        add_value += int((63 - pawn - 8) / 8) / 50.0 * 1
    return add_value


def value_white(board: IBoard) -> float:
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


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function of a given input.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 1 / (1 + math.exp(-x))


def value_player_to_move(board: IBoard) -> float:
    """Calculate the value of the player to move.

    This function calculates the value of the player to move based on the difference in piece values
    between the two players on the board.

    Args:
        board (BoardChi): The chess board.

    Returns:
        float: The value of the player to move.
    """
    value_white_pieces: int = value_base(board, chess.WHITE)
    value_black_pieces: int = value_base(board, chess.BLACK)
    # value_white_pieces += add_pawns_value_white(board)
    # value_black_pieces += add_pawns_value_black(board)
    value: float
    if board.turn == chess.WHITE:
        value = sigmoid((value_white_pieces - value_black_pieces) * 0.2)
    else:
        value = sigmoid((value_black_pieces - value_white_pieces) * 0.2)
    return value


class BasicEvaluation(BoardEvaluator):
    """A basic board evaluator that calculates the value of the board for the white player.

    Args:
        BoardEvaluator (type): The base class for board evaluators.

    Attributes:
        None

    Methods:
        value_white: Calculates the value of the board for the white player.

    """

    def value_white(self, board: IBoard) -> float:
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

        return (
            value_white_pieces - value_black_pieces
        )  # + 100 * board.chess_board.is_check() - 200 * queen_atta
