"""_
Module for the BoardEvaluation class and the PointOfView enumeration.
"""

from enum import Enum


class PointOfView(Enum):
    """Represents the point of view in a game.

    This enumeration defines the possible points of view in a game, including:
    - WHITE: Represents the white player's point of view.
    - BLACK: Represents the black player's point of view.
    - PLAYER_TO_MOVE: Represents the point of view of the player who is currently making a move.
    - NOT_PLAYER_TO_MOVE: Represents the point of view of the player who is not currently making a move.

    Attributes:
        WHITE (PointOfView): The white player's point of view.
        BLACK (PointOfView): The black player's point of view.
        PLAYER_TO_MOVE (PointOfView): The point of view of the player who is currently making a move.
        NOT_PLAYER_TO_MOVE (PointOfView): The point of view of the player who is not currently making a move.
    """
    WHITE = 0
    BLACK = 1
    PLAYER_TO_MOVE = 2
    NOT_PLAYER_TO_MOVE = 3



class BoardEvaluation:
    """
    The class to defines what is an evaluation of a board.
    By convention is it always evaluated from the view point of the white side.
    """

    def __init__(self, value_white: float) -> None:
        """
        Initializes a BoardEvaluation object.

        Args:
            value_white (float): The evaluation value for the white side.
        """
        self.value_white = value_white
