"""_
Module for the BoardEvaluation class and the PointOfView enumeration.
"""

from enum import Enum


class PointOfView(str, Enum):
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

    WHITE = "white"
    BLACK = "black"
    PLAYER_TO_MOVE = "player_to_move"
    NOT_PLAYER_TO_MOVE = "not_player_to_move"
