"""_
Module for the BoardEvaluation class and the PointOfView enumeration.
"""

from dataclasses import dataclass
from enum import Enum

from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.over_event import OverEvent


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


@dataclass
class ForcedOutcome:
    """
    The class
    """

    # The forced outcome with optimal play by both sides.
    outcome: OverEvent

    # the line
    line: list[moveKey]


@dataclass
class FloatyBoardEvaluation:
    """
    The class to defines what is an evaluation of a board.
    By convention is it always evaluated from the view point of the white side.
    """

    # The evaluation value for the white side when the outcome is not certain. Typically, a float.
    # todo can we remove the None option?
    value_white: float | None


BoardEvaluation = FloatyBoardEvaluation | ForcedOutcome
