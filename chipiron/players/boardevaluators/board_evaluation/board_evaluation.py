from enum import Enum


class PointOfView(Enum):
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
        self.value_white = value_white
