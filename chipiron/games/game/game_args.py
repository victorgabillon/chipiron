"""
Module that contains the GameArgs class.
"""

from dataclasses import dataclass

from chipiron.environments.chess.board.starting_position import AllStartingPositionArgs


@dataclass
class GameArgs:
    """
    Represents the arguments for a game.

    Attributes:
        starting_position (AllStartingPositionArgs): The starting position of the game.
        max_half_moves (int | None, optional): The maximum number of half moves allowed in the game. Defaults to None.
        each_player_has_its_own_thread (bool, optional): Whether each player has its own thread. Defaults to False.
    """

    starting_position: AllStartingPositionArgs
    max_half_moves: int | None = None
    each_player_has_its_own_thread: bool = False
