"""
Module that contains the GameStatusMessage class.
"""

from dataclasses import dataclass

from chipiron.games.game.game_playing_status import PlayingStatus


@dataclass
class BackMessage:
    """
    Represents a message indicating a request to go back one move.
    """


@dataclass
class GameStatusMessage:
    """
    Represents a message containing the current game status.

    Attributes:
        status (PlayingStatus): The playing status of the game.
    """

    status: PlayingStatus
