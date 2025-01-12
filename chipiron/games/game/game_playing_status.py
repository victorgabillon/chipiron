"""
Module containing the class GamePlaying
"""

from dataclasses import dataclass
from enum import Enum


class PlayingStatus(Enum):
    """
    Class containing the possible playing status of a game.
    """

    PLAY = 1
    PAUSE = 0


@dataclass
class GamePlayingStatus:
    """
    Object containing the playing status of a game
    and the board.
    """

    _status: PlayingStatus = PlayingStatus.PLAY

    @property
    def status(self) -> PlayingStatus:
        """
        Get the current playing status of the game.

        Returns:
            PlayingStatus: The current playing status.
        """
        return self._status

    @status.setter
    def status(self, new_status: PlayingStatus) -> None:
        """
        Set the playing status of the game.

        Args:
            new_status (PlayingStatus): The new playing status.
        """
        self._status = new_status

    def play(self) -> None:
        """
        Set the playing status to 'PLAY'.
        """
        self.status = PlayingStatus.PLAY

    def pause(self) -> None:
        """
        Set the playing status to 'PAUSE'.
        """
        self.status = PlayingStatus.PAUSE

    def is_paused(self) -> bool:
        """
        Check if the game is currently paused.

        Returns:
            bool: True if the game is paused, False otherwise.
        """
        return self.status == PlayingStatus.PAUSE

    def is_play(self) -> bool:
        """
        Check if the game is currently playing.

        Returns:
            bool: True if the game is playing, False otherwise.
        """
        return self.status == PlayingStatus.PLAY
