from dataclasses import dataclass
from enum import Enum


class PlayingStatus(Enum):
    PLAY: int = 1
    PAUSE: int = 0


@dataclass
class GamePlayingStatus:
    """
    Objet containing the playing status of a game
    and the board
    """
    _status: PlayingStatus = PlayingStatus.PLAY

    @property
    def status(self):
        return self._status

    @status.setter
    # what is the point?
    def status(
            self,
            new_status: PlayingStatus
    ):
        self._status = new_status

    def play(self):
        self.status = PlayingStatus.PLAY

    def pause(self):
        self.status = PlayingStatus.PAUSE

    def is_paused(self):
        return self.status == PlayingStatus.PAUSE

    def is_play(self):
        return self.status == PlayingStatus.PLAY
