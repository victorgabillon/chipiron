from enum import Enum
from dataclasses import dataclass
from typing import List
import queue
import copy


class PlayingStatus(Enum):
    PLAY = 0
    PAUSE = 1


@dataclass
class GamePlayingStatus:
    """
    Objet containing the playing status of a game
    """
    _status: PlayingStatus = PlayingStatus.PLAY

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status: PlayingStatus.PLAY):
        self._status = new_status

    def play(self):
        self.status = PlayingStatus.PLAY

    def pause(self):
        self.status = PlayingStatus.PAUSE

    def is_paused(self):
        return self.status == PlayingStatus.PAUSE

    def is_play(self):
        return self.status == PlayingStatus.PLAY


class ObservableGamePlayingStatus:

    def __init__(self, game_playing_status: GamePlayingStatus):
        self.game_playing_status = game_playing_status
        self._mailboxes = []

    def subscribe(self, mailboxes: List[queue.Queue]):
        self._mailboxes += mailboxes

    @property
    def status(self):
        return self.game_playing_status.status

    @status.setter
    def status(self, new_status: PlayingStatus.PLAY):
        self.game_playing_status.status = new_status
        self.notify()

    def play(self):
        self.game_playing_status.play()
        self.notify()

    def pause(self):
        self.game_playing_status.pause()
        self.notify()

    def is_paused(self):
        return self.game_playing_status.is_paused()

    def is_play(self):
        return self.game_playing_status.is_play()

    def notify(self):
        observable_copy = copy.copy(self.game_playing_status)
        message: dict = {
            'type': 'GamePlayingStatus',
            'GamePlayingStatus': observable_copy
        }
        for mailbox in self._mailboxes:
            mailbox.put(message)
