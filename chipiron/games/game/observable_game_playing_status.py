from typing import List
import queue
from chipiron.utils.communication.gui_messages import GameStatusMessage
from .game_playing_status import GamePlayingStatus, PlayingStatus


class ObservableGamePlayingStatus:
    """
    observable version of GamePlayingStatus that notifies subscribers
    players and gui, for instance, can then decide what to do with this info.
    """

    def __init__(
            self,
            game_playing_status: GamePlayingStatus
    ) -> None:
        self.game_playing_status = game_playing_status
        self._mailboxes: list[queue.Queue[GameStatusMessage]] = []

    def subscribe(
            self,
            mailboxes: list[queue.Queue[GameStatusMessage]]
    ) -> None:
        self._mailboxes += mailboxes

    @property
    def status(self):
        return self.game_playing_status.status

    @status.setter
    def status(
            self,
            new_status: PlayingStatus
    ) -> None:
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
        print('notify observable game playing')
        message: GameStatusMessage = GameStatusMessage(status=self.game_playing_status.status)
        for mailbox in self._mailboxes:
            mailbox.put(item=message)
