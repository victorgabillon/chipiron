"""
Module that defines an observable version of GamePlayingStatus
"""

import queue

from chipiron.utils.communication.gui_messages import GameStatusMessage

from .game_playing_status import GamePlayingStatus, PlayingStatus


class ObservableGamePlayingStatus:
    """
    An observable version of GamePlayingStatus that notifies subscribers.
    Players and GUI can then decide what to do with this information.
    """

    def __init__(self, game_playing_status: GamePlayingStatus) -> None:
        self.game_playing_status = game_playing_status
        self._mailboxes: list[queue.Queue[GameStatusMessage]] = []

    def subscribe(self, mailboxes: list[queue.Queue[GameStatusMessage]]) -> None:
        """
        Subscribes the given mailboxes to receive game status updates.

        Args:
            mailboxes (list[queue.Queue[GameStatusMessage]]): The mailboxes to subscribe.
        """
        self._mailboxes += mailboxes

    @property
    def status(self) -> PlayingStatus:
        """
        Gets the current playing status.

        Returns:
            PlayingStatus: The current playing status.
        """
        return self.game_playing_status.status

    @status.setter
    def status(self, new_status: PlayingStatus) -> None:
        """
        Sets the playing status and notifies the subscribers.

        Args:
            new_status (PlayingStatus): The new playing status.
        """
        self.game_playing_status.status = new_status
        self.notify()

    def play(self) -> None:
        """
        Plays the game and notifies the subscribers.
        """
        self.game_playing_status.play()
        self.notify()

    def pause(self) -> None:
        """
        Pauses the game and notifies the subscribers.
        """
        self.game_playing_status.pause()
        self.notify()

    def is_paused(self) -> bool:
        """
        Checks if the game is currently paused.

        Returns:
            bool: True if the game is paused, False otherwise.
        """
        return self.game_playing_status.is_paused()

    def is_play(self) -> bool:
        """
        Checks if the game is currently being played.

        Returns:
            bool: True if the game is being played, False otherwise.
        """
        return self.game_playing_status.is_play()

    def notify(self) -> None:
        """
        Notifies the subscribers with the current game status.
        """
        print("notify observable game playing")
        message: GameStatusMessage = GameStatusMessage(
            status=self.game_playing_status.status
        )
        for mailbox in self._mailboxes:
            mailbox.put(item=message)
