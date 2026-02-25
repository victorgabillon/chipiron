"""Module that defines an observable version of GamePlayingStatus."""

from chipiron.displays.gui_protocol import UpdGameStatus
from chipiron.displays.gui_publisher import GuiPublisher

from .game_playing_status import GamePlayingStatus, PlayingStatus


class ObservableGamePlayingStatus:
    """An observable version of GamePlayingStatus that notifies subscribers.

    Players and GUI can then decide what to do with this information.
    """

    game_playing_status: GamePlayingStatus
    _publishers: list[GuiPublisher]

    def __init__(self, game_playing_status: GamePlayingStatus) -> None:
        """Initialize the instance."""
        self.game_playing_status = game_playing_status
        self._publishers: list[GuiPublisher] = []

    def subscribe(self, publishers: list[GuiPublisher]) -> None:
        """Subscribe the given mailboxes to receive game status updates.

        Args:
            publishers (list[GuiPublisher]): The publishers to subscribe.

        """
        self._publishers += publishers

    @property
    def status(self) -> PlayingStatus:
        """Get the current playing status.

        Returns:
            PlayingStatus: The current playing status.

        """
        return self.game_playing_status.status

    @status.setter
    def status(self, new_status: PlayingStatus) -> None:
        """Set the playing status and notifies the subscribers.

        Args:
            new_status (PlayingStatus): The new playing status.

        """
        self.game_playing_status.status = new_status
        self.notify()

    def play(self) -> None:
        """Plays the game and notifies the subscribers."""
        self.game_playing_status.play()
        self.notify()

    def pause(self) -> None:
        """Pauses the game and notifies the subscribers."""
        self.game_playing_status.pause()
        self.notify()

    def is_paused(self) -> bool:
        """Check if the game is currently paused.

        Returns:
            bool: True if the game is paused, False otherwise.

        """
        return self.game_playing_status.is_paused()

    def is_play(self) -> bool:
        """Check if the game is currently being played.

        Returns:
            bool: True if the game is being played, False otherwise.

        """
        return self.game_playing_status.is_play()

    def notify(self) -> None:
        """Notifies the subscribers with the current game status."""
        print("notify observable game playing")
        payload = UpdGameStatus(status=self.game_playing_status.status)
        for pub in self._publishers:
            pub.publish(payload)
