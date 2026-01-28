"""
Module collecting the progress of computing moves by each player
"""

from dataclasses import dataclass, field
from typing import Protocol

from valanga import Color

from chipiron.displays.gui_protocol import UpdPlayerProgress
from chipiron.displays.gui_publisher import GuiPublisher


class PlayerProgressCollectorP(Protocol):
    """
    Object defining the protocol for setting the progress values
    """

    def progress_white(self, value: int | None) -> None: ...

    def progress_black(self, value: int | None) -> None: ...


@dataclass
class PlayerProgressCollector:
    """
    Object in charge of collecting the progress of computing moves by each player
    """

    progress_white_: int | None = None
    progress_black_: int | None = None

    @property
    def progress_white(self) -> int | None:
        """Gets the progress of the white player.

        Returns:
            int | None: The progress of the white player.
        """
        return self.progress_white_

    @progress_white.setter
    def progress_white(self, value: int | None) -> None:
        """Sets the progress of the white player.

        Args:
            value (int | None): The progress of the white player.
        """
        self.progress_white_ = value

    @property
    def progress_black(self) -> int | None:
        """Gets the progress of the black player.

        Returns:
            int | None: The progress of the black player.
        """
        return self.progress_black_

    @progress_black.setter
    def progress_black(self, value: int | None) -> None:
        """Sets the progress of the black player.

        Args:
            value (int | None): The progress of the black player.
        """
        self.progress_black_ = value


def make_publishers() -> list[GuiPublisher]:
    return []


@dataclass(slots=True)
class PlayerProgressCollectorObservable(PlayerProgressCollectorP):
    """Collects progress and publishes GUI payloads."""

    publishers: list[GuiPublisher] = field(default_factory=make_publishers)
    progress_collector: PlayerProgressCollector = field(
        default_factory=PlayerProgressCollector
    )

    def progress_white(self, value: int | None) -> None:
        self.progress_collector.progress_white = value
        self._publish(color=Color.WHITE, value=value)

    def progress_black(self, value: int | None) -> None:
        self.progress_collector.progress_black = value
        self._publish(color=Color.BLACK, value=value)

    # If you still receive chess.Color from elsewhere, keep this helper:
    def progress_for_chess_color(self, color: Color, value: int | None) -> None:
        if color == Color.WHITE:
            self.progress_collector.progress_white = value
        else:
            self.progress_collector.progress_black = value
        self._publish(color=color, value=value)

    def _publish(self, color: Color, value: int | None) -> None:
        payload = UpdPlayerProgress(
            player_color=color,
            progress_percent=value,
        )
        for pub in self.publishers:
            pub.publish(payload)
