"""
Module collecting the progress of computing moves by each player
"""

import queue
from dataclasses import dataclass, field
from typing import Protocol

import chess

from chipiron.utils.communication.gui_messages import PlayerProgressMessage
from chipiron.utils.dataclass import IsDataclass


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
        return self.progress_white_

    @progress_white.setter
    def progress_white(self, value: int | None) -> None:
        self.progress_white_ = value

    @property
    def progress_black(self) -> int | None:
        return self.progress_black_

    @progress_black.setter
    def progress_black(self, value: int | None) -> None:
        self.progress_black_ = value


@dataclass
class PlayerProgressCollectorObservable:
    """
    Object in charge of collecting the progress of computing moves by each player
    """

    progress_collector: PlayerProgressCollector = field(
        default_factory=PlayerProgressCollector
    )
    subscribers: list[queue.Queue[IsDataclass]] = field(default_factory=list)

    def progress_white(self, value: int | None) -> None:
        self.progress_collector.progress_white = value
        self.notify(color=chess.WHITE, value=value)

    def progress_black(self, value: int | None) -> None:
        self.progress_collector.progress_black = value
        self.notify(color=chess.BLACK, value=value)

    def notify(self, color: chess.Color, value: int | None) -> None:
        for subscriber in self.subscribers:
            subscriber.put(
                PlayerProgressMessage(player_color=color, progress_percent=value)
            )
