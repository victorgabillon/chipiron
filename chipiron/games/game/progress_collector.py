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

    def progress_white(self, value: float | None):
        ...

    def progress_black(self, value: float | None):
        ...


@dataclass
class PlayerProgressCollector:
    """
    Object in charge of collecting the progress of computing moves by each player
    """

    progress_white_: float | None = None
    progress_black_: float | None = None

    @property
    def progress_white(self):
        return self.progress_white_

    @property
    def progress_black(self):
        return self.progress_black_

    @progress_white.setter
    def progress_white(self, value: float | None):
        self.progress_white_ = value

    @progress_black.setter
    def progress_black(self, value: float | None):
        self.progress_black_ = value


@dataclass
class PlayerProgressCollectorObservable:
    """
    Object in charge of collecting the progress of computing moves by each player
    """

    progress_collector: PlayerProgressCollectorP = field(default_factory=PlayerProgressCollector)
    subscribers: list[queue.Queue[IsDataclass]] = field(default_factory=list)

    def progress_white(self, value: float | None):

        self.progress_collector.progress_white = value
        self.notify(color=chess.WHITE, value=value)

    def progress_black(self, value: float | None):
        self.progress_collector.progress_black = value
        self.notify(color=chess.BLACK, value=value)

    def notify(self, color: chess.Color, value: float | None):
        for subscriber in self.subscribers:
            subscriber.put(
                item=PlayerProgressMessage(
                    player_color=color,
                    progress_percent=value
                )
            )
