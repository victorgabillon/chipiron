from abc import abstractmethod
from typing import Protocol

from .utils import moveUci

moveKey = int


class IMove(Protocol):

    @abstractmethod
    def uci(self) -> moveUci: ...
