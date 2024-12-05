from abc import abstractmethod
from typing import Protocol

moveKey = int


class IMove(Protocol):

    @abstractmethod
    def uci(self) -> str:
        ...
