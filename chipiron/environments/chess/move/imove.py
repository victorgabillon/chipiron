from abc import abstractmethod
from typing import Protocol


class IMove(Protocol):

    @abstractmethod
    def uci(self) -> str:
        ...


