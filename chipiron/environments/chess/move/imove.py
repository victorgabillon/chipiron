from abc import abstractmethod
from typing import Protocol


moveKey = str| int

class IMove(Protocol):

    @abstractmethod
    def uci(self) -> str:
        ...
