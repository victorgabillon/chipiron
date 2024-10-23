from typing import Protocol


class IMove(Protocol):

    def uci(self) -> str:
        ...


