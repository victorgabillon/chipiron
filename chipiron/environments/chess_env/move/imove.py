from abc import abstractmethod
from typing import Protocol

from .utils import moveUci

# numbering scheme for actions in the node of the trees
moveKey = int


class IMove(Protocol):

    @abstractmethod
    def uci(self) -> moveUci: ...
