"""
Defines the IMove interface for chess moves.
"""

from abc import abstractmethod
from typing import Protocol

from .utils import moveUci

# numbering scheme for actions in the node of the trees
moveKey = int


class IMove(Protocol):
    """Interface for a chess move.

    Args:
        Protocol (Protocol): Protocol for type checking.
    """

    @abstractmethod
    def uci(self) -> moveUci:
        """Returns the UCI string representation of the move.
        Returns: moveUci: The UCI string representation of the move.
        """
        ...
