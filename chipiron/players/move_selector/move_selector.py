from typing import Protocol
import environments.chess.board as boards
from enum import Enum
from dataclasses import dataclass
import players.move_selector.treevalue as treevalue
from treevalue.

AllMoveSelectorArgs = treevalue.TreeAndValuePlayerArgs


class MoveSelectorType(Enum):
    RandomPlayer: str = 'RandomPlayer'
    TreeAndValue: str = 'TreeAndValue'
    Stockfish: str = 'Stockfish'
    Human: str = 'Human'


@dataclass
class MoveSelectorArgs:
    type: MoveSelectorType


class MoveSelector(Protocol):

    def select_move(self, board: boards.BoardChi):
        ...
