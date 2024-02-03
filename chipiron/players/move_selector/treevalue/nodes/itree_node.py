from __future__ import annotations  # To be removed in python 3.10 (helping with recursive type annocatation)
from typing import Protocol
from bidict import bidict
from chipiron.environments.chess.board.board import BoardChi


class ITreeNode(Protocol):

    @property
    def id(self) -> int:
        ...

    @property
    def board(self) -> BoardChi:
        ...

    @property
    def half_move(self) -> int:
        ...

    @property
    def moves_children(self) -> bidict:
        ...

    @property
    def parent_nodes(self) -> set[ITreeNode]:
        ...

    def add_parent(self, new_parent_node: ITreeNode):
        ...
