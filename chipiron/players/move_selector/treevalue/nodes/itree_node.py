from __future__ import annotations  # To be removed in python 3.10 (helping with recursive type annocatation)
from typing import Protocol
from bidict import bidict


class ITreeNode(Protocol):

    @property
    def board(self):
        ...

    @property
    def half_move(self) -> int:
        ...

    @property
    def moves_children(self) -> bidict:
        ...

    def add_parent(self, new_parent_node: ITreeNode):
        ...
