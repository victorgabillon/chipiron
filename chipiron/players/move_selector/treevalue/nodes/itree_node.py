from __future__ import annotations  # (helping with recursive type annotation)

from typing import Protocol

import chess
from bidict import bidict

from chipiron.environments.chess.board.board import BoardChi


class ITreeNode(Protocol):

    @property
    def id(
            self
    ) -> int:
        ...

    # actually giving access to the boars gives access to a lot of sub fucntion so might
    # be no need to ask for them in the interfacec expicitly
    @property
    def board(
            self
    ) -> BoardChi:
        ...

    @property
    def half_move(self) -> int:
        ...

    @property
    def moves_children(
            self
    ) -> bidict[chess.Move, ITreeNode | None]:
        ...

    @property
    def parent_nodes(
            self
    ) -> set[ITreeNode]:
        ...

    def add_parent(
            self,
            new_parent_node: ITreeNode
    ) -> None:
        ...

    def dot_description(self) -> str:
        ...

    @property
    def all_legal_moves_generated(self) -> bool:  # todo looks not clean, more like a hack no?
        ...

    @all_legal_moves_generated.setter
    def all_legal_moves_generated(self) -> None:  # todo looks not clean, more like a hack no?
        ...

    @property
    def legal_moves(self) -> chess.LegalMoveGenerator:
        ...

    @property
    def fast_rep(self) -> str:
        ...

    def is_over(self) -> bool:
        ...
