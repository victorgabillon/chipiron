from typing import Protocol


class TreeNodeFactory(Protocol):

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node,
               board_depth: int):
        ...
