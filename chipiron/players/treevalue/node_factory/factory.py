from typing import Protocol
from chipiron.players.treevalue.nodes.tree_node import TreeNode


class TreeNodeFactory(Protocol):

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node,
               board_depth: int) -> TreeNode:
        ...
