from typing import Protocol
from chipiron.players.treevalue.nodes.tree_node import TreeNode
from chipiron.players.treevalue.nodes.itree_node import ITreeNode
import chipiron.chessenvironment.board as board_mod


class TreeNodeFactory(Protocol):

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node: ITreeNode,
               board_depth: int,
               modifications: board_mod.BoardModification
               ) -> TreeNode:
        ...
