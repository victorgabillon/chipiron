"""
TreeNodeFactory Protocol
"""
from typing import Protocol
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
import chipiron.environments.chess.board as boards
import chipiron.environments.chess.board as board_mod


class TreeNodeFactory(Protocol):
    """
    Interface for Tree Node Factories
    """

    def create(
            self,
            board: boards.BoardChi,
            half_move: int,
            count: int,
            parent_node: ITreeNode,
            modifications: board_mod.BoardModification
    ) -> ITreeNode:
        """
        The main method to create a Tree Node
        Args:
            modifications:
            board: a board
            half_move: its half move
            count: the number of the node in the tree
            parent_node: the parent node

        Returns: a Tree Node

        """
        ...
