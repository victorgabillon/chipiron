"""
TreeNodeFactory Protocol
"""
from typing import Protocol

import chipiron.environments.chess.board as board_mod
import chipiron.environments.chess.board as boards
import chipiron.players.move_selector.treevalue.nodes as node


class TreeNodeFactory(Protocol):
    """
    Interface for Tree Node Factories
    """

    def create(
            self,
            board: boards.BoardChi,
            half_move: int,
            count: int,
            parent_node: node.ITreeNode | None,
            modifications: board_mod.BoardModification | None
    ) -> node.ITreeNode:
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
