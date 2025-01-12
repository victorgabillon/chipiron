"""
TreeNodeFactory Protocol
"""

from typing import Any, Protocol

import chipiron.environments.chess.board as board_mod
import chipiron.environments.chess.board as boards
import chipiron.players.move_selector.treevalue.nodes as node
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode


class TreeNodeFactory[T: ITreeNode[Any]](Protocol):
    """
    Interface for Tree Node Factories
    """

    def create(
        self,
        board: boards.IBoard,
        half_move: int,
        count: int,
        parent_node: node.ITreeNode[Any] | None,
        move_from_parent: moveKey | None,
        modifications: board_mod.BoardModificationP | None,
    ) -> node.ITreeNode[T]:
        """
        Creates a new TreeNode object.

        Args:
            board (boards.BoardChi): The current state of the chess board.
            half_move (int): The number of half moves made so far in the game.
            count (int): The number of times this position has occurred in the game.
            parent_node (node.ITreeNode | None): The parent node of the new TreeNode.
            move_from_parent (chess.Move | None): The move that led to this position, if any.
            modifications (board_mod.BoardModification | None): The modifications made to the board, if any.

        Returns:
            node.TreeNode: The newly created TreeNode object.
        """
        ...
