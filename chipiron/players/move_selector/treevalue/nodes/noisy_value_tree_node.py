"""
This module contains the NoisyValueTreeNode class, which is a subclass of TreeNode.
"""

from typing import Any

import chess

from chipiron.environments.chess.board.board_chi import BoardChi
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


class NoisyValueTreeNode(TreeNode[Any]):
    """
    A class representing a node in a noisy value tree.
    Inherits from TreeNode.
    """

    def __init__(
        self,
        board: BoardChi,
        half_move: int,
        id_number: int,
        parent_node: ITreeNode[Any],
        last_move: chess.Move,
    ) -> None:
        """
        Initializes a NoisyValueTreeNode object.

        Args:
            board (BoardChi): The chess board.
            half_move (int): The half move number.
            id_number (int): The ID number of the node.
            parent_node (ITreeNode): The parent node.
            last_move (chess.Move): The last move made.

        """
        # super(TreeNode).__init__(board, half_move, id_number, parent_node, last_move)
        self.number_of_samples = 0
        self.variance = 0

    def test(self) -> None:
        """
        Performs a test.

        """
        super().test()

    def dot_description(self) -> str:
        """
        Returns the dot description of the node.

        Returns:
            str: The dot description.

        """
        super_description: str = super().dot_description()
        return (
            super_description
            + "\n num_sampl: "
            + str(self.number_of_samples)
            + "\n variance: "
            + str(self.variance)
        )
