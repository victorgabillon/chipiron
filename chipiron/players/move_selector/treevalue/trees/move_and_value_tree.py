"""
MoveAndValueTree
"""

from typing import Any

import chipiron.players.move_selector.treevalue.nodes as nodes
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    BoardEvaluation,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode

from .descendants import RangedDescendants

# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999


class MoveAndValueTree:
    """
    This class defines the Tree that is builds out of all the combinations of moves given a starting board position.
    The root node contains the starting board.
    Each node contains a board and has as many children node as there are legal move in the board.
    A children node then contains the board that is obtained by playing a particular moves in the board of the parent
    node.

    It is  pointer to the root node with some counters and keeping track of descendants.
    """

    _root_node: AlgorithmNode
    descendants: RangedDescendants
    tree_root_half_move: int

    def __init__(
        self, root_node: AlgorithmNode, descendants: RangedDescendants
    ) -> None:
        """

        Args:
            board_evaluator (object):
        """
        self.tree_root_half_move = root_node.half_move

        # number of nodes in the tree (already one as we have the root node provided)
        self.nodes_count = 1

        # integer counting the number of moves in the tree.
        # the interest of self.move_count over the number of nodes in the descendants
        # is that is always increasing at each opening,
        # while self.node_count can stay the same if the nodes already existed.
        self.move_count = 0

        self._root_node = root_node
        self.descendants = descendants

    @property
    def root_node(self) -> AlgorithmNode:
        """
        Returns the root node of the move and value tree.

        Returns:
            AlgorithmNode: The root node of the move and value tree.
        """
        return self._root_node

    def node_depth(self, node: nodes.ITreeNode[Any]) -> int:
        """
        Calculates the depth of a given node in the tree.

        Args:
            node (nodes.ITreeNode): The node for which to calculate the depth.

        Returns:
            int: The depth of the node.
        """
        return node.half_move - self.tree_root_half_move

    def is_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self._root_node.is_over()

    def evaluate(self) -> BoardEvaluation:
        return self._root_node.minmax_evaluation.evaluate()
