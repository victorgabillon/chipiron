"""
MoveAndValueTree
"""
import chipiron.players.treevalue.nodes as nodes
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

    _root_node: nodes.AlgorithmNode
    descendants: RangedDescendants
    tree_root_half_move: int

    def __init__(
            self,
            root_node: nodes.AlgorithmNode,
            descendants: RangedDescendants
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
    def root_node(self):
        return self._root_node

    def node_depth(self, node: nodes.ITreeNode) -> int:
        return node.half_move - self.tree_root_half_move

    def is_over(self) -> bool:
        return self._root_node.is_over()
