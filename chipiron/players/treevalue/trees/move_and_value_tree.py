import chipiron.chessenvironment.board as board_mod
from chipiron.players.treevalue.nodes.tree_node import TreeNode

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
    """

    def __init__(self,
                 board_evaluator,
                 starting_board: board_mod.IBoard = None) -> None:
        """

        Args:
            board_evaluator (object):
        """
        if starting_board is not None:  # for tree visualizer...
            # the tree is built at half_move  self.half_move
            self.tree_root_half_move = starting_board.ply()

        # number of nodes in the tree
        self.nodes_count = 0


        # integer counting the number of moves in the tree.
        # the interest of self.move_count over the number of nodes in the descendants
        # is that is always increasing at each opening,
        # while self.node_count can stay the same if the nodes already existed.
        self.move_count = 0

        self.board_evaluator = board_evaluator

        # to be defined later ...
        self._root_node = None
        self.descendants = None

    @property
    def root_node(self):
        return self._root_node

    def node_depth(self, node: TreeNode) -> int:
        return node.half_move - self.tree_root_half_move





