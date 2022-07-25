from typing import List
from chipiron.players.treevalue.node_selector.node_selector import NodeSelector
import chipiron.players.treevalue.trees as trees
import chipiron.chessenvironment.board as chp_board
import chipiron.players.treevalue.nodes as nodes
import chess


class MoveAndValueTreeObservable:
    """
    This class makes an object of the class MoveAndValueTree observable by subscribers,
     notified whenever a node or a move is added.
    """

    def __init__(self,
                 move_and_vale_tree: trees.MoveAndValueTree,
                 subscribers: List[NodeSelector]) -> None:
        self.move_and_vale_tree = move_and_vale_tree
        self.subscribers = subscribers

    @property
    def root_node(self):
        return self.move_and_vale_tree.root_node

    # forward
    def node_depth(self, node):
        self.move_and_vale_tree.node_depth(node)

    def add_root_node(self, board):
        self.move_and_vale_tree.add_root_node(board=board)

    def create_tree_node(self,
                         board,
                         board_modifications,
                         half_move: int,
                         parent_node):
        self.move_and_vale_tree.create_tree_node(board=board,
                                                 board_modifications=board_modifications,
                                                 half_move=half_move,
                                                 parent_node=parent_node)
        print('tytytyty')
        raise
        self.notify_create_tree_node()

    def notify_create_tree_node(self):
        node_selector: NodeSelector
        for node_selector in self.subscribers:
            node_selector.update_after_node_creation()

    def find_or_create_node(self, board: chp_board.IBoard,
                            modifications: chp_board.BoardModification,
                            half_move: int,
                            parent_node: nodes.TreeNode) -> nodes.TreeNode:
        ...

    def open_node_move(self, parent_node: nodes.TreeNodeWithValue, move: chess.Move) -> object:
        ...

    def open_and_update(self,
                        opening_instructions_batch):  # set of nodes and moves to open
        ...

    def batch_opening(self, opening_instructions_batch):
        ...

    def update_backward(self, update_instructions_batch):
        ...

    def print_best_line(self):
        ...
