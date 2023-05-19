from typing import List
from src.players.treevalue.node_selector.node_selector import NodeSelector
import src.chessenvironment.board as chp_board
import src.players.treevalue.nodes as nodes
import chess
from .move_and_value_tree import MoveAndValueTreeBuilder


class MoveAndValueTreeBuilderObservable:
    """
    This class makes an object of the class MoveAndValueTree observable by subscribers,
     notified whenever a node or a move is added.
    """

    move_and_vale_tree: MoveAndValueTreeBuilder

    def __init__(self,
                 move_and_vale_tree: MoveAndValueTreeBuilder,
                 subscribers: List[NodeSelector]) -> None:
        self.move_and_vale_tree = move_and_vale_tree
        self.subscribers = subscribers


    def add_root_node(self, board, tree):
        self.move_and_vale_tree.add_root_node(board=board, tree=tree)

    def create_tree_node(self,
                         board,
                         board_modifications,
                         half_move: int,
                         parent_node):
        self.move_and_vale_tree.create_tree_node(board=board,
                                                 board_modifications=board_modifications,
                                                 half_move=half_move,
                                                 parent_node=parent_node)
        self.notify_create_tree_node()

    def notify_create_tree_node(self):
        node_selector: NodeSelector
        for node_selector in self.subscribers:
            node_selector.update_after_node_creation()

    def find_or_create_node(self,
                            tree,
                            board: chp_board.IBoard,
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
