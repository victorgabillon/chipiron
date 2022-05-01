import chess
import src.chessenvironment.board as board_mod
from graphviz import Digraph
from src.players.treevalue.nodes.tree_node import TreeNode
from src.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
import pickle
from src.players.treevalue.trees.updates import UpdateInstructionsBatch
from src.players.treevalue.node_factory.factory import TreeNodeFactory


class IMoveAndValueTree:
    """
    This class is the interface for a MoveAndValueTree
    """

    def node_depth(self, node):
        ...

    def add_root_node(self, board):
        ...

    def create_tree_node(self,
                         board,
                         board_modifications,
                         half_move: int,
                         parent_node):
        ...

    def find_or_create_node(self, board: board_mod.IBoard,
                            modifications: board_mod.BoardModification,
                            half_move: int,
                            parent_node: TreeNode) -> TreeNode:
        ...

    def open_node_move(self, parent_node: TreeNodeWithValue, move: chess.Move) -> object:
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
