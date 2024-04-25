"""
Basic class for Creating Tree nodes
"""
import chess

import chipiron.environments.chess.board as board_mod
import chipiron.environments.chess.board as boards
from chipiron.players.move_selector.treevalue.node_factory.node_factory import TreeNodeFactory
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


class Base(TreeNodeFactory):
    """
    Basic class for Creating Tree nodes
    """

    def create(
            self,
            board: boards.BoardChi,
            half_move: int,
            count: int,
            parent_node: ITreeNode | None,
            move_from_parent: chess.Move | None,
            modifications: board_mod.BoardModification | None
    ) -> TreeNode:
        """
        Creates a new TreeNode object.

        Args:
            board (boards.BoardChi): The current board state.
            half_move (int): The half-move count.
            count (int): The ID of the new node.
            parent_node (ITreeNode | None): The parent node of the new node.
            move_from_parent (chess.Move | None): The move that leads to the new node.
            modifications (board_mod.BoardModification | None): The modifications applied to the board.

        Returns:
            TreeNode: The newly created TreeNode object.
        """

        parent_nodes: dict[ITreeNode, chess.Move]
        if parent_node is None:
            parent_nodes = {}
        else:
            assert move_from_parent is not None
            parent_nodes = {parent_node: move_from_parent}

        tree_node: TreeNode = TreeNode(
            board_=board,
            half_move_=half_move,
            id_=count,
            parent_nodes_=parent_nodes,
        )
        return tree_node
