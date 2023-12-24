"""
Basic class for Creating Tree nodes
"""
from players.move_selector.treevalue.node_factory.factory import TreeNodeFactory
from players.move_selector.treevalue.nodes.tree_node import TreeNode
import chipiron.environments.chess.board as boards


class Base(TreeNodeFactory):
    """
    Basic class for Creating Tree nodes
    """

    def create(self,
               board: boards.BoardChi,
               half_move: int,
               count: int,
               parent_node
               ) -> TreeNode:
        """
        creating a Tree Node
        """
        tree_node: TreeNode = TreeNode(
            board=board,
            half_move=half_move,
            id_number=count,
            parent_node=parent_node,
        )
        return tree_node
