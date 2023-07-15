"""
Basic class for Creating Tree nodes
"""
from chipiron.players.treevalue.node_factory.factory import TreeNodeFactory
from chipiron.players.treevalue.nodes.tree_node import TreeNode
import chipiron.chessenvironment.board as boards


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
