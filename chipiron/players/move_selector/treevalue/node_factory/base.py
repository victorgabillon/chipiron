"""
Basic class for Creating Tree nodes
"""
from chipiron.players.move_selector.treevalue.node_factory.factory import TreeNodeFactory
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode, ITreeNode
import chipiron.environments.chess.board as boards


class Base(TreeNodeFactory):
    """
    Basic class for Creating Tree nodes
    """

    def create(self,
               board: boards.BoardChi,
               half_move: int,
               count: int,
               parent_node: ITreeNode
               ) -> TreeNode:
        """
        creating a Tree Node
        """
        tree_node: TreeNode = TreeNode(
            board_=board,
            half_move_=half_move,
            id=count,
            parent_nodes=[parent_node],
        )
        return tree_node
