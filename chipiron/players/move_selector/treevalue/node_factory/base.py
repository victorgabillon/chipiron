"""
Basic class for Creating Tree nodes
"""
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode, ITreeNode
from chipiron.players.move_selector.treevalue.node_factory.node_factory import TreeNodeFactory
import chipiron.environments.chess.board as boards


class Base(TreeNodeFactory):
    """
    Basic class for Creating Tree nodes
    """

    def create(
            self,
            board: boards.BoardChi,
            half_move: int,
            count: int,
            parent_node: ITreeNode | None
    ) -> TreeNode:
        """
        creating a Tree Node

        Args:
            board:
            half_move:
            count:
            parent_node: the parent node that can be None if no parent which means it is the rootnode
        """

        parent_nodes: set[ITreeNode]
        if parent_node is None:
            parent_nodes = set()
        else:
            parent_nodes = {parent_node}

        tree_node: TreeNode = TreeNode(
            board_=board,
            half_move_=half_move,
            id=count,
            parent_nodes=parent_nodes,
        )
        return tree_node
