"""
Basic class for Creating Tree nodes
"""
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.node_factory.node_factory import TreeNodeFactory
import chipiron.environments.chess.board as boards
import chipiron.environments.chess.board as board_mod

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
            modifications: board_mod.BoardModification
    ) -> TreeNode:
        """
        creating a Tree Node

        Args:
            modifications:
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
            id_=count,
            parent_nodes_=parent_nodes,
        )
        return tree_node
