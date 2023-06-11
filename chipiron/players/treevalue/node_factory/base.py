from chipiron.players.treevalue.node_factory.factory import TreeNodeFactory
from chipiron.players.treevalue.nodes.tree_node import TreeNode


class Base(TreeNodeFactory):
    def create(self, board, half_move, count, parent_node, board_depth):
        tree_node = TreeNode(
            board=board,
            half_move=half_move,
            id_number=count,
            parent_node=parent_node)
        return tree_node
