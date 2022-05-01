from src.players.treevalue.node_factory.factory import TreeNodeFactory
from src.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
from src.players.treevalue.nodes.tree_node_with_descendants import NodeWithDescendants


class RecurZipfBase(TreeNodeFactory):
    def create(self, board, half_move, count, parent_node, board_depth):
        if board_depth == 0:
            return NodeWithDescendants(board, half_move, count, parent_node)
        else:
            return TreeNodeWithValue(board, half_move, count, parent_node)
