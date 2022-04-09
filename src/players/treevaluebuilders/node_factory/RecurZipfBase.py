from src.players.treevaluebuilders.node_factory.factory import TreeNodeFactory
from src.players.treevaluebuilders.nodes.tree_node_with_values import TreeNodeWithValue
from src.players.treevaluebuilders.nodes.tree_node_with_descendants import NodeWithDescendants


class RecurZipfBase(TreeNodeFactory):
    def create(self, board, half_move, count, parent_node, board_depth):
        if board_depth == 0:
            return NodeWithDescendants(board, half_move, count, parent_node)
        else:
            return TreeNodeWithValue(board, half_move, count, parent_node)

    def update_after_node_creation(self, node, parent_node):
        node_depth = node.half_move - self.tree_root_half_move
        if node_depth >= 1:
            self.root_node.descendants.add_descendant(node)
