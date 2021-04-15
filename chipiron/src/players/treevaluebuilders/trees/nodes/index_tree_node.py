from src.players.treevaluebuilders.trees.nodes.tree_node_with_values import TreeNodeWithValue


class IndexTreeNode(TreeNodeWithValue):

    def __init__(self, board, half_move, id_number, parent_node):
        super().__init__(board, half_move, id_number, parent_node)
        # the index is here to give information on how much we want to explore this node further
        self.index = None

    def test(self):
        super().test()


    def dot_description(self):
        super_description = super().dot_description()
        return super_description + '\n index: ' + str(self.index)
