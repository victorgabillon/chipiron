from players.treevaluebuilders.trees.nodes.tree_node import TreeNode


class IndexTreeNode(TreeNode):

    def __init__(self, board, half_move, count, father_node):
        super().__init__(board, half_move, count, father_node)
        # the index is here to give information on how much we want to explore this node further
        self.index = None

    def test(self):
        super().test()


    def dot_description(self):
        super_description = super().dot_description()
        return super_description + '\n index: ' + str(self.index)
