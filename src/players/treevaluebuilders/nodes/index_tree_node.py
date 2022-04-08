from src.players.treevaluebuilders.nodes.tree_node_with_values import TreeNodeWithValue


class IndexTreeNode(TreeNodeWithValue):

    def __init__(self, board, half_move, id_number, parent_node):
        super().__init__(board, half_move, id_number, parent_node)
        # the index is here to give information on how much we want to explore this node further
        self.index = None

    def test(self):
        super().test()

    def compute_index(self):
        a_parent  = self.parent_nodes[0]
        #print('###',self.id,self.half_move,a_parent)
        if a_parent is None:
            return 0
        else:
            return abs(a_parent.best_child().get_value_white()-self.get_value_white())\
                   + self.parent_nodes[0].compute_index()

    def dot_description(self):
        super_description = super().dot_description()
        return super_description + '\n index: ' + str(self.index)
