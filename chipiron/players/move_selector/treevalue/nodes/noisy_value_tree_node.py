from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


class NoisyValueTreeNode(TreeNode):

    def __init__(self, board, half_move, id_number, parent_node, last_move):
        super().__init__(board, half_move, id_number, parent_node, last_move)
        self.number_of_samples = 0
        self.variance = 0

    def test(self) -> None:
        super().test()

    def dot_description(self) -> str:
        super_description: str = super().dot_description()
        return super_description + '\n num_sampl: ' + str(self.number_of_samples) + '\n variance: ' + str(self.variance)
