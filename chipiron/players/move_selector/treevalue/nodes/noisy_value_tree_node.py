import chess

from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode
from chipiron.players.move_selector.treevalue.nodes.tree_node import TreeNode


class NoisyValueTreeNode(TreeNode):

    def __init__(
            self,
            board: BoardChi,
            half_move: int,
            id_number: int,
            parent_node: ITreeNode,
            last_move: chess.Move
    ) -> None:
        # super(TreeNode).__init__(board, half_move, id_number, parent_node, last_move)
        self.number_of_samples = 0
        self.variance = 0

    def test(self) -> None:
        super().test()

    def dot_description(self) -> str:
        super_description: str = super().dot_description()
        return super_description + '\n num_sampl: ' + str(self.number_of_samples) + '\n variance: ' + str(self.variance)
