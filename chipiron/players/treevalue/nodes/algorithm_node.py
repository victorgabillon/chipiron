from chipiron.players.treevalue.nodes.tree_node import TreeNode
from chipiron.players.treevalue.nodes.itree_node import ITreeNode
from .node_minmax_evaluation import NodeMinmaxEvaluation
from bidict import bidict
from players.boardevaluators.neural_networks.input_converters.board_representation import BoardRepresentation
from chipiron.players.treevalue.nodes.node_exploration_manager import NodeExplorationManager


class AlgorithmNode:
    """
    The generic Node used by the tree and value algorithm.
    It wraps tree nodes with values, minimax computation and exploration tools
    """

    tree_node: TreeNode
    minmax_evaluation: NodeMinmaxEvaluation
    exploration_manager: NodeExplorationManager
    board_representation: BoardRepresentation

    def __init__(self,
                 tree_node: TreeNode,
                 minmax_evaluation: NodeMinmaxEvaluation,
                 exploration_manager: NodeExplorationManager,
                 board_representation: BoardRepresentation) -> None:
        self.tree_node = tree_node
        self.minmax_evaluation = minmax_evaluation
        self.exploration_manager = exploration_manager
        self.board_representation = board_representation

    @property
    def player_to_move(self):
        return self.tree_node.player_to_move

    @property
    def id(self):
        return self.tree_node.id

    @property
    def half_move(self) -> int:
        return self.tree_node.half_move

    @property
    def fast_rep(self):
        return self.tree_node.fast_rep

    @property
    def moves_children(self) -> bidict:
        return self.tree_node.moves_children

    @property
    def parent_nodes(self):
        return self.tree_node.parent_nodes

    @property
    def board(self):
        return self.tree_node.board

    def is_over(self) -> bool:
        return self.minmax_evaluation.is_over()

    def add_parent(self, new_parent_node: ITreeNode):
        self.tree_node.add_parent(new_parent_node=new_parent_node)

    @property
    def all_legal_moves_generated(self):
            return self.tree_node.all_legal_moves_generated

    @property
    def non_opened_legal_moves(self):
            return self.tree_node.non_opened_legal_moves

