from src.players.treevaluebuilders.nodes.tree_node import TreeNode
from src.players.treevaluebuilders.nodes.node_minmax_evaluation import NodeMinmaxEvaluation
from src.players.treevaluebuilders.nodes.node_exploration_manager import NodeExplorationManager


class AlgorithmNode:
    """
    The generic Node used by the tree and value algorithm.
    It wraps tree nodes with values, minimax computation and exploration tools
    """

    def __init__(self,
                 tree_node: TreeNode,
                 minmax_evaluation: object,
                 exploration_manager: object) -> None:
        self.tree_node = tree_node
        self.minmax_evaluation = minmax_evaluation
        self.exploration_manager = exploration_manager


class AlgorithmNodeFactory:
    """

    """

    def __init__(self,
                 minmax_evaluation: object,
                 exploration_manager: object) -> None:
        self.minmax_evaluation = minmax_evaluation
        self.exploration_manager = exploration_manager

    def create(self, tree_node: TreeNode) -> AlgorithmNode:
        minmax_evaluation: object = NodeMinmaxEvaluation(tree_node)
        exploration_manager: object = NodeExplorationManager(tree_node)

        return AlgorithmNode(tree_node=tree_node,
                             minmax_evaluation=minmax_evaluation,
                             exploration_manager=exploration_manager)
