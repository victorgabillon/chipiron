from chipiron.players.treevalue.nodes.node_minmax_evaluation import NodeMinmaxEvaluation
from chipiron.players.treevalue.nodes.node_exploration_manager import NodeExplorationManager
import chipiron.players.treevalue.node_factory as node_fac
from chipiron.players.treevalue.nodes.tree_node import TreeNode
from chipiron.players.treevalue.nodes.algorithm_node import AlgorithmNode

class AlgorithmNodeFactory:
    """

    """
    tree_node_factory: node_fac.TreeNodeFactory

    def __init__(self,
                 tree_node_factory: node_fac.TreeNodeFactory
                 ) -> None:
        self.tree_node_factory = tree_node_factory


    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node: AlgorithmNode,
               board_depth: int
               ) -> AlgorithmNode:
        tree_node: TreeNode = self.tree_node_factory.create(board=board,
                                                            half_move=half_move,
                                                            count=count,
                                                            parent_node=parent_node,
                                                            board_depth=board_depth)
        minmax_evaluation: NodeMinmaxEvaluation = NodeMinmaxEvaluation(tree_node)
        exploration_manager: object = NodeExplorationManager(tree_node)
        return AlgorithmNode(tree_node=tree_node,
                             minmax_evaluation=minmax_evaluation,
                             exploration_manager=exploration_manager)
