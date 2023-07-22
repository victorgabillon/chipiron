""""
AlgorithmNodeFactory
"""
from chipiron.players.treevalue.nodes.node_minmax_evaluation import NodeMinmaxEvaluation
from chipiron.players.treevalue.nodes.node_exploration_manager import NodeExplorationManager
import chipiron.players.treevalue.node_factory as node_fac
import chipiron.players.treevalue.nodes as node
import chipiron.environments.chess.board as board_mod
from players.boardevaluators.neural_networks.input_converters.board_representation import BoardRepresentation
from players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory


class AlgorithmNodeFactory:
    """
    The classe creating Algorithm Nodes
    """
    tree_node_factory: node_fac.TreeNodeFactory
    board_representation_factory: Representation364Factory

    def __init__(self,
                 tree_node_factory: node_fac.TreeNodeFactory,
                 board_representation_factory: Representation364Factory
                 ) -> None:
        self.tree_node_factory = tree_node_factory
        self.board_representation_factory = board_representation_factory

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node: node.AlgorithmNode,
               board_depth: int,
               modifications: board_mod.BoardModification
               ) -> node.AlgorithmNode:
        tree_node: node.TreeNode = self.tree_node_factory.create(
            board=board,
            half_move=half_move,
            count=count,
            parent_node=parent_node,
        )
        minmax_evaluation: NodeMinmaxEvaluation = NodeMinmaxEvaluation(tree_node=tree_node)
        exploration_manager: NodeExplorationManager = NodeExplorationManager(tree_node=tree_node)
        board_representation: BoardRepresentation = self.board_representation_factory.create_from_transition(
            tree_node=tree_node,
            parent_node=parent_node,
            modifications=modifications
        )

        return node.AlgorithmNode(
            tree_node=tree_node,
            minmax_evaluation=minmax_evaluation,
            exploration_manager=exploration_manager,
            board_representation=board_representation
        )
