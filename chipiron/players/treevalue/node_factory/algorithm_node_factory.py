from chipiron.players.treevalue.nodes.node_minmax_evaluation import NodeMinmaxEvaluation
from chipiron.players.treevalue.nodes.node_exploration_manager import NodeExplorationManager
import chipiron.players.treevalue.node_factory as node_fac
import chipiron.players.treevalue.nodes as node
import chipiron.chessenvironment.board as board_mod
from chipiron.players.boardevaluators.board_representations.board_representation import BoardRepresentation


class AlgorithmNodeFactory:
    """

    """
    tree_node_factory: node_fac.TreeNodeFactory
    board_representation_factory: object

    def __init__(self,
                 tree_node_factory: node_fac.TreeNodeFactory,
                 board_representation_factory: object
                 ) -> None:
        self.tree_node_factory = tree_node_factory
        self.board_representation_factory = board_representation_factory

    def create(self,
               board,
               half_move: int,
               count: int,
               parent_node: node.ITreeNode,
               board_depth: int,
               modifications: board_mod.BoardModification
               ) -> node.AlgorithmNode:
        tree_node: node.TreeNode = self.tree_node_factory.create(
            board=board,
            half_move=half_move,
            count=count,
            parent_node=parent_node,
            board_depth=board_depth,
            modifications=modifications
        )
        minmax_evaluation: NodeMinmaxEvaluation = NodeMinmaxEvaluation(tree_node=tree_node)
        exploration_manager: NodeExplorationManager = NodeExplorationManager(tree_node=tree_node)
        board_representation: BoardRepresentation = self.board_representation_factory.create(
            tree_node=tree_node,
            parent_node=parent_node,
            modifications=modifications
        )

        return node.AlgorithmNode(tree_node=tree_node,
                                  minmax_evaluation=minmax_evaluation,
                                  exploration_manager=exploration_manager,
                                  board_representation=board_representation)
