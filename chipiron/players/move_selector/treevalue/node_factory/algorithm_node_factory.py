""""
AlgorithmNodeFactory
"""
from chipiron.players.move_selector.treevalue.nodes.node_minmax_evaluation import NodeMinmaxEvaluation
import chipiron.players.move_selector.treevalue.indices.node_indices as node_indices
import chipiron.players.move_selector.treevalue.node_factory as node_fac
import chipiron.players.move_selector.treevalue.nodes as node
import chipiron.environments.chess.board as board_mod
from chipiron.players.boardevaluators.neural_networks.input_converters.board_representation import BoardRepresentation
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory
from dataclasses import dataclass


@dataclass
class AlgorithmNodeFactory:
    """
    The classe creating Algorithm Nodes
    """
    tree_node_factory: node_fac.Base
    board_representation_factory: Representation364Factory | None
    exploration_index_data_create: node_indices.ExplorationIndexDataFactory



    def create(
            self,
            board : board_mod.BoardChi,
            half_move: int,
            count: int,
            parent_node: node.AlgorithmNode | None,
            modifications: board_mod.BoardModification | None
    ) -> node.AlgorithmNode:
        tree_node: node.TreeNode = self.tree_node_factory.create(
            board=board,
            half_move=half_move,
            count=count,
            parent_node=parent_node,
            modifications=modifications
        )
        minmax_evaluation: NodeMinmaxEvaluation = NodeMinmaxEvaluation(tree_node=tree_node)

        exploration_index_data: NodeExplorationData = self.exploration_index_data_create(tree_node)

        board_representation: BoardRepresentation | None = None
        if self.board_representation_factory is not None:
            board_representation: BoardRepresentation = self.board_representation_factory.create_from_transition(
                tree_node=tree_node,
                parent_node=parent_node,
                modifications=modifications
            )

        return node.AlgorithmNode(
            tree_node=tree_node,
            minmax_evaluation=minmax_evaluation,
            exploration_index_data=exploration_index_data,
            board_representation=board_representation
        )
