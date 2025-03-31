""" "
AlgorithmNodeFactory
"""

from dataclasses import dataclass
from typing import Any

import chipiron.environments.chess.board as board_mod
import chipiron.players.move_selector.treevalue.indices.node_indices as node_indices
import chipiron.players.move_selector.treevalue.nodes as node
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.neural_networks.input_converters.board_representation import (
    BoardRepresentation,
    Representation364,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
)
from chipiron.players.move_selector.treevalue.node_factory.base import Base
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode


@dataclass
class AlgorithmNodeFactory:
    """
    The classe creating Algorithm Nodes
    """

    tree_node_factory: Base[Any]
    board_representation_factory: RepresentationFactory[Any] | None
    exploration_index_data_create: node_indices.ExplorationIndexDataFactory

    def create(
        self,
        board: board_mod.IBoard,
        half_move: int,
        count: int,
        parent_node: ITreeNode[Any] | None,
        move_from_parent: moveKey | None,
        modifications: board_mod.BoardModificationP | None,
    ) -> AlgorithmNode:
        """
        Creates an AlgorithmNode object.

        Args:
            move_from_parent (chess.Move | None): the move that led to the node from the parent node
            board: The board object.
            half_move: The half move count.
            count: The count.
            parent_node: The parent node object.
            modifications: The board modifications object.

        Returns:
            An AlgorithmNode object.

        """
        tree_node: node.TreeNode[Any] = self.tree_node_factory.create(
            board=board,
            half_move=half_move,
            count=count,
            move_from_parent=move_from_parent,
            parent_node=parent_node,
            modifications=modifications,
        )
        minmax_evaluation: NodeMinmaxEvaluation = NodeMinmaxEvaluation(
            tree_node=tree_node
        )

        exploration_index_data: node_indices.NodeExplorationData | None = (
            self.exploration_index_data_create(tree_node)
        )

        board_representation: BoardRepresentation | None = None
        if self.board_representation_factory is not None:
            assert isinstance(parent_node, AlgorithmNode) or parent_node is None
            parent_node_representation: Representation364 | None
            if parent_node is not None:
                assert isinstance(parent_node, AlgorithmNode)
                assert isinstance(parent_node.board_representation, Representation364)
                # todo remove all these ugly assert!!
                parent_node_representation = parent_node.board_representation
            else:
                parent_node_representation = None

            board_representation = (
                self.board_representation_factory.create_from_transition(
                    tree_node=tree_node,
                    parent_node_representation=parent_node_representation,
                    modifications=modifications,
                )
            )

        return AlgorithmNode(
            tree_node=tree_node,
            minmax_evaluation=minmax_evaluation,
            exploration_index_data=exploration_index_data,
            board_representation=board_representation,
        )
