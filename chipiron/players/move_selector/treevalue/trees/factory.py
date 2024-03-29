"""
MoveAndValueTreeFactory
"""

import chipiron.environments.chess.board as boards
import chipiron.players.move_selector.treevalue.node_factory as nod_fac
from chipiron.players.move_selector.treevalue.node_evaluator import NodeEvaluator, EvaluationQueries
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from .descendants import RangedDescendants
from .move_and_value_tree import MoveAndValueTree


class MoveAndValueTreeFactory:
    """
    MoveAndValueTreeFactory
    """
    node_factory: nod_fac.AlgorithmNodeFactory
    node_evaluator: NodeEvaluator

    def __init__(
            self,
            node_factory: nod_fac.AlgorithmNodeFactory,
            node_evaluator: NodeEvaluator
    ) -> None:
        """
        creates the tree factory
        Args:
            node_factory:
            node_evaluator:
        """
        self.node_factory = node_factory
        self.node_evaluator = node_evaluator

    def create(
            self,
            starting_board: boards.BoardChi
    ) -> MoveAndValueTree:
        """
        creates the tree
        Args:
            starting_board: the starting position

        Returns:

        """
        print()
        root_node: AlgorithmNode = self.node_factory.create(
            board=starting_board,
            half_move=starting_board.ply(),
            count=0,
            parent_node=None,
            modifications=None
        )

        evaluation_queries: EvaluationQueries = EvaluationQueries()

        self.node_evaluator.add_evaluation_query(
            node=root_node,
            evaluation_queries=evaluation_queries
        )

        self.node_evaluator.evaluate_all_queried_nodes(evaluation_queries=evaluation_queries)
        # is this needed? used outside?

        descendants: RangedDescendants = RangedDescendants()
        descendants.add_descendant(root_node)

        move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                                 descendants=descendants)

        return move_and_value_tree
