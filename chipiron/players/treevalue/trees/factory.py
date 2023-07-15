"""
MoveAndValueTreeFactory
"""
import chipiron as ch
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from .descendants import RangedDescendants
import chipiron.players.treevalue.node_factory as nod_fac
from chipiron.players.treevalue.tree_manager.node_evaluator import NodeEvaluator, EvaluationQueries
import chipiron.players.treevalue.nodes as nodes


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
            starting_board: ch.chess.BoardChi
    ) -> MoveAndValueTree:
        """
        creates the tree
        Args:
            starting_board: the starting position

        Returns:

        """
        root_node: nodes.AlgorithmNode = self.node_factory.create(
            board=starting_board,
            half_move=starting_board.ply(),
            count=0,
            parent_node=None,
            board_depth=0,
            modifications=None
        )

        evaluation_queries: EvaluationQueries = EvaluationQueries()

        self.node_evaluator.add_evaluation_query(
            node=root_node,
            evaluation_queries=evaluation_queries
        )

        self.node_evaluator.evaluate_all_queried_nodes(evaluation_queries=evaluation_queries)

        descendants: RangedDescendants = RangedDescendants()
        descendants.add_descendant(root_node)

        move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node,
                                                                 descendants=descendants)

        return move_and_value_tree
