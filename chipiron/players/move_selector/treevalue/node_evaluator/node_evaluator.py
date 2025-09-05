"""
This module contains the implementation of the NodeEvaluator class, which is responsible for evaluating the value of
 nodes in a tree-based move selector.

The NodeEvaluator class wraps a board evaluator and a syzygy table to provide more complex evaluations of chess
 positions. It handles queries for evaluating nodes and manages obvious over events.

Classes:
- NodeEvaluator: Wrapping node evaluator with syzygy and obvious over event.

Enums:
- NodeEvaluatorTypes: Types of node evaluators.

Constants:
- DISCOUNT: Discount factor used in the evaluation.

Functions:
- None

"""

from enum import Enum
from typing import TYPE_CHECKING

from chipiron.players.boardevaluators.master_board_evaluator import MasterBoardEvaluator
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

if TYPE_CHECKING:
    from chipiron.players.boardevaluators.over_event import OverEvent

DISCOUNT = 0.99999999  # lokks like at the moment the use is to break ties in the evaluation (not sure if needed or helpful now)


class NodeEvaluatorTypes(str, Enum):
    """
    Enum class representing different types of node evaluators.
    """

    NEURAL_NETWORK = "neural_network"


class EvaluationQueries:
    """
    A class that represents evaluation queries for algorithm nodes.

    Attributes:
        over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are considered "over".
        not_over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are not considered "over".
    """

    over_nodes: list[AlgorithmNode]
    not_over_nodes: list[AlgorithmNode]

    def __init__(self) -> None:
        """
        Initializes a new instance of the NodeEvaluator class.
        """
        self.over_nodes = []
        self.not_over_nodes = []

    def clear_queries(self) -> None:
        """
        Clears the evaluation queries by resetting the over_nodes and not_over_nodes lists.
        """
        self.over_nodes = []
        self.not_over_nodes = []


class NodeEvaluator:
    """
    The NodeEvaluator class is responsible for evaluating the value of nodes in a tree structure.
    It uses a board evaluator and a syzygy evaluator to calculate the value of the nodes.
    """

    master_board_evaluator: MasterBoardEvaluator

    def __init__(
        self,
        master_board_evaluator: MasterBoardEvaluator,
    ) -> None:
        """
        Initializes a NodeEvaluator object.

        Args:
            board_evaluator (MasterBoardEvaluator): The board evaluator used to evaluate the chess board.
        """
        self.master_board_evaluator = master_board_evaluator

    def check_obvious_over_events(self, node: AlgorithmNode) -> None:
        """
        Updates the node.over object if the game is obviously over.
        """
        over_event: OverEvent | None
        evaluation: float | None
        over_event, evaluation = self.master_board_evaluator.check_obvious_over_events(
            node.board
        )
        if over_event is not None:
            node.minmax_evaluation.over_event.becomes_over(
                how_over=over_event.how_over,
                who_is_winner=over_event.who_is_winner,
                termination=over_event.termination,
            )
            assert evaluation is not None, (
                "Evaluation should not be None for over nodes"
            )
            node.minmax_evaluation.set_evaluation(evaluation=evaluation)

    def evaluate_all_queried_nodes(self, evaluation_queries: EvaluationQueries) -> None:
        """
        Evaluates all the queried nodes.
        """
        # node_over: AlgorithmNode
        # for node_over in evaluation_queries.over_nodes:
        # assert isinstance(node_over, AlgorithmNode)
        #    self.evaluate_over(node_over)

        if evaluation_queries.not_over_nodes:
            self.evaluate_all_not_over(evaluation_queries.not_over_nodes)

        evaluation_queries.clear_queries()

    def add_evaluation_query(
        self, node: AlgorithmNode, evaluation_queries: EvaluationQueries
    ) -> None:
        """
        Adds an evaluation query for a node.
        """
        assert node.minmax_evaluation.value_white_evaluator is None
        self.check_obvious_over_events(node)
        if node.is_over():
            evaluation_queries.over_nodes.append(node)
        else:
            evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_not_over(self, not_over_nodes: list[AlgorithmNode]) -> None:
        """
        Evaluates all the nodes that are not over.
        """

        node_not_over: AlgorithmNode
        for node_not_over in not_over_nodes:
            evaluation = self.master_board_evaluator.value_white(
                board=node_not_over.board
            )
            processed_evaluation = self.process_evalution_not_over(
                evaluation=evaluation, node=node_not_over
            )
            # assert isinstance(node_not_over, AlgorithmNode)
            node_not_over.minmax_evaluation.set_evaluation(processed_evaluation)

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode
    ) -> float:
        """
        Processes the evaluation for a node that is not over.
        """
        processed_evaluation = (1 / DISCOUNT) ** node.half_move * evaluation
        return processed_evaluation
