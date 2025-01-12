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
from typing import Any

import chess

import chipiron.environments.chess.board as boards
import chipiron.players.boardevaluators as board_evals
from chipiron.players.boardevaluators.over_event import HowOver, OverEvent, Winner
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode

DISCOUNT = 0.999999999999  # todo play with this


class NodeEvaluatorTypes(str, Enum):
    """
    Enum class representing different types of node evaluators.
    """

    NeuralNetwork = "neural_network"


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

    board_evaluator: board_evals.BoardEvaluator
    syzygy_evaluator: SyzygyTable[Any] | None

    def __init__(
        self,
        board_evaluator: board_evals.BoardEvaluator,
        syzygy: SyzygyTable[Any] | None,
    ) -> None:
        """
        Initializes a NodeEvaluator object.

        Args:
            board_evaluator (board_evals.BoardEvaluator): The board evaluator used to evaluate the chess board.
            syzygy (SyzygyTable | None): The Syzygy table used for endgame tablebase evaluations, or None if not available.
        """
        self.board_evaluator = board_evaluator
        self.syzygy_evaluator = syzygy

    def value_white(self, node: ITreeNode[Any]) -> float:
        """
        Calculates the value for the white player of a given node.
        If the value can be obtained from the syzygy evaluator, it is used.
        Otherwise, the board evaluator is used.
        """
        value_white: float | None = self.syzygy_value_white(node.board)
        value_white_float: float
        if value_white is None:
            value_white_float = self.board_evaluator.value_white(node.board)
        else:
            value_white_float = value_white
        return value_white_float

    def syzygy_value_white(self, board: boards.IBoard) -> float | None:
        """
        Calculates the value for the white player of a given board using the syzygy evaluator.
        If the syzygy evaluator is not available or the board is not in the syzygy table, None is returned.
        """
        if self.syzygy_evaluator is None or not self.syzygy_evaluator.fast_in_table(
            board
        ):
            return None
        else:
            val: int = self.syzygy_evaluator.val(board)
            return val

    def check_obvious_over_events(self, node: AlgorithmNode) -> None:
        """
        Updates the node.over object if the game is obviously over.
        """
        game_over: bool = node.tree_node.board.is_game_over()
        if game_over:
            value_as_string: str = node.board.result(claim_draw=True)
            how_over_: HowOver
            who_is_winner_: Winner
            match value_as_string:
                case "0-1":
                    how_over_ = HowOver.WIN
                    who_is_winner_ = Winner.BLACK
                case "1-0":
                    how_over_ = HowOver.WIN
                    who_is_winner_ = Winner.WHITE
                case "1/2-1/2":
                    how_over_ = HowOver.DRAW
                    who_is_winner_ = Winner.NO_KNOWN_WINNER
                case other:
                    raise ValueError(f"value {other} not expected in {__name__}")

            node.minmax_evaluation.over_event.becomes_over(
                how_over=how_over_,
                who_is_winner=who_is_winner_,
                termination=node.board.termination(),
            )

        elif self.syzygy_evaluator and self.syzygy_evaluator.fast_in_table(
            node.tree_node.board
        ):
            who_is_winner_, how_over_ = self.syzygy_evaluator.get_over_event(
                board=node.board
            )
            node.minmax_evaluation.over_event.becomes_over(
                how_over=how_over_,
                who_is_winner=who_is_winner_,
                termination=None,  # not sure how to retrieve this info more precisely atm
            )

    def value_white_from_over_event(
        self, over_event: OverEvent
    ) -> board_evals.ValueWhiteWhenOver:
        """
        Returns the value white given an over event.
        """
        assert over_event.is_over()
        if over_event.is_win():
            assert not over_event.is_draw()
            if over_event.is_winner(chess.WHITE):
                return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert over_event.is_draw()
            return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_DRAW

    def evaluate_over(self, node: AlgorithmNode) -> None:
        """
        Evaluates the node when the game is over.
        """
        evaluation = DISCOUNT**node.half_move * self.value_white_from_over_event(
            node.minmax_evaluation.over_event
        )
        node.minmax_evaluation.set_evaluation(evaluation)

    def evaluate_all_queried_nodes(self, evaluation_queries: EvaluationQueries) -> None:
        """
        Evaluates all the queried nodes.
        """
        node_over: AlgorithmNode
        for node_over in evaluation_queries.over_nodes:
            # assert isinstance(node_over, AlgorithmNode)
            self.evaluate_over(node_over)

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
            evaluation = self.value_white(node_not_over)
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
