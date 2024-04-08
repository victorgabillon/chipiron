from enum import Enum

import chess

import chipiron.players.boardevaluators as board_evals
from chipiron.environments.chess.board.board import BoardChi
from chipiron.players.boardevaluators.over_event import HowOver, Winner, OverEvent
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from chipiron.players.move_selector.treevalue.nodes.itree_node import ITreeNode

DISCOUNT = .999999999999  # todo play with this


class NodeEvaluatorTypes(str, Enum):
    NeuralNetwork: str = 'neural_network'


class EvaluationQueries:
    over_nodes: list[AlgorithmNode]
    not_over_nodes: list[AlgorithmNode]

    def __init__(self) -> None:
        self.over_nodes = []
        self.not_over_nodes = []

    def clear_queries(self) -> None:
        self.over_nodes = []
        self.not_over_nodes = []


class NodeEvaluator:
    # VALUE_WHITE_WHEN_OVER is the value_white default value when the node is over
    # set atm to be symmetric and high to be preferred

    """ Wrapping node evaluator with syzygy and obvious over event. I think the idea is this class builds
     on top of BoardEvaluator which is th elementary class to build something more complex
      (similar to the relation between player and move selector)
       it also manages the evaluation querrys it seems"""

    board_evaluator: board_evals.BoardEvaluator
    syzygy_evaluator: SyzygyTable | None

    def __init__(
            self,
            board_evaluator: board_evals.BoardEvaluator,
            syzygy: SyzygyTable | None
    ) -> None:
        self.board_evaluator = board_evaluator
        self.syzygy_evaluator = syzygy

    def value_white(
            self,
            node: ITreeNode
    ) -> float:
        value_white: float | None = self.syzygy_value_white(node.board)
        value_white_float: float
        if value_white is None:
            value_white_float = self.board_evaluator.value_white(node.board)
        else:
            value_white_float = value_white
        return value_white_float

    def syzygy_value_white(
            self,
            board: BoardChi
    ) -> float | None:
        # Todo probalby should use the function below value form over evnt
        if self.syzygy_evaluator is None or not self.syzygy_evaluator.fast_in_table(board):
            return None
        else:
            val: int = self.syzygy_evaluator.val(board)

            return val

    def check_obvious_over_events(
            self,
            node: AlgorithmNode
    ) -> None:
        """ updates the node.over object
         if the game is obviously over"""
        game_over: bool = node.tree_node.board.is_game_over()
        if game_over:
            value_as_string: str = node.board.board.result()
            how_over_: HowOver
            who_is_winner_: Winner
            match value_as_string:
                case '0-1':
                    how_over_ = HowOver.WIN
                    who_is_winner_ = Winner.BLACK
                case '1-0':
                    how_over_ = HowOver.WIN
                    who_is_winner_ = Winner.WHITE
                case '1/2-1/2':
                    how_over_ = HowOver.DRAW
                    who_is_winner_ = Winner.NO_KNOWN_WINNER
                case other:
                    raise ValueError(f'value {other} not expected in {__name__}')
            node.minmax_evaluation.over_event.becomes_over(
                how_over=how_over_,
                who_is_winner=who_is_winner_
            )

        elif self.syzygy_evaluator and self.syzygy_evaluator.fast_in_table(node.tree_node.board):
            who_is_winner_, how_over_ = self.syzygy_evaluator.get_over_event(board=node.board)
            node.minmax_evaluation.over_event.becomes_over(
                how_over=how_over_,
                who_is_winner=who_is_winner_
            )

    def value_white_from_over_event(
            self,
            over_event: OverEvent
    ) -> board_evals.ValueWhiteWhenOver:
        """ returns the value white given an over event"""
        assert over_event.is_over()
        if over_event.is_win():
            assert (not over_event.is_draw())
            if over_event.is_winner(chess.WHITE):
                return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert (over_event.is_draw())
            return board_evals.ValueWhiteWhenOver.VALUE_WHITE_WHEN_OVER_DRAW

    def evaluate_over(
            self,
            node: AlgorithmNode
    ) -> None:
        evaluation = DISCOUNT ** node.half_move * self.value_white_from_over_event(node.minmax_evaluation.over_event)
        node.minmax_evaluation.set_evaluation(evaluation)

    def evaluate_all_queried_nodes(
            self,
            evaluation_queries: EvaluationQueries
    ) -> None:

        node_over: AlgorithmNode
        for node_over in evaluation_queries.over_nodes:
            # assert isinstance(node_over, AlgorithmNode)
            self.evaluate_over(node_over)

        if evaluation_queries.not_over_nodes:
            self.evaluate_all_not_over(evaluation_queries.not_over_nodes)

        evaluation_queries.clear_queries()

    def add_evaluation_query(
            self,
            node: AlgorithmNode,
            evaluation_queries: EvaluationQueries
    ) -> None:
        assert (node.minmax_evaluation.value_white_evaluator is None)
        self.check_obvious_over_events(node)
        if node.is_over():
            evaluation_queries.over_nodes.append(node)
        else:
            evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_not_over(
            self,
            not_over_nodes: list[AlgorithmNode]
    ) -> None:
        node_not_over: AlgorithmNode
        for node_not_over in not_over_nodes:
            evaluation = self.value_white(node_not_over)
            processed_evaluation = self.process_evalution_not_over(
                evaluation=evaluation,
                node=node_not_over
            )
            # assert isinstance(node_not_over, AlgorithmNode)
            node_not_over.minmax_evaluation.set_evaluation(processed_evaluation)

    def process_evalution_not_over(
            self,
            evaluation: float,
            node: AlgorithmNode
    ) -> float:
        processed_evaluation = (1 / DISCOUNT) ** node.half_move * evaluation
        return processed_evaluation
