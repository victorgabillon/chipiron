from dataclasses import dataclass
from typing import Any

import chess

import chipiron.environments.chess_env.board as boards
import chipiron.players.boardevaluators.basic_evaluation as basic_evaluation
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    AllBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.board_evaluator_type import BoardEvalTypes
from chipiron.players.boardevaluators.evaluation_scale import (
    EvaluationScale,
    ValueOverEnum,
    get_value_over_enum,
)
from chipiron.players.boardevaluators.neural_networks.factory import (
    create_nn_board_eval_from_nn_parameters_file_and_existing_model,
)
from chipiron.players.boardevaluators.over_event import HowOver, OverEvent, Winner
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable

from .board_evaluator import BoardEvaluator


@dataclass
class MasterBoardEvaluatorArgs:
    """
    Represents the arguments for a master board evaluator.
    """

    # Whether to use syzygy table for evaluation.
    syzygy_evaluation: bool

    # The evaluation scale used by the node evaluator. (Default values when nodes are found to be over)
    evaluation_scale: EvaluationScale

    board_evaluator: AllBoardEvaluatorArgs


class MasterBoardEvaluator:
    """
    The MasterBoardEvaluator class is responsible for evaluating the value of chess positions (that are IBoard).
    It uses a board evaluator and a syzygy evaluator to calculate the value of the positions.
    """

    # The board evaluator used to evaluate the chess board.
    board_evaluator: BoardEvaluator

    # The Syzygy table used for endgame tablebase evaluations, or None if not available.
    syzygy_evaluator: SyzygyTable[Any] | None

    # The value over enum used to determine the value of the node when it is over.
    value_over_enum: ValueOverEnum

    def __init__(
        self,
        board_evaluator: BoardEvaluator,
        syzygy: SyzygyTable[Any] | None,
        value_over_enum: ValueOverEnum,
    ) -> None:
        """
        Initializes a MasterBoardEvaluator object.

        Args:
            board_evaluator (board_evals.BoardEvaluator): The board evaluator used to evaluate the chess board.
            syzygy (SyzygyTable | None): The Syzygy table used for endgame tablebase evaluations, or None if not available.
            value_over_enum (ValueOverEnum): The value over enum used to determine the value of the node when it is over.
        """
        self.board_evaluator = board_evaluator
        self.syzygy_evaluator = syzygy
        self.value_over_enum = value_over_enum

    def value_white(self, board: boards.IBoard) -> float:
        """
        Calculates the value for the white player of a given node.
        If the value can be obtained from the syzygy evaluator, it is used.
        Otherwise, the board evaluator is used.
        """
        value_white: float | None = self.syzygy_value_white(board)
        value_white_float: float
        if value_white is None:
            value_white_float = self.board_evaluator.value_white(board)
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

    def check_obvious_over_events(
        self, board: boards.IBoard
    ) -> tuple[OverEvent | None, float | None]:
        """
        Checks if the given board is in an obvious game-over state and returns the corresponding OverEvent and evaluation.

        Args:
            board (boards.IBoard): The board to evaluate for game-over conditions.

        Raises:
            ValueError: If the board result string is not recognized.

        Returns:
            tuple[OverEvent | None, float]: A tuple containing the OverEvent
            (if the game is over or can be determined from Syzygy tables, otherwise None) and the evaluation score from White's perspective.
            The evaluation is especially useful when training models.
        """
        game_over: bool = board.is_game_over()
        over_event: OverEvent | None = None
        evaluation: float | None = None
        if game_over:
            value_as_string: str = board.result(claim_draw=True)
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

            over_event = OverEvent(
                how_over=how_over_,
                who_is_winner=who_is_winner_,
                termination=board.termination(),
            )

        elif self.syzygy_evaluator and self.syzygy_evaluator.fast_in_table(board):
            who_is_winner_, how_over_ = self.syzygy_evaluator.get_over_event(
                board=board
            )
            over_event = OverEvent(
                how_over=how_over_,
                who_is_winner=who_is_winner_,
                termination=None,  # not sure how to retrieve this info more precisely atm
            )
        if over_event is not None:
            evaluation = self.value_white_from_over_event(over_event=over_event)
        return over_event, evaluation

    def value_white_from_over_event(self, over_event: OverEvent) -> float:
        """
        Returns the value white given an over event.
        """
        assert over_event.is_over()
        white_value: Any
        if over_event.is_win():
            assert not over_event.is_draw()
            if over_event.is_winner(chess.WHITE):
                white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert over_event.is_draw()
            white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_DRAW
        assert isinstance(white_value, float)
        return white_value


def create_master_board_evaluator(
    board_evaluator: BoardEvaluator,
    syzygy: SyzygyTable[Any] | None,
    evaluation_scale: EvaluationScale,
) -> MasterBoardEvaluator:
    """
    Factory function to create a MasterBoardEvaluator instance.

    Args:
        board_evaluator (BoardEvaluator): The board evaluator to use.
        syzygy (SyzygyTable | None): The syzygy table for endgame evaluations.
        value_over_enum (ValueOverEnum): The value over enum for evaluation.

    Returns:
        MasterBoardEvaluator: An instance of MasterBoardEvaluator.
    """

    value_over_enum: ValueOverEnum = get_value_over_enum(
        evaluation_scale=evaluation_scale
    )
    return MasterBoardEvaluator(
        board_evaluator=board_evaluator, syzygy=syzygy, value_over_enum=value_over_enum
    )


def create_master_board_evaluator_from_args(
    master_board_evaluator: MasterBoardEvaluatorArgs,
    syzygy: SyzygyTable[Any] | None,
) -> MasterBoardEvaluator:
    if master_board_evaluator.syzygy_evaluation:
        syzygy_ = syzygy
    else:
        syzygy_ = None

    board_evaluator: BoardEvaluator

    match master_board_evaluator.board_evaluator.type:
        case BoardEvalTypes.BASIC_EVALUATION_EVAL:
            board_evaluator = basic_evaluation.BasicEvaluation()
        case BoardEvalTypes.NEURAL_NET_BOARD_EVAL:
            board_evaluator = create_nn_board_eval_from_nn_parameters_file_and_existing_model(
                model_weights_file_name=master_board_evaluator.board_evaluator.neural_nets_model_and_architecture.model_weights_file_name,
                nn_architecture_args=master_board_evaluator.board_evaluator.neural_nets_model_and_architecture.nn_architecture_args,
            )

    return create_master_board_evaluator(
        board_evaluator=board_evaluator,
        syzygy=syzygy_,
        evaluation_scale=master_board_evaluator.evaluation_scale,
    )
