from typing import Any

import chess

import chipiron.environments.chess_env.board as boards
import chipiron.players.boardevaluators as board_evals
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    BoardEvaluation,
)
from chipiron.players.boardevaluators.over_event import HowOver, OverEvent, Winner
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable

from .board_evaluator import BoardEvaluator


class MasterBoardEvaluator:
    """
    The MasterBoardEvaluator class is responsible for evaluating the value of nodes in a tree structure.
    It uses a board evaluator and a syzygy evaluator to calculate the value of the nodes.
    """

    board_evaluator: BoardEvaluator
    syzygy_evaluator: SyzygyTable[Any] | None

    def __init__(
        self,
        board_evaluator: BoardEvaluator,
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
