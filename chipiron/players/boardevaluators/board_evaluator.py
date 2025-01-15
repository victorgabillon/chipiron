"""
Module representing the board evaluators.
"""

import queue
from enum import Enum
from typing import Any, Protocol

import chess

import chipiron.environments.chess.board as boards
from chipiron.players.boardevaluators.board_evaluation.board_evaluation import (
    BoardEvaluation,
)
from chipiron.utils.communication.gui_messages import EvaluationMessage
from chipiron.utils.dataclass import IsDataclass


class ValueWhiteWhenOver(float, Enum):
    """
    Enum class representing the default values for `value_white` when the node is over.
    """

    VALUE_WHITE_WHEN_OVER_WHITE_WINS = 1000.0
    VALUE_WHITE_WHEN_OVER_DRAW = 0.0
    VALUE_WHITE_WHEN_OVER_BLACK_WINS = -1000.0


class BoardEvaluator(Protocol):
    """
    Protocol representing a board evaluator.
    """

    def value_white(self, board: boards.IBoard) -> float:
        """
        Evaluates a board and returns the value for white.
        """
        ...


class IGameBoardEvaluator(Protocol):
    """
    Protocol representing a game board evaluator.
    """

    def evaluate(self, board: boards.IBoard) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """

    def add_evaluation(
        self, player_color: chess.Color, evaluation: BoardEvaluation
    ) -> None:
        """
        Adds an evaluation value for a player.
        """


class GameBoardEvaluator:
    """
    This class is a collection of evaluators that display their analysis during the game.
    They are not players, just external analysis and display.
    """

    board_evaluator_stock: BoardEvaluator | None
    board_evaluator_chi: BoardEvaluator

    def __init__(
        self,
        board_evaluator_stock: BoardEvaluator | None,
        board_evaluator_chi: BoardEvaluator,
    ):
        self.board_evaluator_stock = board_evaluator_stock
        self.board_evaluator_chi = board_evaluator_chi

    def evaluate(self, board: boards.IBoard) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """
        evaluation_chi = self.board_evaluator_chi.value_white(board=board)
        evaluation_stock = (
            self.board_evaluator_stock.value_white(board=board)
            if self.board_evaluator_stock is not None
            else None
        )

        return evaluation_stock, evaluation_chi

    def add_evaluation(
        self, player_color: chess.Color, evaluation: BoardEvaluation
    ) -> None:
        """
        Adds an evaluation value for a player.
        """


class ObservableBoardEvaluator:
    """
    This class represents an observable board evaluator.
    """

    game_board_evaluator: GameBoardEvaluator
    mailboxes: list[queue.Queue[IsDataclass]]
    evaluation_stock: Any
    evaluation_chi: Any
    evaluation_player_black: Any
    evaluation_player_white: Any

    def __init__(self, game_board_evaluator: GameBoardEvaluator):
        self.game_board_evaluator = game_board_evaluator
        self.mailboxes = []
        self.evaluation_stock = None
        self.evaluation_chi = None
        self.evaluation_player_black = None
        self.evaluation_player_white = None

    def subscribe(self, mailbox: queue.Queue[IsDataclass]) -> None:
        """
        Subscribe to the ObservableBoardEvaluator to get the EvaluationMessage.

        Args:
            mailbox: The mailbox queue.
        """
        self.mailboxes.append(mailbox)

    def evaluate(self, board: boards.IBoard) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """
        self.evaluation_stock, self.evaluation_chi = self.game_board_evaluator.evaluate(
            board=board
        )

        self.notify_new_results()
        return self.evaluation_stock, self.evaluation_chi

    def add_evaluation(
        self, player_color: chess.Color, evaluation: BoardEvaluation
    ) -> None:
        """
        Adds an evaluation value for a player.
        """
        if player_color == chess.BLACK:
            self.evaluation_player_black = evaluation
        if player_color == chess.WHITE:
            self.evaluation_player_white = evaluation
        self.notify_new_results()

    def notify_new_results(self) -> None:
        """
        Notifies the subscribers about the new evaluation results.
        """
        for mailbox in self.mailboxes:
            message: EvaluationMessage = EvaluationMessage(
                evaluation_stock=self.evaluation_stock,
                evaluation_chipiron=self.evaluation_chi,
                evaluation_player_white=self.evaluation_player_white,
                evaluation_player_black=self.evaluation_player_black,
            )
            mailbox.put(item=message)
