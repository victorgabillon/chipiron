"""
Module representing the board evaluators.
"""

from typing import Any, Protocol

from valanga import Color, State, StateEvaluation

from chipiron.utils.communication.gui_messages.gui_messages import UpdEvaluation
from chipiron.utils.communication.gui_publisher import GuiPublisher


class StateEvaluator[StateT](Protocol):
    """
    Protocol representing a board evaluator.
    """

    def value_white(self, state: StateT) -> float:
        """
        Evaluates a board and returns the value for white.
        """
        ...


class IGameStateEvaluator[StateT](Protocol):
    """
    Protocol representing a game board evaluator.
    """

    def evaluate(self, state: StateT) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """
        ...

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """
        Adds an evaluation value for a player.
        """
        ...


class GameStateEvaluator:
    """
    This class is a collection of evaluators that display their analysis during the game.
    They are not players, just external analysis and display.
    """

    board_evaluator_stock: StateEvaluator | None
    board_evaluator_chi: StateEvaluator

    def __init__(
        self,
        board_evaluator_stock: StateEvaluator | None,
        board_evaluator_chi: StateEvaluator,
    ):
        self.board_evaluator_stock = board_evaluator_stock
        self.board_evaluator_chi = board_evaluator_chi

    def evaluate(self, state: State) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """

        evaluation_chi = self.board_evaluator_chi.value_white(state=state)
        evaluation_stock = (
            self.board_evaluator_stock.value_white(state=state)
            if self.board_evaluator_stock is not None
            else None
        )

        return evaluation_stock, evaluation_chi

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """
        Adds an evaluation value for a player.
        """


class ObservableBoardEvaluator:
    """
    This class represents an observable board evaluator.
    """

    publishers: list[GuiPublisher]

    game_board_evaluator: GameStateEvaluator
    evaluation_stock: Any
    evaluation_chi: Any
    evaluation_player_black: Any
    evaluation_player_white: Any

    def __init__(self, game_board_evaluator: GameStateEvaluator):
        self.game_board_evaluator = game_board_evaluator
        self.publishers = []
        self.evaluation_stock = None
        self.evaluation_chi = None
        self.evaluation_player_black = None
        self.evaluation_player_white = None

    def subscribe(self, pub: GuiPublisher) -> None:
        """
        Subscribe to the ObservableBoardEvaluator to get the EvaluationMessage.

        Args:
            mailbox: The mailbox queue.
        """
        self.publishers.append(pub)

    def evaluate(self, state: State) -> tuple[float | None, float]:
        """
        Evaluates a board and returns the evaluation values for stock and chi.
        """
        self.evaluation_stock, self.evaluation_chi = self.game_board_evaluator.evaluate(
            state=state
        )

        self.notify_new_results()
        return self.evaluation_stock, self.evaluation_chi

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """
        Adds an evaluation value for a player.
        """
        if player_color == Color.BLACK:
            self.evaluation_player_black = evaluation
        if player_color == Color.WHITE:
            self.evaluation_player_white = evaluation
        self.notify_new_results()

    def notify_new_results(self) -> None:
        """
        Notifies the subscribers about the new evaluation results.
        """
        payload = UpdEvaluation(
            stock=self.evaluation_stock,
            chipiron=self.evaluation_chi,
            white=self.evaluation_player_white,
            black=self.evaluation_player_black,
        )
        for pub in self.publishers:
            pub.publish(payload=payload)
