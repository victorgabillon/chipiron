"""Module representing the board evaluators."""

from typing import Protocol, TypeVar

from valanga import Color
from valanga.evaluations import FloatyStateEvaluation, StateEvaluation

from chipiron.displays.gui_protocol import UpdEvaluation
from chipiron.displays.gui_publisher import GuiPublisher

StateT_contra = TypeVar("StateT_contra", contravariant=True)
StateT = TypeVar("StateT")


class StateEvaluator(Protocol[StateT_contra]):
    """Protocol representing a board evaluator."""

    def value_white(self, state: StateT_contra) -> float:
        """Evaluate a board and returns the value for white."""
        ...


class IGameStateEvaluator(Protocol[StateT_contra]):
    """Protocol representing a game board evaluator."""

    def evaluate(
        self, state: StateT_contra
    ) -> tuple[StateEvaluation | None, StateEvaluation]:
        """Evaluate a board and returns the evaluation values for oracle and chi."""
        ...

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """Add an evaluation value for a player."""
        ...


class GameStateEvaluator[StateT]:
    """Evaluate positions using a primary evaluator and optional oracle."""

    def __init__(
        self,
        *,
        chi: StateEvaluator[StateT],
        oracle: StateEvaluator[StateT] | None,
    ) -> None:
        """Store the primary evaluator and optional oracle."""
        self._chi = chi
        self._oracle = oracle

    def evaluate(self, state: StateT) -> tuple[StateEvaluation | None, StateEvaluation]:
        """Evaluate a state and return oracle and primary evaluations."""
        chi_value = self._chi.value_white(state)
        chi = FloatyStateEvaluation(value_white=chi_value)
        oracle = (
            FloatyStateEvaluation(value_white=self._oracle.value_white(state))
            if self._oracle
            else None
        )
        return oracle, chi

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """Accept an external evaluation for a given player color."""
        _ = player_color
        _ = evaluation


class ObservableGameStateEvaluator[StateT]:
    """Observable evaluator that publishes GUI updates."""

    publishers: list[GuiPublisher]

    game_state_evaluator: IGameStateEvaluator[StateT]
    evaluation_oracle: StateEvaluation | None = None
    evaluation_chi: StateEvaluation | None = None
    evaluation_player_black: StateEvaluation | None = None
    evaluation_player_white: StateEvaluation | None = None

    def __init__(self, game_state_evaluator: IGameStateEvaluator[StateT]):
        """Initialize with a wrapped game-state evaluator."""
        self.game_state_evaluator = game_state_evaluator
        self.publishers = []
        self.evaluation_oracle = None
        self.evaluation_chi = None
        self.evaluation_player_black = None
        self.evaluation_player_white = None

    def subscribe(self, pub: GuiPublisher) -> None:
        """Subscribe to the ObservableBoardEvaluator to get the EvaluationMessage.

        Args:
            mailbox: The mailbox queue.

        """
        self.publishers.append(pub)

    def evaluate(self, state: StateT) -> tuple[StateEvaluation | None, StateEvaluation]:
        """Evaluate a board and returns the evaluation values for oracle and chi."""
        evaluation_oracle, evaluation_chi = self.game_state_evaluator.evaluate(
            state=state
        )
        self.evaluation_oracle = evaluation_oracle
        self.evaluation_chi = evaluation_chi

        self.notify_new_results()
        return evaluation_oracle, evaluation_chi

    def add_evaluation(self, player_color: Color, evaluation: StateEvaluation) -> None:
        """Add an evaluation value for a player."""
        if player_color == Color.BLACK:
            self.evaluation_player_black = evaluation
        if player_color == Color.WHITE:
            self.evaluation_player_white = evaluation
        self.notify_new_results()

    def notify_new_results(self) -> None:
        """Notifies the subscribers about the new evaluation results."""
        payload = UpdEvaluation(
            oracle=self.evaluation_oracle,
            chipiron=self.evaluation_chi,
            white=self.evaluation_player_white,
            black=self.evaluation_player_black,
        )
        for pub in self.publishers:
            pub.publish(payload=payload)
