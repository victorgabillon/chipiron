"""Module for master board evaluator."""

from typing import Any, cast

from valanga import Color, Outcome
from valanga.evaluations import Value
from valanga.over_event import OverEvent

from chipiron.core.evaluation_scale import (
    EvaluationScale,
    ValueOverEnum,
    get_value_over_enum,
)
from chipiron.core.oracles import TerminalOracle, ValueOracle
from chipiron.environments.chess.players.evaluators.boardevaluators import (
    basic_evaluation,
)
from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks.chess_model_bundle_evaluator import (
    create_chess_nn_state_eval_from_model_bundle,
)
from chipiron.environments.chess.types import ChessState
from chipiron.models.model_bundle import resolve_model_bundle
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    BasicEvaluationBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.board_evaluator import StateEvaluator
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.neural_networks.neural_net_board_eval_args import (
    NeuralNetBoardEvalArgs,
)


class MasterBoardEvaluatorError(ValueError):
    """Base error for master board evaluator issues."""


class UnexpectedGameResultError(MasterBoardEvaluatorError):
    """Raised when a game result string is not recognized."""

    def __init__(self, result: str) -> None:
        """Initialize the error with the unexpected game result."""
        super().__init__(f"value {result} not expected in {__name__}")


class UnsupportedBoardEvaluatorArgsError(MasterBoardEvaluatorError):
    """Raised when board evaluator args are unsupported."""

    def __init__(self, args_type: str) -> None:
        """Initialize the error with the unsupported args type."""
        super().__init__(
            f"unknown type of message received by master board evaluator {args_type} in {__name__}"
        )


class MasterBoardEvaluator:
    """The MasterBoardEvaluator class is responsible for evaluating the value of chess positions (that are IBoard).

    It uses a board evaluator and optional oracle-based evaluation to calculate values.
    """

    # The board evaluator used to evaluate the chess board.
    board_evaluator: StateEvaluator[ChessState]

    # The optional value oracle used for exact evaluations.
    value_oracle: ValueOracle[ChessState] | None

    # The optional terminal oracle used for endgame metadata.
    terminal_oracle: TerminalOracle[ChessState, Color] | None

    # The value over enum used to determine the value of the node when it is over.
    value_over_enum: ValueOverEnum

    def __init__(
        self,
        board_evaluator: StateEvaluator[ChessState],
        value_oracle: ValueOracle[ChessState] | None,
        terminal_oracle: TerminalOracle[ChessState, Color] | None,
        value_over_enum: ValueOverEnum,
    ) -> None:
        """Initialize a MasterBoardEvaluator object.

        Args:
            board_evaluator (board_evals.BoardEvaluator): The board evaluator used to evaluate the chess board.
            value_oracle (ValueOracle | None): The value oracle used for exact evaluations, or None if not available.
            terminal_oracle (TerminalOracle | None): The terminal oracle used for endgame metadata.
            value_over_enum (ValueOverEnum): The value over enum used to determine the value of the node when it is over.

        """
        self.board_evaluator = board_evaluator
        self.value_oracle = value_oracle
        self.terminal_oracle = terminal_oracle
        self.value_over_enum = value_over_enum

    def evaluate(self, state: ChessState) -> Value:
        """Calculate the value for the white player of a given node.

        If the value can be obtained from the syzygy evaluator, it is used.
        Otherwise, the board evaluator is used.
        """
        oracle_value: Value | None = self.oracle_evaluate(state)
        value: Value
        if oracle_value is None:
            value = self.board_evaluator.evaluate(state)
        else:
            value = oracle_value
        return value

    def oracle_evaluate(self, state: ChessState) -> Value | None:
        """Calculate the value for the white player of a given state using the value oracle.

        If the value oracle is not available or the state is not supported, None is returned.
        """
        if self.value_oracle is None or not self.value_oracle.supports(state):
            return None
        return cast("Value", self.value_oracle.evaluate(state))

    def check_obvious_over_events(
        self, state: ChessState
    ) -> tuple[OverEvent[Color] | None, float | None]:
        """Check if the given board is in an obvious game-over state and returns the corresponding OverEvent and evaluation.

        Args:
            state (ChessState): The state to evaluate for game-over conditions.

        Raises:
            ValueError: If the board result string is not recognized.

        Returns:
            tuple[OverEvent[Color] | None, float | None]: A tuple containing the OverEvent
            (if the game is over or can be determined from Syzygy tables, otherwise None) and the evaluation score from White's perspective.
            The evaluation is especially useful when training models.

        """
        board = state.board
        game_over: bool = board.is_game_over()
        over_event: OverEvent[Color] | None = None
        evaluation: float | None = None
        if game_over:
            value_as_string: str = board.result(claim_draw=True)
            how_over_: Outcome
            who_is_winner_: Color | None
            match value_as_string:
                case "0-1":
                    how_over_ = Outcome.WIN
                    who_is_winner_ = Color.BLACK
                case "1-0":
                    how_over_ = Outcome.WIN
                    who_is_winner_ = Color.WHITE
                case "1/2-1/2":
                    how_over_ = Outcome.DRAW
                    who_is_winner_ = None
                case _:
                    raise UnexpectedGameResultError(value_as_string)

            over_event = OverEvent(
                outcome=how_over_,
                winner=who_is_winner_,
                termination=board.termination(),
            )

        elif self.terminal_oracle and self.terminal_oracle.supports(state):
            over_event = self.terminal_oracle.over_event(state)
        if over_event is not None:
            evaluation = self.value_white_from_over_event(over_event=over_event)
        return over_event, evaluation

    def value_white_from_over_event(self, over_event: OverEvent[Color]) -> float:
        """Return the value white given an over event."""
        assert over_event.is_over()
        white_value: Any
        if over_event.is_win():
            assert not over_event.is_draw()
            if over_event.is_win_for(Color.WHITE):
                white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_WHITE_WINS
            else:
                white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_BLACK_WINS
        else:  # draw
            assert over_event.is_draw()
            white_value = self.value_over_enum.VALUE_WHITE_WHEN_OVER_DRAW
        assert isinstance(white_value, float)
        return white_value


def create_master_state_evaluator(
    board_evaluator: StateEvaluator[ChessState],
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState, Color] | None,
    evaluation_scale: EvaluationScale,
) -> MasterBoardEvaluator:
    """Create a MasterBoardEvaluator instance.

    Args:
        board_evaluator (BoardEvaluator): The board evaluator to use.
        value_oracle (ValueOracle | None): The value oracle for exact evaluations.
        terminal_oracle (TerminalOracle | None): The terminal oracle for endgame metadata.
        value_over_enum (ValueOverEnum): The value over enum for evaluation.

    Returns:
        MasterBoardEvaluator: An instance of MasterBoardEvaluator.

    """
    value_over_enum: ValueOverEnum = get_value_over_enum(
        evaluation_scale=evaluation_scale
    )
    return MasterBoardEvaluator(
        board_evaluator=board_evaluator,
        value_oracle=value_oracle,
        terminal_oracle=terminal_oracle,
        value_over_enum=value_over_enum,
    )


def create_master_state_evaluator_from_args(
    master_board_evaluator: MasterBoardEvaluatorArgs,
    value_oracle: ValueOracle[ChessState] | None,
    terminal_oracle: TerminalOracle[ChessState, Color] | None,
) -> MasterBoardEvaluator:
    """Create a MasterBoardEvaluator instance from the given arguments.

    Args:
        master_board_evaluator (MasterBoardEvaluatorArgs): args to build the MasterBoardEvaluator
        value_oracle (ValueOracle | None): a value oracle object
        terminal_oracle (TerminalOracle | None): a terminal oracle object

    Returns:
        MasterBoardEvaluator: _description_

    """
    if master_board_evaluator.oracle_evaluation:
        value_oracle_ = value_oracle
        terminal_oracle_ = terminal_oracle
    else:
        value_oracle_ = None
        terminal_oracle_ = None

    board_evaluator: StateEvaluator[ChessState]
    args = master_board_evaluator.board_evaluator
    match args:
        case BasicEvaluationBoardEvaluatorArgs():
            board_evaluator = basic_evaluation.BasicEvaluation()
        case NeuralNetBoardEvalArgs(neural_nets_model_and_architecture=model):
            bundle = resolve_model_bundle(model.model_bundle)
            board_evaluator = create_chess_nn_state_eval_from_model_bundle(bundle)
        case _:
            raise UnsupportedBoardEvaluatorArgsError(args.type)

    return create_master_state_evaluator(
        board_evaluator=board_evaluator,
        value_oracle=value_oracle_,
        terminal_oracle=terminal_oracle_,
        evaluation_scale=master_board_evaluator.evaluation_scale,
    )
