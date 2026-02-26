"""Chess-specific rules adapter."""

from dataclasses import dataclass

from valanga import Color
from valanga.over_event import HowOver, Winner

from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import (
    AnySyzygyTable,
)
from chipiron.environments.chess.types import ChessState
from chipiron.games.domain.game.game_rules import (
    GameOutcome,
    GameRules,
    OutcomeKind,
    PositionAssessment,
    VerdictKind,
)


class ChessRulesError(ValueError):
    """Base error for chess rules failures."""


class UnexpectedChessResultError(ChessRulesError):
    """Raised when the board result string is not recognized."""

    def __init__(self, result: str, owner: str) -> None:
        """Initialize the error with the unexpected result string."""
        super().__init__(f"unexpected result value {result} in {owner}")


@dataclass(frozen=True)
class ChessRules(GameRules[ChessState]):
    """Chessrules implementation."""

    syzygy: AnySyzygyTable | None = None

    def outcome(self, state: ChessState) -> GameOutcome | None:
        """Outcome."""
        if state.board.is_game_over():
            return self._outcome_from_board(state)
        return None

    def pretty_result(self, state: ChessState, outcome: GameOutcome) -> str:
        """Pretty result."""
        result_str = self._result_string_outcome(outcome)
        reason = outcome.reason
        if reason is None and state.board.is_game_over():
            reason = self._termination_reason(state)
        message = f"Result: {result_str}"
        if reason:
            message = f"{message} ({reason})"
        return message

    def assessment(self, state: ChessState) -> PositionAssessment | None:
        """Assessment."""
        if self.syzygy is None or not self.syzygy.fast_in_table(state.board):
            return None
        winner, how_over = self.syzygy.get_over_event(board=state.board)
        return self._assessment_from_over_event(winner, how_over, reason="syzygy")

    def pretty_assessment(
        self, state: ChessState, assessment: PositionAssessment
    ) -> str:
        """Pretty assessment."""
        result_str = self._result_string_assessment(assessment)
        message = f"Assessment: {result_str}"
        if assessment.reason:
            message = f"{message} ({assessment.reason})"
        if self.syzygy is not None and self.syzygy.fast_in_table(state.board):
            syzygy_str = self.syzygy.string_result(state.board)
            message = f"{message} | Syzygy: {syzygy_str}"
        return message

    def _outcome_from_board(self, state: ChessState) -> GameOutcome:
        """Derive a GameOutcome from the current board result."""
        result = state.board.result(claim_draw=True)
        reason = self._termination_reason(state)
        match result:
            case "1-0":
                return GameOutcome(
                    kind=OutcomeKind.WIN, winner=Color.WHITE, reason=reason
                )
            case "0-1":
                return GameOutcome(
                    kind=OutcomeKind.WIN, winner=Color.BLACK, reason=reason
                )
            case "1/2-1/2":
                return GameOutcome(kind=OutcomeKind.DRAW, reason=reason)
            case "*":
                return GameOutcome(kind=OutcomeKind.UNKNOWN, reason=reason)
            case _:
                raise UnexpectedChessResultError(result, self.__class__.__name__)

    def _assessment_from_over_event(
        self, winner: Winner, how_over: HowOver, reason: str | None = None
    ) -> PositionAssessment:
        """Map a terminal event into a position assessment."""
        if how_over is HowOver.DRAW:
            return PositionAssessment(kind=VerdictKind.DRAW, reason=reason)
        if how_over is HowOver.WIN:
            mapped_winner: Color | None = None
            if winner is Winner.WHITE:
                mapped_winner = Color.WHITE
            elif winner is Winner.BLACK:
                mapped_winner = Color.BLACK
            if mapped_winner is None:
                return PositionAssessment(kind=VerdictKind.UNKNOWN, reason=reason)
            return PositionAssessment(
                kind=VerdictKind.WIN, winner=mapped_winner, reason=reason
            )
        return PositionAssessment(kind=VerdictKind.UNKNOWN, reason=reason)

    @staticmethod
    def _result_string_outcome(outcome: GameOutcome) -> str:
        """Return the PGN result string for a GameOutcome."""
        if outcome.kind is OutcomeKind.WIN and outcome.winner is Color.WHITE:
            return "1-0"
        if outcome.kind is OutcomeKind.WIN and outcome.winner is Color.BLACK:
            return "0-1"
        if outcome.kind is OutcomeKind.DRAW:
            return "1/2-1/2"
        return "*"

    @staticmethod
    def _result_string_assessment(assessment: PositionAssessment) -> str:
        """Return the PGN result string for a PositionAssessment."""
        if assessment.kind is VerdictKind.WIN and assessment.winner is Color.WHITE:
            return "1-0"
        if assessment.kind is VerdictKind.WIN and assessment.winner is Color.BLACK:
            return "0-1"
        if assessment.kind is VerdictKind.DRAW:
            return "1/2-1/2"
        return "*"

    @staticmethod
    def _termination_reason(state: ChessState) -> str | None:
        """Return a termination reason string when the game is over."""
        termination = state.board.termination()
        if termination is None:
            return None
        return str(termination)
