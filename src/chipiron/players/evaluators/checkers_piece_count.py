"""Simple checkers evaluator wiring for tree search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from valanga import Color, State
from valanga.over_event import HowOver, OverEvent, Winner

from chipiron.environments.checkers.checkers_rules import CheckersRules
from chipiron.environments.checkers.types import CheckersState
from chipiron.games.game.game_rules import OutcomeKind

KING_WEIGHT = 2


@dataclass(frozen=True)
class CheckersPieceCountEvaluator:
    """Material-count evaluator from White's perspective."""

    king_weight: int = KING_WEIGHT

    def value_white(self, state: CheckersState) -> float:
        """Return material score from White's perspective."""
        white_men = state.wm.bit_count()
        white_kings = state.wk.bit_count()
        black_men = state.bm.bit_count()
        black_kings = state.bk.bit_count()

        return float(
            (white_men + self.king_weight * white_kings)
            - (black_men + self.king_weight * black_kings)
        )


@dataclass(frozen=True)
class CheckersOverEventDetector:
    """Detect terminal outcomes using chipiron's checkers rules adapter."""

    rules: CheckersRules

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return terminal over-event + value when available."""
        checkers_state = cast("CheckersState", state)
        outcome = self.rules.outcome(state=checkers_state)
        if outcome is None:
            return None, None

        if outcome.kind is OutcomeKind.WIN:
            who_is_winner = (
                Winner.WHITE if outcome.winner is Color.WHITE else Winner.BLACK
            )
            over_event = OverEvent(
                how_over=HowOver.WIN,
                who_is_winner=who_is_winner,
                termination=None,
            )
            value_white = 1.0 if outcome.winner is Color.WHITE else -1.0
            return over_event, value_white

        over_event = OverEvent(
            how_over=HowOver.DRAW,
            who_is_winner=Winner.NO_KNOWN_WINNER,
            termination=None,
        )
        return over_event, 0.0


@dataclass(frozen=True)
class CheckersMasterEvaluator(MasterStateEvaluator):
    """Anemone-compatible master evaluator for checkers."""

    evaluator: CheckersPieceCountEvaluator
    over: CheckersOverEventDetector

    def value_white(self, state: State) -> float:
        """Evaluate state from White's perspective."""
        _over_event, terminal_value_white = self.over.check_obvious_over_events(state)
        if terminal_value_white is not None:
            return float(terminal_value_white)
        return self.evaluator.value_white(cast("CheckersState", state))
