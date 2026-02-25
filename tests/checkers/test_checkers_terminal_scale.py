"""Tests for checkers terminal values across evaluation scales."""

from dataclasses import dataclass

import pytest

pytest.importorskip("valanga")
from valanga import Color

from chipiron.games.domain.game.game_rules import GameOutcome, OutcomeKind
from chipiron.players.boardevaluators.evaluation_scale import (
    EvaluationScale,
    get_value_over_enum,
)
from chipiron.environments.checkers.players.evaluators.checkers_piece_count import CheckersOverEventDetector


@dataclass(frozen=True)
class _StubRules:
    outcome_to_return: GameOutcome

    def outcome(self, state: object) -> GameOutcome:
        del state
        return self.outcome_to_return


def _value_for_outcome(
    outcome: GameOutcome, evaluation_scale: EvaluationScale
) -> float:
    detector = CheckersOverEventDetector(
        rules=_StubRules(outcome),
        value_over_enum=get_value_over_enum(evaluation_scale),
    )
    _, value = detector.check_obvious_over_events(state=object())
    assert value is not None
    return value


@pytest.mark.parametrize(
    "evaluation_scale",
    [
        EvaluationScale.ENTIRE_REAL_AXIS,
        EvaluationScale.SYMMETRIC_UNIT_INTERVAL,
    ],
)
def test_checkers_terminal_values_match_value_over_enum(
    evaluation_scale: EvaluationScale,
) -> None:
    """Terminal outcomes match values defined by the selected scale enum."""
    value_over_enum = get_value_over_enum(evaluation_scale)

    assert _value_for_outcome(
        GameOutcome(kind=OutcomeKind.WIN, winner=Color.WHITE),
        evaluation_scale,
    ) == float(value_over_enum.VALUE_WHITE_WHEN_OVER_WHITE_WINS)
    assert _value_for_outcome(
        GameOutcome(kind=OutcomeKind.DRAW),
        evaluation_scale,
    ) == float(value_over_enum.VALUE_WHITE_WHEN_OVER_DRAW)
    assert _value_for_outcome(
        GameOutcome(kind=OutcomeKind.WIN, winner=Color.BLACK),
        evaluation_scale,
    ) == float(value_over_enum.VALUE_WHITE_WHEN_OVER_BLACK_WINS)
