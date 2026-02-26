"""Adapter between Chipiron board evaluators and Anemone's evaluator protocol."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from valanga import State

from chipiron.environments.chess.players.evaluators.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluator,
)

if TYPE_CHECKING:
    from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
        OverEventDetector,
    )
    from valanga.over_event import OverEvent

    from chipiron.environments.chess.types import ChessState


@dataclass
class MasterBoardOverEventDetector:
    """Bridge Chipiron's 'check_obvious_over_events' into Anemone's OverEventDetector protocol."""

    evaluator: MasterBoardEvaluator

    def check_obvious_over_events(
        self, state: State
    ) -> tuple["OverEvent | None", float | None]:
        """Check obvious over events."""
        return self.evaluator.check_obvious_over_events(cast("ChessState", state))


@dataclass
class MasterBoardEvaluatorAsAnemone(MasterStateEvaluator):
    """Adapter: Chipiron MasterBoardEvaluator -> Anemone MasterStateEvaluator."""

    inner: MasterBoardEvaluator
    over: "OverEventDetector"

    def value_white(self, state: State) -> float:
        """Value white."""
        return self.inner.value_white(cast("ChessState", state))
