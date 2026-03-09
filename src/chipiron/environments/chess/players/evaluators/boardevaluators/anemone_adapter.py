"""Adapter between Chipiron board evaluators and Anemone's evaluator protocol."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from anemone.node_evaluation.node_direct_evaluation.protocols import MasterStateValueEvaluator
from valanga import State
from valanga.evaluations import Value

from chipiron.environments.chess.players.evaluators.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluator,
)

if TYPE_CHECKING:
    from anemone.node_evaluation.node_direct_evaluation.protocols import (
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
class MasterBoardEvaluatorAsAnemone(MasterStateValueEvaluator):
    """Adapter: Chipiron MasterBoardEvaluator -> Anemone MasterStateValueEvaluator."""

    inner: MasterBoardEvaluator
    over: "OverEventDetector"

    def evaluate(self, state: State) -> Value:
        """Value white."""
        return self.inner.value_white(cast("ChessState", state))
