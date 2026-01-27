"""Adapter between Chipiron board evaluators and Anemone's evaluator protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.master_board_evaluator import MasterBoardEvaluator
from chipiron.players.oracles import TerminalOracle

if TYPE_CHECKING:
    from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
        OverEventDetector,
    )
    from valanga.over_event import OverEvent


@dataclass(frozen=True)
class TerminalOracleOverDetector:
    """Bridge a terminal oracle into an Anemone-style over detector."""

    terminal_oracle: TerminalOracle[ChessState] | None

    def __call__(self, state: ChessState) -> OverEvent | None:
        if self.terminal_oracle is None:
            return None
        if self.terminal_oracle.supports(state):
            return self.terminal_oracle.over_event(state)
        return None


@dataclass
class MasterBoardEvaluatorAsAnemone(MasterStateEvaluator):
    """Adapter: Chipiron MasterBoardEvaluator -> Anemone MasterStateEvaluator."""

    inner: MasterBoardEvaluator
    over: "OverEventDetector"

    def value_white(self, state: ChessState) -> float:
        return self.inner.value_white(state)
