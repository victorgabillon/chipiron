"""
Protocols for wiring game-specific evaluators.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

StateT = TypeVar("StateT")


class EvaluatorWiring(Protocol[StateT]):
    def build_chi(self) -> StateEvaluator[StateT]: ...

    def build_oracle(self) -> StateEvaluator[StateT] | None: ...
