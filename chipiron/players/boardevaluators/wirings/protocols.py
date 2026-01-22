"""
Protocols for wiring game-specific evaluators.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

StateT_contra = TypeVar("StateT_contra", contravariant=True)


class EvaluatorWiring(Protocol[StateT_contra]):
    def build_chi(self) -> StateEvaluator[StateT_contra]: ...

    def build_oracle(self) -> StateEvaluator[StateT_contra] | None: ...
