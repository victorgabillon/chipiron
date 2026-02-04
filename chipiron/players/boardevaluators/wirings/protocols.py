"""Protocols for wiring game-specific evaluators."""

from typing import Protocol, TypeVar

from chipiron.players.boardevaluators.board_evaluator import StateEvaluator

StateT_contra = TypeVar("StateT_contra", contravariant=True)


class EvaluatorWiring(Protocol[StateT_contra]):
    """Protocol for wiring a state evaluator and optional oracle."""

    def build_chi(self) -> StateEvaluator[StateT_contra]:
        """Build the main evaluator."""
        ...

    def build_oracle(self) -> StateEvaluator[StateT_contra] | None:
        """Build the optional oracle evaluator."""
        ...
