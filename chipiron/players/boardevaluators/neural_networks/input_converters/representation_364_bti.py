"""
This module provides a class for converting a chess board into a representation input.
"""

from __future__ import annotations

from typing import Callable

from valanga import State
from valanga.representation_factory import RepresentationFactory
from valanga.represention_for_evaluation import ContentRepresentation


class RepresentationBTI[StateT: State, EvalIn, ModsT]:
    """Converts a content state into a representation input."""

    def __init__(
        self,
        representation_factory: RepresentationFactory[
            StateT, ContentRepresentation[StateT, EvalIn], ModsT
        ],
        postprocess: Callable[[EvalIn], EvalIn] | None = None,
    ) -> None:
        """Initialize the RepresentationBTI instance."""
        self.representation_factory = representation_factory
        self._postprocess = postprocess

    def convert(self, state: StateT) -> EvalIn:
        """Convert a state into an evaluator input."""
        representation: ContentRepresentation[StateT, EvalIn] = (
            self.representation_factory.create_from_state(state=state)
        )
        output: EvalIn = representation.get_evaluator_input(state=state)
        return self._postprocess(output) if self._postprocess else output
