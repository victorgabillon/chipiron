"""
This module provides a class for converting a chess board into a tensor representation using a 364-dimensional input.

Classes:
- Representation364BTI: Converts a chess board into a tensor representation.

"""

from typing import TYPE_CHECKING

import torch
from valanga import State
from valanga.representation_factory import RepresentationFactory

if TYPE_CHECKING:
    from valanga.represention_for_evaluation import ContentRepresentation


class RepresentationBTI[StateT: State, EvalIn, StateModT]:
    """
    Converts a chess board into a tensor representation using a 364-dimensional input.

    Methods:
    - __init__: Initializes the Representation364BTI object.
    - convert: Converts the chess board into a tensor representation.

    """

    def __init__(
        self, representation_factory: RepresentationFactory[StateT, EvalIn, StateModT]
    ) -> None:
        """
        Initializes the Representation364BTI object.

        Parameters:
        - representation_factory (RepresentationFactory[StateT, EvalIn, StateModT]): The factory object for creating the board representation.

        """
        self.representation_factory = representation_factory

    def convert(self, state: StateT) -> torch.Tensor:
        """
        Converts the chess board into a tensor representation.

        Parameters:
        - board (BoardChi): The chess board to convert.

        Returns:
        - tensor (torch.Tensor): The tensor representation of the chess board.

        """
        representation: ContentRepresentation[StateT, EvalIn] = (
            self.representation_factory.create_from_state(state=state)
        )
        tensor: torch.Tensor = representation.get_evaluator_input(state=state)
        return tensor.float()
