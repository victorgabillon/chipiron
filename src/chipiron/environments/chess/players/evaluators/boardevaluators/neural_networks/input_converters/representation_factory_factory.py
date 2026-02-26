"""Document the module contains a factory to create board representations."""

import torch
from atomheart.games.chess.board import BoardModificationP
from valanga.representation_factory import RepresentationFactory

from chipiron.environments.chess.types import ChessState

from .model_input_representation_type import InternalTensorRepresentationType
from .rep_364 import create_from_board as create_from_board_364_no_bug
from .rep_364 import (
    create_from_state_and_modifications as create_from_state_and_modifications_364_no_bug,
)
from .rep_364_bug import create_from_board as create_from_board_364_bug
from .rep_364_bug import (
    create_from_state_and_modifications as create_from_state_and_modifications_364_bug,
)


class RepresentationFactoryError(ValueError):
    """Raised when a representation factory cannot be created."""

    def __init__(self, representation: InternalTensorRepresentationType) -> None:
        """Initialize the error with the unsupported representation."""
        super().__init__(f"trying to create {representation} in file {__name__}")


def create_board_representation_factory(
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> RepresentationFactory[ChessState, torch.Tensor, BoardModificationP] | None:
    """Create a board representation based on the given string.

    Args:
        internal_tensor_representation_type (InternalTensorRepresentationType): The string representing the board representation.

    Returns:
        RepresentationFactory[ChessState, torch.Tensor, BoardModificationP] | None: The created board representation object, or None if the string is 'no'.

    Raises:
        Exception: If the string is not '364'.

    """
    board_representation_factory: (
        RepresentationFactory[ChessState, torch.Tensor, BoardModificationP] | None
    )
    match internal_tensor_representation_type:
        case InternalTensorRepresentationType.BUG364:
            board_representation_factory = RepresentationFactory[
                ChessState, torch.Tensor, BoardModificationP
            ](
                create_from_state=create_from_board_364_bug,
                create_from_state_and_modifications=create_from_state_and_modifications_364_bug,
            )
        case InternalTensorRepresentationType.NOBUG364:
            board_representation_factory = RepresentationFactory[
                ChessState, torch.Tensor, BoardModificationP
            ](
                create_from_state=create_from_board_364_no_bug,
                create_from_state_and_modifications=create_from_state_and_modifications_364_no_bug,
            )
        case InternalTensorRepresentationType.NO:
            board_representation_factory = None
        case _:
            raise RepresentationFactoryError(internal_tensor_representation_type)

    return board_representation_factory
