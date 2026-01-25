"""
This module contains a factory to create board representations.
"""

from atomheart.board import BoardModificationP
from valanga.representation_factory import RepresentationFactory

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.neural_networks.input_converters.board_representation import (
    Representation364,
)

from .ModelInputRepresentationType import InternalTensorRepresentationType
from .rep_364 import create_from_board as create_from_board_364_no_bug
from .rep_364 import (
    create_from_state_and_modifications as create_from_state_and_modifications_364_no_bug,
)
from .rep_364_bug import create_from_board as create_from_board_364_bug
from .rep_364_bug import (
    create_from_state_and_modifications as create_from_state_and_modifications_364_bug,
)


def create_board_representation_factory(
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> RepresentationFactory[ChessState, BoardModificationP, Representation364] | None:
    """
    Create a board representation based on the given string.

    Args:
        internal_tensor_representation_type (InternalTensorRepresentationType): The string representing the board representation.

    Returns:
        RepresentationFactory[ChessState, BoardModificationP, Representation364] | None: The created board representation object, or None if the string is 'no'.

    Raises:
        Exception: If the string is not '364'.

    """
    board_representation_factory: (
        RepresentationFactory[ChessState, BoardModificationP, Representation364] | None
    )
    match internal_tensor_representation_type:
        case InternalTensorRepresentationType.BUG364:
            board_representation_factory = RepresentationFactory[
                ChessState, BoardModificationP, Representation364
            ](
                create_from_state=create_from_board_364_bug,
                create_from_state_and_modifications=create_from_state_and_modifications_364_bug,
            )
        case InternalTensorRepresentationType.NOBUG364:
            board_representation_factory = RepresentationFactory[
                ChessState, BoardModificationP, Representation364
            ](
                create_from_state=create_from_board_364_no_bug,
                create_from_state_and_modifications=create_from_state_and_modifications_364_no_bug,
            )
        case InternalTensorRepresentationType.NO:
            board_representation_factory = None
        case _:
            raise ValueError(
                f"trying to create {internal_tensor_representation_type} in file {__name__}"
            )

    return board_representation_factory
