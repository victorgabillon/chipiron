from typing import Any

from .factory import RepresentationFactory
from .rep_364 import create_from_board as create_from_board_364_no_bug
from .rep_364 import (
    create_from_board_and_from_parent as create_from_board_and_from_parent_364_no_bug,
)
from .rep_364_bug import create_from_board as create_from_board_364_bug
from .rep_364_bug import (
    create_from_board_and_from_parent as create_from_board_and_from_parent_364_bug,
)
from .TensorRepresentationType import InternalTensorRepresentationType


def create_board_representation_factory(
    board_representation_factory_type: InternalTensorRepresentationType,
) -> RepresentationFactory[Any] | None:
    """
    Create a board representation based on the given string.

    Args:
        board_representation_factory_str (str): The string representing the board representation.

    Returns:
        Representation364Factory | None: The created board representation object, or None if the string is 'no'.

    Raises:
        Exception: If the string is not '364'.

    """
    board_representation_factory: RepresentationFactory[Any] | None
    match board_representation_factory_type:
        case InternalTensorRepresentationType.BUG364:
            board_representation_factory = RepresentationFactory(
                create_from_board=create_from_board_364_bug,
                create_from_board_and_from_parent=create_from_board_and_from_parent_364_bug,
            )
        case InternalTensorRepresentationType.NOBUG364:
            board_representation_factory = RepresentationFactory(
                create_from_board=create_from_board_364_no_bug,
                create_from_board_and_from_parent=create_from_board_and_from_parent_364_no_bug,
            )
        case InternalTensorRepresentationType.NO:
            board_representation_factory = None
        case other:
            raise Exception(f"trying to create {other} in file {__name__}")

    return board_representation_factory
