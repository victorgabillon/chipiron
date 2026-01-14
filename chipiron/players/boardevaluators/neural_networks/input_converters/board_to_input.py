"""
Module for the BoardToInput protocol and ContentToInputFunction protocol.
"""

from typing import TYPE_CHECKING

import torch
from atomheart.board import IBoard
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInput,
    ContentToInputFunction,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
)

from chipiron.players.boardevaluators.neural_networks.board_to_tensor import (
    transform_board_pieces_one_side,
    transform_board_pieces_two_sides,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_transformer_input import (
    build_transformer_input,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    InternalTensorRepresentationType,
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_364_bti import (
    RepresentationBTI,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)

if TYPE_CHECKING:
    from valanga.representation_factory import (
        RepresentationFactory,
    )


def create_board_to_input_from_representation(
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> ContentToInputFunction:
    """Creates a ContentToInputFunction from an InternalTensorRepresentationType.

    Args:
        internal_tensor_representation_type (InternalTensorRepresentationType): The internal tensor representation type to use.

    Returns:
        ContentToInputFunction: A function that converts a chess board to a tensor input.
    """

    representation_factory: RepresentationFactory | None = (
        create_board_representation_factory(
            internal_tensor_representation_type=internal_tensor_representation_type
        )
    )
    assert representation_factory is not None
    board_to_input_convert: ContentToInput = RepresentationBTI(
        representation_factory=representation_factory
    )
    return board_to_input_convert.convert


def create_board_to_input(
    model_input_representation_type: ModelInputRepresentationType,
) -> ContentToInputFunction:
    """Creates a ContentToInputFunction from a ModelInputRepresentationType.

    Args:
        model_input_representation_type (ModelInputRepresentationType): The model input representation type to use.

    Raises:
        Exception: If the model input representation type is not supported.

    Returns:
        ContentToInputFunction: A function that converts a chess board to a tensor input.
    """

    board_to_input_convert: ContentToInputFunction

    match model_input_representation_type:
        case ModelInputRepresentationType.BUG364:
            board_to_input_convert = create_board_to_input_from_representation(
                internal_tensor_representation_type=InternalTensorRepresentationType.BUG364
            )
        case ModelInputRepresentationType.NOBUG364:
            board_to_input_convert = create_board_to_input_from_representation(
                internal_tensor_representation_type=InternalTensorRepresentationType.NOBUG364
            )

        case ModelInputRepresentationType.PIECE_MAP:

            def board_to_input_convert_transformer(board: IBoard) -> torch.Tensor:
                return build_transformer_input(
                    piece_map=board.piece_map(),
                    board_turn=board.turn,
                    transformer_args=TransformerArgs(),
                )

            board_to_input_convert = board_to_input_convert_transformer

        case ModelInputRepresentationType.PIECE_DIFFERENCE:

            def board_to_input_convert_one_side(board: IBoard) -> torch.Tensor:
                return transform_board_pieces_one_side(board, False)

            board_to_input_convert = board_to_input_convert_one_side

        case ModelInputRepresentationType.BOARD_PIECES_TWO_SIDES:

            def board_to_input_convert_two_sides(board: IBoard) -> torch.Tensor:
                return transform_board_pieces_two_sides(board, False)

            board_to_input_convert = board_to_input_convert_two_sides

        case _:
            raise Exception(
                f"no matching case for {model_input_representation_type} in {__name__}"
            )

    return board_to_input_convert
