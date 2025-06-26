"""
Module for the BoardToInput protocol and BoardToInputFunction protocol.
"""

from typing import Any, Protocol, runtime_checkable

import torch

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.board import IBoard
from chipiron.players.boardevaluators.neural_networks.board_to_tensor import (
    transform_board_pieces_one_side,
    transform_board_pieces_two_sides,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_transformer_input import (
    build_transformer_input,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
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
from chipiron.players.boardevaluators.neural_networks.models.transformer_one import (
    TransformerArgs,
)


@runtime_checkable
class BoardToInputFunction(Protocol):
    """
    Protocol for a callable object that converts a chess board to a tensor input for a neural network.
    """

    def __call__(self, board: boards.IBoard) -> Any:
        """
        Converts the given chess board to a tensor input.

        Args:
            board (BoardChi): The chess board to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...


class BoardToInput(Protocol):
    """
    Protocol for converting a chess board to a tensor input for a neural network.
    """

    def convert(self, board: boards.IBoard) -> torch.Tensor:
        """
        Converts the given chess board to a tensor input.

        Args:
            board (BoardChi): The chess board to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...


def create_board_to_input_from_representation(
    internal_tensor_representation_type: InternalTensorRepresentationType,
) -> BoardToInputFunction:
    representation_factory: RepresentationFactory[Any] | None = (
        create_board_representation_factory(
            internal_tensor_representation_type=internal_tensor_representation_type
        )
    )
    assert representation_factory is not None
    board_to_input_convert: BoardToInput = RepresentationBTI(
        representation_factory=representation_factory
    )
    return board_to_input_convert.convert


def create_board_to_input(
    model_input_representation_type: ModelInputRepresentationType,
) -> BoardToInputFunction:
    board_to_input_convert: BoardToInputFunction

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

            def board_to_input_convert(board: IBoard) -> torch.Tensor:
                return build_transformer_input(
                    piece_map=board.piece_map(),
                    board_turn=board.turn,
                    transformer_args=TransformerArgs(),
                )

        case ModelInputRepresentationType.PIECE_DIFFERENCE:

            def board_to_input_convert(board: IBoard) -> torch.Tensor:
                return transform_board_pieces_one_side(board, False)

        case ModelInputRepresentationType.BOARD_PIECES_TWO_SIDES:

            def board_to_input_convert(board: IBoard) -> torch.Tensor:
                return transform_board_pieces_two_sides(board, False)

        case other:
            raise Exception(f"no matching case for {other} in {__name__}")
    return board_to_input_convert
