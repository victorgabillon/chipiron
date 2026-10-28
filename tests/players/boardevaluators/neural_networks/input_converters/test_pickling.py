"""Check converter serialization required by native forkserver DataLoaders."""

import pickle

import pytest
import torch
from atomheart.games.chess.board import create_board

from chipiron.environments.chess.players.evaluators.boardevaluators.neural_networks.input_converters.board_to_input import (
    create_board_to_input,
)
from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    ModelInputRepresentationType,
)


@pytest.mark.fast
@pytest.mark.parametrize(
    "representation",
    [
        ModelInputRepresentationType.BUG364,
        ModelInputRepresentationType.NOBUG364,
        ModelInputRepresentationType.PIECE_MAP,
        ModelInputRepresentationType.PIECE_DIFFERENCE,
        ModelInputRepresentationType.BOARD_PIECES_TWO_SIDES,
    ],
)
def test_converter_survives_worker_serialization(
    representation: ModelInputRepresentationType,
) -> None:
    """Round-trip every chess converter and preserve exact input tensors."""
    state = ChessState(board=create_board(use_rust_boards=False))
    converter = create_board_to_input(representation)
    restored = pickle.loads(pickle.dumps(converter))
    torch.testing.assert_close(restored(state), converter(state), rtol=0, atol=0)
