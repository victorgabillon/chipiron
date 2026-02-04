"""Test for the board representation."""

from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

import chess
import pytest
import torch
from atomheart.board import (
    BoardModificationP,
    IBoard,
    create_board,
)
from atomheart.board.utils import FenPlusHistory, bitboard_rotate

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    InternalTensorRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)

if TYPE_CHECKING:
    from atomheart.move.imove import MoveKey
    from valanga.represention_for_evaluation import ContentRepresentation


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
@pytest.mark.parametrize(("board_representation_factory_type"), ["364_no_bug"])
def test_representation(
    use_rust_boards: bool,
    board_representation_factory_type: InternalTensorRepresentationType,
) -> None:
    """Test the board representation factory.

    Args:
        use_rust_boards (bool): Whether to use Rust boards.
        board_representation_factory_type (InternalTensorRepresentationType): The type of board representation factory to use.

    """
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )
    state = ChessState(board=board)

    representation_factory = create_board_representation_factory(
        internal_tensor_representation_type=board_representation_factory_type
    )
    assert representation_factory is not None

    parent_node_board_representation: ContentRepresentation[
        ChessState, torch.Tensor
    ] = representation_factory.create_from_state(state=state)

    all_moves_keys_chi: Sequence[MoveKey] = board.branch_keys.get_all()
    board_modification: BoardModificationP | None = board.play_move_key(
        move=all_moves_keys_chi[0]
    )
    state_after_move = ChessState(board=board)

    assert board_modification is not None
    direct_rep: ContentRepresentation[ChessState, torch.Tensor] = (
        representation_factory.create_from_state(state=state_after_move)
    )
    rep_from_parents: ContentRepresentation[ChessState, torch.Tensor] = (
        representation_factory.create_from_state_and_modifications(
            state=state_after_move,
            state_modifications=board_modification,
            previous_state_representation=parent_node_board_representation,
        )
    )

    assert direct_rep == rep_from_parents


@pytest.mark.parametrize(("board_representation_factory_type"), ["364_no_bug"])
@pytest.mark.parametrize(("use_rust_boards"), (True, False))
@pytest.mark.parametrize(("use_board_modification"), (True, False))
def test_representation364(
    use_rust_boards: bool,
    board_representation_factory_type: InternalTensorRepresentationType,
    use_board_modification: bool,
) -> None:
    # 'rnb2bnr/ppp2ppp/2k3q1/8/8/1Q3K2/PPP2PPP/RNB2BNR w - - 0 1'
    # '8/ppp2ppp/2k3q1/8/8/1Q3K2/PPP2PPP/8 w - - 0 1'
    """Test representation364."""
    board_one: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=use_board_modification,
        fen_with_history=FenPlusHistory(
            current_fen="rnb2bnr/ppp2ppp/2k3q1/8/8/1Q3K2/PPP2PPP/RNB2BNR w - - 0 1"
        ),
    )

    board_two: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=use_board_modification,
        fen_with_history=FenPlusHistory(
            current_fen="rnb2bnr/ppp2ppp/2k3q1/8/8/1Q3K2/PPP2PPP/RNB2BNR b - - 0 1"
        ),
    )
    state_one = ChessState(board=board_one)
    state_two = ChessState(board=board_two)

    representation_factory = create_board_representation_factory(
        internal_tensor_representation_type=board_representation_factory_type
    )
    assert representation_factory is not None

    assert board_one.occupied == bitboard_rotate(board_two.occupied)

    board_representation_one_copy: ContentRepresentation[ChessState, torch.Tensor] = (
        deepcopy(representation_factory.create_from_state(state=state_one))
    )
    board_representation_two_copy: ContentRepresentation[ChessState, torch.Tensor] = (
        deepcopy(representation_factory.create_from_state(state=state_two))
    )

    board_representation_one: ContentRepresentation[ChessState, torch.Tensor]
    board_representation_two: ContentRepresentation[ChessState, torch.Tensor]
    if use_board_modification:
        board_modification_one: BoardModificationP | None = board_one.play_move_uci(
            move_uci="a2a3"
        )
        board_modification_two: BoardModificationP | None = board_two.play_move_uci(
            move_uci="h7h6"
        )
        state_one = ChessState(board=board_one)
        state_two = ChessState(board=board_two)
        assert board_modification_one is not None
        assert board_modification_two is not None
        assert board_one.occupied == bitboard_rotate(board_two.occupied)
        board_representation_one = (
            representation_factory.create_from_state_and_modifications(
                state=state_one,
                state_modifications=board_modification_one,
                previous_state_representation=board_representation_one_copy,
            )
        )
        board_representation_two = (
            representation_factory.create_from_state_and_modifications(
                state=state_two,
                state_modifications=board_modification_two,
                previous_state_representation=board_representation_two_copy,
            )
        )
    else:
        board_representation_one = representation_factory.create_from_state(
            state=state_one
        )
        board_representation_two = representation_factory.create_from_state(
            state=state_two
        )

    inputs_one = board_representation_one.get_evaluator_input(state_one)

    inputs_two = board_representation_two.get_evaluator_input(state_two)

    assert torch.equal(inputs_one, inputs_two)


if __name__ == "__main__":
    use_rusty_board: bool
    board_representation_factory_type: InternalTensorRepresentationType
    use_board_modification: bool
    for use_rusty_board in [True, False]:
        for board_representation_factory_type in [
            InternalTensorRepresentationType.NOBUG364
        ]:
            test_representation(
                use_rust_boards=use_rusty_board,
                board_representation_factory_type=board_representation_factory_type,
            )
            for use_board_modification in [True, False]:
                test_representation364(
                    use_rust_boards=use_rusty_board,
                    board_representation_factory_type=board_representation_factory_type,
                    use_board_modification=use_board_modification,
                )
