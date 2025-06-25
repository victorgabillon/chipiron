from copy import deepcopy
from typing import Any

import chess
import pytest
import torch

from chipiron.environments.chess.board import BoardModificationP, IBoard, create_board
from chipiron.environments.chess.board.utils import FenPlusHistory, bitboard_rotate
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.neural_networks.input_converters.board_representation import (
    BoardRepresentation,
    Representation364,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    InternalTensorRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)


@pytest.mark.parametrize(("use_rust_boards"), (True, False))
@pytest.mark.parametrize(("board_representation_factory_type"), ["364_no_bug"])
def test_representation(
    use_rust_boards: bool,
    board_representation_factory_type: InternalTensorRepresentationType,
) -> None:
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
    )

    representation_factory: RepresentationFactory[Any] | None = (
        create_board_representation_factory(
            internal_tensor_representation_type=board_representation_factory_type
        )
    )
    assert representation_factory is not None

    parent_node_board_representation: Representation364 = (
        representation_factory.create_from_board(board=board)
    )

    all_moves_keys_chi: list[moveKey] = board.legal_moves.get_all()
    board_modification: BoardModificationP | None = board.play_move_key(
        move=all_moves_keys_chi[0]
    )

    assert board_modification is not None
    direct_rep: BoardRepresentation = representation_factory.create_from_board(
        board=board
    )
    rep_from_parents: BoardRepresentation = (
        representation_factory.create_from_board_and_from_parent(
            board=board,
            board_modifications=board_modification,
            parent_node_board_representation=parent_node_board_representation,
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

    representation_factory: RepresentationFactory[Any] | None = (
        create_board_representation_factory(
            internal_tensor_representation_type=board_representation_factory_type
        )
    )
    assert representation_factory is not None

    assert board_one.occupied == bitboard_rotate(board_two.occupied)

    board_representation_one_copy: Representation364 = deepcopy(
        representation_factory.create_from_board(board=board_one)
    )
    board_representation_two_copy: Representation364 = deepcopy(
        representation_factory.create_from_board(board=board_two)
    )

    board_representation_one: Representation364
    board_representation_two: Representation364
    if use_board_modification:
        board_modification_one: BoardModificationP | None = board_one.play_move_uci(
            move_uci="a2a3"
        )
        board_modification_two: BoardModificationP | None = board_two.play_move_uci(
            move_uci="h7h6"
        )
        assert board_modification_one is not None
        assert board_modification_two is not None
        assert board_one.occupied == bitboard_rotate(board_two.occupied)
        board_representation_one = (
            representation_factory.create_from_board_and_from_parent(
                board=board_one,
                board_modifications=board_modification_one,
                parent_node_board_representation=board_representation_one_copy,
            )
        )
        board_representation_two = (
            representation_factory.create_from_board_and_from_parent(
                board=board_two,
                board_modifications=board_modification_two,
                parent_node_board_representation=board_representation_two_copy,
            )
        )
    else:
        board_representation_one = representation_factory.create_from_board(
            board=board_one
        )
        board_representation_two = representation_factory.create_from_board(
            board=board_two
        )

    inputs_one = board_representation_one.get_evaluator_input(board_one.turn)

    inputs_two = board_representation_two.get_evaluator_input(board_two.turn)

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
