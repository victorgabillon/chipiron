import chess
import pytest

from chipiron.environments.chess.board import IBoard, create_board, BoardModificationP
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.environments.chess.move.imove import moveKey
from chipiron.players.boardevaluators.neural_networks.input_converters.board_representation import Representation364
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import Representation364Factory, \
    node_to_tensors_pieces_square_from_parent


@pytest.mark.parametrize(
    ('use_rust_boards'),
    (True, False)
)
def test_representation(use_rust_boards: bool) -> None:
    board: IBoard = create_board(
        use_rust_boards=use_rust_boards,
        use_board_modification=True,
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN)
    )

    representation_364_factory: Representation364Factory = Representation364Factory()

    parent_node_board_representation: Representation364 = representation_364_factory.create_from_board(board=board)

    all_moves_keys_chi: list[moveKey] = board.legal_moves.get_all()
    board_modification: BoardModificationP | None = board.play_move_key(move=all_moves_keys_chi[0])

    assert (board_modification is not None)
    direct_rep: Representation364 = representation_364_factory.create_from_board(board=board)
    rep_from_parents: Representation364 = node_to_tensors_pieces_square_from_parent(
        board=board,
        board_modifications=board_modification,
        parent_node_board_representation=parent_node_board_representation
    )

    assert (direct_rep == rep_from_parents)


if __name__ == '__main__':
    use_rusty_board: bool
    for use_rusty_board in [True, False]:
        test_representation(use_rust_boards=use_rusty_board)
