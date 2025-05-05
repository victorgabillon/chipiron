import random

import chess

from chipiron.environments.chess.board import create_board_chi
from chipiron.environments.chess.board.factory import create_rust_board
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.scripts.chipiron_args import ImplementationArgs


def test_universal_behavior() -> None:
    board_chi = create_board_chi(
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
        sort_legal_moves=True,
    )

    all_moves_keys_chi = board_chi.legal_moves.get_all()
    all_moves_uci_chi = [
        board_chi.get_uci_from_move_key(move_key=move_key)
        for move_key in all_moves_keys_chi
    ]
    print(all_moves_keys_chi, all_moves_uci_chi)

    board_rust = create_rust_board(
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN),
        sort_legal_moves=True,
    )

    all_moves_keys_rust = board_rust.legal_moves.get_all()
    all_moves_uci_rust = [
        board_rust.get_uci_from_move_key(move_key=move_key)
        for move_key in all_moves_keys_rust
    ]
    print(all_moves_keys_rust, all_moves_uci_rust)

    assert all_moves_uci_rust == all_moves_uci_chi

    board_chi.play_move_key(move=all_moves_keys_chi[0])
    board_rust.play_move_key(move=all_moves_keys_rust[0])

    assert board_chi.fast_representation_ == board_rust.fast_representation_

    # redo the check after the move
    all_moves_keys_chi = board_chi.legal_moves.get_all()
    all_moves_uci_chi = [
        board_chi.get_uci_from_move_key(move_key=move_key)
        for move_key in all_moves_keys_chi
    ]
    print(all_moves_keys_chi, all_moves_uci_chi)

    all_moves_keys_rust = board_rust.legal_moves.get_all()
    all_moves_uci_rust = [
        board_rust.get_uci_from_move_key(move_key=move_key)
        for move_key in all_moves_keys_rust
    ]
    print(all_moves_keys_rust, all_moves_uci_rust)

    assert all_moves_uci_rust == all_moves_uci_chi

    random_generator_rust: random.Random = random.Random(0)
    player_rust: Player = create_chipiron_player(
        implementation_args=ImplementationArgs(use_rust_boards=True),
        universal_behavior=True,
        random_generator=random_generator_rust,
    )

    move_reco_rust: MoveRecommendation = player_rust.select_move(
        fen_plus_history=board_rust.into_fen_plus_history(), seed_int=0
    )

    random_generator_chi: random.Random = random.Random(0)
    player_chi: Player = create_chipiron_player(
        implementation_args=ImplementationArgs(use_rust_boards=False),
        universal_behavior=True,
        random_generator=random_generator_chi,
    )

    move_reco_chi: MoveRecommendation = player_chi.select_move(
        fen_plus_history=board_chi.into_fen_plus_history(), seed_int=0
    )

    assert move_reco_chi.evaluation == move_reco_rust.evaluation

    chi_move_uci = move_reco_chi.move
    chi_move_rust = move_reco_rust.move

    assert chi_move_uci == chi_move_rust


if __name__ == "__main__":
    test_universal_behavior()
