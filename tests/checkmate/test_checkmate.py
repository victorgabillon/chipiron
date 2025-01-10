import pickle
import random

import chess
import pytest

from chipiron.environments.chess.board import create_board, IBoard
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.environments.chess.move import moveUci
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player, create_player_from_file
from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.players.player_ids import PlayerConfigFile


@pytest.mark.parametrize(
    ('use_rusty_board'),
    (True, False)
)
def test_check_in_one(use_rusty_board: bool):
    board: IBoard = create_board(
        use_rust_boards=use_rusty_board,
        fen_with_history=FenPlusHistory(
            current_fen='1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5')
    )

    random_generator: random.Random = random.Random(0)

    player: Player = create_chipiron_player(
        depth=1,
        use_rusty_board=use_rusty_board,
        random_generator=random_generator
    )

    move_reco: MoveRecommendation = player.select_move(
        board=board,
        seed_int=0
    )
    move_reco_uci: moveUci = board.get_uci_from_move_key(move_key=move_reco.move)

    assert (move_reco_uci == 'g6h7' or move_reco_uci == 'g6g7')


@pytest.mark.parametrize(
    ('use_rusty_board'),
    (True, False)
)
def test_check_in_two(use_rusty_board: bool):
    with open('data/puzzles/mate_in_2_db_small.pickle', 'rb') as file:
        dict_fen_move = pickle.load(file=file)

    assert dict_fen_move

    random_generator: random.Random = random.Random()

    fen: str
    moves: list[chess.Move]
    print(f'Testing  check in two on {len(dict_fen_move)} boards.')
    for fen, moves in dict_fen_move.items():
        print('fen', fen)
        board: IBoard = create_board(
            use_rust_boards=use_rusty_board,
            fen_with_history=FenPlusHistory(current_fen=fen)
        )
        player = create_player_from_file(
            player_args_file=PlayerConfigFile.UniformDepth3,
            random_generator=random_generator,
            use_rusty_board=use_rusty_board
        )
        move_reco: MoveRecommendation = player.select_move(
            board=board,
            seed_int=0
        )
        move_reco_uci: moveUci = board.get_uci_from_move_key(move_key=move_reco.move)

        assert (move_reco_uci == moves[0].uci())


if __name__ == '__main__':
    test_check_in_one(use_rusty_board=True)
    test_check_in_one(use_rusty_board=False)

    test_check_in_two(use_rusty_board=True)
    test_check_in_two(use_rusty_board=False)
