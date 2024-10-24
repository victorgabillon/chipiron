import pickle
import random

import chess
import pytest

from chipiron.environments.chess.board import create_board, BoardChi
from chipiron.environments.chess.board.utils import FenPlusMoveHistory
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player, create_player_from_file
from chipiron.players.move_selector.move_selector import MoveRecommendation


@pytest.mark.parametrize(
    ('use_rusty_board'),
    (True, False)
)
def test_check_in_one(use_rusty_board: bool):
    board: BoardChi = create_board(
        fen_with_history=FenPlusMoveHistory(
            current_fen='1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5')
    )

    player: Player = create_chipiron_player(
        depth=1,
        use_rusty_board=use_rusty_board
    )

    move_reco: MoveRecommendation = player.select_move(board=board, seed_int=0)

    assert (move_reco.move == chess.Move.from_uci('g6h7') or move_reco.move == chess.Move.from_uci('g6g7'))


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
        board = create_board(fen_with_history=FenPlusMoveHistory(current_fen=fen))
        player = create_player_from_file(
            player_args_file='UniformDepth.yaml',
            random_generator=random_generator,
            use_rusty_board=use_rusty_board
        )
        move_reco = player.select_move(board=board, seed_int=0)
        assert (move_reco.move == moves[0])


test_check_in_one(use_rusty_board=True)
test_check_in_one(use_rusty_board=False)

test_check_in_two(use_rusty_board=True)
test_check_in_two(use_rusty_board=False)
