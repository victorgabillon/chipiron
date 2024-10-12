import pickle
import random

import chess

from chipiron.environments.chess.board import create_board, BoardChi
from chipiron.environments.chess.board.utils import FenPlusMoveHistory
from chipiron.players import Player
from chipiron.players.factory import create_chipiron_player, create_player_from_file
from chipiron.players.move_selector.move_selector import MoveRecommendation

def test_check():
    board: BoardChi = create_board(fen_with_history=FenPlusMoveHistory(
        current_fen='1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5')
    )

    player: Player = create_chipiron_player(depth=1)

    move_reco: MoveRecommendation = player.select_move(board=board, seed_int=0)

    print('move', move_reco.move)
    assert (move_reco.move == chess.Move.from_uci('g6h7'))


def test_check_in_two():
    with open('data/puzzles/mate_in_2_db_small.pickle', 'rb') as file:
        dict_fen_move = pickle.load(file=file)

    assert dict_fen_move

    random_generator: random.Random = random.Random()

    fen: str
    moves: list[chess.Move]
    for fen, moves in dict_fen_move.items():
        board = create_board(fen_with_history=FenPlusMoveHistory(current_fen=fen))
        player = create_player_from_file(
            player_args_file='UniformDepth.yaml',
            random_generator=random_generator
        )
        move_reco = player.select_move(board=board, seed_int=0)
        assert (move_reco.move == moves[0])


test_check_in_two()
