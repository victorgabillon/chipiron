import chess
import pickle

from chipiron.environments.chess.board import create_board
from chipiron.players.factory import create_chipiron_player, create_player_from_file
import random


def test_check():
    board = create_board(fen='1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5')

    player = create_chipiron_player(depth=1)

    move_reco = player.select_move(board=board, seed_=0)

    print('move', move_reco.move)
    assert (move_reco.move == chess.Move.from_uci('g6h7'))


def test_check_in_two():
    dict_fen_move: dict[str, list[chess.Move]] = {}
    with open('data/puzzles/mate_in_2_db.pickle', 'rb') as file:
        dict_fen_move = pickle.load(file=file)

    assert dict_fen_move

    random_generator: random.Random = random.Random()

    fen: str
    moves: list[chess.Move]
    for fen, moves in dict_fen_move.items():
        board = create_board(fen=fen)

        #player = create_chipiron_player(depth=1)
        print('FEEEN',fen)
        player = create_player_from_file(
            player_args_file='UniformDepth.yaml',
            random_generator=random_generator
        )
        move_reco = player.select_move(board=board, seed_=0)

        print('MMMMMMMMM move', move_reco.move, moves[0], move_reco.evaluation)
        assert (move_reco.move == moves[0])


test_check_in_two()
