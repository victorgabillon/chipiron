import chess

from chipiron.environments.chess.board import create_board
from chipiron.players.factory import create_chipiron_player


def test_check():
    board = create_board(fen='1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5')

    player = create_chipiron_player(depth=1)

    move_reco = player.select_move(board=board, seed_=0)

    print('move', move_reco.move)
    assert (move_reco.move == chess.Move.from_uci('g6h7'))
