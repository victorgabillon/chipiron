import sys
from src.players.RandomPlayer import RandomPlayer
from src.players.treevaluebuilders.create_tree_and_value import create_tree_and_value_builders
from src.players.human import Human
from src.players.stockfish import Stockfish


def create_player(arg, chess_simulator, syzygy):
    if arg['type'] == 'RandomPlayer':
        player = RandomPlayer()
    elif arg['type'] == 'TreeAndValue':
        player = create_tree_and_value_builders(arg, chess_simulator, syzygy)
    elif arg['type'] == 'Human':
        player = Human(arg, chess_simulator)
    elif arg['type'] == 'Stockfish':
        player = Stockfish(arg, chess_simulator)
    else:
        sys.exit('player creator: can not find ' + arg['type'])

    player.player_name = arg['name']
    player.syzygy_play = arg['syzygy_play']
    if player.syzygy_play:
        player.syzygy_player = syzygy
    player.print_info()
    return player
